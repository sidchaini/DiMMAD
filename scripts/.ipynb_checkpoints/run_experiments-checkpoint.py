from sklearn.base import clone
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mycolors = {
 'DiMMAD (min-med)': "#FFB000",
 'DiMMAD (med-med)': "#785EF0",
 'iForest': "#648FFF",
 'LOF': "#FE6100",
 'OC-SVM': "#DC267F",
 'Autoencoder': "#5F3331",
 'MCSVDD': "#6BC59D"
}

mylinestyles = {
 'DiMMAD (min-med)': 'solid',
 'DiMMAD (med-med)': 'solid',
 'iForest': 'dotted',
 'LOF': 'dashdot',
 'OC-SVM': 'dashed',
 'Autoencoder': (0, (3, 1, 1, 1)),
 'MCSVDD': (0, (5, 1))
}



def run_experiment(
    N_RUNS,
    models_to_test,
    full_inlier_df,
    full_outlier_df,
    new_knowns,
    features_to_use,
    seed_val,
):
    smallest_class_count = full_inlier_df["class"].value_counts().min()
    outlier_counts = full_outlier_df["class"].value_counts()
    num_outlier_classes = len(outlier_counts)
    all_run_results = []

    for i in tqdm(range(N_RUNS), desc="Running Experiments"):
        current_seed = seed_val + i

        # Balance inlier classes
        keep_indices = (
            full_inlier_df.groupby("class")
            .sample(smallest_class_count, random_state=current_seed)
            .index
        )
        inlier_df = full_inlier_df[full_inlier_df.index.isin(keep_indices)]
        thrown_inlier_df = inlier_df[~inlier_df.index.isin(keep_indices)]

        # Split balanced inliers into train/test
        inlier_train_df, inlier_test_df = train_test_split(
            inlier_df,
            test_size=0.5,
            random_state=current_seed,
            stratify=inlier_df["class"],
        )

        tot_inliers_in_test = inlier_test_df["class"].value_counts().sum()
        frac_outliers_in_test = 0.9
        tot_outliers_in_test = int(frac_outliers_in_test * tot_inliers_in_test)

        samples_to_take = {}
        active_classes = list(outlier_counts.index)
        remaining_to_sample = tot_outliers_in_test
        available_counts = outlier_counts.copy()

        while True:
            if not active_classes:
                break

            ideal_samples_per_class = remaining_to_sample / len(active_classes)

            # Find classes that have fewer members than the ideal number
            small_classes = [
                c
                for c in active_classes
                if available_counts[c] < ideal_samples_per_class
            ]

            if not small_classes:
                # No more small classes, proceed to final distribution
                break

            # For each small class, take all available members
            for class_name in small_classes:
                num_to_take = available_counts[class_name]
                samples_to_take[class_name] = num_to_take

                remaining_to_sample -= num_to_take
                active_classes.remove(class_name)

        # Distribute the rest of the samples among the larger classes
        if active_classes:
            base_samples = remaining_to_sample // len(active_classes)
            remainder = remaining_to_sample % len(active_classes)

            for i, class_name in enumerate(active_classes):
                if i < remainder:
                    samples_to_take[class_name] = base_samples + 1
                else:
                    samples_to_take[class_name] = base_samples

        # Perform the actual sampling
        sampled_dfs = []
        for class_name, n_samples in samples_to_take.items():
            class_df = full_outlier_df[full_outlier_df["class"] == class_name].sample(
                n=n_samples, random_state=current_seed
            )
            sampled_dfs.append(class_df)

        # Combine all the sampled data into a final DataFrame
        final_sampled_df = pd.concat(sampled_dfs)

        X_train_df = inlier_train_df[features_to_use]
        y_train_df = inlier_train_df["class"]

        X_test_df = pd.concat(
            [inlier_test_df[features_to_use], final_sampled_df[features_to_use]]
        ).sample(frac=1, random_state=current_seed)
        y_test_df = pd.concat([inlier_test_df["class"], final_sampled_df["class"]]).loc[
            X_test_df.index
        ]

        results_df = pd.DataFrame(index=X_test_df.index)
        results_df["class"] = y_test_df
        mask = results_df["class"].isin(new_knowns)
        results_df["status"] = np.where(mask, "normal", "anomalous")

        for model_name, model_template in models_to_test.items():
            model = clone(model_template)

            if "random_state" in model.get_params():
                model.set_params(random_state=current_seed)

            if "DiMMAD" in model_name or "MCSVDD" in model_name:
                model.fit(X_train_df.to_numpy(), y_train_df.to_numpy())
            else:
                model.fit(X_train_df.to_numpy())

            scores = model.decision_function(X_test_df.to_numpy())
            if model_name in ["iForest", "LOF", "OC-SVM"]:
                scores = -scores

            results_df[f"score_{model_name}"] = scores

        all_run_results.append(results_df)

    return all_run_results


def analysis_with_errors(models_to_test, all_fold_results, budget=300, metric="purity"):
    for model_name in tqdm(models_to_test.keys(), desc="Analyzing Models", leave=False):
        fold_curves = []
        max_anom_in_folds = [
            fold_res["status"].value_counts().get("anomalous", 0)
            for fold_res in all_fold_results
        ]

        budget = min(budget, min(max_anom_in_folds))
        top_ns = np.arange(1, budget + 1)

        for fold_res in all_fold_results:
            curve = []
            topresults_df = fold_res.sort_values(
                by=f"score_{model_name}", ascending=False
            )
            # print(topresults_df["status"].value_counts(normalize=True))

            for i in top_ns:
                top_candidates = topresults_df.head(i)

                if metric == "purity":
                    valcounts = top_candidates["status"].value_counts(normalize=True)
                    value = valcounts.get("anomalous", 0)
                elif metric == "diversity":
                    anomalous_found = top_candidates[
                        top_candidates["status"] == "anomalous"
                    ]
                    value = len(anomalous_found["class"].unique())
                else:
                    raise ValueError("Metric must be 'purity' or 'diversity'")
                curve.append(value)
            fold_curves.append(curve)

        # Convert list of curves to a 2D numpy array for easy stats
        fold_curves_arr = np.array(fold_curves)

        mean_curve = np.mean(fold_curves_arr, axis=0)
        std_curve = np.std(fold_curves_arr, axis=0)

        # Plotting
        lw = 3.25 if "dimmad".lower() in model_name.lower() else 1.75
        label_name = model_name.replace(r"DiMMAD", r"$DiMMAD$")
        try:
            mycolor = mycolors[model_name]
            myls = mylinestyles[model_name]
        except Exception as e:
            mycolor = None
            myls = None
        p = plt.plot(
            top_ns,
            mean_curve,
            label=model_name,
            linewidth=lw,
            color=mycolor,
            linestyle=myls,
        )
        plt.fill_between(
            top_ns,
            np.clip(mean_curve - 0.5 * std_curve, a_min=0, a_max=1),
            np.clip(mean_curve + 0.5 * std_curve, a_min=0, a_max=1),
            color=p[0].get_color(),
            alpha=0.1,
        )

    if metric == "purity":
        plt.ylabel("Mean Purity", fontsize=16)
    elif metric == "diversity":
        plt.ylabel("New Classes Discovered", fontsize=16)

    plt.xlabel("Follow-up Budget (No. of sources)", fontsize=16)
    plt.gca().tick_params(axis="both", which="major", labelsize=16)
    plt.gca().tick_params(axis="both", which="minor", labelsize=12)
    legend = plt.legend(loc="lower right", fontsize=12, framealpha=1)

    for text in legend.get_texts():
        if "dimmad" in text.get_text().lower():
            text.set_fontweight("bold")

    # plt.grid(False)
    plt.grid(True, which="both", linestyle=":", linewidth=0.5)
    return plt.gcf()
