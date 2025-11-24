import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
from tqdm.auto import tqdm
import distclassipy as dcpy

from mlxtend.feature_selection import (
    SequentialFeatureSelector,
    ExhaustiveFeatureSelector as EFS,
)

myname = "sid"


def get_2digits(num):
    """
    Function to get simplified string form of from abc * 10^(xyz) -> (a,x)
    Examples
    --------
    >>> get_2digits(547.123)
    '5 \\times 10^{2}'
    """
    scinum_ls = "{:.0e}".format(num).split("e+")
    if scinum_ls[1] == "00":
        label = r"{0}".format(scinum_ls[0])
    else:
        label = r"{0} \times 10^{{{1}}}".format(
            scinum_ls[0], scinum_ls[1].replace("0", "")
        )
    return label


def plot_two_chen_lightcurves(df1, df2):
    # Plot the lightcurves for the given object
    dflist = [df1, df2]
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    for i, ax in enumerate(axs):
        df = dflist[i]
        df_g = df[df["band"] == "g"].sort_values(by="HJD", ascending=True)
        df_r = df[df["band"] == "r"].sort_values(by="HJD", ascending=True)
        objid = df["SourceID"].iloc[0]

        ax.errorbar(df_g["HJD"], df_g["mag"], yerr=df_g["e_mag"], fmt="go", label="g")
        ax.errorbar(df_r["HJD"], df_r["mag"], yerr=df_r["e_mag"], fmt="ro", label="r")
        ax.invert_yaxis()
        ax.set_xlabel("HJD")
        ax.set_ylabel("Magnitude")
        ax.set_title("Lightcurve for SourceID " + str(objid))
        ax.legend()
    plt.show()


def get_individual_lc(lcs, objid):
    return lcs[(lcs["SourceID"] == objid)]


def local_alerce_data_cleaner(alerce_features_path, chen_features_path):
    """
    Note: This is a highly specific function built only for preprocessing and combining
    the features calculatef from ALeRCE's package lc_classifier along with chen's features,
    for a objects from Chen's catalog ONLY.

    For objects from different sources (e.g ZTF DRs), a new function will be needed.
    Albeit, a lot of code here can be copied over.
    Nevertheless,
    this function is for internal purposes only.
    """

    ### 1.1 Load alerce featues calculated in step 2 and prep
    alercefeatures_df = pd.read_csv(alerce_features_path)
    alercefeatures_df = alercefeatures_df.set_index("oid")
    ### 1.2 Load chen features from Chen subset downloaded in step 1 and prep them
    chenfeatures_df = pd.read_csv(chen_features_path)
    chenfeatures_df = chenfeatures_df.rename(
        columns={
            "SourceID": "oid",
            "Type": "class",
            "213-elta_min_g": "Delta_min_g",
            "219-elta_min_r": "Delta_min_r",
        }
    )
    chenfeatures_df = chenfeatures_df.set_index("oid")
    ### 1.3 Drop redundant and irrelevant columns
    ####### (refer siddharth/archived/removing irrelevant and redundant features.ipynb for why)
    alercefeatures_df = alercefeatures_df.drop(
        [
            # Irrelevant:
            "MHPS_non_zero_g",
            "MHPS_non_zero_r",
            # Redundant:
            "iqr_r",
            "iqr_g",
        ],
        axis=1,
    )

    chenfeatures_df = chenfeatures_df.drop(
        [
            # Irrelevant:
            "ID",
            "RAdeg",
            "DEdeg",
            "T_0",
            "Num_g",
            "Num_r",
            # Redundant:
            "Per",
            "Per_g",
            "Per_r",
            "rmag",
            "gmag",
            "Amp_g",
            "Amp_r",
            "phi21",
            "R21",
        ],
        axis=1,
    )
    ### 1.4 Drop all NA from everything - important to remove objects where ALeRCE feature extraction failed
    chenfeatures_df = chenfeatures_df.dropna()
    alercefeatures_df = alercefeatures_df.dropna()
    #### 1.4.1 Remove the same NA rows objects from both the dataframes
    com_idx = np.intersect1d(
        alercefeatures_df.index.to_numpy(), chenfeatures_df.index.to_numpy()
    )
    chenfeatures_df = chenfeatures_df.loc[com_idx]
    alercefeatures_df = alercefeatures_df.loc[com_idx]
    ### 1.5 Create a dataframe with just the "oid" and "class"
    class_df = chenfeatures_df[["class"]]
    #### 1.5.1 Remove classes from chenfeatures
    chenfeatures_df = chenfeatures_df.drop(["class"], axis=1)
    ### 1.6 Create a final combined feature dataset
    chenalerce_df = pd.concat([alercefeatures_df, chenfeatures_df], axis=1)
    print(f"Total of {alercefeatures_df.shape[1]} features in ALeRCE")
    print(f"Total of {chenfeatures_df.shape[1]} features in Chen")
    print("*" * 40)
    print(f"Total of {chenalerce_df.shape[1]} features in ALL")

    print("List of features being used:")
    print(
        "https://docs.google.com/spreadsheets/d/1EDzk51Nzk6jhGbZhimN3iQiRkba_NCRN6Uoxi6eTzoU/edit?usp=sharing"
    )

    # Standardising
    # chenalerce_df_meannorm = (chenalerce_df-chenalerce_df.mean())/chenalerce_df.std()

    # Normalising:
    # chenalerce_df_minmax = (chenalerce_df-chenalerce_df.min())/(chenalerce_df.max()-chenalerce_df.min())

    return alercefeatures_df, chenfeatures_df, chenalerce_df, class_df


def get_metric_name(metric):
    if callable(metric):
        metric_str = metric.__name__
    else:
        metric_str = metric
    return metric_str.title()


def load_best_features(sfs_df):
    """

    Choose n features for this metric such that n is the
    lowest number of features whose F1 score is within 1
    standard deviation of the maximum F1 score.

    Input:
    sfs_df: Sequential feature selection dataframe with objid as 'num_feats' as index.

    Returns:
    feats_idx: Index of the best features
    feats: Name of the best features
    """

    idx_maxscore = sfs_df["avg_score"].idxmax()
    max_score = sfs_df.loc[idx_maxscore, "avg_score"]
    max_score_std = sfs_df.loc[idx_maxscore, "std_dev"]

    sfs_df = sfs_df.loc[(sfs_df["avg_score"]) >= max_score - max_score_std]
    sfs_df = sfs_df.sort_index()

    feats_idx = list(ast.literal_eval(sfs_df.iloc[0]["feature_idx"]))
    feats = list(ast.literal_eval(sfs_df.iloc[0]["feature_names"]))

    return feats_idx, feats


def plot_cm(y_true, y_pred, annot_fmt="pn", label_strings=None):
    """
    annot_fmt: n is numeric, p is percentage and np is both no.(%)
    """
    if label_strings is None:
        label_strings = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=label_strings)

    fig, ax = plt.subplots()

    if annot_fmt == "n":
        df_cm = pd.DataFrame(cm, index=label_strings, columns=label_strings)
        sns.heatmap(
            df_cm, annot=True, cmap="Blues", square=True, fmt="d", ax=ax, cbar=False
        )  # fmt='.2%'
    elif annot_fmt == "p":
        cm = confusion_matrix(y_true, y_pred, normalize="true", labels=label_strings)
        df_cm = pd.DataFrame(cm, index=label_strings, columns=label_strings)
        sns.heatmap(
            df_cm, annot=True, cmap="Blues", square=True, fmt=".1%", ax=ax, cbar=False
        )

    elif annot_fmt == "np":
        df_cm = pd.DataFrame(cm, index=label_strings, columns=label_strings)
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                annot[i, j] = f"{cm[i, j]}\n({cm_perc[i, j]:.0f}%)"
        sns.heatmap(
            df_cm, annot=annot, cmap="Blues", square=True, fmt="", ax=ax, cbar=False
        )

    elif annot_fmt == "pn":
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        df_cm = pd.DataFrame(cm_perc, index=label_strings, columns=label_strings)
        df_cm_rounded = df_cm.round(decimals=0).astype("int")
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                annot[i, j] = f"{cm_perc[i, j]:.0f}%\n({cm[i, j]})"
        sns.heatmap(
            df_cm, annot=annot, cmap="Blues", square=True, fmt="", ax=ax, cbar=False
        )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # plt.xticks(rotation=45, ha='right')
    # plt.yticks(rotation=45)

    return ax

def remove_correlated_features(features_df, class_df, corr_thresh=0.9, rebalance=True):
    assert (features_df.index == class_df.index).all()
    assert features_df.index.name == class_df.index.name
    
    print("Calculating correlations...")
    corr_matrix = features_df.corr()
    print("Done!")

    corr_matrix = corr_matrix.abs()
    
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype("bool"))
    upper = upper[upper >= corr_thresh].dropna(axis=1, how="all")
    upper = upper.sort_values(by=list(upper.columns), ascending=False)
    
    lists = []
    
    for idx in upper.index:
        row = upper.loc[idx].dropna()
        ls = sorted(list(row.index) + [row.name])
        lists.append(ls)
    
    # lists = [[1, 3], [3, 5], [5, 6], [7, 8]]
    merged = []
    for lst in lists:
        for m in merged:
            if any(item in m for item in lst):
                m.extend(lst)
                break
        else:
            merged.append(lst)
    
    merged_alt = []
    for m in merged:
        merged_alt.append(list(set(m)))
    merged = merged_alt
    # print(merged)
    
    choose_corrfeats = []
    for m in merged:
        features_df[m].isna()
        argmin_feat = features_df[m].isna().sum().argmin()
        feat_chosen = m[argmin_feat]
        choose_corrfeats.append(feat_chosen)
    
    bigset = []
    for m in merged:
        if len(m) > 1:
            bigset.append(m)
    bigset = [item for subset in bigset for item in subset]
    
    drop_corrfeats = list(set(bigset) - set(choose_corrfeats))
    
    smallfeatures = features_df.loc[:, choose_corrfeats].dropna(axis=1)
    # drop duplicated cols
    smallfeatures = smallfeatures.loc[:, ~smallfeatures.columns.duplicated()].copy()

    if rebalance:
        class_df = class_df.loc[smallfeatures.index]
        new_n = class_df.groupby(class_df).count().min()
        class_df=class_df.groupby(class_df).sample(new_n)
        smallfeatures = smallfeatures.loc[class_df.index]
    return smallfeatures, class_df

def get_lcdc_features(X_df, y_df, all_metrics, k_features=10):
    scoring = "f1_macro"

    good_feats = []

    metric_maxscore = {"metric":[],"max_score":[]}
    
    for metric in tqdm(all_metrics, desc="SFS", leave=True):
        # Sequential Feature Selection 1-31 features
        lcdc = dcpy.DistanceMetricClassifier(
            scale=True,
            central_stat="median",
            dispersion_stat="std",
            metric=metric,
        )

        feat_selector = SequentialFeatureSelector(
            lcdc,
            k_features=k_features,
            scoring=scoring,
            forward=True,
            n_jobs=-1,
            verbose=0,
        ).fit(X_df, y_df)

        sfs_df = pd.DataFrame.from_dict(feat_selector.get_metric_dict()).T
        sfs_df.index.name = "num_feats"
        sfs_df["avg_score"] = sfs_df["avg_score"].astype("float")
        sfs_df = sfs_df.sort_values(by="avg_score", ascending=False)

        # load_best_features()
        idx_maxscore = sfs_df["avg_score"].idxmax()
        max_score = sfs_df.loc[idx_maxscore, "avg_score"]
        max_score_std = sfs_df.loc[idx_maxscore, "std_dev"]
        sfs_df = sfs_df.loc[(sfs_df["avg_score"]) >= max_score - max_score_std]
        sfs_df = sfs_df.sort_index()
        feats = list(sfs_df.iloc[0]["feature_names"])
        
        metric_maxscore["metric"].append(metric)
        metric_maxscore["max_score"].append(max_score)

        good_feats = good_feats + feats

    topfeats = pd.Series(np.array(good_feats))

    superset_feats = topfeats.value_counts().iloc[:k_features].index

    X_df = X_df.loc[:, superset_feats]

    metric_maxscore = pd.DataFrame(metric_maxscore)

    for metric in tqdm([metric_maxscore.iloc[metric_maxscore["max_score"].argmax()]["metric"]], desc="EFS", leave=False):
        lcdc = dcpy.DistanceMetricClassifier(
            scale=True,
            central_stat="median",
            dispersion_stat="std",
            metric=metric,
        )

        feat_selector = EFS(
            lcdc,
            min_features=1,
            max_features=len(superset_feats),
            scoring=scoring,
            print_progress=False,
            n_jobs=-1,
        ).fit(X_df, y_df)
        
        return sorted(list(feat_selector.best_feature_names_))