import numpy as np

from tqdm.auto import tqdm
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.model_selection import train_test_split
from types import SimpleNamespace
import copy


# EarlyStopping Helper Class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_model_state_dict = None

    def __call__(self, val_loss, model):
        # Check if the validation loss has improved
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save the best model state using a deep copy
            self.best_model_state_dict = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(
                    f"Validation loss decreased ({self.best_loss:.6f}). Saving model..."
                )
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                print("Early stopping triggered.")
                self.early_stop = True


# PyTorch Models
class ae(nn.Module):
    def __init__(self, args):
        super(ae, self).__init__()
        self.args = args

        # Encoder Architecture
        self.enc1 = nn.Linear(args.in_dim, 512)
        self.encbn1 = nn.BatchNorm1d(512)
        self.enc2 = nn.Linear(512, 256)
        self.encbn2 = nn.BatchNorm1d(256)
        self.enc3 = nn.Linear(256, 128)
        self.encbn3 = nn.BatchNorm1d(128)
        self.enc4 = nn.Linear(128, args.z_dim, bias=False)

        # Decoder Architecture
        self.dec1 = nn.Linear(args.z_dim, 128)
        self.decbn1 = nn.BatchNorm1d(128)
        self.dec2 = nn.Linear(128, 256)
        self.decbn2 = nn.BatchNorm1d(256)
        self.dec3 = nn.Linear(256, 512)
        self.decbn3 = nn.BatchNorm1d(512)
        self.dec4 = nn.Linear(512, args.in_dim)

    def encode(self, x):
        h = F.leaky_relu(self.encbn1(self.enc1(x)))
        h = F.leaky_relu(self.encbn2(self.enc2(h)))
        h = F.leaky_relu(self.encbn3(self.enc3(h)))
        return self.enc4(h)

    def decode(self, x):
        h = F.leaky_relu(self.decbn1(self.dec1(x)))
        h = F.leaky_relu(self.decbn2(self.dec2(h)))
        h = F.leaky_relu(self.decbn3(self.dec3(h)))
        return torch.tanh(self.dec4(h))

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat

    def compute_loss(self, x):
        _, x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x, reduction="mean")
        return loss

    def compute_anomaly_score(self, x):
        _, x_hat = self.forward(x)
        # Calculates reconstruction error for each sample
        score = F.mse_loss(x_hat, x, reduction="none")
        # The anomaly score is the sum of squared errors across all features
        return torch.sum(score, dim=1)


class vae(nn.Module):
    def __init__(self, args):
        super(vae, self).__init__()

        # Encoder Architecture
        self.enc1 = nn.Linear(args.in_dim, 512)
        self.encbn1 = nn.BatchNorm1d(512)
        self.enc2 = nn.Linear(512, 256)
        self.encbn2 = nn.BatchNorm1d(256)
        self.enc3 = nn.Linear(256, 128)
        self.encbn3 = nn.BatchNorm1d(128)

        self.mu = nn.Linear(128, args.z_dim)
        self.log_var = nn.Linear(128, args.z_dim)

        # Decoder Architecture
        self.dec1 = nn.Linear(args.z_dim, 128)
        self.decbn1 = nn.BatchNorm1d(128)
        self.dec2 = nn.Linear(128, 256)
        self.decbn2 = nn.BatchNorm1d(256)
        self.dec3 = nn.Linear(256, 512)
        self.decbn3 = nn.BatchNorm1d(512)
        self.dec4 = nn.Linear(512, args.in_dim)

    def encode(self, x):
        h = F.leaky_relu(self.encbn1(self.enc1(x)))
        h = F.leaky_relu(self.encbn2(self.enc2(h)))
        h = F.leaky_relu(self.encbn3(self.enc3(h)))
        return self.mu(h), self.log_var(h)

    def decode(self, x):
        h = F.leaky_relu(self.decbn1(self.dec1(x)))
        h = F.leaky_relu(self.decbn2(self.dec2(h)))
        h = F.leaky_relu(self.decbn3(self.dec3(h)))
        return torch.tanh(self.dec4(h))

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return mu, log_var, z, x_hat

    def compute_loss(self, x, L=1):
        mu, log_var = self.encode(x)

        # KL Divergence
        dkl = torch.mean(-0.5 * (1 + log_var - mu.pow(2) - log_var.exp()))

        # Reconstruction Loss (averaged over L samples)
        rec_loss = 0
        for _ in range(L):
            z = self.reparameterize(mu, log_var)
            x_hat = self.decode(z)
            rec_loss += F.mse_loss(x_hat, x, reduction="mean")

        reconstruction = rec_loss / L

        loss = reconstruction + dkl
        return loss

    def compute_anomaly_score(self, x):
        # The forward pass in evaluation mode will use a single z sample
        _, _, _, x_hat = self.forward(x)
        score = F.mse_loss(x_hat, x, reduction="none")
        # The anomaly score is the sum of squared errors across all features
        return torch.sum(score, dim=1)


class DeepSVDD_torch(nn.Module):
    def __init__(self, args):
        super(DeepSVDD_torch, self).__init__()
        self.args = args

        # Encoder Architecture
        self.enc1 = nn.Linear(args.in_dim, 512)
        self.encbn1 = nn.BatchNorm1d(512)
        self.enc2 = nn.Linear(512, 256)
        self.encbn2 = nn.BatchNorm1d(256)
        self.enc3 = nn.Linear(256, 128)
        self.encbn3 = nn.BatchNorm1d(128)
        self.enc4 = nn.Linear(128, args.z_dim, bias=False)

    def encode(self, x):
        h = F.leaky_relu(self.encbn1(self.enc1(x)))
        h = F.leaky_relu(self.encbn2(self.enc2(h)))
        h = F.leaky_relu(self.encbn3(self.enc3(h)))
        return self.enc4(h)

    def forward(self, x):
        z = self.encode(x)
        return z

    def compute_loss(self, x, c):
        z = self.forward(x)
        loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
        return loss

    def compute_anomaly_score(self, x, c):
        z = self.forward(x)
        score = torch.sum((z - c) ** 2, dim=1)
        return score


class ClassSVDD(nn.Module):
    def __init__(self, args):
        super(ClassSVDD, self).__init__()

        self.args = args
        self.c = None  # Center(s) of the hypersphere(s)

        # Encoder Architecture
        self.enc1 = nn.Linear(args.in_dim, 512)
        self.encbn1 = nn.BatchNorm1d(512)
        self.enc2 = nn.Linear(512, 256)
        self.encbn2 = nn.BatchNorm1d(256)
        self.enc3 = nn.Linear(256, 128)
        self.encbn3 = nn.BatchNorm1d(128)
        self.enc4 = nn.Linear(128, args.z_dim, bias=False)

    def encode(self, x):
        h = F.leaky_relu(self.encbn1(self.enc1(x)))
        h = F.leaky_relu(self.encbn2(self.enc2(h)))
        h = F.leaky_relu(self.encbn3(self.enc3(h)))
        return self.enc4(h)

    def forward(self, x):
        z = self.encode(x)
        return z

    def compute_loss(self, x, y):
        if self.c is None:
            raise ValueError(
                "Center 'c' is not initialized. Call 'init_center_c' before training."
            )

        z = self.forward(x)
        # Calculate distance to the center corresponding to the label 'y'
        loss = torch.mean(torch.sum((z - self.c[y]) ** 2, dim=1))
        return loss

    def compute_anomaly_score(self, x):
        if self.c is None:
            raise ValueError("Center 'c' is not initialized.")

        z = self.forward(x)
        # Calculate distance to all centers and take the minimum
        distances_to_centers = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)
        score = torch.min(distances_to_centers, dim=1)[0]
        return score


# scikit-learn compatible Anomaly Detectors using the PyTorch classes
class AutoencoderAnomalyDetector(BaseEstimator, OutlierMixin):
    def __init__(
        self,
        encoding_dim=16,
        epochs=50,
        batch_size=32,
        lr=1e-4,
        validation_split=0.1,
        patience=5,
        verbose=False,
    ):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.validation_split = validation_split
        self.patience = patience
        self.verbose = verbose
        self.model_ = None

    def fit(self, X, y=None):
        # Infer input dimension from data
        input_dim = X.shape[1]

        # Create a namespace for model arguments, mimicking the 'args' object
        model_args = SimpleNamespace(in_dim=input_dim, z_dim=self.encoding_dim)

        # Instantiate the PyTorch model
        self.model_ = ae(model_args)

        # Split data into training and validation sets
        X_train, X_val = train_test_split(
            X, test_size=self.validation_split, random_state=42
        )

        # Convert numpy data to PyTorch tensors
        X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
        X_val_tensor = torch.from_numpy(X_val.astype(np.float32))

        # Create DataLoaders for batching
        train_dataset = TensorDataset(X_train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_dataset = TensorDataset(X_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        early_stopper = EarlyStopping(patience=self.patience, verbose=self.verbose)

        # Training loop
        for epoch in tqdm(range(self.epochs), leave=False, desc="Training AE"):
            # --- Training Step ---
            self.model_.train()
            for batch in train_loader:
                inputs = batch[0]
                optimizer.zero_grad()
                loss = self.model_.compute_loss(inputs)
                loss.backward()
                optimizer.step()

            # --- Validation Step ---
            self.model_.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch[0]
                    loss = self.model_.compute_loss(inputs)
                    validation_loss += loss.item()

            avg_val_loss = validation_loss / len(val_loader)

            # --- Early Stopping Check ---
            early_stopper(avg_val_loss, self.model_)
            if early_stopper.early_stop:
                break

        # Load the best model state found during training
        if early_stopper.best_model_state_dict:
            self.model_.load_state_dict(early_stopper.best_model_state_dict)

        return self

    def decision_function(self, X):
        if self.model_ is None:
            raise RuntimeError("The model has not been fitted yet.")

        self.model_.eval()
        X_tensor = torch.from_numpy(X.astype(np.float32))

        with torch.no_grad():
            scores = self.model_.compute_anomaly_score(X_tensor)

        return scores.detach().cpu().numpy()


class VAEAnomalyDetector(BaseEstimator, OutlierMixin):
    def __init__(
        self,
        encoding_dim=16,
        epochs=50,
        batch_size=32,
        lr=1e-4,
        loss_samples=1,
        validation_split=0.1,
        patience=5,
        verbose=False,
    ):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.loss_samples = loss_samples  # Number of MC samples (L) for loss calc
        self.validation_split = validation_split
        self.patience = patience
        self.verbose = verbose
        self.model_ = None

    def fit(self, X, y=None):
        # Infer input dimension from data
        input_dim = X.shape[1]

        # Create a namespace for model arguments, mimicking the 'args' object
        model_args = SimpleNamespace(in_dim=input_dim, z_dim=self.encoding_dim)

        # Instantiate the PyTorch model
        self.model_ = vae(model_args)

        # Split data into training and validation sets
        X_train, X_val = train_test_split(
            X, test_size=self.validation_split, random_state=42
        )

        # Convert numpy data to PyTorch tensors
        X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
        X_val_tensor = torch.from_numpy(X_val.astype(np.float32))

        # Create DataLoaders for batching
        train_dataset = TensorDataset(X_train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_dataset = TensorDataset(X_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        early_stopper = EarlyStopping(patience=self.patience, verbose=self.verbose)

        # Training loop
        for epoch in tqdm(range(self.epochs), leave=False, desc="Training VAE"):
            # --- Training Step ---
            self.model_.train()
            for batch in train_loader:
                inputs = batch[0]
                optimizer.zero_grad()
                loss = self.model_.compute_loss(inputs, L=self.loss_samples)
                loss.backward()
                optimizer.step()

            # --- Validation Step ---
            self.model_.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch[0]
                    # Use a single sample for validation loss for speed and consistency
                    loss = self.model_.compute_loss(inputs, L=1)
                    validation_loss += loss.item()

            avg_val_loss = validation_loss / len(val_loader)

            # --- Early Stopping Check ---
            early_stopper(avg_val_loss, self.model_)
            if early_stopper.early_stop:
                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load the best model state found during training
        if early_stopper.best_model_state_dict:
            self.model_.load_state_dict(early_stopper.best_model_state_dict)

        return self

    def decision_function(self, X):
        if self.model_ is None:
            raise RuntimeError("The model has not been fitted yet.")

        self.model_.eval()
        X_tensor = torch.from_numpy(X.astype(np.float32))

        with torch.no_grad():
            scores = self.model_.compute_anomaly_score(X_tensor)

        return scores.detach().cpu().numpy()


class DeepSVDDAnomalyDetector(BaseEstimator, OutlierMixin):
    def __init__(
        self,
        encoding_dim=16,
        epochs=50,
        batch_size=32,
        lr=1e-4,
        validation_split=0.1,
        patience=5,
        verbose=False,
    ):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.validation_split = validation_split
        self.patience = patience
        self.verbose = verbose
        self.model_ = None
        self.c_ = None  # Center of the hypersphere

    def _init_center_c(self, train_loader, eps=0.01):
        self.model_.eval()
        zs = []
        with torch.no_grad():
            for batch in train_loader:
                inputs = batch[0]
                z = self.model_(inputs)
                zs.append(z.detach())
        zs = torch.cat(zs)
        c = torch.mean(zs, dim=0)
        # If c is too close to 0, set to a small value eps to avoid trivial solution
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def fit(self, X, y=None):
        # --- 0. Label Encoding ---
        # Convert string labels (like 'RRL') to integer labels (0, 1, 2, ...)
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(
            y.ravel()
        )  # Use ravel() to ensure 1D array

        # Infer input dimension from data
        input_dim = X.shape[1]

        # Create a namespace for model arguments
        model_args = SimpleNamespace(in_dim=input_dim, z_dim=self.encoding_dim)

        # Instantiate the PyTorch model
        self.model_ = DeepSVDD_torch(model_args)

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y_encoded,
            test_size=self.validation_split,
            random_state=42,
            stratify=y_encoded,
        )

        # Convert numpy data to PyTorch tensors
        X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
        X_val_tensor = torch.from_numpy(X_val.astype(np.float32))

        # Create DataLoaders for batching
        train_dataset = TensorDataset(X_train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_dataset = TensorDataset(X_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Initialize hypersphere center c
        self.c_ = self._init_center_c(train_loader)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        early_stopper = EarlyStopping(patience=self.patience, verbose=self.verbose)

        # Training loop
        for epoch in tqdm(range(self.epochs), leave=False, desc="Training DeepSVDD"):
            # --- Training Step ---
            self.model_.train()
            for batch in train_loader:
                inputs = batch[0]
                optimizer.zero_grad()
                loss = self.model_.compute_loss(inputs, self.c_)
                loss.backward()
                optimizer.step()

            # --- Validation Step ---
            self.model_.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch[0]
                    loss = self.model_.compute_loss(inputs, self.c_)
                    validation_loss += loss.item()

            avg_val_loss = validation_loss / len(val_loader)

            # --- Early Stopping Check ---
            early_stopper(avg_val_loss, self.model_)
            if early_stopper.early_stop:
                break

        # Load the best model state found during training
        if early_stopper.best_model_state_dict:
            self.model_.load_state_dict(early_stopper.best_model_state_dict)

        return self

    def decision_function(self, X):
        if self.model_ is None or self.c_ is None:
            raise RuntimeError("The model has not been fitted yet.")

        self.model_.eval()
        X_tensor = torch.from_numpy(X.astype(np.float32))

        with torch.no_grad():
            scores = self.model_.compute_anomaly_score(X_tensor, self.c_)

        return scores.detach().cpu().numpy()


class ClassSVDDAnomalyDetector(BaseEstimator, OutlierMixin):
    def __init__(
        self,
        z_dim=32,
        epochs=100,
        batch_size=64,
        lr=1e-4,
        validation_split=0.1,
        patience=10,
        verbose=False,
        eps=0.01,
    ):
        self.z_dim = z_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.validation_split = validation_split
        self.patience = patience
        self.verbose = verbose
        self.eps = eps
        self.model_ = None
        self.label_encoder_ = None

    def _init_center_c(self, X, y):
        self.model_.eval()

        dataset = TensorDataset(
            torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.int64))
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        latents = []
        labels = []
        with torch.no_grad():
            for inputs, targets in loader:
                z = self.model_(inputs)
                latents.append(z.detach().cpu())
                labels.append(targets.cpu())

        latents = torch.cat(latents)
        labels = torch.cat(labels).numpy()

        centers = []
        num_classes = len(self.label_encoder_.classes_)
        for label_idx in range(num_classes):
            class_latents = latents[labels == label_idx]
            if len(class_latents) > 0:
                centers.append(torch.mean(class_latents, dim=0))
            else:
                # Handle case where a class might not be in the initial data (unlikely but safe)
                # Initialize center as a zero vector of the correct dimension
                centers.append(torch.zeros(self.z_dim))

        centers_tensor = torch.stack(centers)

        # Stabilize centers to avoid collapse
        centers_tensor[(abs(centers_tensor) < self.eps) & (centers_tensor < 0)] = (
            -self.eps
        )
        centers_tensor[(abs(centers_tensor) < self.eps) & (centers_tensor > 0)] = (
            self.eps
        )

        # Set the centers on the model
        self.model_.c = centers_tensor

    def fit(self, X, y):
        # --- 0. Label Encoding ---
        # Convert string labels (like 'RRL') to integer labels (0, 1, 2, ...)
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(
            y.ravel()
        )  # Use ravel() to ensure 1D array

        # Infer input dimension from data
        input_dim = X.shape[1]

        # Create a namespace for model arguments
        model_args = SimpleNamespace(in_dim=input_dim, z_dim=self.z_dim)

        # Instantiate the PyTorch model
        self.model_ = ClassSVDD(model_args)

        # --- 1. Initialize Centers ---
        # Note: The centers are initialized with a non-trained network,
        # then the network is trained to map points closer to these fixed centers.
        if self.verbose:
            print("Initializing class centers...")
        self._init_center_c(X, y_encoded)  # Pass the encoded labels

        # --- 2. Train the Network ---
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y_encoded,
            test_size=self.validation_split,
            random_state=42,
            stratify=y_encoded,
        )

        # Convert numpy data to PyTorch tensors
        X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
        y_train_tensor = torch.from_numpy(y_train.astype(np.int64))
        X_val_tensor = torch.from_numpy(X_val.astype(np.float32))
        y_val_tensor = torch.from_numpy(y_val.astype(np.int64))

        # Create DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        early_stopper = EarlyStopping(patience=self.patience, verbose=self.verbose)

        # Training loop
        for epoch in tqdm(range(self.epochs), leave=False, desc="Training ClassSVDD"):
            # --- Training Step ---
            self.model_.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                loss = self.model_.compute_loss(inputs, labels)
                loss.backward()
                optimizer.step()

            # --- Validation Step ---
            self.model_.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    loss = self.model_.compute_loss(inputs, labels)
                    validation_loss += loss.item()

            avg_val_loss = validation_loss / len(val_loader)

            # --- Early Stopping Check ---
            early_stopper(avg_val_loss, self.model_)
            if early_stopper.early_stop:
                if self.verbose:
                    print("Early stopping triggered.")
                break

        # Load the best model state found during training
        if early_stopper.best_model_state_dict:
            self.model_.load_state_dict(early_stopper.best_model_state_dict)

        return self

    def decision_function(self, X):
        if self.model_ is None:
            raise RuntimeError("The model has not been fitted yet.")

        self.model_.eval()
        X_tensor = torch.from_numpy(X.astype(np.float32))

        with torch.no_grad():
            scores = self.model_.compute_anomaly_score(X_tensor)

        return scores.detach().cpu().numpy()
