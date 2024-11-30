"""Traditional classifier experiments with support for clean and poisoned data."""

import logging
from pathlib import Path
import time
from typing import List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.experiment_config import create_experiment_config
from models.factory import get_model
from utils.logging import get_logger

logger = get_logger(__name__)


class TraditionalExperiment:
    """Traditional classifier experiment."""

    def __init__(
        self,
        config=None,
        checkpoint_path=None,
        subset_size: Optional[int] = None,
        classifiers: Optional[List[str]] = None,
    ):
        """Initialize experiment."""
        if config is None:
            config = create_experiment_config(
                dataset_name="cifar100",
                model_name="wrn-28-10",
                checkpoint_path=checkpoint_path,
            )
        self.config = config
        self.checkpoint_path = checkpoint_path or config.model.checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = (
            config.model.name if hasattr(config.model, "name") else None
        )
        self.subset_size = subset_size
        self.classifiers = classifiers or ["knn", "lr", "rf", "svm"]

    def get_classifier(self, name: str, n_samples: int):
        """Get classifier instance."""
        if name == "knn":
            return KNeighborsClassifier(
                n_neighbors=min(30, n_samples // 50),
                weights="distance",
                metric="cosine",
                n_jobs=-1,
            )
        elif name == "lr":
            return LogisticRegression(
                max_iter=2000,
                multi_class="multinomial",
                solver="lbfgs",
                C=10.0,
                class_weight="balanced",
                n_jobs=-1,
            )
        elif name == "rf":
            return RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                bootstrap=True,
                n_jobs=-1,
                class_weight="balanced",
                random_state=42,
            )
        elif name == "svm":
            return SVC(
                kernel="rbf",
                C=100.0,
                gamma="scale",
                class_weight="balanced",
                cache_size=2000,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown classifier: {name}")

    def train_and_evaluate(
        self,
        clf_name: str,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Train and evaluate classifier."""
        # Scale features
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)

        # Apply PCA if needed
        if train_features.shape[1] > 512:
            logger.info("Applying PCA: components=512")
            pca = PCA(n_components=512)
            train_features = pca.fit_transform(train_features)
            test_features = pca.transform(test_features)

        # Get classifier
        clf = self.get_classifier(clf_name, len(train_labels))

        # Train classifier
        train_start = time.time()
        clf.fit(train_features, train_labels)
        train_time = time.time() - train_start

        # Evaluate classifier
        inference_start = time.time()
        accuracy = clf.score(test_features, test_labels) * 100
        inference_time = time.time() - inference_start

        logger.info(
            f"{clf_name} - Accuracy: {accuracy:.2f}% "
            f"(Train: {train_time:.2f}s, Inference: {inference_time:.2f}s)"
        )

        return train_time, inference_time, accuracy

    def extract_features(self, dataset, desc="Extracting features"):
        """Extract features from dataset using CNN."""
        model = get_model(self.config.dataset.name, self.feature_extractor).to(
            self.device
        )

        # Load checkpoint if available
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=True
            )

            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Try to load state dict
            try:
                model.load_state_dict(state_dict)
                logger.info("Successfully loaded state dict")
            except Exception as e:
                logger.warning(f"Failed to load state dict directly: {str(e)}")
                logger.info("Attempting to remap state dict keys...")

                # Try to remap keys
                new_state_dict = {}
                for k, v in state_dict.items():
                    # Handle WideResNet key differences
                    if k.startswith("layer"):
                        # Convert layer1.0.conv1 -> block1.layer.0.conv1
                        parts = k.split(".")
                        if len(parts) > 1 and parts[1].isdigit():
                            layer_num = int(parts[0].replace("layer", ""))
                            block_num = parts[1]
                            rest = ".".join(parts[2:])
                            new_key = f"block{layer_num}.layer.{block_num}.{rest}"
                            new_state_dict[new_key] = v
                        else:
                            new_state_dict[k] = v
                    elif k == "bn.weight":
                        new_state_dict["bn.weight"] = v
                    elif k == "bn.bias":
                        new_state_dict["bn.bias"] = v
                    elif k == "bn.running_mean":
                        new_state_dict["bn.running_mean"] = v
                    elif k == "bn.running_var":
                        new_state_dict["bn.running_var"] = v
                    else:
                        new_state_dict[k] = v

                try:
                    model.load_state_dict(new_state_dict)
                    logger.info("Successfully loaded remapped state dict")
                except Exception as e:
                    logger.warning(f"Failed to load remapped state dict: {str(e)}")
                    logger.info("Attempting direct key mapping...")

                    # Try direct key mapping
                    direct_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith("layer"):
                            # Convert layer1.0.conv1 -> layer1.layer.0.conv1
                            parts = k.split(".")
                            if len(parts) > 1 and parts[1].isdigit():
                                layer_num = parts[0]
                                block_num = parts[1]
                                rest = ".".join(parts[2:])
                                new_key = f"{layer_num}.layer.{block_num}.{rest}"
                                direct_state_dict[new_key] = v
                            else:
                                direct_state_dict[k] = v
                        else:
                            direct_state_dict[k] = v

                    try:
                        model.load_state_dict(direct_state_dict, strict=False)
                        logger.info(
                            "Successfully loaded state dict with non-strict matching"
                        )
                    except Exception as e:
                        logger.error(f"Failed to load state dict: {str(e)}")
                        logger.warning("Using untrained model for feature extraction")
        else:
            logger.warning(
                "No checkpoint found. Using untrained model for feature extraction"
            )

        model.eval()
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=False,
            num_workers=self.config.hardware.num_workers,
            pin_memory=self.config.hardware.pin_memory,
        )

        features = []
        labels = []

        with torch.no_grad():
            for batch, targets in tqdm(dataloader, desc=desc):
                batch = batch.to(self.device)
                batch_features = model.extract_features(batch)
                features.append(batch_features.cpu().numpy())
                labels.append(
                    targets.numpy() if isinstance(targets, torch.Tensor) else targets
                )

        return np.vstack(features), np.concatenate(labels)

    def run_clean(self):
        """Run experiment on clean data."""
        logger.info("Running clean data experiment")

        # Get datasets
        train_dataset = get_dataset(
            self.config.dataset.name,
            train=True,
            subset_size=self.config.dataset.subset_size,
        )
        test_dataset = get_dataset(
            self.config.dataset.name,
            train=False,
            subset_size=self.config.dataset.subset_size,
        )

        # Extract features
        X_train, y_train = self.extract_features(
            train_dataset, "Extracting training features"
        )
        X_test, y_test = self.extract_features(test_dataset, "Extracting test features")

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Apply PCA if specified
        if self.pca_components:
            n_components = min(X_train.shape[0], X_train.shape[1], self.pca_components)
            logger.info(f"Applying PCA: components={n_components}")
            pca = PCA(n_components=n_components, random_state=42)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            logger.info(
                f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}"
            )

        results = []
        for clf_name in self.classifiers:
            logger.info(f"Training {clf_name}")
            clf = self.get_classifier(clf_name, len(y_train))

            # Train and evaluate
            start_time = time.time()
            clf.fit(X_train, y_train)
            train_time = time.time() - start_time

            start_time = time.time()
            y_pred = clf.predict(X_test)
            inference_time = time.time() - start_time

            accuracy = (y_pred == y_test).mean()

            result = {
                "dataset": self.config.dataset.name,
                "classifier": clf_name,
                "subset_size": len(y_train),
                "accuracy": accuracy,
                "train_time": train_time,
                "inference_time": inference_time,
            }
            results.append(result)

            logger.info(
                f"{clf_name} - Accuracy: {accuracy*100:.2f}% "
                f"(Train: {train_time:.2f}s, Inference: {inference_time:.2f}s)"
            )

        return results

    def run_poisoned(self, poisoned_dataset, clean_test_dataset=None):
        """Run experiment on poisoned data."""
        logger.info("Running poisoned data experiment")

        # Use clean test set if provided, otherwise get it
        if clean_test_dataset is None:
            clean_test_dataset = get_dataset(
                self.config.dataset.name,
                train=False,
                subset_size=self.config.dataset.subset_size,
            )

        # Extract features
        X_train, y_train = self.extract_features(
            poisoned_dataset, "Extracting poisoned features"
        )
        X_test, y_test = self.extract_features(
            clean_test_dataset, "Extracting clean test features"
        )

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Apply PCA if specified
        if self.pca_components:
            n_components = min(X_train.shape[0], X_train.shape[1], self.pca_components)
            logger.info(f"Applying PCA: components={n_components}")
            pca = PCA(n_components=n_components, random_state=42)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        results = []
        for clf_name in self.classifiers:
            logger.info(f"Training {clf_name} on poisoned data")
            clf = self.get_classifier(clf_name, len(y_train))

            # Train and evaluate
            start_time = time.time()
            clf.fit(X_train, y_train)
            train_time = time.time() - start_time

            start_time = time.time()
            y_pred = clf.predict(X_test)
            inference_time = time.time() - start_time

            accuracy = (y_pred == y_test).mean()

            result = {
                "dataset": self.config.dataset.name,
                "classifier": clf_name,
                "subset_size": len(y_train),
                "accuracy": accuracy,
                "train_time": train_time,
                "inference_time": inference_time,
                "poison_type": self.config.poison.poison_type,
                "poison_ratio": self.config.poison.poison_ratio,
            }
            results.append(result)

            logger.info(
                f"{clf_name} - Accuracy: {accuracy*100:.2f}% "
                f"(Train: {train_time:.2f}s, Inference: {inference_time:.2f}s)"
            )

        return results
