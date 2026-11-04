"""
ml_model.py
===========
Machine Learning module for Handwritten Digit Recognition.
Implements the dataset loading, 6-model training, 5-fold cross-validation,
PCA dimensionality reduction experiment, noise robustness evaluation,
confusion matrix and error analysis, and drawing canvas preprocessing.

College Mini Project: Handwritten Digit Recognition Using Machine Learning
Dataset: Optical Recognition of Handwritten Digits (sklearn load_digits)
"""

import io
import base64
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class DigitMLPipeline:
    """
    Manages the complete machine learning lifecycle for handwritten digit recognition.
    Trains and caches models, performs cross-validation, benchmarks PCA reduction,
    evaluates noise robustness, and preprocesses user canvas drawings.
    """

    def __init__(self):
        self.is_initialized = False

        # Raw data and splits
        self.digits = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Metadata
        self.dataset_stats = {}

        # Trained models
        self.models = {}

        # Evaluation results
        self.comparison_df = None
        self.model_comparison_list = []
        self.best_model_name = None
        self.best_model = None

        # Experiments
        self.pca_results = []
        self.noise_results = {}
        self.noise_table = []

        # Error analysis
        self.classification_report_dict = {}
        self.confusion_matrix = []
        self.misclassified_samples = []

    def initialize(self):
        """Loads data, trains the models, runs experiments, and caches all results."""
        if self.is_initialized:
            return

        print("[ML Pipeline] Loading dataset...")
        self._load_data()

        print("[ML Pipeline] Training 6 models and computing 5-fold CV...")
        self._train_and_evaluate_models()

        print("[ML Pipeline] Running PCA dimensionality reduction experiment...")
        self._run_pca_experiment()

        print("[ML Pipeline] Running Noise Robustness experiment...")
        self._run_noise_experiment()

        print("[ML Pipeline] Running error analysis on best model...")
        self._run_error_analysis()

        self.is_initialized = True
        print("[ML Pipeline] Initialization complete. All models and experiments ready.")

    def _load_data(self):
        """Loads sklearn load_digits and creates an 80/20 stratified train/test split."""
        self.digits = load_digits()
        self.X = self.digits.data
        self.y = self.digits.target

        # Stratified split with random_state=42 (matches college report specification)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            stratify=self.y,
            random_state=42
        )

        # Calculate exact dataset statistics
        unique_classes, class_counts = np.unique(self.y, return_counts=True)
        class_distribution = {int(k): int(v) for k, v in zip(unique_classes, class_counts)}

        # Sample images (one representative per digit 0-9)
        sample_digit_previews = []
        for digit in range(10):
            idx = np.where(self.y == digit)[0][0]
            img_8x8 = self.digits.images[idx].tolist()
            sample_digit_previews.append({
                "digit": digit,
                "pixels": img_8x8,
                "preview_png": self._matrix_to_base64_png(self.digits.images[idx])
            })

        self.dataset_stats = {
            "total_samples": int(self.X.shape[0]),
            "n_features": int(self.X.shape[1]),
            "image_height": 8,
            "image_width": 8,
            "n_classes": int(len(unique_classes)),
            "pixel_min": float(self.X.min()),
            "pixel_max": float(self.X.max()),
            "train_samples": int(self.X_train.shape[0]),
            "test_samples": int(self.X_test.shape[0]),
            "train_ratio": 0.8,
            "test_ratio": 0.2,
            "class_distribution": class_distribution,
            "sample_previews": sample_digit_previews
        }

    def _train_and_evaluate_models(self):
        """
        Defines and trains the 6 classical ML models:
        1. Logistic Regression (max_iter=3000)
        2. K-Nearest Neighbors (k=3)
        3. Support Vector Machine (RBF kernel, random_state=42)
        4. Decision Tree (random_state=42)
        5. Random Forest (n_estimators=200, random_state=42)
        6. Gaussian Naive Bayes
        """
        self.models = {
            "SVM (RBF)": SVC(kernel="rbf", random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=3),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=3000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Naive Bayes": GaussianNB(),
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rows = []

        for name, model in self.models.items():
            # 5-fold Stratified Cross-Validation on training split
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=skf, scoring="accuracy")
            cv_mean = float(cv_scores.mean())

            # Train on the complete training split
            model.fit(self.X_train, self.y_train)

            # Predict on the held-out 20% test split
            pred = model.predict(self.X_test)

            acc = float(accuracy_score(self.y_test, pred))
            prec = float(precision_score(self.y_test, pred, average="macro", zero_division=0))
            rec = float(recall_score(self.y_test, pred, average="macro", zero_division=0))
            f1 = float(f1_score(self.y_test, pred, average="macro", zero_division=0))

            rows.append({
                "name": name,
                "cv_acc": round(cv_mean, 4),
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            })

        # Sort models strictly by actual test accuracy descending
        self.comparison_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
        self.model_comparison_list = self.comparison_df.to_dict(orient="records")

        # Dynamically select the best model based on calculated test accuracy
        self.best_model_name = self.comparison_df.loc[0, "name"]
        self.best_model = self.models[self.best_model_name]

    def _run_pca_experiment(self):
        """
        Evaluates dimensionality reduction using PCA + SVM (RBF)
        for components: 5, 10, 15, 20, 30, 40, 64.
        """
        component_counts = [5, 10, 15, 20, 30, 40, 64]
        self.pca_results = []

        for k in component_counts:
            pipeline = make_pipeline(PCA(n_components=k), SVC(kernel="rbf", random_state=42))
            pipeline.fit(self.X_train, self.y_train)
            pred = pipeline.predict(self.X_test)
            acc = float(accuracy_score(self.y_test, pred))

            self.pca_results.append({
                "components": int(k),
                "accuracy": round(acc, 4),
                "accuracy_pct": round(acc * 100, 2),
                "features_retained_pct": round((k / 64.0) * 100, 1),
                "features_removed_pct": round(((64 - k) / 64.0) * 100, 1)
            })

    def _run_noise_experiment(self):
        """
        Evaluates noise robustness for all 6 models across noise levels 0, 1, 2, 3, 4.
        Follows reference code: np.random.RandomState(0), np.clip(X_test + noise, 0, 16).
        """
        rng = np.random.RandomState(0)
        noise_levels = [0, 1, 2, 3, 4]
        self.noise_results = {name: [] for name in self.models}

        for s in noise_levels:
            # Add Gaussian noise and clip to original pixel range [0, 16]
            X_noisy = np.clip(self.X_test + rng.normal(0, s, self.X_test.shape), 0, 16)
            for name, model in self.models.items():
                acc = float(accuracy_score(self.y_test, model.predict(X_noisy)))
                self.noise_results[name].append(round(acc, 4))

        # Format structured table for frontend display
        self.noise_table = []
        for name in self.models:
            row = {"model": name}
            for idx, s in enumerate(noise_levels):
                row[f"noise_{s}"] = self.noise_results[name][idx]
            self.noise_table.append(row)

    def _run_error_analysis(self):
        """
        Generates classification report, confusion matrix, and extracts up to 8
        misclassified samples with true/predicted labels and images.
        """
        pred = self.best_model.predict(self.X_test)

        # Classification report as structured dictionary
        raw_report = classification_report(
            self.y_test,
            pred,
            output_dict=True,
            zero_division=0
        )
        # Format for clean display
        formatted_report = {}
        for key, val in raw_report.items():
            if isinstance(val, dict):
                formatted_report[key] = {
                    "precision": round(val.get("precision", 0), 4),
                    "recall": round(val.get("recall", 0), 4),
                    "f1_score": round(val.get("f1-score", 0), 4),
                    "support": int(val.get("support", 0))
                }
            else:
                formatted_report[key] = round(val, 4) if isinstance(val, (float, int)) else val

        self.classification_report_dict = formatted_report

        # 10x10 Confusion Matrix
        cm = confusion_matrix(self.y_test, pred)
        self.confusion_matrix = cm.tolist()

        # Find misclassified sample indices
        wrong_indices = np.where(pred != self.y_test)[0]
        self.misclassified_samples = []

        n_samples = min(len(wrong_indices), 8)
        for i in range(n_samples):
            idx = wrong_indices[i]
            img_8x8 = self.X_test[idx].reshape(8, 8)
            self.misclassified_samples.append({
                "test_index": int(idx),
                "true_label": int(self.y_test[idx]),
                "predicted_label": int(pred[idx]),
                "pixels": img_8x8.tolist(),
                "preview_png": self._matrix_to_base64_png(img_8x8)
            })

    def _matrix_to_base64_png(self, matrix_8x8, upscale_factor=16):
        """
        Converts an 8x8 numpy array with values 0-16 to a high-contrast base64 PNG data URL.
        Note: load_digits has 0 = white background, 16 = dark stroke (matching gray_r).
        """
        arr = np.array(matrix_8x8, dtype=np.float32)
        # In load_digits: 0 is white, 16 is black/dark stroke
        # Map 0 -> 255 (white), 16 -> 0 (black)
        clamped = np.clip(arr, 0, 16)
        pixel_255 = 255 - (clamped / 16.0 * 255.0)
        img = Image.fromarray(pixel_255.astype(np.uint8), mode="L")

        # Upscale with nearest-neighbor to display crisp 8x8 pixels
        if upscale_factor > 1:
            img = img.resize((8 * upscale_factor, 8 * upscale_factor), Image.Resampling.NEAREST)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def preprocess_canvas_image(self, base64_image_str):
        """
        Takes a base64 image string from an HTML5 canvas drawing and processes it into
        the 64-feature vector compatible with the 8x8 sklearn load_digits models.

        Methodology:
        1. Decode base64 image and convert to RGBA / Grayscale.
        2. Detect foreground digit stroke (canvas drawings have black/colored stroke on white,
           or light stroke on dark).
        3. Reject empty canvases (return specific error message).
        4. Find bounding box of foreground strokes.
        5. Crop tightly to bounding box while preserving aspect ratio.
        6. Resize to fit inside 8x8 with a border margin (typically 6x6 centered within 8x8).
        7. Normalize pixel values to the 0-16 range expected by the models (0=background, 16=stroke).
        """
        if not base64_image_str:
            raise ValueError("No image data received.")

        # Strip data URL prefix if present
        if "base64," in base64_image_str:
            base64_image_str = base64_image_str.split("base64,")[1]

        try:
            image_bytes = base64.b64decode(base64_image_str)
            img = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise ValueError(f"Could not decode image data: {str(e)}")

        # Convert to RGBA first to handle transparent backgrounds
        img_rgba = img.convert("RGBA")
        bg = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(bg, img_rgba)
        gray = composite.convert("L")
        arr = np.array(gray, dtype=np.float32)

        # Detect drawing stroke:
        # In a standard canvas, background is white (~255) and stroke is dark (<200)
        # Foreground intensity: dark strokes -> higher foreground intensity
        fg_intensity = 255.0 - arr

        # Check if the canvas is practically empty
        # If max stroke intensity is very low or total active stroke pixels is tiny
        stroke_mask = fg_intensity > 35
        if np.sum(stroke_mask) < 20:
            raise ValueError("Canvas appears to be empty. Please draw a digit before predicting.")

        # Find bounding box coordinates
        rows = np.any(stroke_mask, axis=1)
        cols = np.any(stroke_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Ensure minimal bounding box dimensions
        rmax = max(rmax, rmin + 2)
        cmax = max(cmax, cmin + 2)

        # Crop the foreground stroke
        cropped_fg = fg_intensity[rmin:rmax + 1, cmin:cmax + 1]

        # Make it a square box by padding the shorter dimension (preserves aspect ratio)
        h, w = cropped_fg.shape
        max_dim = max(h, w)
        pad_top = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top
        pad_left = (max_dim - w) // 2
        pad_right = max_dim - w - pad_left

        square_fg = np.pad(
            cropped_fg,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0
        )

        # Convert back to PIL image for high-quality antialiased resizing
        square_pil = Image.fromarray(square_fg.astype(np.uint8))

        # In load_digits, digits usually occupy roughly 6x6 inside the 8x8 grid (with 1-pixel margin)
        inner_size = 6
        resized_inner = square_pil.resize((inner_size, inner_size), Image.Resampling.BILINEAR)
        inner_arr = np.array(resized_inner, dtype=np.float32)

        # Place inner 6x6 inside 8x8 canvas with 1-pixel border
        target_8x8 = np.zeros((8, 8), dtype=np.float32)
        target_8x8[1:7, 1:7] = inner_arr

        # Normalize intensities to 0.0 - 16.0 range (matching load_digits)
        # 0 = background, 16 = strongest stroke
        max_val = target_8x8.max()
        if max_val > 0:
            # Scale non-linear contrast slightly to match digit sample sharpness
            scaled_8x8 = (target_8x8 / max_val) * 16.0
        else:
            scaled_8x8 = target_8x8

        scaled_8x8 = np.clip(scaled_8x8, 0.0, 16.0)

        # Flatten into 64 features
        features_64 = scaled_8x8.flatten()

        # Generate preview data URL
        preview_png = self._matrix_to_base64_png(scaled_8x8, upscale_factor=16)

        return features_64, scaled_8x8.tolist(), preview_png

    def predict_digit(self, features_64, model_name=None):
        """
        Predicts digit for a 64-feature vector using the specified model (or default best model).
        Returns predicted digit, confidence information (where valid), and per-class decision scores.
        """
        if not self.is_initialized:
            self.initialize()

        if model_name is None or model_name not in self.models:
            model_name = self.best_model_name

        model = self.models[model_name]
        features = np.array(features_64).reshape(1, -1)

        prediction = int(model.predict(features)[0])

        confidence_info = {
            "type": "none",
            "score": None,
            "label": "Not Applicable",
            "explanation": ""
        }

        # Check if model supports predict_proba or decision_function
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            confidence_val = float(probs[prediction])
            confidence_info = {
                "type": "probability",
                "score": round(confidence_val * 100, 2),
                "label": f"{round(confidence_val * 100, 1)}%",
                "explanation": "Estimated posterior class probability (predict_proba)."
            }
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(features)[0]
            # SVM RBF decision function output represents distance to the separating hyperplanes
            best_score = float(scores[prediction]) if scores.ndim > 0 else float(scores)
            confidence_info = {
                "type": "decision_score",
                "score": round(best_score, 2),
                "label": f"{round(best_score, 2)} (Margin Score)",
                "explanation": "SVM signed distance to decision boundary. Higher indicates greater margin separation."
            }

        return {
            "prediction": prediction,
            "model_used": model_name,
            "confidence": confidence_info
        }

    def get_preset_digit(self, digit_value):
        """
        Returns a sample test image for a given digit (0-9) from the held-out test split,
        allowing the user to click and test the models on actual dataset samples.
        """
        if not self.is_initialized:
            self.initialize()

        indices = np.where(self.y_test == digit_value)[0]
        if len(indices) == 0:
            indices = np.where(self.y == digit_value)[0]

        idx = indices[0]
        pixels_8x8 = self.X_test[idx].reshape(8, 8)
        preview_png = self._matrix_to_base64_png(pixels_8x8)

        return {
            "digit": int(digit_value),
            "features_64": self.X_test[idx].tolist(),
            "pixels_8x8": pixels_8x8.tolist(),
            "preview_png": preview_png
        }


# Global pipeline instance
pipeline_instance = DigitMLPipeline()
