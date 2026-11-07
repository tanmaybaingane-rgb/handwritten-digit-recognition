"""
test_app.py
===========
Automated test suite for the Handwritten Digit Recognition Flask application.
Tests all API endpoints, model predictions, preprocessing edge cases, and data outputs.
"""

import unittest
import json
import base64
import io
import numpy as np
from PIL import Image, ImageDraw

from app import app
from ml_model import pipeline_instance


class TestDigitRecognitionApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n--- Starting Digit Recognition Test Suite ---")
        pipeline_instance.initialize()
        cls.client = app.test_client()

    def test_01_index_page(self):
        """Test that index homepage loads with 200 OK."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Handwritten Digit Recognition", response.data)
        self.assertIn(b"Drawing Canvas", response.data)
        print("[PASS] Test 1: Index page loads successfully.")

    def test_02_api_overview(self):
        """Test dataset overview API endpoint."""
        response = self.client.get("/api/overview")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        ds = data["dataset"]
        self.assertEqual(ds["total_samples"], 1797)
        self.assertEqual(ds["n_features"], 64)
        self.assertEqual(ds["n_classes"], 10)
        self.assertEqual(ds["train_samples"], 1437)
        self.assertEqual(ds["test_samples"], 360)
        self.assertEqual(len(ds["sample_previews"]), 10)
        print("[PASS] Test 2: API overview returns exact dataset statistics.")

    def test_03_api_models(self):
        """Test model comparison API endpoint."""
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(len(data["models"]), 6)
        # Winner must be SVM (RBF)
        self.assertEqual(data["best_model"], "SVM (RBF)")
        # Check sorted order
        accuracies = [m["accuracy"] for m in data["models"]]
        self.assertEqual(accuracies, sorted(accuracies, reverse=True))
        # Top accuracy should be 0.9917
        self.assertAlmostEqual(data["models"][0]["accuracy"], 0.9917, places=3)
        print("[PASS] Test 3: API models returns 6 models correctly sorted with SVM as winner.")

    def test_04_api_pca(self):
        """Test PCA experiment API endpoint."""
        response = self.client.get("/api/pca")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        results = data["results"]
        self.assertEqual(len(results), 7)
        expected_ks = [5, 10, 15, 20, 30, 40, 64]
        self.assertEqual([r["components"] for r in results], expected_ks)
        # Check plateau at 15
        k15 = [r for r in results if r["components"] == 15][0]
        self.assertAlmostEqual(k15["accuracy"], 0.9889, places=3)
        print("[PASS] Test 4: API PCA experiment verified across all 7 component counts.")

    def test_05_api_noise(self):
        """Test noise robustness API endpoint."""
        response = self.client.get("/api/noise")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["noise_levels"], [0, 1, 2, 3, 4])
        self.assertEqual(len(data["table"]), 6)
        print("[PASS] Test 5: API noise robustness returns data for 6 models across 5 noise levels.")

    def test_06_api_error_analysis(self):
        """Test error analysis API endpoint."""
        response = self.client.get("/api/error-analysis")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        # Confusion matrix is 10x10
        cm = data["confusion_matrix"]
        self.assertEqual(len(cm), 10)
        self.assertEqual(len(cm[0]), 10)
        # Check misclassified count (exactly 3 for SVM)
        self.assertEqual(data["total_misclassified"], 3)
        self.assertEqual(len(data["misclassified_samples"]), 3)
        for s in data["misclassified_samples"]:
            self.assertIn("test_index", s)
            self.assertIn("true_label", s)
            self.assertIn("predicted_label", s)
            self.assertIn("preview_png", s)
        print("[PASS] Test 6: API error analysis returns 10x10 confusion matrix and 3 misclassified samples.")

    def test_07_preset_digit_endpoint(self):
        """Test preset digit retrieval endpoint."""
        response = self.client.get("/api/preset-digit/7")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["digit"], 7)
        self.assertEqual(len(data["data"]["features_64"]), 64)
        print("[PASS] Test 7: Preset digit endpoint returns digit 7 features correctly.")

    def test_08_predict_empty_canvas(self):
        """Test that predicting an empty canvas returns 400 with a clean error message."""
        # Create an entirely white blank image
        img = Image.new("RGBA", (280, 280), (255, 255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        response = self.client.post("/api/predict", json={
            "image": f"data:image/png;base64,{b64}",
            "model": "SVM (RBF)"
        })
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("empty", data["error"].lower())
        print("[PASS] Test 8: Empty canvas correctly rejected with 400 Bad Request.")

    def test_09_predict_preset_features(self):
        """Test predicting on direct preset test features."""
        preset_resp = self.client.get("/api/preset-digit/4")
        features = preset_resp.get_json()["data"]["features_64"]

        pred_resp = self.client.post("/api/predict", json={
            "features": features,
            "model": "SVM (RBF)"
        })
        self.assertEqual(pred_resp.status_code, 200)
        pred_data = pred_resp.get_json()
        self.assertTrue(pred_data["success"])
        self.assertEqual(pred_data["prediction"], 4)
        self.assertEqual(pred_data["model_used"], "SVM (RBF)")
        print("[PASS] Test 9: Prediction on direct features classified digit 4 correctly.")

    def test_10_predict_drawn_digit(self):
        """Test predicting on a synthetically drawn canvas digit."""
        # Draw a synthetic stroke '1' (vertical line) on 280x280 white canvas
        img = Image.new("RGBA", (280, 280), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.line([(140, 50), (140, 230)], fill=(0, 0, 0, 255), width=24)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        pred_resp = self.client.post("/api/predict", json={
            "image": f"data:image/png;base64,{b64}",
            "model": "SVM (RBF)"
        })
        self.assertEqual(pred_resp.status_code, 200)
        pred_data = pred_resp.get_json()
        self.assertTrue(pred_data["success"])
        self.assertEqual(pred_data["prediction"], 1)
        self.assertIn("preview_png", pred_data)
        print("[PASS] Test 10: Canvas preprocessing correctly classified drawn digit 1.")


if __name__ == "__main__":
    unittest.main()
