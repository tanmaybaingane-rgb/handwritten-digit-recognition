"""
app.py
======
Flask Web Application for Handwritten Digit Recognition.
Provides an academic dashboard interface with interactive drawing canvas,
dataset exploration, 6-model comparison, PCA dimensionality reduction analysis,
noise robustness testing, and deep error analysis.

College Mini Project: Handwritten Digit Recognition Using Machine Learning
Department of Artificial Intelligence & Data Science
"""

import os
import sys
from flask import Flask, render_template, request, jsonify
from ml_model import pipeline_instance

# Create Flask application
app = Flask(__name__, static_folder="static", template_folder="templates")

# Configure max upload size (16 MB)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


@app.before_request
def ensure_ml_initialized():
    """Ensure ML pipeline is trained and cached before handling requests."""
    if not pipeline_instance.is_initialized:
        pipeline_instance.initialize()


@app.route("/")
def index():
    """Renders the main academic dashboard."""
    return render_template("index.html")


@app.route("/api/overview", methods=["GET"])
def api_overview():
    """Returns dataset statistics, class balance, and sample previews."""
    return jsonify({
        "success": True,
        "dataset": pipeline_instance.dataset_stats
    })


@app.route("/api/models", methods=["GET"])
def api_models():
    """Returns the 6-model comparison table sorted by actual test accuracy."""
    return jsonify({
        "success": True,
        "models": pipeline_instance.model_comparison_list,
        "best_model": pipeline_instance.best_model_name
    })


@app.route("/api/pca", methods=["GET"])
def api_pca():
    """Returns the PCA dimensionality reduction experiment results."""
    return jsonify({
        "success": True,
        "results": pipeline_instance.pca_results,
        "description": (
            "PCA (Principal Component Analysis) reduces the original 64 pixel features "
            "into orthogonal components that capture the maximum data variance. "
            "Our experiment tests SVM (RBF) accuracy across 5, 10, 15, 20, 30, 40, and 64 components."
        )
    })


@app.route("/api/noise", methods=["GET"])
def api_noise():
    """Returns noise robustness experiment results across 5 noise levels for all 6 models."""
    return jsonify({
        "success": True,
        "noise_levels": [0, 1, 2, 3, 4],
        "table": pipeline_instance.noise_table,
        "results_by_model": pipeline_instance.noise_results,
        "interpretation": (
            "SVM and KNN demonstrate superior noise tolerance because their margin and "
            "distance metrics are robust to distributed Gaussian noise without feature standardization. "
            "Conversely, Naive Bayes drops sharply because border pixels in clean images are almost always zero; "
            "noise makes these pixels non-zero, severely penalizing the Gaussian likelihood."
        )
    })


@app.route("/api/error-analysis", methods=["GET"])
def api_error_analysis():
    """Returns the confusion matrix, classification report, and misclassified digit samples."""
    return jsonify({
        "success": True,
        "best_model": pipeline_instance.best_model_name,
        "confusion_matrix": pipeline_instance.confusion_matrix,
        "classification_report": pipeline_instance.classification_report_dict,
        "misclassified_samples": pipeline_instance.misclassified_samples,
        "total_misclassified": len(pipeline_instance.misclassified_samples)
    })


@app.route("/api/preset-digit/<int:digit>", methods=["GET"])
def api_preset_digit(digit):
    """Returns an actual test sample for a given digit (0-9) to test immediately."""
    if digit < 0 or digit > 9:
        return jsonify({"success": False, "error": "Digit must be between 0 and 9."}), 400

    try:
        data = pipeline_instance.get_preset_digit(digit)
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accepts drawn canvas data (or 64-feature array), preprocesses to 8x8 with 0-16 scale,
    and returns model prediction with preview image and confidence metadata.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Invalid request payload. Expected JSON."}), 400

    selected_model = data.get("model", pipeline_instance.best_model_name)
    image_b64 = data.get("image")
    direct_features = data.get("features")

    try:
        if direct_features and len(direct_features) == 64:
            # Direct feature vector input (e.g., from sample preset)
            features_64 = direct_features
            scaled_8x8 = [features_64[i * 8:(i + 1) * 8] for i in range(8)]
            preview_png = pipeline_instance._matrix_to_base64_png(scaled_8x8)
        elif image_b64:
            # Canvas drawing input - preprocess image
            features_64, scaled_8x8, preview_png = pipeline_instance.preprocess_canvas_image(image_b64)
        else:
            return jsonify({
                "success": False,
                "error": "No drawing or features provided. Please draw a digit on the canvas."
            }), 400

        result = pipeline_instance.predict_digit(features_64, model_name=selected_model)

        return jsonify({
            "success": True,
            "prediction": result["prediction"],
            "model_used": result["model_used"],
            "confidence": result["confidence"],
            "preview_png": preview_png,
            "features_summary": {
                "active_features": int(sum(1 for v in features_64 if v > 0)),
                "max_intensity": float(max(features_64)),
                "mean_intensity": round(float(sum(features_64) / 64.0), 2)
            }
        })

    except ValueError as ve:
        # Handles empty canvas and preprocessing validation errors
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as ex:
        return jsonify({"success": False, "error": f"Internal prediction error: {str(ex)}"}), 500


@app.route("/api/about", methods=["GET"])
def api_about():
    """Provides project documentation content matching the college submission report."""
    return jsonify({
        "success": True,
        "title": "Handwritten Digit Recognition Using Machine Learning",
        "institution": "Yeshwantrao Chavan College of Engineering, Nagpur",
        "department": "Department of Artificial Intelligence & Data Science",
        "academic_year": "2026-27",
        "subject": "Fundamentals Of Machine Learning",
        "faculty": "Ganesh Yenurkar Sir",
        "students": [
            {"name": "Tanmay Baingane", "roll": "166", "division": "B"},
            {"name": "Tanmay Kamthekar", "roll": "167", "division": "B"},
            {"name": "Tavakkal Sheikh", "roll": "168", "division": "B"}
        ]
    })


if __name__ == "__main__":
    # Ensure initialization before serving
    pipeline_instance.initialize()
    print("\n* Handwritten Digit Recognition App is starting...")
    print("* Local URL: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
