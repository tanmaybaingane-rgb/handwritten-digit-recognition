# Handwritten Digit Recognition Using Machine Learning

An end-to-end, beginner-friendly Machine Learning mini project and interactive academic web dashboard for classifying handwritten digits (0–9) using classical machine learning algorithms.

Developed as a Project-Based Learning (PBL) submission for **Fundamentals of Machine Learning (FML)**, Department of Artificial Intelligence & Data Science, Yeshwantrao Chavan College of Engineering, Nagpur.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Technology Stack](#technology-stack)
4. [Dataset Details](#dataset-details)
5. [Machine Learning Algorithms & Results](#machine-learning-algorithms--results)
6. [Innovative Experiments](#innovative-experiments)
   - [PCA Dimensionality Reduction](#pca-dimensionality-reduction)
   - [Noise Robustness Stress Test](#noise-robustness-stress-test)
   - [Error Analysis & Confusion Matrix](#error-analysis--confusion-matrix)
7. [Project Structure](#project-structure)
8. [Windows Setup & Installation Guide](#windows-setup--installation-guide)
9. [Running the Web Application](#running-the-web-application)
10. [Offline Standalone Benchmark Script](#offline-standalone-benchmark-script)
11. [Running the Automated Test Suite](#running-the-automated-test-suite)
12. [Troubleshooting & Common Questions](#troubleshooting--common-questions)
13. [Academic Submission Details](#academic-submission-details)

---

## 1. Project Overview

The objective of this project is to recognize handwritten digits (0 through 9) from 8×8 grayscale images using classical machine learning algorithms. Instead of relying solely on baseline accuracy, this project performs a rigorous scientific comparison:
- Trains and evaluates **6 distinct machine learning models** on an identical stratified split.
- Validates model stability using **5-fold Stratified Cross-Validation**.
- Evaluates feature compression using **Principal Component Analysis (PCA)**.
- Benchmarks sensor degradation via **Gaussian Noise Robustness Testing**.
- Conducts in-depth **Error Analysis** via a 10×10 confusion matrix and misclassified sample inspection.
- Provides a live, interactive **Drawing Canvas** allowing real-time digit recognition in the browser.

---

## 2. Key Features

- **Interactive Drawing Canvas:** Draw any digit using a mouse or touchscreen with responsive scaling, aspect-ratio-preserving centering, and automatic empty-canvas rejection.
- **Model Selector:** Choose between all 6 trained algorithms in real-time to compare their predictions on the same drawing.
- **Sample Digits Quick-Test:** Test pre-loaded held-out test samples (digits 0–9) with a single click.
- **Comprehensive Academic Dashboard:** Clean light-theme user interface with restrained blue accents, responsive navigation, and interactive charts powered by Chart.js.
- **Zero Cloud Dependencies:** Operates 100% offline without requiring databases, API keys, or Docker containers.

---

## 3. Technology Stack

- **Backend:** Python, Flask
- **Machine Learning:** scikit-learn (`SVC`, `KNeighborsClassifier`, `RandomForestClassifier`, `LogisticRegression`, `DecisionTreeClassifier`, `GaussianNB`, `PCA`, `StratifiedKFold`)
- **Data Handling:** NumPy, pandas
- **Image Preprocessing:** Pillow (PIL)
- **Frontend:** HTML5, CSS3, vanilla JavaScript (ES6+)
- **Interactive Visualizations:** Chart.js

---

## 4. Dataset Details

The application uses scikit-learn's built-in **Optical Recognition of Handwritten Digits** dataset (`sklearn.datasets.load_digits`):

| Property | Value | Description |
| :--- | :--- | :--- |
| **Total Samples** | 1,797 | Normalized grayscale images |
| **Dimensions** | 8 × 8 pixels | 64 pixel features per image |
| **Classes** | 10 | Digits 0 through 9 (~180 images per class) |
| **Pixel Range** | 0 to 16 | 0 = White background, 16 = Darkest ink stroke |
| **Split Ratio** | 80% / 20% | 1,437 training samples, 360 testing samples |
| **Partitioning** | Stratified | `test_size=0.2, stratify=y, random_state=42` |
| **Feature Scaling** | None | Raw [0, 16] scale preserved (critical for noise robustness) |

---

## 5. Machine Learning Algorithms & Results

All models were evaluated on the held-out test set (360 samples) and cross-validated with 5-fold stratified cross-validation on the training set:

| Rank | Model | 5-Fold CV Accuracy | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **SVM (RBF Kernel)** | **98.82%** | **99.17%** | **0.9920** | **0.9915** | **0.9916** | **Best Model (Winner)** |
| 2 | K-Nearest Neighbors ($k=3$) | 98.68% | 98.61% | 0.9866 | 0.9859 | 0.9859 | Top Performer |
| 3 | Random Forest ($200$ trees) | 97.63% | 96.39% | 0.9644 | 0.9635 | 0.9634 | Ensemble Robustness |
| 4 | Logistic Regression (`max_iter=3000`) | 96.31% | 95.83% | 0.9587 | 0.9580 | 0.9581 | Linear Baseline |
| 5 | Decision Tree | 84.48% | 82.50% | 0.8236 | 0.8242 | 0.8230 | High Variance |
| 6 | Gaussian Naive Bayes | 82.26% | 81.11% | 0.8463 | 0.8103 | 0.8137 | Feature Independence Flaw |

### Key Takeaways:
- **SVM (RBF)** is the highest-performing model, achieving **99.17% test accuracy** and committing only **3 errors out of 360 test samples**.
- **KNN** is a very close second at **98.61%**.
- **Naive Bayes** performs the weakest because it assumes all 64 pixels are conditionally independent, which is fundamentally invalid for spatial image data where adjacent pixels are strongly correlated.

---

## 6. Innovative Experiments

### A. PCA Dimensionality Reduction
PCA was integrated into an SVM (RBF) pipeline to determine how many features can be discarded without sacrificing accuracy:

| PCA Components | Retained Features | Removed Features | Actual Test Accuracy |
| :---: | :---: | :---: | :---: |
| 5 | 7.8% | 92.2% | 92.78% |
| 10 | 15.6% | 84.4% | 97.78% |
| **15** | **23.4%** | **76.6%** | **98.89% (Optimal Tradeoff)** |
| 20 | 31.2% | 68.8% | 98.61% |
| 30 | 46.9% | 53.1% | 98.89% |
| 40 | 62.5% | 37.5% | 98.89% |
| 64 (All) | 100.0% | 0.0% | 98.89% |

**Finding:** With only **15 components**, accuracy reaches **98.89%**, removing **76.6%** of all features while maintaining virtually identical performance to the full 64-pixel representation.

### B. Noise Robustness Stress Test
To simulate sensor noise or optical scanner degradation, zero-mean Gaussian noise with standard deviation $\sigma \in \{0, 1, 2, 3, 4\}$ was added to the test set:

| Model | Clean ($\sigma = 0$) | $\sigma = 1$ | $\sigma = 2$ | $\sigma = 3$ | Severe ($\sigma = 4$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SVM (RBF)** | **99.17%** | **99.44%** | **97.22%** | **98.33%** | **95.83%** |
| **KNN** | 98.61% | 98.61% | 97.78% | 98.06% | **97.50%** |
| **Random Forest** | 96.39% | 96.11% | 94.17% | 89.44% | 88.33% |
| **Logistic Regression** | 95.83% | 96.39% | 92.78% | 89.44% | 83.61% |
| **Decision Tree** | 82.50% | 67.22% | 58.61% | 51.94% | 46.67% |
| **Naive Bayes** | 81.11% | 13.33% | 13.61% | 12.22% | 12.22% |

**The Feature Scaling Discovery:**
In initial experiments using `StandardScaler`, noise caused SVM accuracy to drop to ~13%. Because outer border pixels in handwritten digit images have near-zero variance, standardization artificially inflates small random noise on those pixels. By keeping the native [0, 16] scale, SVM and KNN demonstrated outstanding noise resistance, remaining above **95%** even under severe noise.

### C. Error Analysis & Confusion Matrix
For the winning SVM model, errors were examined on the 360-sample test set:
- **Total Mistakes:** 3 images misclassified.
- **Digit 1:** 1 sample confused with digit 8.
- **Digit 8:** 1 sample confused with digit 1.
- **Digit 9:** 1 sample confused with digit 7.
- Visual inspection confirms that these 3 samples possess severe handwriting ambiguity that could easily mislead human inspection.

---

## 7. Project Structure

```
digits_recognition/
│
├── app.py                     # Flask application entry point and REST API routes
├── ml_model.py                # Core ML pipeline (training, CV, PCA, noise, preprocessing)
├── test_app.py                # Automated test suite (10 test cases)
├── generate_plots.py          # Generates reference figures into shots/ directory
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation and setup instructions
│
├── templates/
│   └── index.html             # Academic dashboard HTML template
│
├── static/
│   ├── css/
│   │   └── style.css          # Professional academic light theme styling
│   └── js/
│       └── app.js             # Canvas drawing, touch handlers, Chart.js integrations
│
├── shots/                     # Generated reference benchmark plots
│   ├── samples.png            # 2x8 dataset sample grid
│   ├── classdist.png          # Class distribution bar chart
│   ├── cm.png                 # 10x10 confusion matrix heatmap
│   ├── wrong.png              # Misclassified test digits
│   ├── pca.png                # Components vs accuracy curve
│   └── noise.png              # Noise robustness degradation comparison
│
├── FML project.docx           # College project submission report
└── project_digits.pdf         # Reference code implementation
```

---

## 8. Windows Setup & Installation Guide

### Prerequisites
- Windows 10 or Windows 11
- Python 3.10, 3.11, 3.12, 3.13, or 3.14 installed and added to PATH.

### Step 1: Open PowerShell or Command Prompt
Open your terminal in the project directory:
```powershell
cd C:\Users\tanma\OneDrive\Documents\digits_recognition
```

### Step 2: (Optional but Recommended) Create a Virtual Environment
```powershell
# Create virtual environment named 'venv'
python -m venv venv

# If PowerShell execution policy restricts scripts, allow local execution:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```
*(Dependencies: `Flask`, `scikit-learn`, `numpy`, `pandas`, `pillow`, `matplotlib`)*

---

## 9. Running the Web Application

To launch the web server:
```powershell
python app.py
```

You will see output indicating model initialization:
```
[ML Pipeline] Loading dataset...
[ML Pipeline] Training 6 models and computing 5-fold CV...
[ML Pipeline] Running PCA dimensionality reduction experiment...
[ML Pipeline] Running Noise Robustness experiment...
[ML Pipeline] Running error analysis on best model...
[ML Pipeline] Initialization complete. All models and experiments ready.

* Handwritten Digit Recognition App is starting...
* Local URL: http://127.0.0.1:5000
```

### Open in your browser:
Navigate to:
```
http://127.0.0.1:5000
```

### Using the Application:
1. **Draw a Digit:** Use your mouse or finger to sketch a single digit (0–9) on the white canvas.
2. **Select Model:** Keep the default `SVM (RBF)` or pick another algorithm from the dropdown.
3. **Click "Predict Digit":** See the predicted digit, confidence/decision metric, and the processed 8×8 preview image.
4. **Try Sample Digits:** Click any number from `0` to `9` in the "Or test real test-set samples" strip to see instant predictions on actual dataset test images.
5. **Explore Dashboard Tabs:** Click the sidebar links to review Dataset Overview, Model Comparison, PCA Analysis, Noise Robustness, Error Analysis, and About Project.

---

## 10. Offline Standalone Benchmark Script

If you want to generate or regenerate the reference static figures (`shots/*.png`) offline without running the web server:
```powershell
python generate_plots.py
```
This produces:
- `shots/samples.png`
- `shots/classdist.png`
- `shots/cm.png`
- `shots/wrong.png`
- `shots/pca.png`
- `shots/noise.png`

---

## 11. Running the Automated Test Suite

A comprehensive test suite is included in `test_app.py` covering 10 functional tests:
```powershell
python test_app.py
```

Expected output:
```
Ran 10 tests in 4.010s

OK
[PASS] Test 1: Index page loads successfully.
[PASS] Test 2: API overview returns exact dataset statistics.
[PASS] Test 3: API models returns 6 models correctly sorted with SVM as winner.
[PASS] Test 4: API PCA experiment verified across all 7 component counts.
[PASS] Test 5: API noise robustness returns data for 6 models across 5 noise levels.
[PASS] Test 6: API error analysis returns 10x10 confusion matrix and 3 misclassified samples.
[PASS] Test 7: Preset digit endpoint returns digit 7 features correctly.
[PASS] Test 8: Empty canvas correctly rejected with 400 Bad Request.
[PASS] Test 9: Prediction on direct features classified digit 4 correctly.
[PASS] Test 10: Canvas preprocessing correctly classified drawn digit 1.
```

---

## 12. Troubleshooting & Common Questions

### 1. `Port 5000 is already in use`
If another process is running on port 5000:
```powershell
# Find and terminate process on port 5000
Get-Process -Id (Get-NetTCPConnection -LocalPort 5000).OwningProcess | Stop-Process
```
Or specify a different port in `app.py`: `app.run(port=5001)`.

### 2. `Execution of scripts is disabled on this system` (PowerShell)
Run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```
Then reactivate your virtual environment.

### 3. "Canvas appears to be empty"
Ensure you draw with a firm stroke inside the canvas boundaries before pressing **Predict Digit**. Blank or faint single-dot drawings are safely rejected.

### 4. Why are drawn digits classified best by SVM and KNN?
The training dataset contains 8×8 low-resolution digits. Non-linear kernel SVM and KNN adapt smoothly to curved handwriting boundaries, whereas single Decision Trees and Naive Bayes are brittle against stroke variations.

---

## 13. Academic Submission Details

- **Project Title:** Handwritten Digit Recognition Using Machine Learning
- **Institution:** Yeshwantrao Chavan College of Engineering, Nagpur
- **Department:** Department of Artificial Intelligence & Data Science
- **Course:** Fundamentals of Machine Learning (FML)
- **Faculty Guide:** Ganesh Yenurkar Sir
- **Academic Session:** 2026-27
- **Submitted by:**
  - Tanmay Baingane — Roll No. 166 (Division B)
  - Tanmay Kamthekar — Roll No. 167 (Division B)
  - Tavakkal Sheikh — Roll No. 168 (Division B)
