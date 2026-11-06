/**
 * app.js
 * ======
 * Frontend JavaScript for Handwritten Digit Recognition Dashboard.
 * Handles interactive drawing canvas, touch/mouse events, API requests,
 * dynamic chart rendering with Chart.js, and navigation.
 */

document.addEventListener("DOMContentLoaded", () => {
    // ---------------- App State ----------------
    const state = {
        activeSection: "recognize",
        charts: {},
        isDrawing: false,
        lastX: 0,
        lastY: 0,
        hasDrawn: false
    };

    // ---------------- DOM References ----------------
    const navItems = document.querySelectorAll(".nav-item");
    const sections = document.querySelectorAll(".dashboard-section");

    // Canvas elements
    const canvas = document.getElementById("digit-canvas");
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const btnClear = document.getElementById("btn-clear");
    const btnPredict = document.getElementById("btn-predict");
    const modelSelector = document.getElementById("model-selector");
    const sampleButtonsContainer = document.getElementById("sample-buttons");

    // Result elements
    const alertBanner = document.getElementById("alert-banner");
    const predictedDigitValue = document.getElementById("predicted-digit-value");
    const predictedModelTag = document.getElementById("predicted-model-tag");
    const processedPreviewImg = document.getElementById("processed-preview-img");
    const previewPlaceholder = document.getElementById("preview-placeholder");
    const confVal = document.getElementById("conf-val");
    const confExplanation = document.getElementById("conf-explanation");
    const activePixelsVal = document.getElementById("active-pixels-val");
    const peakIntensityVal = document.getElementById("peak-intensity-val");

    // ---------------- Navigation ----------------
    function initNavigation() {
        navItems.forEach(item => {
            item.addEventListener("click", (e) => {
                e.preventDefault();
                const targetSection = item.getAttribute("data-section");
                switchSection(targetSection);
            });
        });

        // Check URL hash
        const initialHash = window.location.hash.replace("#", "");
        if (initialHash && document.getElementById(initialHash)) {
            switchSection(initialHash);
        }
    }

    function switchSection(sectionId) {
        state.activeSection = sectionId;
        window.location.hash = sectionId;

        navItems.forEach(item => {
            if (item.getAttribute("data-section") === sectionId) {
                item.classList.add("active");
            } else {
                item.classList.remove("active");
            }
        });

        sections.forEach(sec => {
            if (sec.id === sectionId) {
                sec.classList.add("active");
            } else {
                sec.classList.remove("active");
            }
        });

        // Trigger chart resize / render on section switch
        if (state.charts[sectionId]) {
            state.charts[sectionId].resize();
        }
    }

    // ---------------- Canvas Setup & Drawing ----------------
    function initCanvas() {
        // Set up white background
        clearCanvas();

        // Canvas stroke styles
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.strokeStyle = "#000000";
        ctx.lineWidth = 20;

        // Mouse events
        canvas.addEventListener("mousedown", startDrawing);
        canvas.addEventListener("mousemove", draw);
        canvas.addEventListener("mouseup", stopDrawing);
        canvas.addEventListener("mouseleave", stopDrawing);

        // Touch events for mobile / tablets
        canvas.addEventListener("touchstart", (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent("mousedown", {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        }, { passive: false });

        canvas.addEventListener("touchmove", (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent("mousemove", {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        }, { passive: false });

        canvas.addEventListener("touchend", (e) => {
            e.preventDefault();
            const mouseEvent = new MouseEvent("mouseup", {});
            canvas.dispatchEvent(mouseEvent);
        }, { passive: false });

        // Controls
        btnClear.addEventListener("click", clearCanvas);
        btnPredict.addEventListener("click", handlePredict);

        // Initialize preset digit buttons (0 through 9)
        sampleButtonsContainer.innerHTML = "";
        for (let d = 0; d <= 9; d++) {
            const btn = document.createElement("button");
            btn.className = "btn-sample-digit";
            btn.textContent = d;
            btn.title = `Test with real test-set digit ${d}`;
            btn.addEventListener("click", () => loadPresetDigit(d));
            sampleButtonsContainer.appendChild(btn);
        }
    }

    function getCanvasCoordinates(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    function startDrawing(e) {
        state.isDrawing = true;
        const coords = getCanvasCoordinates(e);
        state.lastX = coords.x;
        state.lastY = coords.y;
        state.hasDrawn = true;
        hideAlert();
    }

    function draw(e) {
        if (!state.isDrawing) return;
        const coords = getCanvasCoordinates(e);

        ctx.beginPath();
        ctx.moveTo(state.lastX, state.lastY);
        ctx.lineTo(coords.x, coords.y);
        ctx.stroke();

        state.lastX = coords.x;
        state.lastY = coords.y;
    }

    function stopDrawing() {
        state.isDrawing = false;
    }

    function clearCanvas() {
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        state.hasDrawn = false;
        hideAlert();
        resetPredictionCard();
    }

    function resetPredictionCard() {
        predictedDigitValue.textContent = "—";
        predictedModelTag.textContent = "Awaiting Input";
        processedPreviewImg.classList.add("hidden");
        processedPreviewImg.src = "";
        previewPlaceholder.classList.remove("hidden");
        confVal.textContent = "—";
        confExplanation.textContent = "Draw a digit above or click a preset to see the classification output.";
        activePixelsVal.textContent = "—";
        peakIntensityVal.textContent = "— / 16.0";
    }

    function showAlert(message, type = "error") {
        alertBanner.className = `alert-banner alert-${type}`;
        alertBanner.textContent = message;
        alertBanner.classList.remove("hidden");
    }

    function hideAlert() {
        alertBanner.classList.add("hidden");
    }

    // ---------------- Prediction API ----------------
    async function handlePredict() {
        hideAlert();

        if (!state.hasDrawn) {
            showAlert("Canvas is empty. Please draw a digit before predicting.");
            return;
        }

        const selectedModel = modelSelector.value;
        const dataUrl = canvas.toDataURL("image/png");

        btnPredict.disabled = true;
        btnPredict.innerHTML = "Processing...";

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image: dataUrl,
                    model: selectedModel
                })
            });

            const result = await response.json();

            if (!result.success) {
                showAlert(result.error || "Failed to recognize digit.");
                resetPredictionCard();
            } else {
                displayPredictionResult(result);
            }
        } catch (error) {
            showAlert("Network or server error during prediction: " + error.message);
        } finally {
            btnPredict.disabled = false;
            btnPredict.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                Predict Digit
            `;
        }
    }

    async function loadPresetDigit(digit) {
        hideAlert();
        const selectedModel = modelSelector.value;

        try {
            const response = await fetch(`/api/preset-digit/${digit}`);
            const resData = await response.json();

            if (!resData.success) {
                showAlert("Could not load preset sample.");
                return;
            }

            // Draw this sample into canvas so user sees it
            renderPresetToCanvas(resData.data.pixels_8x8);

            // Directly send features for classification
            const predResponse = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    features: resData.data.features_64,
                    model: selectedModel
                })
            });

            const predResult = await predResponse.json();
            if (predResult.success) {
                displayPredictionResult(predResult);
                showAlert(`Loaded held-out test sample for digit ${digit} successfully.`, "success");
            }
        } catch (e) {
            showAlert("Error loading sample: " + e.message);
        }
    }

    function renderPresetToCanvas(pixels8x8) {
        clearCanvas();
        // Scale 8x8 onto 280x280 canvas with rounded blocks
        const cellSize = canvas.width / 8;
        for (let r = 0; r < 8; r++) {
            for (let c = 0; c < 8; c++) {
                const val = pixels8x8[r][c]; // 0 to 16
                if (val > 0) {
                    const brightness = Math.round(255 - (val / 16.0 * 255));
                    ctx.fillStyle = `rgb(${brightness}, ${brightness}, ${brightness})`;
                    ctx.beginPath();
                    ctx.arc(
                        c * cellSize + cellSize / 2,
                        r * cellSize + cellSize / 2,
                        (cellSize / 2) * (val / 16.0),
                        0,
                        Math.PI * 2
                    );
                    ctx.fill();
                }
            }
        }
        state.hasDrawn = true;
    }

    function displayPredictionResult(result) {
        predictedDigitValue.textContent = result.prediction;
        predictedModelTag.textContent = result.model_used;

        if (result.preview_png) {
            processedPreviewImg.src = result.preview_png;
            processedPreviewImg.classList.remove("hidden");
            previewPlaceholder.classList.add("hidden");
        }

        const conf = result.confidence;
        if (conf && conf.type !== "none") {
            confVal.textContent = conf.label;
            confExplanation.textContent = conf.explanation;
        } else {
            confVal.textContent = "N/A";
            confExplanation.textContent = "Model does not provide probabilistic margins.";
        }

        if (result.features_summary) {
            activePixelsVal.textContent = `${result.features_summary.active_features} / 64`;
            peakIntensityVal.textContent = `${result.features_summary.max_intensity.toFixed(1)} / 16.0`;
        }
    }

    // ---------------- Load Section Data & Charts ----------------
    async function loadDatasetOverview() {
        try {
            const res = await fetch("/api/overview");
            const data = await res.json();
            if (!data.success) return;

            const ds = data.dataset;
            document.getElementById("stat-total-samples").textContent = ds.total_samples.toLocaleString();
            document.getElementById("stat-train-samples").textContent = ds.train_samples.toLocaleString();
            document.getElementById("stat-test-samples").textContent = ds.test_samples.toLocaleString();

            // Samples Gallery
            const gallery = document.getElementById("samples-gallery");
            gallery.innerHTML = "";
            ds.sample_previews.forEach(item => {
                const div = document.createElement("div");
                div.className = "sample-item";
                div.innerHTML = `
                    <img src="${item.preview_png}" alt="Digit ${item.digit}">
                    <span>Digit ${item.digit}</span>
                `;
                gallery.appendChild(div);
            });

            // Class distribution chart
            const labels = Object.keys(ds.class_distribution).map(k => `Digit ${k}`);
            const counts = Object.values(ds.class_distribution);

            const ctxChart = document.getElementById("chart-class-dist").getContext("2d");
            state.charts["overview"] = new Chart(ctxChart, {
                type: "bar",
                data: {
                    labels: labels,
                    datasets: [{
                        label: "Number of Images",
                        data: counts,
                        backgroundColor: "#3b82f6",
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 200,
                            ticks: { stepSize: 50 }
                        }
                    }
                }
            });
        } catch (e) {
            console.error("Error loading dataset overview:", e);
        }
    }

    async function loadModelComparison() {
        try {
            const res = await fetch("/api/models");
            const data = await res.json();
            if (!data.success) return;

            const tbody = document.getElementById("table-models-body");
            tbody.innerHTML = "";

            const modelNames = [];
            const testAccs = [];
            const cvAccs = [];

            data.models.forEach((m, idx) => {
                const isWinner = m.name === data.best_model;
                modelNames.push(m.name);
                testAccs.push((m.accuracy * 100).toFixed(2));
                cvAccs.push((m.cv_acc * 100).toFixed(2));

                const tr = document.createElement("tr");
                if (isWinner) tr.className = "winner-row";

                tr.innerHTML = `
                    <td>#${idx + 1}</td>
                    <td><strong>${m.name}</strong></td>
                    <td class="numeric">${(m.cv_acc * 100).toFixed(2)}%</td>
                    <td class="numeric"><strong>${(m.accuracy * 100).toFixed(2)}%</strong></td>
                    <td class="numeric">${(m.precision * 100).toFixed(2)}%</td>
                    <td class="numeric">${(m.recall * 100).toFixed(2)}%</td>
                    <td class="numeric">${(m.f1 * 100).toFixed(2)}%</td>
                    <td>${isWinner ? '<span class="badge-best">Winner</span>' : '<span style="color:var(--text-subtle);">Evaluated</span>'}</td>
                `;
                tbody.appendChild(tr);
            });

            // Model Comparison Chart
            const ctxChart = document.getElementById("chart-model-comparison").getContext("2d");
            state.charts["models"] = new Chart(ctxChart, {
                type: "bar",
                data: {
                    labels: modelNames,
                    datasets: [
                        {
                            label: "Test Accuracy (%)",
                            data: testAccs,
                            backgroundColor: "#2563eb",
                            borderRadius: 4
                        },
                        {
                            label: "5-Fold CV Accuracy (%)",
                            data: cvAccs,
                            backgroundColor: "#93c5fd",
                            borderRadius: 4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            min: 75,
                            max: 100,
                            ticks: { stepSize: 5 }
                        }
                    }
                }
            });
        } catch (e) {
            console.error("Error loading model comparison:", e);
        }
    }

    async function loadPcaExperiment() {
        try {
            const res = await fetch("/api/pca");
            const data = await res.json();
            if (!data.success) return;

            const tbody = document.getElementById("table-pca-body");
            tbody.innerHTML = "";

            const components = [];
            const accuracies = [];

            data.results.forEach(r => {
                components.push(r.components);
                accuracies.push(r.accuracy_pct);

                const tr = document.createElement("tr");
                const isOptimal = r.components === 15;
                if (isOptimal) tr.className = "winner-row";

                tr.innerHTML = `
                    <td><strong>${r.components}</strong> components</td>
                    <td class="numeric"><strong>${r.accuracy_pct}%</strong></td>
                    <td class="numeric">${r.features_retained_pct}%</td>
                    <td class="numeric">${r.features_removed_pct}%</td>
                    <td>${isOptimal ? '<span class="badge-best">Optimal Tradeoff</span>' : 'Standard'}</td>
                `;
                tbody.appendChild(tr);
            });

            // PCA Chart
            const ctxChart = document.getElementById("chart-pca").getContext("2d");
            state.charts["pca"] = new Chart(ctxChart, {
                type: "line",
                data: {
                    labels: components.map(c => `${c}`),
                    datasets: [{
                        label: "Test Accuracy (%)",
                        data: accuracies,
                        borderColor: "#059669",
                        backgroundColor: "rgba(5, 150, 105, 0.1)",
                        pointBackgroundColor: "#059669",
                        pointRadius: 5,
                        fill: true,
                        tension: 0.2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: "Number of Principal Components (out of 64)" } },
                        y: { min: 90, max: 100, title: { display: true, text: "Test Accuracy (%)" } }
                    }
                }
            });
        } catch (e) {
            console.error("Error loading PCA data:", e);
        }
    }

    async function loadNoiseExperiment() {
        try {
            const res = await fetch("/api/noise");
            const data = await res.json();
            if (!data.success) return;

            const tbody = document.getElementById("table-noise-body");
            tbody.innerHTML = "";

            const datasets = [];
            const colors = [
                "#2563eb", // SVM (blue)
                "#059669", // KNN (green)
                "#d97706", // Random Forest (amber)
                "#7c3aed", // Logistic Regression (purple)
                "#dc2626", // Decision Tree (red)
                "#64748b"  // Naive Bayes (gray)
            ];

            let colorIdx = 0;
            for (const [modelName, accList] of Object.entries(data.results_by_model)) {
                const color = colors[colorIdx % colors.length];
                colorIdx++;

                const accPercentages = accList.map(a => (a * 100).toFixed(2));
                datasets.push({
                    label: modelName,
                    data: accPercentages,
                    borderColor: color,
                    backgroundColor: "transparent",
                    pointBackgroundColor: color,
                    pointRadius: 4,
                    tension: 0.2
                });
            }

            data.table.forEach(row => {
                const tr = document.createElement("tr");
                const degradation = ((row.noise_0 - row.noise_4) * 100).toFixed(1);
                tr.innerHTML = `
                    <td><strong>${row.model}</strong></td>
                    <td class="numeric">${(row.noise_0 * 100).toFixed(1)}%</td>
                    <td class="numeric">${(row.noise_1 * 100).toFixed(1)}%</td>
                    <td class="numeric">${(row.noise_2 * 100).toFixed(1)}%</td>
                    <td class="numeric">${(row.noise_3 * 100).toFixed(1)}%</td>
                    <td class="numeric"><strong>${(row.noise_4 * 100).toFixed(1)}%</strong></td>
                    <td class="numeric" style="color:${degradation > 20 ? 'var(--danger)' : 'var(--text-muted)'};">
                        -${degradation}%
                    </td>
                `;
                tbody.appendChild(tr);
            });

            // Noise Chart
            const ctxChart = document.getElementById("chart-noise").getContext("2d");
            state.charts["noise"] = new Chart(ctxChart, {
                type: "line",
                data: {
                    labels: ["σ = 0 (Clean)", "σ = 1", "σ = 2", "σ = 3", "σ = 4 (Severe)"],
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: "Gaussian Noise Level (Standard Deviation σ)" } },
                        y: { min: 0, max: 100, title: { display: true, text: "Classification Accuracy (%)" } }
                    }
                }
            });
        } catch (e) {
            console.error("Error loading noise data:", e);
        }
    }

    async function loadErrorAnalysis() {
        try {
            const res = await fetch("/api/error-analysis");
            const data = await res.json();
            if (!data.success) return;

            // 1. Confusion Matrix (10x10)
            const cmGrid = document.getElementById("cm-grid");
            cmGrid.innerHTML = "";
            const cm = data.confusion_matrix;

            for (let r = 0; r < 10; r++) {
                for (let c = 0; c < 10; c++) {
                    const count = cm[r][c];
                    const cell = document.createElement("div");
                    cell.className = "cm-cell";
                    cell.textContent = count;
                    cell.title = `True: ${r}, Predicted: ${c} (Count: ${count})`;

                    if (r === c) {
                        // Diagonal correct predictions
                        cell.style.backgroundColor = count > 30 ? "#1d4ed8" : "#3b82f6";
                        cell.style.color = "#ffffff";
                    } else if (count > 0) {
                        // Off-diagonal mistake
                        cell.style.backgroundColor = "#fee2e2";
                        cell.style.color = "#b91c1c";
                        cell.style.fontWeight = "bold";
                    } else {
                        // Zero mistake
                        cell.style.backgroundColor = "#f8fafc";
                        cell.style.color = "#94a3b8";
                    }
                    cmGrid.appendChild(cell);
                }
            }

            // 2. Misclassified samples gallery
            const gallery = document.getElementById("misclassified-gallery");
            gallery.innerHTML = "";
            if (data.misclassified_samples.length === 0) {
                gallery.innerHTML = "<p style='color:var(--success); font-weight:600;'>Zero misclassified images on test set!</p>";
            } else {
                data.misclassified_samples.forEach(s => {
                    const card = document.createElement("div");
                    card.className = "misclassified-card";
                    card.innerHTML = `
                        <img src="${s.preview_png}" alt="Test sample ${s.test_index}">
                        <div class="misclassified-meta">
                            <div>True: <span class="badge-true">${s.true_label}</span></div>
                            <div>Pred: <span class="badge-pred">${s.predicted_label}</span></div>
                            <small style="color:var(--text-subtle);">Idx: #${s.test_index}</small>
                        </div>
                    `;
                    gallery.appendChild(card);
                });
            }

            // 3. Classification Report Table
            const crBody = document.getElementById("table-cr-body");
            crBody.innerHTML = "";
            const cr = data.classification_report;

            for (let d = 0; d <= 9; d++) {
                const key = String(d);
                if (cr[key]) {
                    const item = cr[key];
                    const tr = document.createElement("tr");
                    tr.innerHTML = `
                        <td><strong>Digit ${d}</strong></td>
                        <td class="numeric">${(item.precision * 100).toFixed(1)}%</td>
                        <td class="numeric">${(item.recall * 100).toFixed(1)}%</td>
                        <td class="numeric"><strong>${(item.f1_score * 100).toFixed(1)}%</strong></td>
                        <td class="numeric">${item.support}</td>
                    `;
                    crBody.appendChild(tr);
                }
            }
        } catch (e) {
            console.error("Error loading error analysis:", e);
        }
    }

    // ---------------- Initialization ----------------
    initNavigation();
    initCanvas();

    // Fetch initial data
    loadDatasetOverview();
    loadModelComparison();
    loadPcaExperiment();
    loadNoiseExperiment();
    loadErrorAnalysis();
});
