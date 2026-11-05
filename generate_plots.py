"""
generate_plots.py
=================
Generates the benchmark plots and figures into the 'shots/' directory
using Pillow and NumPy. This ensures all figures referenced in the report
are pre-generated and accessible offline.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ml_model import pipeline_instance

os.makedirs("shots", exist_ok=True)
pipeline_instance.initialize()

# Try loading a basic font, fallback to default
try:
    font_sm = ImageFont.truetype("arial.ttf", 12)
    font_md = ImageFont.truetype("arial.ttf", 14)
    font_lg = ImageFont.truetype("arial.ttf", 18)
    font_bold = ImageFont.truetype("arialbd.ttf", 14)
except Exception:
    font_sm = ImageFont.load_default()
    font_md = ImageFont.load_default()
    font_lg = ImageFont.load_default()
    font_bold = ImageFont.load_default()


def save_samples_plot():
    """Generates shots/samples.png: 2x8 grid of sample images with labels."""
    digits = pipeline_instance.digits
    cols = 8
    rows = 2
    cell_w, cell_h = 80, 95
    pad = 10
    total_w = cols * cell_w + pad * 2
    total_h = rows * cell_h + pad * 2 + 30

    img = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.text((pad, 10), "Sample images from the dataset (label above each image)", fill=(30, 30, 30), font=font_md)

    for i in range(16):
        r = i // cols
        c = i % cols
        x = pad + c * cell_w
        y = pad + 35 + r * cell_h

        lab = str(pipeline_instance.y[i])
        draw.text((x + 28, y), f"Digit {lab}", fill=(40, 40, 40), font=font_sm)

        # 8x8 image
        arr = digits.images[i]
        pix_255 = 255 - (np.clip(arr, 0, 16) / 16.0 * 255.0).astype(np.uint8)
        thumb = Image.fromarray(pix_255, mode="L").resize((56, 56), Image.Resampling.NEAREST).convert("RGB")
        img.paste(thumb, (x + 10, y + 20))
        draw.rectangle([(x + 9, y + 19), (x + 66, y + 76)], outline=(200, 200, 200))

    img.save("shots/samples.png", dpi=(160, 160))
    print("[Plots] Saved shots/samples.png")


def save_classdist_plot():
    """Generates shots/classdist.png: Bar chart of class distribution."""
    dist = pipeline_instance.dataset_stats["class_distribution"]
    w, h = 500, 350
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.text((30, 20), "Images per digit (Class Distribution)", fill=(20, 20, 20), font=font_lg)

    origin_x, origin_y = 60, 290
    chart_w, chart_h = 400, 220

    draw.line([(origin_x, origin_y), (origin_x + chart_w, origin_y)], fill=(150, 150, 150), width=2)
    draw.line([(origin_x, origin_y), (origin_x, origin_y - chart_h)], fill=(150, 150, 150), width=2)

    bar_w = 26
    spacing = (chart_w - (10 * bar_w)) // 11
    max_val = 200

    for digit in range(10):
        count = dist[digit]
        bar_h = int((count / max_val) * chart_h)
        bx = origin_x + spacing + digit * (bar_w + spacing)
        by = origin_y - bar_h

        draw.rectangle([(bx, by), (bx + bar_w, origin_y)], fill=(70, 130, 180), outline=(50, 100, 150))
        draw.text((bx + 8, origin_y + 8), str(digit), fill=(30, 30, 30), font=font_md)
        draw.text((bx + 2, by - 16), str(count), fill=(70, 70, 70), font=font_sm)

    img.save("shots/classdist.png", dpi=(160, 160))
    print("[Plots] Saved shots/classdist.png")


def save_cm_plot():
    """Generates shots/cm.png: 10x10 Confusion Matrix heatmap."""
    cm = np.array(pipeline_instance.confusion_matrix)
    w, h = 560, 520
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    best_name = pipeline_instance.best_model_name
    draw.text((30, 15), f"Confusion Matrix ({best_name}) - 360 Test Samples", fill=(20, 20, 20), font=font_lg)

    start_x, start_y = 70, 60
    cell_size = 42

    for r in range(10):
        draw.text((start_x - 30, start_y + r * cell_size + 12), f"T{r}", fill=(60, 60, 60), font=font_bold)
        for c in range(10):
            if r == 0:
                draw.text((start_x + c * cell_size + 14, start_y - 25), f"P{c}", fill=(60, 60, 60), font=font_bold)

            val = cm[r, c]
            x = start_x + c * cell_size
            y = start_y + r * cell_size

            if r == c:
                bg = (37, 99, 235) if val > 25 else (96, 165, 250)
                fg = (255, 255, 255)
            elif val > 0:
                bg = (239, 68, 68)
                fg = (255, 255, 255)
            else:
                bg = (248, 250, 252)
                fg = (148, 163, 184)

            draw.rectangle([(x, y), (x + cell_size - 2, y + cell_size - 2)], fill=bg)
            draw.text((x + 12, y + 12), str(val), fill=fg, font=font_md)

    draw.text((start_x + 90, start_y + 10 * cell_size + 15), "Columns: Predicted | Rows: True", fill=(100, 100, 100), font=font_sm)
    img.save("shots/cm.png", dpi=(160, 160))
    print("[Plots] Saved shots/cm.png")


def save_wrong_plot():
    """Generates shots/wrong.png: Images misclassified by best model."""
    wrong = pipeline_instance.misclassified_samples
    n = max(len(wrong), 1)
    cell_w, cell_h = 100, 130
    w = max(500, n * cell_w + 40)
    h = cell_h + 80

    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.text((20, 15), f"Images the {pipeline_instance.best_model_name} model got wrong ({len(wrong)} total)", fill=(20, 20, 20), font=font_lg)

    for idx, s in enumerate(wrong):
        x = 30 + idx * cell_w
        y = 55

        draw.text((x + 10, y), f"True: {s['true_label']} | Pred: {s['predicted_label']}", fill=(180, 20, 20), font=font_bold)

        arr = np.array(s["pixels"])
        pix_255 = 255 - (np.clip(arr, 0, 16) / 16.0 * 255.0).astype(np.uint8)
        thumb = Image.fromarray(pix_255, mode="L").resize((70, 70), Image.Resampling.NEAREST).convert("RGB")
        img.paste(thumb, (x + 10, y + 25))
        draw.rectangle([(x + 9, y + 24), (x + 80, y + 95)], outline=(200, 200, 200))

    img.save("shots/wrong.png", dpi=(160, 160))
    print("[Plots] Saved shots/wrong.png")


def save_pca_plot():
    """Generates shots/pca.png: PCA Components vs Test Accuracy."""
    pca = pipeline_instance.pca_results
    w, h = 550, 400
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.text((30, 20), "Accuracy with fewer features (SVM + PCA)", fill=(20, 20, 20), font=font_lg)

    ox, oy = 70, 330
    cw, ch = 430, 240

    draw.line([(ox, oy), (ox + cw, oy)], fill=(150, 150, 150), width=2)
    draw.line([(ox, oy), (ox, oy - ch)], fill=(150, 150, 150), width=2)

    # Grid lines
    for pct in [90, 92, 94, 96, 98, 100]:
        gy = oy - int(((pct - 90) / 10.0) * ch)
        draw.line([(ox, gy), (ox + cw, gy)], fill=(240, 240, 240), width=1)
        draw.text((ox - 35, gy - 6), f"{pct}%", fill=(120, 120, 120), font=font_sm)

    points = []
    for item in pca:
        k = item["components"]
        acc = item["accuracy"] * 100
        px = ox + int((k / 64.0) * cw)
        py = oy - int(((acc - 90) / 10.0) * ch)
        points.append((px, py))
        draw.ellipse([(px - 4, py - 4), (px + 4, py + 4)], fill=(5, 150, 105), outline=(0, 100, 60))
        draw.text((px - 8, py - 18), f"{acc:.1f}%", fill=(20, 80, 50), font=font_sm)
        draw.text((px - 6, oy + 8), str(k), fill=(80, 80, 80), font=font_sm)

    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=(5, 150, 105), width=3)

    draw.text((ox + 120, oy + 32), "Number of PCA components (out of 64)", fill=(60, 60, 60), font=font_md)
    img.save("shots/pca.png", dpi=(160, 160))
    print("[Plots] Saved shots/pca.png")


def save_noise_plot():
    """Generates shots/noise.png: Noise robustness curves for 6 models."""
    res = pipeline_instance.noise_results
    w, h = 600, 420
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.text((30, 20), "How well do the models handle noisy images?", fill=(20, 20, 20), font=font_lg)

    ox, oy = 70, 330
    cw, ch = 380, 250

    draw.line([(ox, oy), (ox + cw, oy)], fill=(150, 150, 150), width=2)
    draw.line([(ox, oy), (ox, oy - ch)], fill=(150, 150, 150), width=2)

    for pct in [0, 20, 40, 60, 80, 100]:
        gy = oy - int((pct / 100.0) * ch)
        draw.line([(ox, gy), (ox + cw, gy)], fill=(240, 240, 240), width=1)
        draw.text((ox - 35, gy - 6), f"{pct}%", fill=(120, 120, 120), font=font_sm)

    colors = {
        "SVM (RBF)": (37, 99, 235),
        "KNN": (5, 150, 105),
        "Random Forest": (217, 119, 6),
        "Logistic Regression": (124, 58, 237),
        "Decision Tree": (220, 38, 38),
        "Naive Bayes": (100, 116, 139)
    }

    leg_x = ox + cw + 15
    leg_y = oy - ch + 20

    for name, accs in res.items():
        col = colors.get(name, (0, 0, 0))
        pts = []
        for s_idx, a in enumerate(accs):
            px = ox + int((s_idx / 4.0) * cw)
            py = oy - int((a * 100 / 100.0) * ch)
            pts.append((px, py))
            draw.ellipse([(px - 3, py - 3), (px + 3, py + 3)], fill=col)

        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=col, width=2)

        # Legend entry
        draw.line([(leg_x, leg_y + 6), (leg_x + 18, leg_y + 6)], fill=col, width=3)
        draw.text((leg_x + 24, leg_y), name, fill=(40, 40, 40), font=font_sm)
        leg_y += 24

    for s in range(5):
        draw.text((ox + int((s / 4.0) * cw) - 4, oy + 8), str(s), fill=(80, 80, 80), font=font_sm)

    draw.text((ox + 90, oy + 30), "Gaussian Noise Level (σ = 0 to 4)", fill=(60, 60, 60), font=font_md)
    img.save("shots/noise.png", dpi=(160, 160))
    print("[Plots] Saved shots/noise.png")


if __name__ == "__main__":
    save_samples_plot()
    save_classdist_plot()
    save_cm_plot()
    save_wrong_plot()
    save_pca_plot()
    save_noise_plot()
    print("All plots successfully generated in shots/")
