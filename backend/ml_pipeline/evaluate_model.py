"""
Krishi Rakshak AI - Model Evaluation Pipeline
Team: Codex Verse
Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Classification Report
Held-Out Data: data/test/ (Guaranteed zero data leakage)
"""

import json
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def evaluate():
    print("=" * 70)
    print("[*] KRISHI RAKSHAK AI - MODEL EVALUATION PIPELINE")
    print(f"Team: {config.TEAM_NAME}")
    print(f"Target Test Dataset: {config.TEST_DIR}")
    print("=" * 70)

    # 1. Check Artifacts
    target_model_path = config.MODEL_PATH if config.MODEL_PATH.exists() else config.MODEL_H5_PATH
    if not target_model_path.exists():
        print(f"[!] Error: Model not found at {config.MODEL_PATH}")
        print("Please train the model first by running: python train_model.py")
        sys.exit(1)

    if not config.TEST_DIR.exists() or not any(config.TEST_DIR.iterdir()):
        print(f"[!] Error: Held-out test data not found at {config.TEST_DIR}")
        print("Please prepare the dataset first by running: python download_dataset.py")
        sys.exit(1)

    # 2. Load Model & Labels
    print(f"Loading trained model from {target_model_path}...")
    model = tf.keras.models.load_model(str(target_model_path))

    with open(config.CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    # Ensure ordered list of class names
    if isinstance(list(class_indices.keys())[0], int) or list(class_indices.keys())[0].isdigit():
        class_names = [class_indices[str(i)] if str(i) in class_indices else class_indices[i] for i in range(len(class_indices))]
    else:
        class_names = list(class_indices.keys())

    # 3. Load Held-Out Test Set (No Data Leakage)
    print(f"Loading held-out test data from {config.TEST_DIR}...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    # Extract all true labels and batch images
    y_true_list = []
    y_pred_probs_list = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true_list.extend(labels.numpy())
        y_pred_probs_list.append(preds)

    y_true = np.array(y_true_list)
    y_pred_probs = np.vstack(y_pred_probs_list)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 4. Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    summary_metrics = {
        "accuracy": round(float(acc), 4),
        "precision_weighted": round(float(prec_w), 4),
        "recall_weighted": round(float(rec_w), 4),
        "f1_score_weighted": round(float(f1_w), 4),
        "precision_macro": round(float(prec_m), 4),
        "recall_macro": round(float(rec_m), 4),
        "f1_score_macro": round(float(f1_m), 4),
        "total_test_samples": len(y_true),
        "number_of_classes": len(class_names)
    }

    full_report = {
        "summary": summary_metrics,
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "class_names": class_names
    }

    # 5. Save Reports
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
        
    with open(config.METRICS_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    # Try saving a clean confusion matrix heatmap plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
        plt.title("Krishi Rakshak AI - Crop Disease Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, [c.split("___")[-1][:12] for c in class_names], rotation=45, ha="right", fontsize=8)
        plt.yticks(tick_marks, [c.split("___")[-1][:12] for c in class_names], fontsize=8)
        plt.ylabel("True Class")
        plt.xlabel("Predicted Class")
        plt.tight_layout()
        plt.savefig(str(config.CONFUSION_MATRIX_PATH), dpi=200)
        plt.close()
        print(f"Saved confusion matrix chart to {config.CONFUSION_MATRIX_PATH}")
    except Exception as e:
        print(f"Notice: Matplotlib plot skipped ({e})")

    # 6. Pretty Print Report for Judges
    print("\n" + "=" * 70)
    print("[RESULTS] FINAL EVALUATION RESULTS ON HELD-OUT TEST DATA")
    print("=" * 70)
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Weighted F1-Score: {f1_w * 100:.2f}%")
    print(f"Weighted Recall:   {rec_w * 100:.2f}%")
    print(f"Weighted Precision:{prec_w * 100:.2f}%")
    print(f"Total Test Images: {len(y_true)}")
    print("-" * 70)
    print(f"{'Class Name':<32} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 70)
    for c_name in class_names:
        c_stats = report_dict.get(c_name, {})
        p = c_stats.get("precision", 0.0)
        r = c_stats.get("recall", 0.0)
        f = c_stats.get("f1-score", 0.0)
        print(f"{c_name:<32} | {p:<10.3f} | {r:<10.3f} | {f:<10.3f}")
    print("=" * 70)
    print(f"Full Report: {config.EVALUATION_REPORT_PATH}")
    print(f"Summary:     {config.METRICS_SUMMARY_PATH}")


if __name__ == "__main__":
    evaluate()
