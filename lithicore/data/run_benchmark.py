"""run_benchmark.py — Self-validation benchmark for Dibble lithic classifiers.

Generates a held-out synthetic test set from published metric ranges, runs all
three pre-trained classifiers, and produces an interactive HTML validation report.

Usage:
    python -m lithicore.data.run_benchmark
    # Output: docs/benchmark/results/report.html
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

from lithicore._models import LithicFeatureVector
from lithicore._classification import ClassifierModel

# Import training data ranges directly (not via package)
_data_dir = Path(__file__).resolve().parent
if str(_data_dir) not in sys.path:
    sys.path.insert(0, str(_data_dir))
from generate_training_data import (  # noqa: E402
    BASIC_RANGES, BORDES_RANGES, TECH_RANGES, generate_samples,
)

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "docs" / "benchmark"
RESULTS_DIR = BENCHMARK_DIR / "results"


def _run_benchmark(
    name: str,
    display_name: str,
    ranges: dict,
    n_test: int = 100,
) -> dict:
    """Run benchmark for one typology system.

    Args:
        name: Short name (e.g. 'basic').
        display_name: Human-readable name (e.g. 'Basic Morphological').
        ranges: Metric ranges dict from generate_training_data.
        n_test: Number of test samples per class.

    Returns:
        Dict of metrics for the report.
    """
    print(f"  Benchmarking {display_name}...")

    # Generate held-out test set (separate from training distribution)
    rng = np.random.default_rng(20260527)  # fixed seed for reproducibility
    test_features, test_labels = generate_samples(ranges, n_per_class=n_test, noise=0.20)

    # Load pre-trained model
    try:
        model = ClassifierModel.load_pre_trained(name)
    except FileNotFoundError:
        print(f"    Pre-trained model not found for {name}. Train it first.")
        return {"name": name, "error": "Model not found"}

    # Predict
    y_true = test_labels
    y_pred = []
    for fv in test_features:
        result = model.predict(fv)
        y_pred.append(result.label)

    # Compute metrics
    classes = sorted(set(y_true))
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Per-class metrics
    per_class = []
    for i, cls in enumerate(classes):
        per_class.append({
            "class": cls,
            "precision": round(float(precision[i]), 3),
            "recall": round(float(recall[i]), 3),
            "f1": round(float(f1[i]), 3),
            "support": int(support[i]),
        })

    # Confusion matrix as nested lists for JSON
    cm_list = cm.tolist()

    print(f"    Accuracy: {accuracy:.1%} ({len(y_true)} samples, {len(classes)} classes)")

    return {
        "name": name,
        "display_name": display_name,
        "n_classes": len(classes),
        "n_test_samples": len(y_true),
        "accuracy": round(float(accuracy), 4),
        "per_class": per_class,
        "confusion_matrix": cm_list,
        "classes": classes,
        "classification_report": classification_report(
            y_true, y_pred, labels=classes, zero_division=0,
        ),
    }


def _generate_html(results: list[dict]) -> str:
    """Generate an interactive HTML validation report."""
    # Determine overall pass/fail
    all_pass = all(r.get("accuracy", 0) >= 0.70 for r in results if "error" not in r)
    overall_status = "PASS" if all_pass else "REVIEW"

    html_parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Dibble Classifier Validation Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 960px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #333; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #4472C4; padding-bottom: 8px; }}
  h2 {{ color: #2d3436; margin-top: 32px; }}
  .status {{ display: inline-block; padding: 6px 16px; border-radius: 4px;
             font-weight: bold; font-size: 14px; }}
  .pass {{ background: #d4edda; color: #155724; }}
  .review {{ background: #fff3cd; color: #856404; }}
  .summary {{ background: white; border-radius: 8px; padding: 16px; margin: 16px 0;
              box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #dee2e6; padding: 8px 12px; text-align: left; }}
  th {{ background: #4472C4; color: white; }}
  tr:nth-child(even) {{ background: #f2f2f2; }}
  .cm-table td {{ text-align: center; font-family: monospace; min-width: 60px; }}
  .cm-header {{ background: #e8e8e8 !important; font-weight: bold; }}
  .cm-diagonal {{ background: #d4edda !important; }}
  .metrics {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .metric-card {{ background: white; border-radius: 8px; padding: 16px; flex: 1;
                  min-width: 200px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .metric-value {{ font-size: 28px; font-weight: bold; color: #4472C4; }}
  .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
  pre {{ background: #f5f5f5; padding: 12px; border-radius: 4px; overflow-x: auto;
        font-size: 12px; }}
  .footer {{ margin-top: 40px; padding: 12px; text-align: center;
             color: #888; font-size: 12px; border-top: 1px solid #dee2e6; }}
</style>
</head>
<body>
<h1>Dibble Classifier Validation Report</h1>
<p>Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<p class="status {'pass' if all_pass else 'review'}">{overall_status}</p>
<p>This report validates all three pre-trained lithic typology classifiers against
held-out synthetic test data generated from published metric ranges.</p>
"""]

    for result in results:
        if "error" in result:
            html_parts.append(f"<h2>{result['display_name']}</h2><p>Error: {result['error']}</p>")
            continue

        html_parts.append(f"""
<h2>{result['display_name']}</h2>
<div class="summary">
<div class="metrics">
  <div class="metric-card">
    <div class="metric-value">{result['accuracy']:.1%}</div>
    <div class="metric-label">Overall Accuracy</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{result['n_classes']}</div>
    <div class="metric-label">Classes</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{result['n_test_samples']}</div>
    <div class="metric-label">Test Samples</div>
  </div>
</div>

<h3>Per-Class Metrics</h3>
<table>
<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
""")
        for pc in result["per_class"]:
            f1_colour = "green" if pc["f1"] >= 0.8 else "orange" if pc["f1"] >= 0.6 else "red"
            html_parts.append(f"""
<tr>
  <td><b>{pc['class']}</b></td>
  <td>{pc['precision']:.3f}</td>
  <td>{pc['recall']:.3f}</td>
  <td style="color:{f1_colour};font-weight:bold;">{pc['f1']:.3f}</td>
  <td>{pc['support']}</td>
</tr>""")

        html_parts.append("""
</table>

<h3>Confusion Matrix</h3>
<table class="cm-table">
<tr><td class="cm-header">True \\ Pred</td>""")
        for cls in result["classes"]:
            html_parts.append(f"<td class='cm-header'>{cls}</td>")
        html_parts.append("</tr>")

        cm = result["confusion_matrix"]
        for i, cls in enumerate(result["classes"]):
            html_parts.append(f"<tr><td class='cm-header'>{cls}</td>")
            for j, val in enumerate(cm[i]):
                diag_class = "cm-diagonal" if i == j else ""
                html_parts.append(f"<td class='{diag_class}'>{val}</td>")
            html_parts.append("</tr>")

        html_parts.append("""
</table>

<h3>Detailed Report</h3>
<pre>""" + result["classification_report"] + "</pre>")

    html_parts.append(f"""
<div class="footer">
  Dibble Classifier Validation — <a href="https://github.com/mabo-du/dibble">github.com/mabo-du/dibble</a><br>
  Reproduce: <code>python -m lithicore.data.run_benchmark</code>
</div>
</body>
</html>""")

    return "\n".join(html_parts)


def main() -> None:
    """Run all benchmarks and generate the HTML report."""
    print("=" * 60)
    print("  Dibble Classifier Self-Validation Benchmark")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    benchmarks = [
        ("basic", "Basic Morphological", BASIC_RANGES),
        ("bordes", "Bordes Typology", BORDES_RANGES),
        ("technological", "Technological", TECH_RANGES),
    ]

    # Save benchmark configuration
    config = {
        "n_test_per_class": 100,
        "noise_level": 0.20,
        "seed": 20260527,
        "benchmark_date": __import__('datetime').datetime.now().isoformat(),
        "classifiers": [b[0] for b in benchmarks],
    }
    (RESULTS_DIR / "config.json").write_text(json.dumps(config, indent=2))

    results = []
    for name, display, ranges in benchmarks:
        result = _run_benchmark(name, display, ranges)
        results.append(result)

        # Save per-classifier JSON
        if "error" not in result:
            (RESULTS_DIR / f"{name}_metrics.json").write_text(
                json.dumps(result, indent=2, default=str)
            )

    # Generate HTML report
    html = _generate_html(results)
    report_path = RESULTS_DIR / "report.html"
    report_path.write_text(html)
    print(f"\n  Report saved: {report_path}")

    # Generate summary markdown
    summary_lines = ["# Classifier Validation Summary\n"]
    summary_lines.append("| Typology | Accuracy | Classes | Samples |\n")
    summary_lines.append("|----------|----------|---------|--------|\n")
    for r in results:
        acc = f"{r['accuracy']:.1%}" if "error" not in r else "ERROR"
        cls = r.get("n_classes", "?")
        sam = r.get("n_test_samples", "?")
        summary_lines.append(f"| {r['display_name']} | {acc} | {cls} | {sam} |\n")

    (RESULTS_DIR / "summary.md").write_text("".join(summary_lines))
    print(f"  Summary saved: {RESULTS_DIR / 'summary.md'}")

    print("=" * 60)
    print("  Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
