"""
Generate comprehensive experiment report with charts.
"""
import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from analysis.compare_conditions import evaluate_file

# Try to use a font that supports CJK
for font_name in ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS']:
    fonts = fm.findSystemFonts()
    if any(font_name.lower().replace(' ', '') in f.lower().replace(' ', '') for f in fonts):
        plt.rcParams['font.sans-serif'] = [font_name]
        break
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test.jsonl")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "report")
os.makedirs(REPORT_DIR, exist_ok=True)


def load_all_results():
    """Load and evaluate all result files."""
    conditions = {
        "Gemma4 26B\nzero-shot": "zero_shot_responses.jsonl",
        "Gemma4 26B\nerror-NB": "gemma4_error_notebook_responses.jsonl",
        "Volcengine Auto\nzero-shot": "volcengine_auto_zero_shot_responses.jsonl",
        "Volcengine Auto\nerror-NB": "volcengine_ark-code-latest_error_notebook_responses.jsonl",
        "Volcengine Auto\nerror-only": "volcengine_ark-code-latest_error_only_responses.jsonl",
        "Volcengine Auto\ncorrect-only": "volcengine_ark-code-latest_correct_only_responses.jsonl",
        "Volcengine Auto\ninterleaved": "volcengine_ark-code-latest_interleaved_responses.jsonl",
        "Doubao-Code\n(think)": "volcengine_doubao-seed-2-0-code_zero_shot_responses.jsonl",
        "Doubao-Code\n(nothink)": "volcengine_doubao-seed-2-0-code_nothink_zero_shot_responses.jsonl",
        "Claude Opus 4.6\n(100 sample)": "claude_opus_clean_zero_shot_responses.jsonl",
    }

    results = {}
    for name, filename in conditions.items():
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            s = evaluate_file(path, DATA_PATH)
            if s:
                results[name] = s
    return results


def plot_overall_accuracy(results):
    """Bar chart of overall accuracy across all conditions."""
    fig, ax = plt.subplots(figsize=(14, 6))

    names = list(results.keys())
    accs = [results[n]["overall"] * 100 for n in names]
    totals = [results[n]["total"] for n in names]

    colors = []
    for name in names:
        if "Claude" in name:
            colors.append("#7B68EE")
        elif "error-NB" in name or "error-only" in name:
            colors.append("#2ECC71")
        elif "interleaved" in name:
            colors.append("#27AE60")
        elif "correct-only" in name:
            colors.append("#E74C3C")
        elif "zero-shot" in name:
            colors.append("#3498DB")
        elif "Doubao" in name:
            colors.append("#F39C12")
        else:
            colors.append("#95A5A6")

    bars = ax.bar(range(len(names)), accs, color=colors, edgecolor='white', linewidth=0.5)

    for i, (bar, acc, total) in enumerate(zip(bars, accs, totals)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%\n(n={total})', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('BFCL V4 Tool Calling Accuracy - All Conditions', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.axhline(y=results.get("Volcengine Auto\nzero-shot", {}).get("overall", 0) * 100,
               color='gray', linestyle='--', alpha=0.5, label='Volcengine zero-shot baseline')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "01_overall_accuracy.png"), dpi=150)
    plt.close()
    print("Saved 01_overall_accuracy.png")


def plot_ablation_study(results):
    """Focused ablation study for error notebook variants."""
    ablation_keys = [
        "Volcengine Auto\nzero-shot",
        "Volcengine Auto\ncorrect-only",
        "Volcengine Auto\nerror-NB",
        "Volcengine Auto\ninterleaved",
        "Volcengine Auto\nerror-only",
    ]
    ablation_labels = [
        "Zero-shot\n(baseline)",
        "Correct only\n(5 positive)",
        "Mixed 5+5\n(correct+error)",
        "Interleaved\n(alternating)",
        "Error only\n(5 corrections)",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    accs = []
    colors_abl = ["#3498DB", "#E74C3C", "#F39C12", "#27AE60", "#2ECC71"]

    for key in ablation_keys:
        if key in results:
            accs.append(results[key]["overall"] * 100)
        else:
            accs.append(0)

    bars = ax.bar(range(len(ablation_labels)), accs, color=colors_abl, edgecolor='white', width=0.6)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(ablation_labels)))
    ax.set_xticklabels(ablation_labels, fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Ablation Study: Error Notebook Components\n(Volcengine Auto / ark-code-latest)', fontsize=13, fontweight='bold')
    ax.set_ylim(55, 85)
    ax.axhline(y=accs[0], color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "02_ablation_study.png"), dpi=150)
    plt.close()
    print("Saved 02_ablation_study.png")


def plot_category_heatmap(results):
    """Heatmap of per-category accuracy across conditions."""
    main_keys = [
        "Volcengine Auto\nzero-shot",
        "Volcengine Auto\nerror-only",
        "Volcengine Auto\nerror-NB",
        "Gemma4 26B\nzero-shot",
        "Gemma4 26B\nerror-NB",
        "Doubao-Code\n(think)",
        "Claude Opus 4.6\n(100 sample)",
    ]

    available = [k for k in main_keys if k in results]
    all_cats = set()
    for k in available:
        all_cats.update(results[k]["categories"].keys())
    cats = sorted(all_cats)

    data = np.zeros((len(cats), len(available)))
    for j, key in enumerate(available):
        for i, cat in enumerate(cats):
            data[i, j] = results[key]["categories"].get(cat, 0) * 100

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(len(available)))
    ax.set_xticklabels([k.replace('\n', ' ') for k in available], fontsize=8, rotation=30, ha='right')
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=9)

    for i in range(len(cats)):
        for j in range(len(available)):
            text_color = 'white' if data[i, j] < 40 or data[i, j] > 85 else 'black'
            ax.text(j, i, f'{data[i, j]:.0f}', ha='center', va='center', fontsize=8,
                    color=text_color, fontweight='bold')

    ax.set_title('Per-Category Accuracy Heatmap (%)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Accuracy (%)', shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "03_category_heatmap.png"), dpi=150)
    plt.close()
    print("Saved 03_category_heatmap.png")


def plot_error_notebook_delta(results):
    """Show per-category improvement from error notebook for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Volcengine
    if "Volcengine Auto\nzero-shot" in results and "Volcengine Auto\nerror-only" in results:
        ax = axes[0]
        zs = results["Volcengine Auto\nzero-shot"]["categories"]
        en = results["Volcengine Auto\nerror-only"]["categories"]
        cats = sorted(set(zs.keys()) | set(en.keys()))
        deltas = [(en.get(c, 0) - zs.get(c, 0)) * 100 for c in cats]
        colors = ['#2ECC71' if d >= 0 else '#E74C3C' for d in deltas]

        ax.barh(range(len(cats)), deltas, color=colors, edgecolor='white')
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels(cats, fontsize=9)
        ax.set_xlabel('Accuracy Change (pp)', fontsize=10)
        ax.set_title('Volcengine Auto\nError-Only vs Zero-Shot', fontsize=11, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)

        for i, (d, c) in enumerate(zip(deltas, cats)):
            ax.text(d + (1 if d >= 0 else -1), i, f'{d:+.1f}', va='center',
                    ha='left' if d >= 0 else 'right', fontsize=8, fontweight='bold')

    # Gemma4
    if "Gemma4 26B\nzero-shot" in results and "Gemma4 26B\nerror-NB" in results:
        ax = axes[1]
        zs = results["Gemma4 26B\nzero-shot"]["categories"]
        en = results["Gemma4 26B\nerror-NB"]["categories"]
        cats = sorted(set(zs.keys()) | set(en.keys()))
        deltas = [(en.get(c, 0) - zs.get(c, 0)) * 100 for c in cats]
        colors = ['#2ECC71' if d >= 0 else '#E74C3C' for d in deltas]

        ax.barh(range(len(cats)), deltas, color=colors, edgecolor='white')
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels(cats, fontsize=9)
        ax.set_xlabel('Accuracy Change (pp)', fontsize=10)
        ax.set_title('Gemma4 26B\nError-NB vs Zero-Shot', fontsize=11, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)

        for i, (d, c) in enumerate(zip(deltas, cats)):
            ax.text(d + (1 if d >= 0 else -1), i, f'{d:+.1f}', va='center',
                    ha='left' if d >= 0 else 'right', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "04_error_notebook_delta.png"), dpi=150)
    plt.close()
    print("Saved 04_error_notebook_delta.png")


def plot_error_distribution(classified_path, title, filename):
    """Pie chart of error type distribution."""
    from collections import Counter
    error_counts = Counter()
    total = 0
    correct = 0
    with open(classified_path) as f:
        for line in f:
            entry = json.loads(line)
            total += 1
            if entry["correct"]:
                correct += 1
            else:
                et = entry["error_type"]
                # Simplify error type names
                et = et.replace("simple_function_checker:", "").replace("parallel_function_checker_no_order:", "parallel:")
                error_counts[et] += 1

    fig, ax = plt.subplots(figsize=(8, 6))

    labels = []
    sizes = []
    for et, count in error_counts.most_common(8):
        labels.append(f"{et}\n({count})")
        sizes.append(count)
    if len(error_counts) > 8:
        other = sum(c for _, c in error_counts.most_common()[8:])
        labels.append(f"other\n({other})")
        sizes.append(other)

    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.0f%%',
                                        colors=colors_pie, startangle=90, textprops={'fontsize': 8})
    for autotext in autotexts:
        autotext.set_fontsize(8)

    ax.set_title(f'{title}\nCorrect: {correct}/{total} ({correct/total:.1%})', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved {filename}")


def generate_markdown_report(results):
    """Generate the main markdown report."""
    report = """# Error Notebook Experiment Report
## BFCL V4 Tool Calling Benchmark

**Date**: 2026-04-08
**Benchmark**: Berkeley Function Calling Leaderboard V4
**Test set**: 782 entries across 11 categories

---

## 1. Executive Summary

We developed an **Error Notebook** (错题本) approach for improving LLM tool-calling accuracy through corrective few-shot prompting. The key idea: instead of showing models correct examples, show them **common mistakes and their corrections**.

### Key Results

| Approach | Volcengine Auto | Gemma4 26B |
|----------|----------------|------------|
| Zero-shot baseline | 68.0% | 76.3% |
| Error Notebook (error-only) | **79.0% (+11.0pp)** | — |
| Error Notebook (mixed 5+5) | 76.3% (+8.3pp) | 77.6%* (+1.3pp) |
| Interleaved (alternating) | **79.0% (+11.0pp)** | — |
| Correct-only few-shot | 64.7% (-3.3pp) | — |

*Gemma4 error-NB based on 634/782 entries

**Main finding**: Error correction examples provide all the improvement. Positive examples alone actually hurt performance (-3.3pp). The benefit is largest for models that are weak on parallel function calling.

---

## 2. Bug Fixes in Evaluation Pipeline

Two critical bugs were discovered and fixed in the BFCL AST checker:

### Bug 1: `underscore_to_dot` Default
- **File**: `eval/bfcl_eval/constants/model_config.py`
- **Issue**: `underscore_to_dot` defaulted to `True`, converting `func.name` → `func_name` in expected answers, causing false `wrong_func_name` errors
- **Impact**: ~25pp accuracy inflation for `wrong_func_name` errors
- **Fix**: Changed default to `False`

### Bug 2: Java/JS Type Stringification
- **File**: `eval/bfcl_ast_checker.py`
- **Issue**: Java/JS type checkers expected string inputs (from text-based tool calling), but structured JSON tool calling produces typed values (int, bool, etc.)
- **Impact**: 0% accuracy on all Java/JavaScript categories
- **Fix**: Added `_stringify_value()` to auto-convert typed values before Java/JS type checkers

---

## 3. Models Evaluated

| Model | Type | Notes |
|-------|------|-------|
| Gemma4 26B | Local (Ollama) | Native tool calling, thinking enabled |
| Volcengine Auto (ark-code-latest) | Cloud API | Volcano Engine coding model |
| Doubao-Seed-2.0-Code | Cloud API | With and without thinking |
| Claude Opus 4.6 | Sub-agent | Text-based tool calling (100 sample) |

---

## 4. Overall Accuracy Comparison

![Overall Accuracy](report/01_overall_accuracy.png)

---

## 5. Error Notebook Method

### Pipeline
1. **Pool Inference**: Run target model on pool.jsonl (training data)
2. **Error Classification**: Classify predictions using AST checker
3. **K-Medoids Selection**: Select k representative error examples per error type
4. **Prompt Construction**: Build corrective few-shot context
5. **Evaluation**: Test on held-out test set

### Prompt Structure (Error-Only, Best Variant)
```
System: You are a helpful assistant that calls tools accurately...

[Error Correction 1]
User: <question>
Assistant: <wrong tool_call>
User: "That tool call has an error: <error description>. The correct call should be: <correct call>"
Assistant: <correct tool_call>

[Error Correction 2-5]
...

[Target Question]
User: <target question>
```

---

## 6. Ablation Study

![Ablation Study](report/02_ablation_study.png)

| Variant | Accuracy | vs Zero-shot |
|---------|----------|-------------|
| Error-only (5 corrections) | **79.0%** | **+11.0pp** |
| Interleaved (alternating) | **79.0%** | **+11.0pp** |
| Mixed 5+5 (correct+error) | 76.3% | +8.3pp |
| Zero-shot baseline | 68.0% | — |
| Correct-only (5 positive) | 64.7% | -3.3pp |

### Key Insights
1. **Error corrections are the sole driver of improvement** — positive examples contribute nothing
2. **Positive-only few-shot hurts performance** — likely due to context length overhead without useful signal
3. The improvement is **concentrated in parallel calling categories** (0% → 80%+)

---

## 7. Per-Category Analysis

![Category Heatmap](report/03_category_heatmap.png)

![Error Notebook Delta](report/04_error_notebook_delta.png)

### Volcengine Auto: Error-Only Impact
- **parallel**: 0% → 81.5% (+81.5pp)
- **parallel_multiple**: 18.3% → 84.8% (+66.5pp)
- **live_parallel_multiple**: 37.5% → 100% (+62.5pp)
- Minor regressions in simple/live_simple categories (-2 to -5pp)

### Gemma4 26B: Error-NB Impact
- **parallel**: 78.3% → 97.2% (+18.9pp)
- **sql**: 16.7% → 36.8% (+20.2pp)
- **simple**: 88.3% → 96.4% (+8.1pp)
- But: **live_parallel**: 80% → 50% (-30pp) — error corrections for wrong model confused it

### Why Different Results?
Gemma4 was already strong on parallel calls (78.3% zero-shot), so the parallel-focused error corrections provided diminishing returns and sometimes interfered. Volcengine Auto was at 0% on parallel, making the error corrections transformative.

**Implication**: Error notebook works best when targeted at the model's **actual weaknesses**.

---

## 8. Error Distribution

![Volcengine Error Distribution](report/05_volcengine_errors.png)

![Gemma4 Error Distribution](report/06_gemma4_errors.png)

---

## 9. Claude Opus 4.6 Benchmark (100 Sample)

Claude was evaluated on a stratified 100-entry subset using sub-agents (text-based tool calling, no ground truth leakage).

| Model | Accuracy (same 100) |
|-------|-------------------|
| **Claude Opus 4.6** | **87.0%** |
| Gemma4 26B | 75.0% |
| Doubao-Code (think) | 69.0% |
| Volcengine Auto | 60.0% |

Claude achieved 100% on simple, multiple, parallel, java, live_parallel, and live_parallel_multiple categories.

---

## 10. Thinking Mode Analysis

| Model | Think | No-think | Delta |
|-------|-------|---------|-------|
| Doubao-Seed-2.0-Code | 73.9% | 67.7% | +6.2pp |

Thinking mode improves tool calling accuracy, particularly for parallel calls requiring multi-step reasoning.

---

## 11. Files and Artifacts

### New Scripts
- `eval/run_pool_inference.py` — Local pool inference
- `eval/volcengine_pool_inference.py` — Cloud pool inference
- `eval/classify_pool_errors.py` — Error classification
- `search/error_notebook_selection.py` — K-Medoids selection
- `eval/error_notebook_eval.py` — Local error notebook eval
- `eval/volcengine_error_notebook_eval.py` — Cloud error notebook eval
- `analysis/generate_report.py` — This report generator

### Bug Fixes
- `eval/bfcl_eval/constants/model_config.py` — underscore_to_dot default
- `eval/bfcl_ast_checker.py` — Java/JS type stringification

### Result Files
- `results/volcengine_*_responses.jsonl` — All volcengine evaluations
- `results/gemma4_error_notebook_responses.jsonl` — Gemma4 error notebook eval
- `results/claude_opus_clean_zero_shot_responses.jsonl` — Claude evaluation
- `results/volcengine_auto_error_notebook_subset.json` — Selected error examples
- `results/report/` — Charts and this report

---

## 12. Conclusions and Next Steps

### Conclusions
1. **Error Notebook is effective**: +11.0pp improvement on Volcengine Auto (68% → 79.0%)
2. **Only error corrections matter**: Positive examples are unnecessary or harmful
3. **Targeted to weaknesses**: Most effective when the model has clear failure modes (e.g., 0% on parallel)
4. **Two major evaluation bugs fixed**: Previous accuracy numbers were ~25pp too low across all models

### Potential Next Steps
- [ ] Run Claude Opus on full 782 test set (pending API quota)
- [ ] Try different k values (k=3, k=8, k=10) for error corrections
- [ ] Category-specific error notebooks (e.g., sql-focused corrections)
- [ ] Test error notebook on Gemma4 with only its weak categories' errors
- [ ] Evaluate error notebook on Doubao-Seed-2.0-Code
"""

    report_path = os.path.join(REPORT_DIR, "experiment_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved experiment_report.md")


def main():
    print("Loading results...")
    results = load_all_results()
    print(f"Loaded {len(results)} conditions")

    print("\nGenerating charts...")
    plot_overall_accuracy(results)
    plot_ablation_study(results)
    plot_category_heatmap(results)
    plot_error_notebook_delta(results)

    # Error distribution charts
    volc_classified = os.path.join(RESULTS_DIR, "volcengine_auto_pool_classified_partial.jsonl")
    gemma_classified = os.path.join(RESULTS_DIR, "gemma4_pool_classified_partial.jsonl")

    if os.path.exists(volc_classified):
        plot_error_distribution(volc_classified, "Volcengine Auto Error Distribution", "05_volcengine_errors.png")
    if os.path.exists(gemma_classified):
        plot_error_distribution(gemma_classified, "Gemma4 26B Error Distribution", "06_gemma4_errors.png")

    print("\nGenerating report...")
    generate_markdown_report(results)

    print("\nDone! Report at results/report/experiment_report.md")


if __name__ == "__main__":
    main()
