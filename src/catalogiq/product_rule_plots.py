"""Standalone charts for fixed-floor product review policies, from saved tables."""
from pathlib import Path
import sys

import pandas as pd


def pyplot(plot_cache):
    if plot_cache:
        sys.path.append(str(Path(plot_cache).resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt": "catalogiq-product-rules-v1", "font.size": 10})
    return plt


def save(figure, path, plt):
    figure.tight_layout(rect=(0, 0, 1, .94))
    figure.savefig(path.with_suffix(".png"), dpi=140)
    figure.savefig(path.with_suffix(".svg"), metadata={"Date": None})
    plt.close(figure)


def plot_discovery(output, *, plot_cache=None):
    import json
    output = Path(output)
    table = pd.read_csv(output/"selection_policy_comparison.csv")
    choices = json.loads((output/"selected_rules.json").read_text())
    plt = pyplot(plot_cache)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for index, (model, data) in enumerate(table.groupby("model", sort=False)):
        left, right = axes[index]
        left.scatter(data.coverage*100, data.accepted_error_rate*100, color="#64748b", alpha=.55,
                     label="Fixed-floor candidate policies")
        right.scatter(data.coverage*100, data.beta_lower*100, color="#2563eb", alpha=.6,
                      label="Nominal 95% lower accuracy")
        right.scatter(data.coverage*100, data.beta_lower_search_adjusted*100, color="#d97706", alpha=.6,
                      label="Search-adjusted lower accuracy")
        for rule_id, color, label in (("all__all", "#111827", "Baseline"),
                                      (choices[model]["rule_id"], "#059669", "Selected before audit")):
            point = data.loc[data.rule_id.eq(rule_id)]
            if len(point):
                left.scatter(point.coverage*100, point.accepted_error_rate*100, color=color, marker="*", s=180, label=label)
        left.axhline(1, color="#dc2626", ls="--", label="1% error goal")
        right.axhline(99, color="#dc2626", ls="--", label="99% lower-bound goal")
        left.set(xlabel="Automatically accepted products (%)", ylabel="Mistakes among accepted (%)",
                 title=f"{model}: selection tradeoff", xlim=(0, 100), ylim=(0, 6))
        right.set(xlabel="Automatically accepted products (%)", ylabel="Lower accuracy bound (%)",
                  title=f"{model}: evidence after searching 60 rules", xlim=(0, 100), ylim=(90, 100.1))
        for ax in (left, right):
            ax.grid(alpha=.2)
            ax.legend(fontsize=7, loc="best")
    fig.suptitle("Rule selection uses existing non-audit development groups; cutoffs are unchanged")
    save(fig, output/"selection_tradeoff", plt)


def plot_audit(output, *, plot_cache=None):
    output = Path(output)
    table = pd.read_csv(output/"audit_policy_comparison.csv")
    plt = pyplot(plot_cache)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (model, data) in zip(axes, table.groupby("model", sort=False)):
        for _, row in data.iterrows():
            selected = row.policy == "selected_experimental_rule"
            color = "#059669" if selected else "#2563eb"
            label = "Frozen experimental rule" if selected else "Fixed-floor baseline"
            risk = 100*row.accepted_error_rate
            x = 100*row.coverage
            ax.scatter([x], [risk], color=color, s=90, label=label)
            if pd.notna(row.beta_lower):
                ax.vlines(x, risk, 100*(1-row.beta_lower), color=color, linestyles="dotted", lw=2)
            ax.annotate(f"{int(row.errors)} mistakes / {int(row.accepted_rows)} accepted", (x, risk),
                        xytext=(0, 12 if selected else -22), textcoords="offset points", ha="center", fontsize=8)
        ax.axhline(1, color="#dc2626", ls="--", label="1% error goal")
        ax.set(xlim=(0, 100), ylim=(0, 4), xlabel="Automatically accepted products (%)",
               ylabel="Mistakes among accepted (%)", title=model)
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Historical audit replay: two frozen policies only\nDotted lines show the preferred 95% upper error bounds; no untouched confirmation")
    save(fig, output/"audit_tradeoff", plt)
