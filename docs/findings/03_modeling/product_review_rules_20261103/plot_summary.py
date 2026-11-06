"""Rebuild this report's aggregate workload chart; no prediction inputs are read."""
from pathlib import Path

import numpy as np
import pandas as pd

from catalogiq.paths import ROOT_PATH
from catalogiq.product_rule_plots import pyplot, save


def main():
    directory = Path(__file__).resolve().parent
    table = pd.read_csv(directory/"audit_policy_comparison.csv")
    output = directory/"workload_and_mistakes"
    if output.with_suffix(".png").exists() or output.with_suffix(".svg").exists():
        raise FileExistsError("Use a fresh output location rather than overwrite a report chart")
    cache = ROOT_PATH/".cache/calibration_plot_deps"
    plt = pyplot(cache if cache.is_dir() else None)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    y = np.arange(len(table))
    labels = [f"{'Segment' if r.model == 'segment' else 'Brand lookup'}: "
              f"{'candidate' if r.policy == 'selected_experimental_rule' else 'baseline'}"
              for r in table.itertuples()]
    auto = table.coverage.to_numpy()*100
    review = 100-auto
    errors = table.errors.to_numpy()/table.population_rows.to_numpy()*100
    axes[0].barh(y, auto, color="#2563eb", label="Eligible for automatic classification")
    axes[0].barh(y, review, left=auto, color="#cbd5e1", label="Human review")
    for i, (a, r) in enumerate(zip(auto, review)):
        axes[0].text(a/2, i, f"{a:.0f}", ha="center", va="center", color="white")
        axes[0].text(a+r/2, i, f"{r:.0f}", ha="center", va="center")
    axes[0].set(yticks=y, yticklabels=labels, xlim=(0, 100), xlabel="Products out of 100")
    axes[0].set_title("Blue: automatic; gray: human review", fontsize=10)
    axes[1].barh(y, errors, color="#d97706")
    for i, e in enumerate(errors):
        axes[1].text(e+.015, i, f"{e:.2f}", va="center")
    axes[1].set(yticks=y, yticklabels=labels, xlim=(0, 1), xlabel="Wrong automatic decisions per 100 incoming products")
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Fewer automatic mistakes require more human review\nHistorical development replay; experimental policies remain disabled")
    save(fig, output, plt)


if __name__ == "__main__":
    main()
