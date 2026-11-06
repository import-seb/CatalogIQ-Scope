"""Standalone scientific charts from saved experiment tables; no source loading."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def plot_experiment(output, names, *, plot_cache=None):
    if plot_cache:
        # Append so the pinned scientific environment keeps its existing NumPy.
        sys.path.append(str(Path(plot_cache).resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"svg.hashsalt": "catalogiq-confidence-v1", "font.size": 10})
    output = Path(output)
    colors = {"raw": "#2563eb", "temperature": "#d97706"}
    figure, axes = plt.subplots(len(names), 2, figsize=(12, 3.5*len(names)), squeeze=False)
    for index, name in enumerate(names):
        directory = output/name
        bins = pd.read_csv(directory/"reliability.csv")
        audit = bins.loc[bins.subset.eq("policy_audit")]
        thresholds = pd.read_csv(directory/"thresholds.csv")
        risks = thresholds.loc[thresholds.subset.eq("policy_audit")]
        fig, row = plt.subplots(1, 2, figsize=(12, 4.3))
        for pair in ((row[0], row[1]), (axes[index, 0], axes[index, 1])):
            left, right = pair
            left.plot([0, 1], [0, 1], "--", color="#94a3b8", label="Perfect confidence")
            for variant, color in colors.items():
                data = audit.loc[audit.variant.eq(variant) & audit.rows.gt(0)]
                left.plot(data.mean_confidence, data.accuracy, "o-", color=color, label=variant)
                curve = risks.loc[risks.variant.eq(variant)].sort_values("coverage")
                right.plot(curve.coverage*100, curve.accepted_error_rate*100, color=color, label=variant)
            right.axhline(1, ls="--", color="#dc2626", label="1% error target")
            left.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Reported confidence", ylabel="Observed accuracy",
                     title=f"{name}: reliability (audit only)")
            right.set(xlim=(0, 100), xlabel="Products accepted automatically (%)", ylabel="Errors among accepted (%)",
                      title=f"{name}: risk vs coverage (audit only)")
            right.set_ylim(0, max(2., min(25., float(np.nanmax(risks.accepted_error_rate))*100 + 1)))
            for ax in pair:
                ax.grid(alpha=.2)
                ax.legend(loc="best", fontsize=8)
        fig.suptitle("Development diagnostics; audit curves must not be used to retune thresholds", fontsize=10)
        fig.tight_layout()
        fig.savefig(directory/"calibration_and_risk.svg", metadata={"Date": None})
        fig.savefig(directory/"calibration_and_risk.png", dpi=140)
        plt.close(fig)
    figure.suptitle("CatalogIQ confidence experiment — group-disjoint development audit", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, .98))
    figure.savefig(output/"calibration_and_risk.svg", metadata={"Date": None})
    figure.savefig(output/"calibration_and_risk.png", dpi=130)
    plt.close(figure)
