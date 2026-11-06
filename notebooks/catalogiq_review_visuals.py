"""Visual-only presentation views over the existing, verified Review object.

No file loading, training, inference, or new estimation. Original evidence views
and calculations remain in catalogiq_review_support and the notebook appendix.
"""

from html import escape
import io

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display

from catalogiq_review_support import NAVY, TEAL, GOLD, GREY


VISUAL_STYLE = """
<style>
.cq-heading {margin:28px 0 20px; max-width:1100px;}
.cq-heading h1 {font-size:clamp(36px,4.1vw,60px);line-height:1.08;letter-spacing:-.04em;color:#18324d;margin:16px 0 22px;font-weight:750;}
.cq-heading p {font-size:clamp(19px,1.9vw,26px);line-height:1.5;color:#526778;max-width:1000px;}
.cq-visual {margin:20px 0 32px;color:#18324d;}
.cq-cards {display:grid;grid-template-columns:repeat(3,1fr);gap:28px;align-items:stretch;}
.cq-card {border:1px solid #dbe5e8;border-radius:22px;background:#f4f8f8;padding:36px 28px;min-height:180px;display:flex;flex-direction:column;justify-content:center;}
.cq-card strong {font-size:30px;line-height:1.25;font-weight:700;}
.cq-card small {font-size:18px;color:#627488;margin-top:18px;}
.cq-icon {width:58px;height:58px;color:#087f8c;margin-bottom:28px;}
.cq-product {font-size:25px;color:#18324d;border-left:5px solid #087f8c;padding:18px 25px;background:#f4f8f8;border-radius:0 16px 16px 0;margin-bottom:30px;}
.cq-taxonomies {display:grid;grid-template-columns:1fr 1fr;gap:36px;}
.cq-taxonomy {padding:35px;border:2px solid #dbe5e8;border-radius:20px;}
.cq-taxonomy span {display:block;color:#627488;font-size:20px;margin-bottom:24px;}
.cq-taxonomy strong {font-size:34px;line-height:1.3;color:#18324d;}
.cq-taxonomy strong.cq-breadcrumb {font-size:26px;line-height:1.5;font-weight:650;}
.cq-taxonomy:last-child {border-color:#087f8c;background:#eff8f6;}
.cq-correction {display:flex;align-items:center;justify-content:center;gap:28px;flex-wrap:wrap;font-size:25px;margin:22px 0;}
.cq-correction b {color:#087f8c;}
.cq-correction .cq-arrow {font-size:34px;color:#087f8c;}
.cq-pipeline {display:flex;gap:16px;align-items:center;justify-content:center;padding:32px 0;}
.cq-pipeline div {padding:36px 22px;background:#eff8f6;border-top:5px solid #087f8c;border-radius:12px;flex:1;text-align:center;font-size:25px;font-weight:650;}
.cq-pipeline span {font-size:28px;color:#8498a7;}
.cq-parameters {display:flex;justify-content:center;gap:60px;margin:28px 0;font-size:25px;color:#526778;}
.cq-parameters b {font-size:39px;color:#18324d;}
.cq-chart svg {width:100%;height:auto;display:block;}
.cq-chart {margin:0 auto;max-width:1120px;}
.cq-chart .cq-card svg {width:100%;height:130px;}
.cq-presenter-notes {margin:16px 0;color:#526778;border:1px solid #dbe5e8;border-radius:10px;padding:12px 18px;}
.cq-presenter-notes summary {cursor:pointer;font-weight:650;}
.cq-presenter-notes[open] {background:#f4f8f8;}
@media(max-width:700px) {.cq-cards {gap:12px;} .cq-card {padding:24px 16px;} .cq-card strong {font-size:23px;} .cq-pipeline {gap:8px;} .cq-pipeline div {padding:22px 10px;font-size:20px;}}
@media print {.cq-presenter-notes {display:none!important;}}
</style>
"""


def _html(content):
    display(HTML(f'<div class="cq-visual">{content}</div>'))


def _figure(fig):
    fig.tight_layout(pad=1.2)
    output = io.StringIO()
    fig.savefig(output, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    _html('<div class="cq-chart">' + output.getvalue() + '</div>')


CHART_STYLE = {
    "font.size": 16, "axes.labelsize": 15, "xtick.labelsize": 15,
    "ytick.labelsize": 17, "legend.fontsize": 15,
    "svg.fonttype": "path", "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False,
    "axes.spines.bottom": False,
}


class VisualReview:
    def __init__(self, review):
        self.review = review

    def problem(self):
        icons = [
            '<rect x="12" y="7" width="34" height="46" rx="5"/><path d="M21 20h16M21 30h16M21 40h10"/>',
            '<path d="M10 30l13 13L49 16"/><circle cx="30" cy="30" r="25"/>',
            '<circle cx="25" cy="20" r="10"/><path d="M7 52v-8c0-16 36-16 36 0v8M43 24l5 5 9-12"/>',
        ]
        cards = []
        for label, icon in zip(["Propose", "Validate", "Review"], icons):
            cards.append(f'<div class="cq-card"><svg class="cq-icon" viewBox="0 0 60 60" fill="none" stroke="currentColor" stroke-width="3" aria-hidden="true">{icon}</svg><strong>{label}</strong></div>')
        _html('<div class="cq-cards">' + ''.join(cards) + '</div>')

    def taxonomy(self):
        record_id = "06d2ca12999a801a09e50c57c1d86d38aeb17aa4a46e419dcbee516450bc63b0"
        product = self.review.dev_index.loc[record_id]
        assert record_id not in self.review.protected_ids
        assert product.ProductName.startswith("Jarrow Formulas Fem-Dophilus")
        assert "Vitamins, Minerals & Supplements" in product.ProductCategory
        assert product.Segment == "Digestive Health"
        _html('<div class="cq-product">Jarrow Formulas Fem-Dophilus</div>'
              '<div class="cq-taxonomies"><div class="cq-taxonomy"><span>Retailer</span>'
              '<strong class="cq-breadcrumb" aria-label="Full retailer category">' + escape(product.ProductCategory) + '</strong></div>'
              '<div class="cq-taxonomy"><span>CatalogIQ</span><strong>Digestive Health</strong></div></div>')

    def cleaning(self):
        result = self.review.cleaning_validation["training"]
        retained = result["counts"]["candidate"]
        quarantined = result["counts"]["quarantine"]
        assert retained + quarantined == result["verified_rows"] == 84552
        with plt.rc_context(CHART_STYLE):
            fig, ax = plt.subplots(figsize=(12.2, 2.65))
            ax.barh([0], [retained], color=TEAL, height=.42)
            ax.barh([0], [quarantined], left=[retained], color=GOLD, height=.42)
            ax.set(xlim=(0, 84552), ylim=(-.7, .7), yticks=[], xticks=[])
            ax.text(0, .40, "84,552 original training listings", fontsize=19, color=NAVY)
            ax.text(0, -.42, f"{retained:,} retained", fontsize=24, weight="bold", color=TEAL)
            ax.text(84552, -.42, f"{quarantined:,} quarantined", ha="right", fontsize=24, weight="bold", color=GOLD)
            _figure(fig)
        change = self.review.changes[self.review.changes.source_row.eq("21582")].iloc[0]
        product = self.review.development[self.review.development.source_row.eq("21582")].iloc[0]
        assert product.ProductName.startswith("Vicks VapoRub")
        _html('<div class="cq-correction"><span>Vicks VapoRub</span><span>' + escape(change.old_value) +
              '</span><span class="cq-arrow">→</span><b>' + escape(change.new_value) + '</b></div>')

    def splitting(self):
        # Existing assignments contain identity/group/split metadata, never test labels.
        counts = self.review.assignments["V4"].split.value_counts()
        assert counts.to_dict() == {"train": 59111, "validation": 12667, "test": 12160}
        assert self.review.equality["crossing_pairs"] == 0
        panels = []
        for role, name, color in [("train", "Train", TEAL), ("validation", "Validation", NAVY), ("test", "Protected test", GREY)]:
            # Illustrative family glyphs; no rows or products are represented by these dots.
            circles = ''.join(f'<circle cx="{38 + x*23}" cy="{24 + y*26}" r="7" fill="{color}"/>'
                              for y in range(2) for x in range(3))
            panels.append(f'<div class="cq-card"><strong>{name}</strong><svg viewBox="0 0 125 80" role="img" aria-label="Illustrative grouped records">{circles}</svg><strong>{counts[role]:,}</strong></div>')
        _html('<div class="cq-chart"><div class="cq-cards">' + ''.join(panels) +
              '</div><div class="cq-parameters"><span><b>0</b> exact-input crossings</span></div></div>')

    def baseline(self):
        text = self.review.classical_text
        assert "| Four agreed product fields | 89.57% | 83.90% | 1,246 |" in text
        assert "| Four fields + concatenated ProductCategory | 94.42% | 91.00% | 667 |" in text
        neural = self.review.category.set_index("arm").loc["with_category", "macro_f1"]
        values = [83.90, 91.00, neural*100]
        with plt.rc_context(CHART_STYLE):
            fig, ax = plt.subplots(figsize=(12.2, 4.05))
            bars = ax.barh([2, 1, 0], values, color=[NAVY, NAVY, TEAL], height=.56)
            ax.set(yticks=[2, 1, 0], yticklabels=["TF-IDF · four fields", "TF-IDF · + category", "MiniLM · + category"],
                   xlim=(0, 107), xticks=[0, 25, 50, 75, 100], xlabel="Segment validation macro-F1 (%) · same 11,952 records")
            ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=9, fontsize=19, weight="bold")
            ax.tick_params(axis="y", length=0, pad=12)
            _figure(fig)

    def architecture(self):
        config = self.review.manifests["segment"]["config"]
        assert (config["sequence_length"], config["epochs"], config["seed"]) == (256, 3, 42)
        _html('<div class="cq-pipeline"><div>Product text</div><span>→</span><div>MiniLM</div><span>→</span>'
              '<div>Pooling</div><span>→</span><div>Task head</div></div>'
              '<div class="cq-parameters"><span><b>256</b> tokens</span><span><b>3</b> epochs</span><span>Seed <b>42</b></span></div>')

    def results(self):
        data = self.review.scores
        with plt.rc_context(CHART_STYLE):
            fig, ax = plt.subplots(figsize=(12.2, 4.45))
            y = np.arange(len(data))
            ax.barh(y-.17, data.Accuracy*100, height=.29, color=NAVY, label="Accuracy")
            ax.barh(y+.17, data["Macro-F1 (full)"]*100, height=.29, color=TEAL, label="Macro-F1 · full vocabulary")
            labels = [f"{row['Target']}   n={row['Validation n']:,}" for _, row in data.iterrows()]
            ax.set(yticks=y, yticklabels=labels, xlim=(0, 112), xticks=[0, 25, 50, 75, 100], xlabel="Validation score (%)")
            ax.invert_yaxis()
            for i, row in data.iterrows():
                ax.text(row.Accuracy*100+1, i-.17, f"{row.Accuracy:.2%}", va="center", fontsize=14)
                ax.text(row["Macro-F1 (full)"]*100+1, i+.17, f"{row['Macro-F1 (full)']:.2%}", va="center", fontsize=14)
            ax.legend(loc="lower center", bbox_to_anchor=(.46, 1.02), ncol=2, frameon=False)
            ax.tick_params(axis="y", length=0, pad=12)
            _figure(fig)

    def brand(self):
        neural = self.review.scores.set_index("Target").loc["Brand"]
        lookup = self.review.brand_result["lookup_baseline"]["full_validation"]
        values = [(neural.Accuracy*100, lookup["accuracy"]*100),
                  (neural["Macro-F1 (full)"]*100, lookup["macro_f1"]*100)]
        with plt.rc_context(CHART_STYLE):
            fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.1))
            for ax, title, (mini, simple) in zip(axes, ["Accuracy", "Macro-F1 · all 143 classes"], values):
                ax.bar([0, 1], [mini, simple], color=[NAVY, TEAL], width=.53)
                ax.set(xticks=[0, 1], xticklabels=["MiniLM", "Lookup"], ylim=(0, 112), yticks=[0, 50, 100])
                ax.set_title(title, fontsize=19, pad=20)
                for x, v in enumerate([mini, simple]):
                    ax.text(x, v+3, f"{v:.2f}%", ha="center", fontsize=23, weight="bold")
                ax.tick_params(axis="x", length=0, pad=12)
            axes[0].set_ylabel("Validation score (%)")
            _figure(fig)

    def decisions(self):
        _html('<div class="cq-cards"><div class="cq-card"><strong>Taxonomy</strong></div>'
              '<div class="cq-card"><strong>Review capacity</strong></div>'
              '<div class="cq-card"><strong>Model choice</strong></div></div>')
