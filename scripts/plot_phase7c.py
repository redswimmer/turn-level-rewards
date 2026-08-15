"""Every figure the README embeds.

Reads only committed data under results/ -- phase7c-summary.json and phase7c-curves.json for the
paper-faithful comparison, the per-run *_eval_metrics.json files for the graded-reward follow-ups.
So the figures regenerate without the gitignored outputs/ and logs/ trees.

Unlike compare_runs.py, which queries a live trackio backend, nothing here needs anything the repo
does not already carry.

Usage: python scripts/plot_phase7c.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Light-mode slots from the dataviz reference palette. Named by role, not by hue, so a
# palette swap touches only this block.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dcdbd6"
OURS = "#2a78d6"  # categorical slot 1
PAPER = "#eb6834"  # categorical slot 2
GAIN = "#1baf7a"  # slot 3 -- reads as "added"
LOSS = "#e34948"  # slot 8 -- reads as "removed"
DEAD = "#8a8a86"  # collapsed arm, deliberately desaturated

RESULTS = Path("results")


def _style(ax: plt.Axes) -> None:
    """Recessive axes and grid: the marks carry the message, the frame should not compete."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.grid(axis="both", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _fig(width: float, height: float, ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), facecolor=SURFACE)
    axes = [axes] if ncols == 1 else list(axes)
    for ax in axes:
        _style(ax)
    return fig, axes


# Paper Table 2, HotpotQA. Format correctness is not reported for the two MR baselines.
PAPER_ANCHORS = {
    "PPO-OR": (0.435, 0.916),
    "PPO-MR": (0.436, None),
    "MT-PPO": (0.453, 0.998),
    "GRPO-OR": (0.331, 0.513),
    "GRPO-MR": (0.416, None),
}
ARM_KEYS = {
    "PPO-OR": "ppo-precollapse",
    "PPO-MR": "ppo_mr_paper",
    "MT-PPO": "mt_ppo",
    "GRPO-OR": "grpo_or",
    "GRPO-MR": "grpo_mr",
}
# Arms whose training died, so their held-out scores are not comparable measurements. GRPO-OR's
# gradient was exactly zero from step 185 on; PPO-OR was stopped once it had stopped answering.
# Their bars are suppressed rather than drawn: GRPO-OR still scores 0.421 on format correctness
# (a frozen policy emits a parseable tag 42% of the time), and plotting that beside the paper's
# 0.513 invites a comparison of two numbers that do not mean the same thing. Exact values for
# every arm, collapsed included, are in the README's results table.
COLLAPSED = {"PPO-OR", "GRPO-OR"}


def plot_vs_paper(summary: dict, out: Path) -> None:
    """Grouped bars: this reproduction against the paper's published numbers, EM and format.

    Two panels rather than one dual-axis chart -- EM and format correctness are different
    measures and sharing a y-axis would imply a comparability they do not have.
    """
    arms = list(PAPER_ANCHORS)
    y = range(len(arms))
    fig, (em_ax, fmt_ax) = _fig(11, 4.2, ncols=2)

    for ax, idx, title in ((em_ax, 0, "Exact match"), (fmt_ax, 1, "Format correctness")):
        ours, paper = [], []
        for arm in arms:
            m = summary[ARM_KEYS[arm]]
            score = m["em"] if idx == 0 else m["fmt"]
            ours.append(0.0 if arm in COLLAPSED else score)
            p = PAPER_ANCHORS[arm][idx]
            paper.append(p if p is not None else 0.0)
        ax.barh([i + 0.19 for i in y], paper, height=0.36, color=PAPER, label="Paper (Qwen2.5-7B)")
        ax.barh([i - 0.19 for i in y], ours, height=0.36, color=OURS, label="This repo (0.8B)")
        for i, arm in enumerate(arms):
            # Coloured to each series it describes: a neutral grey label sitting between two bars
            # gives the reader no way to tell which of them it belongs to.
            if PAPER_ANCHORS[arm][idx] is None:
                ax.text(0.008, i + 0.19, "not reported", va="center", fontsize=7.5, color=PAPER)
            if arm in COLLAPSED:
                ax.text(
                    0.012,
                    i - 0.19,
                    "training collapsed",
                    va="center",
                    fontsize=7.5,
                    color=OURS,
                    weight="bold",
                )
        ax.set_yticks(list(y), arms, fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_title(title, fontsize=11, color=INK, pad=8, loc="left")

    em_ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK_SOFT)
    fig.suptitle(
        "This reproduction vs. the paper's published results",
        fontsize=12.5,
        color=INK,
        x=0.007,
        ha="left",
        y=0.99,
    )
    # No footnote: the in-chart "collapsed" labels already say which arms died, and the surrounding
    # prose carries the model-size caveat. Repeating either here just competes with the bars.
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_decomposition(summary: dict, out: Path) -> None:
    """Waterfall: the paper's PPO-MR -> MT-PPO step split into reward content and placement.

    A waterfall rather than three bars, because the point is that two opposing contributions sum
    to the published net -- which grouped bars would leave the reader to compute. Bars are
    directly labelled, so no legend box: the colours are explained by the axis labels beneath.
    """
    base = summary["ppo_mr_paper"]["em"]
    mid = summary["ppo_mr"]["em"]
    end = summary["mt_ppo"]["em"]
    fig, (ax,) = _fig(8.6, 4.6)

    w = 0.5
    ax.bar(0, base, width=w, color=OURS)
    ax.bar(1, mid - base, bottom=base, width=w, color=GAIN)  # rises base -> mid
    ax.bar(2, mid - end, bottom=end, width=w, color=LOSS)  # falls mid -> end
    ax.bar(3, end, width=w, color=OURS)

    # Step connectors: each bar's exit level carried across to the next bar's entry.
    for x, level in ((0, base), (1, mid), (2, end)):
        ax.plot(
            [x + w / 2, x + 1 - w / 2],
            [level, level],
            color=INK_SOFT,
            linewidth=1,
            linestyle=(0, (3, 2)),
            zorder=0,
        )

    ax.text(0, base + 0.008, f"{base:.3f}", ha="center", fontsize=10.5, color=INK)
    ax.text(
        1, mid + 0.008, f"+{mid - base:.3f}", ha="center", fontsize=11, color=GAIN, weight="bold"
    )
    ax.text(
        2,
        mid + 0.008,
        f"\u2212{mid - end:.3f}",
        ha="center",
        fontsize=11,
        color=LOSS,
        weight="bold",
    )
    ax.text(3, end + 0.008, f"{end:.3f}", ha="center", fontsize=10.5, color=INK)

    ax.set_xticks(
        range(4),
        [
            "PPO-MR\n(the paper's baseline)",
            "add \u03bbs + per-turn\nformat CONTENT",
            "move it to turn\nboundaries (Eq. 9)",
            "MT-PPO",
        ],
        fontsize=9.5,
    )
    ax.set_ylabel("Held-out exact match", fontsize=10, color=INK_SOFT)
    ax.set_ylim(0, 0.35)
    ax.set_title(
        "What the paper's +1.7-point MT-PPO gain actually contains",
        fontsize=12.5,
        color=INK,
        loc="left",
        pad=10,
    )
    fig.text(
        0.012,
        0.05,
        "Net +0.039 EM, reproducing the paper's +0.017. Its own PPO-MR \u2192 MT-PPO comparison moves "
        "reward content and placement",
        fontsize=8.5,
        color=INK_SOFT,
    )
    fig.text(
        0.012,
        0.015,
        "together, so it cannot separate them. n=1 seed: the content effect is large, "
        "the placement effect suggestive.",
        fontsize=8.5,
        color=INK_SOFT,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def _smooth(ys: list[float], window: int = 15) -> list[float]:
    """Centred rolling mean. GRPO reward is inherently noisy step-to-step; smoothing is for
    readability only and does not change the underlying values."""
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - window // 2), min(len(ys), i + window // 2 + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out


def plot_collapse(curves: dict, out: Path) -> None:
    """The GRPO-OR collapse, and the mechanism behind it.

    Second panel is frac_reward_zero_std rather than grad_norm: it is bounded [0, 1], so it reads
    directly as "what fraction of prompt-groups produced no gradient at all", where grad_norm's
    scale is dominated by a single pre-collapse spike. grad_norm hitting exactly 0.0000 is stated
    in the caption instead.
    """
    fig, (r_ax, z_ax) = _fig(11, 4.2, ncols=2)

    panels = (
        (r_ax, "reward", "Mean reward", None),
        (z_ax, "zero_std", "Fraction of groups with zero reward variance", (0, 1.05)),
    )
    for ax, key, label, ylim in panels:
        for arm, color, name in (
            ("grpo_mr", OURS, "GRPO-MR (merged reward)"),
            ("grpo_or", DEAD, "GRPO-OR (binary outcome only)"),
        ):
            xs = [p[0] for p in curves[arm][key]]
            ys = [p[1] for p in curves[arm][key]]
            # Smoothed only: the raw per-step series is pure noise at this density, and in the
            # zero-variance panel it is a binary 0/1 flicker that obscures rather than informs.
            ax.plot(xs, _smooth(ys), color=color, linewidth=2.2, label=name)
        ax.axvline(184, color=LOSS, linewidth=1.1, linestyle="--", alpha=0.8)
        ax.set_xlabel("Training step", fontsize=10, color=INK_SOFT)
        ax.set_title(label, fontsize=11, color=INK, loc="left", pad=8)
        ax.set_xlim(0, 500)
        if ylim:
            ax.set_ylim(*ylim)

    r_ax.set_ylim(-0.1, 1.15)
    r_ax.annotate(
        "step 184\nlast nonzero reward",
        xy=(190, 0.05),
        xytext=(250, 0.30),
        fontsize=8.5,
        color=LOSS,
        arrowprops={"arrowstyle": "->", "color": LOSS, "linewidth": 1},
    )
    z_ax.annotate(
        "pinned at 1.000 \u2014 every group\nscores identically, so no gradient",
        xy=(340, 1.0),
        xytext=(215, 0.72),
        fontsize=8.5,
        color=LOSS,
        arrowprops={"arrowstyle": "->", "color": LOSS, "linewidth": 1},
    )
    r_ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK_SOFT)
    fig.suptitle(
        "A binary reward can stop training permanently",
        fontsize=12.5,
        color=INK,
        x=0.007,
        ha="left",
        y=0.99,
    )
    fig.text(
        0.007,
        0.05,
        "Same trainer, same seed, same hyperparameters \u2014 only the reward differs. After step 184, "
        "GRPO-OR's grad_norm is exactly 0.0000 for 300 straight",
        fontsize=8.5,
        color=INK_SOFT,
    )
    fig.text(
        0.007,
        0.014,
        "steps and checkpoints 450 and 500 are byte-identical. On held-out data the frozen "
        "policy never called search once. Curves are a 15-step rolling mean.",
        fontsize=8.5,
        color=INK_SOFT,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# Categorical slots for the four PPO arms. ppo is deliberately the desaturated one -- it is the
# arm that died, and it should read as background against the three that trained.
PPO_ARMS = [
    ("MT-PPO", "mt_ppo", OURS),
    ("PPO-MR (paper)", "ppo_mr_paper", PAPER),
    ("PPO-MR + content (ours)", "ppo_mr", GAIN),
    ("PPO-OR", "ppo", DEAD),
]


def plot_training(curves: dict, out: Path) -> None:
    """Two panels that between them replace most of the section's prose.

    Left: format compliance separates the arms cleanly and shows PPO-OR flatlining at zero.
    Right: searches per episode, which is the whole of the "a search penalty is not what bounds
    search" finding -- PPO-MR runs the penalty at zero and still sits near 2, while PPO-OR pins
    itself to the 4-turn cap. Stated as numbers that comparison needs three sentences.
    """
    fig, (fmt_ax, turn_ax) = _fig(11, 4.2, ncols=2)

    for ax, key, title in (
        (fmt_ax, "format_compliance", "Answers in the required format"),
        (turn_ax, "mean_tool_turns", "Searches per episode"),
    ):
        for name, arm, colour in PPO_ARMS:
            series = curves[arm]
            ax.plot(
                series["step"],
                _smooth(series[key]),
                color=colour,
                linewidth=2.0,
                label=name if key == "format_compliance" else None,
            )
        ax.set_xlabel("training step", fontsize=9, color=INK_SOFT)
        ax.set_title(title, fontsize=11, color=INK, pad=8, loc="left")

    fmt_ax.set_ylim(0, 1.02)
    fmt_ax.legend(frameon=False, fontsize=8.5, loc="center right", labelcolor=INK_SOFT)
    # The cap is what makes "uncontrolled search" legible -- without it the reader cannot tell
    # whether 4 turns per episode is a lot.
    turn_ax.axhline(4, color=LOSS, linewidth=1.0, linestyle="--")
    turn_ax.text(20, 3.82, "4-turn cap", fontsize=8, color=LOSS, va="top")
    turn_ax.set_ylim(0, 4.4)

    fig.suptitle(
        "PPO-OR stops answering and searches to the cap; the reward-shaped arms do neither",
        fontsize=12.5,
        color=INK,
        x=0.007,
        ha="left",
        y=0.99,
    )
    fig.text(
        0.007,
        0.015,
        "PPO-OR's curve ends at step 160, where it was stopped. 15-step rolling mean.",
        fontsize=8.5,
        color=INK_SOFT,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# The graded-reward GRPO follow-ups (seed 123, 600 steps). Separate track from the arms above:
# different reward, seed and step count, so these never share an axis with the paper comparison.
FOLLOWUPS = [
    ("Baseline\n(no penalty)", "seed123_600steps"),
    ("Length\npenalty", "lengthpenalty"),
    ("Search-count\npenalty", "searchpenalty"),
    ("Remove prompt cap\n(no penalty)", "nocapprompt"),
]


def plot_followups(out: Path) -> None:
    """Grouped bars: four reward configurations, outcome-only vs. merged.

    Two panels because exact match alone shows only *that* the outcome-only arms collapsed. Mean
    completion length shows *how*: the length-penalised arm answers in ~12 tokens, which is what
    "degenerate" actually looks like and is not recoverable from an accuracy bar.
    """
    labels = [label for label, _ in FOLLOWUPS]
    x = range(len(FOLLOWUPS))

    def read(condition: str, suffix: str) -> dict:
        return json.loads((RESULTS / f"{condition}_{suffix}_eval_metrics.json").read_text())

    runs = {c: [read(c, suffix) for _, suffix in FOLLOWUPS] for c in ("outcome_only", "turn_level")}
    fig, (em_ax, len_ax) = _fig(11, 4.2, ncols=2)

    for ax, key, title in (
        (em_ax, "eval_exact_match", "Held-out exact match"),
        (len_ax, "eval_completions/mean_length", "Mean completion length (tokens)"),
    ):
        for offset, (condition, colour, label) in enumerate(
            (("outcome_only", PAPER, "Outcome only"), ("turn_level", OURS, "Merged reward"))
        ):
            values = [m[key] for m in runs[condition]]
            ax.bar(
                [i + (offset - 0.5) * 0.38 for i in x],
                values,
                width=0.36,
                color=colour,
                label=label,
            )
            for i, v in enumerate(values):
                ax.text(
                    i + (offset - 0.5) * 0.38,
                    v,
                    f"{v:.3f}" if key.endswith("exact_match") else f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=INK_SOFT,
                )
        ax.set_xticks(list(x), labels, fontsize=8.5)
        ax.set_title(title, fontsize=11, color=INK, pad=8, loc="left")
        ax.grid(axis="x", visible=False)
        # Headroom for the value labels, and for the legend that sits over the left panel.
        ax.set_ylim(0, max(m[key] for r in runs.values() for m in r) * 1.22)

    # The baseline is what every other bar is being judged against -- mark it rather than
    # relying on the reader to keep the leftmost pair in mind across two panels.
    for ax in (em_ax, len_ax):
        ax.axvline(0.5, color=GRID, linewidth=1.0, linestyle="--")
    # Upper-left, not upper-right: the tallest bar in this panel is the rightmost one.
    em_ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK_SOFT)

    fig.suptitle(
        "Our own experiments: none of three added pressures beat the baseline",
        fontsize=12.5,
        color=INK,
        x=0.007,
        ha="left",
        y=0.99,
    )
    fig.text(
        0.007,
        0.015,
        "Graded reward (F1 + exact-match bonus), seed 123, 600 steps — a separate track from the "
        "paper reproduction, not comparable to it.\nOutcome-only collapses under either penalty, "
        "answering in ~12-21 tokens; the merged reward degrades but stays coherent.",
        fontsize=8.5,
        color=INK_SOFT,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    summary = json.loads((RESULTS / "phase7c-summary.json").read_text())
    curves = json.loads((RESULTS / "phase7c-curves.json").read_text())
    plot_vs_paper(summary, RESULTS / "phase7c_vs_paper.png")
    plot_decomposition(summary, RESULTS / "phase7c_decomposition.png")
    plot_collapse(curves, RESULTS / "phase7c_collapse.png")
    plot_training(curves, RESULTS / "phase7c_training.png")
    plot_followups(RESULTS / "followup_experiments_comparison.png")
    print("wrote 5 figures to results/")


if __name__ == "__main__":
    main()
