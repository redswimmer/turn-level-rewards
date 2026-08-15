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
# Ordered to match the README's "Reward approaches explored" section, which builds GRPO first and
# then switches algorithms. Both dicts drive plot order, so they must stay in step.
PAPER_ANCHORS = {
    "GRPO-OR": (0.331, 0.513),
    "GRPO-MR": (0.416, None),
    "PPO-OR": (0.435, 0.916),
    "PPO-MR": (0.436, None),
    "MT-PPO": (0.453, 0.998),
}
ARM_KEYS = {
    "GRPO-OR": "grpo_or",
    "GRPO-MR": "grpo_mr",
    "PPO-OR": "ppo-precollapse",
    "PPO-MR": "ppo_mr_paper",
    "MT-PPO": "mt_ppo",
}
# Arms whose training died, so their held-out scores are not comparable measurements. GRPO-OR's
# gradient was exactly zero from step 185 on; PPO-OR was stopped once it had stopped answering.
# Their bars are suppressed rather than drawn: GRPO-OR still scores 0.421 on format correctness
# (a frozen policy emits a parseable tag 42% of the time), and plotting that beside the paper's
# 0.513 invites a comparison of two numbers that do not mean the same thing. Suppressing them is
# also why every surviving bar is value-labelled below -- this figure replaces the results table
# it used to sit above, so it has to carry the exact numbers itself.
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
        # Both series name their model in full: "0.8B" alone reads as a size rather than a model,
        # which hides that these are different base models and not one model at two scales.
        ax.barh(
            [i - 0.19 for i in y], ours, height=0.36, color=OURS, label="This repo (Qwen3.5-0.8B)"
        )
        # Each panel gets its own x-range. They measure different things, and a shared axis would
        # squeeze every exact-match bar into the left third to accommodate format's near-1.0 bars.
        span = max(max(ours), max(paper)) * 1.12
        for i, arm in enumerate(arms):
            # Coloured to each series it describes: a neutral grey label sitting between two bars
            # gives the reader no way to tell which of them it belongs to.
            if PAPER_ANCHORS[arm][idx] is None:
                ax.text(
                    0.008 * span, i + 0.19, "not reported", va="center", fontsize=7.5, color=PAPER
                )
            if arm in COLLAPSED:
                ax.text(
                    0.012 * span,
                    i - 0.19,
                    "training collapsed",
                    va="center",
                    fontsize=7.5,
                    color=OURS,
                    weight="bold",
                )
            # Print the value on every bar that exists, so the figure carries the exact numbers
            # and needs no results table beside it. Long bars take the label inside in the surface
            # colour; short ones would not fit, so those sit just past the end.
            for value, offset, colour in ((paper[i], 0.19, PAPER), (ours[i], -0.19, OURS)):
                if value <= 0:
                    continue
                inside = value > 0.3 * span
                ax.text(
                    value - 0.015 * span if inside else value + 0.012 * span,
                    i + offset,
                    f"{value:.3f}",
                    va="center",
                    ha="right" if inside else "left",
                    fontsize=8,
                    color=SURFACE if inside else colour,
                    weight="bold",
                )
        ax.set_yticks(list(y), arms, fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlim(0, span)
        ax.set_title(title, fontsize=11, color=INK, pad=8, loc="left")

    fig.suptitle(
        "This reproduction vs. the paper's published results",
        fontsize=12.5,
        color=INK,
        x=0.007,
        ha="left",
        y=0.985,
    )
    # Figure-level and horizontal, under the title rather than inside a panel. Every corner of the
    # exact-match axes is occupied -- bars, value labels, or the collapsed-arm notes -- so an
    # in-axes legend has nowhere to sit without crowding something.
    fig.legend(
        *em_ax.get_legend_handles_labels(),
        loc="upper left",
        bbox_to_anchor=(0.006, 0.945),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_SOFT,
        handlelength=1.1,
        handleheight=0.9,
        columnspacing=1.6,
    )
    # No footnote: the in-chart "collapsed" labels already say which arms died, and the surrounding
    # prose carries the model-size caveat. Repeating either here just competes with the bars.
    fig.tight_layout(rect=(0, 0.01, 1, 0.90))
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


def plot_collapses(curves: dict, out: Path) -> None:
    """Both collapsed runs next to the runs that held, one panel per track.

    One figure rather than two: the section's claim is that exactly two of our five runs
    collapsed, so both collapses belong in a single glance. They cannot share an axis -- the
    tracks logged different metrics (TRL's GRPOTrainer emits per-step reward; MTPPOTrainer logs
    per-episode format compliance), and GRPO-OR's death never shows in its answer rate anyway,
    only in its training signal. Collapsed runs in red, survivors in gray, direct labels.
    """
    fig, (g_ax, p_ax) = _fig(11, 4.2, ncols=2)

    for arm, colour in (("grpo_mr", DEAD), ("grpo_or", LOSS)):
        xs = [p[0] for p in curves[arm]["reward"]]
        ys = [p[1] for p in curves[arm]["reward"]]
        g_ax.plot(xs, _smooth(ys), color=colour, linewidth=2.2)
    g_ax.set_title("GRPO track", fontsize=11, color=INK, loc="left", pad=8)
    g_ax.set_ylabel("mean training reward", fontsize=9.5, color=INK_SOFT)
    g_ax.set_ylim(0, 0.78)
    g_ax.text(300, 0.045, "GRPO-OR", fontsize=9.5, color=LOSS)
    g_ax.text(255, 0.60, "GRPO-MR", fontsize=9.5, color=INK_SOFT)

    for arm, colour in (("ppo_mr_paper", DEAD), ("mt_ppo", DEAD), ("ppo", LOSS)):
        series = curves[arm]
        p_ax.plot(
            series["step"],
            _smooth(series["format_compliance"]),
            color=colour,
            linewidth=2.2 if colour == LOSS else 1.8,
        )
    p_ax.set_title("PPO track", fontsize=11, color=INK, loc="left", pad=8)
    p_ax.set_ylabel("share of rollouts with a parseable answer", fontsize=9.5, color=INK_SOFT)
    p_ax.set_ylim(0, 1.05)
    p_ax.text(145, 0.07, "PPO-OR", fontsize=9.5, color=LOSS)
    p_ax.text(190, 0.97, "PPO-MR / MT-PPO", fontsize=9.5, color=INK_SOFT)

    for ax in (g_ax, p_ax):
        ax.set_xlabel("training step", fontsize=9.5, color=INK_SOFT)
        ax.set_xlim(0, 500)

    fig.suptitle("Our five training runs", fontsize=12.5, color=INK, x=0.007, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0.01, 1, 0.94))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# Same section order as PAPER_ANCHORS, restricted to the PPO track. PPO-OR is drawn first so the
# collapsed arm sits behind the three that trained rather than over them.
# All series are this repo's own runs of the paper's three PPO arms. The extra `ppo_mr` control
# run exists in the results data but is deliberately absent here: the README presents only the
# paper's five arms (see docs/phase-7c-paper-faithful-ppo.md §15c for the control's write-up).
PPO_ARMS = [
    ("PPO-OR", "ppo", DEAD),
    ("PPO-MR", "ppo_mr_paper", PAPER),
    ("MT-PPO", "mt_ppo", OURS),
]


def _ppo_curve(curves: dict, key: str, title: str):
    """One metric, all four PPO arms. Returns (fig, ax) for the caller to finish and save.

    Format compliance and searches per episode used to share a figure, but they support two
    different findings in the README, and a section whose only evidence is half a chart filed
    under the previous heading reads as having no evidence at all.
    """
    fig, (ax,) = _fig(7.4, 4.0)
    for name, arm, colour in PPO_ARMS:
        series = curves[arm]
        ax.plot(series["step"], _smooth(series[key]), color=colour, linewidth=2.0, label=name)
    ax.set_xlabel("training step", fontsize=9, color=INK_SOFT)
    fig.suptitle(title, fontsize=12.5, color=INK, x=0.011, ha="left", y=0.985)
    return fig, ax


def plot_searches(curves: dict, out: Path) -> None:
    """Searches per episode: the arm with no search penalty is not the one that runs away."""
    fig, ax = _ppo_curve(
        curves, "mean_tool_turns", "The unpenalised arm is not the one that searches to the cap"
    )
    ax.set_ylabel("searches per episode", fontsize=9, color=INK_SOFT)
    ax.set_ylim(0, 4.4)
    # The cap is what makes "uncontrolled search" legible -- without it the reader cannot tell
    # whether 4 turns per episode is a lot.
    ax.axhline(4, color=LOSS, linewidth=1.0, linestyle="--")
    ax.text(20, 3.82, "4-turn cap", fontsize=8, color=LOSS, va="top")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK_SOFT)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
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
        # Kept because it travels with the image: these bars invite comparison against the paper
        # arms, and nothing in the chart itself says they were run under a different reward, seed
        # and step count. What the bars already show is left to the bars.
        "Graded reward (F1 + exact-match bonus), seed 123, 600 steps — a separate track from the "
        "paper reproduction, not comparable to it.",
        fontsize=8.5,
        color=INK_SOFT,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    summary = json.loads((RESULTS / "phase7c-summary.json").read_text())
    curves = json.loads((RESULTS / "phase7c-curves.json").read_text())
    plot_vs_paper(summary, RESULTS / "phase7c_vs_paper.png")
    plot_collapses(curves, RESULTS / "phase7c_collapses.png")
    plot_searches(curves, RESULTS / "phase7c_searches.png")
    plot_followups(RESULTS / "followup_experiments_comparison.png")
    print("wrote 4 figures to results/")


if __name__ == "__main__":
    main()
