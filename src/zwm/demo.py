"""End-to-end demo runner for the ZWM Trinity world-model planner.

This is the consumer for the per-tick telemetry that ``TickReport`` produces:
it runs a closed OODA loop and actually *reads* ``jepa_loss``, ``router_loss``,
``surprise``, ``reward``, ``codon``, ``codon_aa``, and ``mutation_class`` off
each tick to report learning progress — so the loop's learning signals are
consumed, not merely produced.

Run it as a module::

    python -m zwm.demo
"""
from __future__ import annotations

import math

from zwm.core.hexagram import hexagram_from_name
from zwm.planner.agent import TrinityAgent


def run_demo(
    ticks: int = 50,
    db_path: str = ":memory:",
    checkpoint_path: str | None = None,
    mcts_iterations: int = 120,
    seed_hexagram: str = "乾为天",
    n_particles: int = 16,
    use_diffusion: bool = True,
) -> dict:
    """Run ``ticks`` OODA steps and return consumed learning metrics.

    Returns a summary dict built by reading each ``TickReport`` — proving the
    JEPA loss, world-model surprise and router loss are consumed downstream.
    """
    h = hexagram_from_name(seed_hexagram)
    jepa_losses: list[float] = []
    router_losses: list[float] = []
    surprises: list[float] = []
    rewards: list[float] = []
    codons: list[str] = []
    mutation_classes: list[str] = []

    with TrinityAgent(
        db_path=db_path,
        checkpoint_path=checkpoint_path,
        mcts_iterations=mcts_iterations,
        n_particles=n_particles,
        use_diffusion=use_diffusion,
    ) as agent:
        for t in range(ticks):
            # A simple non-stationary reward so the learners have signal to chase.
            reward = 0.5 + 0.4 * math.sin(t / 5.0)
            report = agent.tick(h_current=h, reward=reward)

            # CONSUME the telemetry rather than discarding it.
            if report.jepa_loss is not None:
                jepa_losses.append(report.jepa_loss)
            if report.router_loss is not None:
                router_losses.append(report.router_loss)
            surprises.append(report.surprise)
            rewards.append(report.reward)
            # P3: Consume codon and mutation classification from the report.
            if report.codon and report.codon != "???":
                codons.append(report.codon)
            if report.mutation_class:
                mutation_classes.append(report.mutation_class)

            # Advance the state along the planner's chosen transition.
            h = report.h_next

        episodes = agent.store.count()

    return {
        "ticks": ticks,
        "episodes_stored": episodes,
        "jepa_losses": jepa_losses,
        "router_losses": router_losses,
        "surprises": surprises,
        "rewards": rewards,
        "codons": codons,
        "mutation_classes": mutation_classes,
        "jepa_loss_first": jepa_losses[0] if jepa_losses else None,
        "jepa_loss_last": jepa_losses[-1] if jepa_losses else None,
        "surprise_mean": sum(surprises) / len(surprises) if surprises else None,
        "unique_codons": len(set(codons)) if codons else 0,
    }


def main() -> None:
    summary = run_demo()
    print("=== ZWM OODA demo ===")
    print(f"ticks            : {summary['ticks']}")
    print(f"episodes stored  : {summary['episodes_stored']}")
    if summary["jepa_loss_first"] is not None:
        print(
            f"JEPA loss        : {summary['jepa_loss_first']:.4f} "
            f"-> {summary['jepa_loss_last']:.4f}"
        )
    print(f"mean surprise    : {summary['surprise_mean']:.4f}")
    print(f"router steps     : {len(summary['router_losses'])}")
    print(f"unique codons    : {summary['unique_codons']}")
    print(f"mutation types   : {len(set(summary['mutation_classes']))}")
    # F10: render the JEPA world-model surprise curve as a quick
    # ASCII chart so the demo is visually self-documenting.  No
    # matplotlib / pillow dependency — plain stdlib only.  50 rows
    # tall, 60 columns wide.
    try:
        _render_surprise_ascii(summary["surprises"])
    except Exception:
        pass


def _render_surprise_ascii(surprises: list[float], height: int = 8, width: int = 50) -> None:
    """F10: low-tech visual of the surprise curve in the terminal.

    Bins the surprise values into ``width`` buckets, scales each
    bucket to ``height`` rows, then prints the resulting bar chart
    top-to-bottom.  Helps a reader see "did surprise decrease over
    training?" at a glance, with zero extra dependencies.
    """
    if not surprises:
        return
    n = len(surprises)
    if n < 2:
        return
    # Bucket into ``width`` columns.
    cols: list[float] = []
    bucket = max(1, n // width)
    for i in range(0, n, bucket):
        chunk = surprises[i:i + bucket]
        if chunk:
            cols.append(sum(chunk) / len(chunk))
    if len(cols) < width:
        cols += [cols[-1]] * (width - len(cols))
    cols = cols[:width]
    lo, hi = min(cols), max(cols)
    span = max(hi - lo, 1e-9)
    print()
    print("Surprise curve (high = uncertain, low = predicted):")
    print("  " + "┐" * width)
    for row in range(height, 0, -1):
        threshold = lo + (row / height) * span
        line = "  "
        for v in cols:
            line += "█" if v >= threshold else " "
        print(line)
    print("  " + "─" * width + f"  first→last ({cols[0]:.3f} → {cols[-1]:.3f})")


if __name__ == "__main__":  # pragma: no cover
    main()
