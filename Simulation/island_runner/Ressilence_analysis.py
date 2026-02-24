"""
standalone_run.py — Phase B (Standalone / No Circular Exchange)

Runs the island simulation in STANDALONE mode:
- No grid transfers (no heat/CO2/basil circular credits)
- Waste tax applies to vented waste
- Imports + order fees + storage surcharges show up clearly

Notebook-safe: ignores Jupyter's injected args.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Optional, Sequence

# Import your CTA runner (must be in same folder or on PYTHONPATH)
from island_runner_cta import simulate_island_day  # uses circular=True/False internally


def _print_standalone_summary(r: Dict[str, Any]) -> None:
    print("\n  [STANDALONE (no circular exchange)]")
    print(f"  Island GDP:       ${r['island_gdp']:,.2f}")
    print(f"  Bakery:           ${r['bakery_profit']:,.2f} (SL={r['bakery_sl']:.1f}%)")
    print(f"  Fish Farm:        ${r['fish_profit']:,.2f} (SL={r['fish_sl']:.1f}%)")
    print(f"  Greenhouse:       ${r['greenhouse_profit']:,.2f} (SL={r['greenhouse_sl']:.1f}%)")
    print(f"  Basil: 0 to bakery / {r['basil_to_market']:.0f} to market / {r['basil_import']:.0f} imported")
    print(f"  Heat: {r['heat_supplied']:.0f} supplied / {r['heat_deficit']:.0f} deficit / {r['heat_surplus']:.0f} surplus")
    print(f"  Waste tax:        ${r['total_waste_tax']:,.2f}")
    print(f"  CTA credits:      ${r['total_credits']:,.2f}")


def run_cli(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Standalone Island Runner (Phase B)")
    parser.add_argument("--ovens", type=int, default=3)
    parser.add_argument("--tanks", type=int, default=6)
    parser.add_argument("--units", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    # Notebook-safe: ignore unknown args like -f kernel.json
    args, _unknown = parser.parse_known_args(args=argv)

    cfg = dict(
        num_ovens=args.ovens,
        num_tanks=args.tanks,
        num_units=args.units,
        # keep flags present, but standalone mode will ignore circular exchange
        use_grid_basil=True,
        use_grid_heat=True,
        use_grid_nutrients=True,
        fish_mode="optimal",
        gh_mode="high",
        phase="PHASE_2",
    )

    r = simulate_island_day(cfg, seed=args.seed, circular=False)  # ✅ standalone
    _print_standalone_summary(r)


def main(argv: Optional[Sequence[str]] = None) -> None:
    print("--- [ISLAND RUNNER] PHASE B STANDALONE SIMULATION ---")
    run_cli(argv)


if __name__ == "__main__":
    # Notebook-safe default behavior
    if "ipykernel" in sys.modules:
        main([])     # force empty argv in Jupyter
    else:
        main(None)   # normal CLI behavior
