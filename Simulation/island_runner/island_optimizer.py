#!/usr/bin/env python3
"""
SOUTHERN REEFS — Full Island Optimizer (Phase B)
Bakery + Fish Farm + Greenhouse — Circular Economy + CTA Directives

Uses simulate_island_day() from island_runner.py (single source of truth
for circular accounting + CTA incentives).  Prices and CTA directives
come from island_grid_B.py.

Finds the optimal unit counts that maximize total island GDP.
"""

import sys
import time
import itertools
from pathlib import Path

import numpy as np

# ── Path setup for island_grid_B (parent dir) ────────────────────────────
_GRID_DIR = Path(__file__).resolve().parents[1]
if str(_GRID_DIR) not in sys.path:
    sys.path.insert(0, str(_GRID_DIR))

from island_grid_B import PHASE_2_PRICES as PRICES, CTA_DIRECTIVES
from island_runner import simulate_island_day


# ══════════════════════════════════════════════════════════════════════════
#  MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════

def island_monte_carlo(config, n_trials=1000):
    results = []
    for i in range(n_trials):
        r = simulate_island_day(config, seed=i)
        results.append(r)

    gdps = [r["island_gdp"] for r in results]
    b_profits = [r["bakery_profit"] for r in results]
    f_profits = [r["fish_profit"] for r in results]
    g_profits = [r["greenhouse_profit"] for r in results]
    b_sls = [r["bakery_sl"] for r in results]
    f_sls = [r["fish_sl"] for r in results]
    g_sls = [r["greenhouse_sl"] for r in results]
    waste_taxes = [r["total_waste_tax"] for r in results]
    basil_imports = [r["basil_import"] for r in results]
    heat_surpluses = [r["heat_surplus"] for r in results]
    total_credits = [r["total_credits"] for r in results]

    return dict(
        config=config,
        gdp_mean=np.mean(gdps),
        gdp_std=np.std(gdps),
        gdp_p5=np.percentile(gdps, 5),
        gdp_p95=np.percentile(gdps, 95),
        bakery_profit_mean=np.mean(b_profits),
        fish_profit_mean=np.mean(f_profits),
        gh_profit_mean=np.mean(g_profits),
        bakery_sl_mean=np.mean(b_sls),
        fish_sl_mean=np.mean(f_sls),
        gh_sl_mean=np.mean(g_sls),
        bakery_sl_ge90=np.mean([1 if s >= 90 else 0 for s in b_sls]) * 100,
        fish_sl_ge90=np.mean([1 if s >= 90 else 0 for s in f_sls]) * 100,
        gh_sl_ge90=np.mean([1 if s >= 90 else 0 for s in g_sls]) * 100,
        waste_tax_mean=np.mean(waste_taxes),
        basil_import_mean=np.mean(basil_imports),
        heat_surplus_mean=np.mean(heat_surpluses),
        total_credits_mean=np.mean(total_credits),
    )


# ══════════════════════════════════════════════════════════════════════════
#  GRID SEARCH — Island-wide
# ══════════════════════════════════════════════════════════════════════════

def island_grid_search(n_trials=500, verbose=True):
    """Search over unit counts for all three enterprises. Maximize island GDP."""
    oven_range = range(2, 8)
    tank_range = range(2, 7)
    unit_range = range(3, 9)
    fish_modes = ["optimal"]

    combos = list(itertools.product(oven_range, tank_range, unit_range, fish_modes))
    total = len(combos)

    if verbose:
        print(f"\n{'='*80}")
        print(f"  ISLAND GRID SEARCH: {total} configurations x {n_trials} trials each")
        print(f"  CTA Directives: oven cap={CTA_DIRECTIVES['MAX_OVENS_PER_BAKER']}, "
              f"heat credit=${CTA_DIRECTIVES['HEAT_CREDIT_PER_UNIT']}/unit, "
              f"basil discount={CTA_DIRECTIVES['BASIL_PARTNER_DISCOUNT']*100:.0f}%")
        print(f"{'='*80}\n")

    summaries = []
    t0 = time.time()

    for idx, (ovens, tanks, units, fm) in enumerate(combos, 1):
        config = dict(
            num_ovens=ovens, bakery_stagger=15, bakery_demand=400,
            num_tanks=tanks, fish_mode=fm, fish_stagger=35, fish_demand=200,
            num_units=units, gh_mode="high", gh_stagger=14, gh_demand=500,
            use_grid_basil=True, use_grid_heat=True, use_grid_nutrients=True,
        )
        s = island_monte_carlo(config, n_trials=n_trials)
        summaries.append(s)

        if verbose and idx % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / idx * (total - idx)
            print(f"  [{idx:4d}/{total}] ovens={ovens} tanks={tanks} units={units}"
                  f"  GDP=${s['gdp_mean']:>8.2f}"
                  f"  B=${s['bakery_profit_mean']:>7.0f}"
                  f"  F=${s['fish_profit_mean']:>7.0f}"
                  f"  G=${s['gh_profit_mean']:>7.0f}"
                  f"  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  Grid search completed in {elapsed:.1f}s\n")

    summaries.sort(key=lambda s: s["gdp_mean"], reverse=True)
    return summaries


# ══════════════════════════════════════════════════════════════════════════
#  STANDALONE: Circular vs Independent comparison
# ══════════════════════════════════════════════════════════════════════════

def island_monte_carlo_standalone(config, n_trials=1000):
    """Monte Carlo WITHOUT circular economy (no CTA benefits)."""
    results = []
    for i in range(n_trials):
        r = simulate_island_day(config, seed=i, circular=False)
        results.append(r)

    gdps = [r["island_gdp"] for r in results]
    b_profits = [r["bakery_profit"] for r in results]
    f_profits = [r["fish_profit"] for r in results]
    g_profits = [r["greenhouse_profit"] for r in results]
    waste_taxes = [r["total_waste_tax"] for r in results]

    return dict(
        config=config,
        gdp_mean=np.mean(gdps),
        gdp_std=np.std(gdps),
        bakery_profit_mean=np.mean(b_profits),
        fish_profit_mean=np.mean(f_profits),
        gh_profit_mean=np.mean(g_profits),
        waste_tax_mean=np.mean(waste_taxes),
    )


def compare_circular_vs_standalone(best_config, n_trials=1000):
    """Compare island GDP with and without circular economy + CTA."""
    circular = island_monte_carlo(best_config, n_trials)
    standalone = island_monte_carlo_standalone(best_config, n_trials)
    return circular, standalone


# ══════════════════════════════════════════════════════════════════════════
#  PRETTY PRINT
# ══════════════════════════════════════════════════════════════════════════

def print_island_report(summaries, top_n=15):
    print(f"\n{'='*110}")
    print(f"  TOP {top_n} ISLAND CONFIGURATIONS — Ranked by Total GDP (with CTA)")
    print(f"{'='*110}\n")

    header = (f"  {'Rank':>4} | {'Ovens':>5} | {'Tanks':>5} | {'GH':>3} | "
              f"{'GDP':>10} | {'Bakery':>8} | {'Fish':>8} | {'GH':>8} | "
              f"{'B_SL%':>5} | {'F_SL%':>5} | {'G_SL%':>5} | "
              f"{'WasteTax':>8} | {'Credits':>8}")
    print(header)
    print("  " + "-" * 106)

    for rank, s in enumerate(summaries[:top_n], 1):
        c = s["config"]
        print(f"  {rank:>4} | {c['num_ovens']:>5} | {c['num_tanks']:>5} | "
              f"{c['num_units']:>3} | "
              f"${s['gdp_mean']:>8.0f} | ${s['bakery_profit_mean']:>6.0f} | "
              f"${s['fish_profit_mean']:>6.0f} | ${s['gh_profit_mean']:>6.0f} | "
              f"{s['bakery_sl_mean']:>4.1f} | {s['fish_sl_mean']:>4.1f} | "
              f"{s['gh_sl_mean']:>4.1f} | ${s['waste_tax_mean']:>6.0f} | "
              f"${s['total_credits_mean']:>6.0f}")


def print_detailed(s):
    c = s["config"]
    print(f"""
  ======================================================================
    OPTIMAL ISLAND CONFIGURATION — Southern Reefs (Phase B + CTA)
  ======================================================================

  Bakery (Flourish & Crumb):     {c['num_ovens']} ovens (CTA cap: {CTA_DIRECTIVES['MAX_OVENS_PER_BAKER']})
  Fish Farm (Blue Horizon):      {c['num_tanks']} tanks (mode: {c['fish_mode']})
  Greenhouse (Verdant Canopy):   {c['num_units']} units (mode: {c.get('gh_mode', 'high')})

  -- Daily Island GDP --
  Total:       ${s['gdp_mean']:>10,.2f}  +/- ${s['gdp_std']:>8,.2f}
  5th/95th:    ${s['gdp_p5']:>10,.2f} / ${s['gdp_p95']:>10,.2f}

  -- Breakdown by Enterprise --
  Bakery:      ${s['bakery_profit_mean']:>10,.2f}   (SL: {s['bakery_sl_mean']:.1f}%, P(>=90): {s['bakery_sl_ge90']:.1f}%)
  Fish Farm:   ${s['fish_profit_mean']:>10,.2f}   (SL: {s['fish_sl_mean']:.1f}%, P(>=90): {s['fish_sl_ge90']:.1f}%)
  Greenhouse:  ${s['gh_profit_mean']:>10,.2f}   (SL: {s['gh_sl_mean']:.1f}%, P(>=90): {s['gh_sl_ge90']:.1f}%)

  -- Circular Economy + CTA Metrics --
  Waste Tax (daily):    ${s['waste_tax_mean']:>8,.2f}
  CTA Credits (daily):  ${s['total_credits_mean']:>8,.2f}
  Basil imported:       {s['basil_import_mean']:.1f} kg/day (lower = better)
  Heat surplus vented:  {s['heat_surplus_mean']:.1f} units/day (lower = better)
""")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("\n" + "=" * 80)
    print("  SOUTHERN REEFS — ISLAND-WIDE OPTIMIZATION")
    print("  Phase B: The Great Re-Alignment — Circular Economy + CTA")
    print("  " + "=" * 76)

    # Phase 1: Grid Search
    summaries = island_grid_search(n_trials=500, verbose=True)

    # Phase 2: Report
    print_island_report(summaries, top_n=15)

    # Phase 3: Detailed best config
    best = summaries[0]
    print_detailed(best)

    # Phase 4: Circular vs Standalone comparison
    print(f"\n{'='*80}")
    print(f"  CIRCULAR + CTA vs STANDALONE COMPARISON")
    print(f"{'='*80}\n")

    circ, standalone = compare_circular_vs_standalone(best["config"], n_trials=1000)
    print(f"  {'':>30} {'CIRC+CTA':>12} {'STANDALONE':>12} {'SAVINGS':>12}")
    print(f"  {'-'*66}")
    print(f"  {'Island GDP':>30} ${circ['gdp_mean']:>10,.0f} "
          f"${standalone['gdp_mean']:>10,.0f} "
          f"${circ['gdp_mean']-standalone['gdp_mean']:>10,.0f}")
    print(f"  {'Bakery profit':>30} ${circ['bakery_profit_mean']:>10,.0f} "
          f"${standalone['bakery_profit_mean']:>10,.0f} "
          f"${circ['bakery_profit_mean']-standalone['bakery_profit_mean']:>10,.0f}")
    print(f"  {'Fish Farm profit':>30} ${circ['fish_profit_mean']:>10,.0f} "
          f"${standalone['fish_profit_mean']:>10,.0f} "
          f"${circ['fish_profit_mean']-standalone['fish_profit_mean']:>10,.0f}")
    print(f"  {'Greenhouse profit':>30} ${circ['gh_profit_mean']:>10,.0f} "
          f"${standalone['gh_profit_mean']:>10,.0f} "
          f"${circ['gh_profit_mean']-standalone['gh_profit_mean']:>10,.0f}")
    print(f"  {'Waste Tax':>30} ${circ['waste_tax_mean']:>10,.0f} "
          f"${standalone['waste_tax_mean']:>10,.0f} "
          f"${standalone['waste_tax_mean']-circ['waste_tax_mean']:>10,.0f}")
    print(f"  {'CTA Credits':>30} ${circ.get('total_credits_mean', 0):>10,.0f}")

    delta = circ['gdp_mean'] - standalone['gdp_mean']
    if standalone['gdp_mean'] != 0:
        pct = delta / abs(standalone['gdp_mean']) * 100
        print(f"\n  Circular + CTA adds ${delta:,.0f}/day ({pct:+.1f}%)")

    elapsed = time.time() - t_start
    print(f"\n  Total pipeline time: {elapsed:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
