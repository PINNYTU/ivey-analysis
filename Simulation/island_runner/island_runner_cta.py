"""Phase B integrated simulation runner for the Southern Reefs island.

Uses the EnterpriseDay ABC hierarchy (enterprise_base.py) so that all
three enterprises share the same machine-process loop, market logic,
and cost-accounting framework.  Each enterprise only implements what
is unique to it (SRP / Open-Closed).

Modes
-----
Option B (standalone): Each enterprise runs in its own SimPy env;
    circular accounting is applied post-hoc.  Used by Monte Carlo /
    grid search for speed.
Option A (shared-env): All three enterprises share ONE SimPy env +
    IslandGrid containers for real-time resource exchange during the
    shift.  Demonstrates the skeleton runner pattern.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import simpy

# ── Path setup: island_grid_B lives in parent dir (docs/simulation_B/) ─────
_GRID_DIR = Path(__file__).resolve().parents[1]   # docs/simulation_B/
if str(_GRID_DIR) not in sys.path:
    sys.path.insert(0, str(_GRID_DIR))
from island_grid_B import (
    IslandGrid, PHASE_2_PRICES as PRICES, CTA_DIRECTIVES,
    compute_tiered_waste_tax, SHIFT_MIN,
)

# local imports (same directory)
from bakery_model_cta import BakeryDay
from fish_farm_model import FishFarmDay
from greenhouse_model import GreenhouseDay


# ══════════════════════════════════════════════════════════════════════════
#  OPTION B — Standalone Simulation + Post-Hoc Circular Accounting
# ══════════════════════════════════════════════════════════════════════════

def simulate_island_day(config: Optional[Dict[str, Any]] = None,
                        seed: Optional[int] = None,
                        circular: bool = True) -> Dict[str, Any]:
    """
    Simulate one day for the entire Southern Reefs island (Option B).

    Each enterprise runs in its own SimPy environment via .simulate().
    Circular resource flows (heat, effluent, basil) are reconciled
    after all three simulations complete.

    CTA Directives applied (circular mode):
      - Bakery heat → Fish Farm: tax exempt + $5.50/unit resilience credit
      - Bakery/Fish CO2 → Greenhouse: $5.00/kg capture credit
      - Fish effluent → Greenhouse: $5.00/kg rebate + $5.00/kg heat rebate
      - Greenhouse → Bakery basil: 15% partner discount
      - Surplus waste: tiered tax ($2→$3→$5 per kg)
      - Max ovens per baker: 3 (CTA cap)
    """
    cfg = _default_config(config)
    p = PRICES
    cta = CTA_DIRECTIVES

    # CTA: enforce oven-to-baker ratio cap
    max_ovens = cta["MAX_OVENS_PER_BAKER"]
    if cfg.get("num_ovens", 3) > max_ovens:
        cfg["num_ovens"] = max_ovens

    bakery = BakeryDay(cfg, prices=p)
    fish = FishFarmDay(cfg, prices=p)
    gh = GreenhouseDay(cfg, prices=p)

    b = bakery.simulate(seed=seed)
    f = fish.simulate(seed=(seed + 10000) if seed is not None else None)
    g = gh.simulate(seed=(seed + 20000) if seed is not None else None)

    if circular:
        # ── Heat: Bakery → Fish Farm ──────────────────────────────────
        heat_supplied = b["heat_produced"]
        heat_needed = f["gas_used"]
        heat_to_farm = min(heat_supplied, heat_needed) if cfg["use_grid_heat"] else 0.0
        heat_deficit = max(0.0, heat_needed - heat_to_farm)
        heat_surplus = max(0.0, heat_supplied - heat_to_farm)
        fish_gas_cost = heat_deficit * p["GAS"]

        # CTA: captured heat exempt from waste tax + resilience credit
        bakery_heat_credit = heat_to_farm * cta["HEAT_CREDIT_PER_UNIT"]
        # CTA: surplus heat → tiered waste tax
        bakery_heat_tax = compute_tiered_waste_tax(heat_surplus)

        # ── CO2: Bakery + Fish → Greenhouse (carbon capture) ─────────
        co2_bakery = b["co2_produced"]
        co2_fish = f["co2_produced"]
        co2_captured = co2_bakery + co2_fish  # greenhouse plants absorb all
        # CTA: CO2 capture credit for both suppliers
        bakery_co2_credit = co2_bakery * cta["CO2_CAPTURE_CREDIT_PER_KG"]
        fish_co2_credit = co2_fish * cta["CO2_CAPTURE_CREDIT_PER_KG"]

        # ── Effluent: Fish Farm → Greenhouse ──────────────────────────
        effluent_produced = f["effluent_produced"]
        effluent_captured = effluent_produced  # all captured via grid
        # CTA: effluent rebate for Fish Farm (waste tax rebate)
        fish_effluent_rebate = effluent_captured * cta["EFFLUENT_REBATE_PER_KG"]
        # CTA: heat rebate for Greenhouse (warm effluent saves heating)
        gh_heat_rebate = effluent_captured * cta["HEAT_REBATE_PER_KG"]

        # ── Basil: Greenhouse → Bakery ────────────────────────────────
        basil_needed = b["basil_kg_needed"]
        basil_available = g["basil_kg"]
        basil_to_bakery = min(basil_needed, basil_available) if cfg["use_grid_basil"] else 0.0
        basil_import = max(0.0, basil_needed - basil_to_bakery)
        basil_to_market = max(0.0, basil_available - basil_to_bakery)

        # CTA: 15% partner discount on local basil
        discount = cta["BASIL_PARTNER_DISCOUNT"]
        local_basil_price = p["BASIL_PRICE"] * (1.0 - discount)
        internal_basil_cost = basil_to_bakery * local_basil_price

        # Greenhouse revenue: market at full price + bakery at discounted
        market_demand = g["demand"]
        market_sold = min(basil_to_market, market_demand)
        market_unsold = max(basil_to_market - market_demand, 0.0)
        market_lost = max(market_demand - basil_to_market, 0.0)
        market_rev = (market_sold * p["BASIL_PRICE"]
                      + market_unsold * p["BASIL_PRICE"] * 0.50)
        bakery_transfer_rev = basil_to_bakery * local_basil_price
        gh_revenue = market_rev + bakery_transfer_rev

        # Imported basil (mainland) — no CTA discount
        import_cost = 0.0
        if basil_import > 0:
            import_cost = basil_import * p["BASIL_IMPORT"]
            eoq = max(1, np.sqrt(
                2 * basil_import * p["ORDER_FEE_BASIL"] / p["STORAGE_FEE_BASIL"]
            ))
            import_cost += (basil_import / eoq) * p["ORDER_FEE_BASIL"]
            import_cost += (eoq / 2) * p["STORAGE_FEE_BASIL"]

        # ── P&L: Bakery ──────────────────────────────────────────────
        b_cost = (
            b["lease_cost"] + b["gas_cost"] + b["flour_cost"]
            + b["repair_cost"] + b["overhead"]
            + import_cost + internal_basil_cost
            + bakery_heat_tax      # tiered tax on surplus heat only
            # CO2 tax = 0 (all captured to greenhouse)
        )
        b_credits = bakery_heat_credit + bakery_co2_credit
        b_profit = b["revenue"] - b_cost + b_credits

        # ── P&L: Fish Farm ───────────────────────────────────────────
        f_cost = (
            f["lease_cost"] + fish_gas_cost
            + f["repair_cost"] + f["overhead"]
            # CO2 tax = 0 (captured to greenhouse)
            # Effluent tax = 0 (captured to greenhouse)
        )
        f_credits = fish_effluent_rebate + fish_co2_credit
        f_profit = f["revenue"] - f_cost + f_credits

        # ── P&L: Greenhouse ──────────────────────────────────────────
        g_cost = (
            g["lease_cost"] + g["gas_cost"]
            + g["repair_cost"] + g["overhead"]
            + g["nutrient_cost"] + g["water_cost"]
            # CO2 tax = 0 (self-consumed)
        )
        g_credits = gh_heat_rebate
        g_profit = gh_revenue - g_cost + g_credits

        g_sl = (market_sold / market_demand * 100) if market_demand > 0 else 100.0
        total_waste_tax = bakery_heat_tax  # only surplus heat is taxed
        total_credits = b_credits + f_credits + g_credits

    else:
        # ── STANDALONE (no circular economy, no CTA benefits) ────────
        heat_supplied = b["heat_produced"]
        heat_needed = f["gas_used"]
        heat_surplus = heat_supplied
        heat_deficit = heat_needed
        fish_gas_cost = heat_needed * p["GAS"]

        # All waste vented → tiered tax per firm
        bakery_vented = b["heat_produced"] + b["co2_produced"]
        fish_vented = f["effluent_produced"] + f["co2_produced"]
        bakery_heat_tax = compute_tiered_waste_tax(bakery_vented)
        fish_waste_tax = compute_tiered_waste_tax(fish_vented)

        basil_to_bakery = 0.0
        basil_to_market = g["basil_kg"]
        basil_import = b["basil_kg_needed"]

        import_cost = basil_import * p["BASIL_IMPORT"]
        if basil_import > 0:
            eoq = max(1, np.sqrt(
                2 * basil_import * p["ORDER_FEE_BASIL"] / p["STORAGE_FEE_BASIL"]
            ))
            import_cost += (basil_import / eoq) * p["ORDER_FEE_BASIL"]
            import_cost += (eoq / 2) * p["STORAGE_FEE_BASIL"]

        b_cost = (
            b["lease_cost"] + b["gas_cost"] + b["flour_cost"]
            + b["repair_cost"] + b["overhead"]
            + import_cost + bakery_heat_tax
        )
        f_cost = (
            f["lease_cost"] + fish_gas_cost
            + f["repair_cost"] + f["overhead"]
            + fish_waste_tax
        )
        g_cost = (
            g["lease_cost"] + g["gas_cost"]
            + g["repair_cost"] + g["overhead"]
            + g["nutrient_cost"] + g["water_cost"]
        )

        b_profit = b["revenue"] - b_cost
        f_profit = f["revenue"] - f_cost
        g_profit = g["revenue"] - g_cost
        g_sl = g["service_level"]
        total_waste_tax = bakery_heat_tax + fish_waste_tax

        # No CTA benefits in standalone mode
        total_credits = 0.0
        b_credits = f_credits = g_credits = 0.0
        bakery_heat_credit = bakery_co2_credit = 0.0
        fish_effluent_rebate = fish_co2_credit = 0.0
        gh_heat_rebate = 0.0
        co2_captured = effluent_captured = 0.0

        market_sold = g["sold"]
        market_unsold = g["unsold"]
        market_lost = g["lost_sales"]

    island_gdp = b_profit + f_profit + g_profit

    return {
        "island_gdp": island_gdp,
        "bakery_profit": b_profit,
        "fish_profit": f_profit,
        "greenhouse_profit": g_profit,
        "bakery_sl": b["service_level"],
        "fish_sl": f["service_level"],
        "greenhouse_sl": g_sl,
        "bakery_loaves": b["loaves"],
        "fish_kg": f["fish_kg"],
        "basil_kg": g["basil_kg"],
        "basil_to_bakery": basil_to_bakery,
        "basil_to_market": basil_to_market,
        "basil_import": basil_import,
        "heat_supplied": heat_supplied,
        "heat_needed": heat_needed,
        "heat_deficit": heat_deficit,
        "heat_surplus": heat_surplus,
        "effluent_produced": f["effluent_produced"],
        "co2_captured": co2_captured,
        "effluent_captured": effluent_captured,
        # CTA incentives breakdown
        "total_credits": total_credits,
        "bakery_credits": b_credits,
        "fish_credits": f_credits,
        "greenhouse_credits": g_credits,
        "bakery_heat_credit": bakery_heat_credit,
        "bakery_co2_credit": bakery_co2_credit,
        "fish_effluent_rebate": fish_effluent_rebate,
        "fish_co2_credit": fish_co2_credit,
        "gh_heat_rebate": gh_heat_rebate,
        "total_waste_tax": total_waste_tax,
        "co2_bakery": b["co2_produced"],
        "co2_fish": f["co2_produced"],
        "co2_greenhouse": 0.0,
        "bakery_detail": b,
        "fish_detail": f,
        "greenhouse_detail": g,
    }


# ══════════════════════════════════════════════════════════════════════════
#  OPTION A — Shared-Environment with Real-Time Grid Exchange
# ══════════════════════════════════════════════════════════════════════════

def simulate_island_day_realtime(config: Optional[Dict[str, Any]] = None,
                                 seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Simulate one day using a SHARED SimPy environment + IslandGrid.

    All three enterprises run concurrently.  After each production cycle,
    resources flow through the grid containers in real time:
      Bakery  -> grid.energy  (heat)
      FishFarm -> grid.water  (effluent)
      Greenhouse -> grid.produce (basil)

    Post-shift, end_of_day() computes each enterprise's individual P&L.
    CTA credits are applied post-hoc at the island level.
    """
    import random as _random
    if seed is not None:
        _random.seed(seed)
        np.random.seed(seed % (2**31))

    cfg = _default_config(config)
    cta = CTA_DIRECTIVES

    # CTA: enforce oven cap
    max_ovens = cta["MAX_OVENS_PER_BAKER"]
    if cfg.get("num_ovens", 3) > max_ovens:
        cfg["num_ovens"] = max_ovens

    env = simpy.Environment()
    grid = IslandGrid(env)

    bakery = BakeryDay(cfg, env=env, grid=grid, prices=PRICES)
    fish = FishFarmDay(cfg, env=env, grid=grid, prices=PRICES)
    gh = GreenhouseDay(cfg, env=env, grid=grid, prices=PRICES)

    env.process(bakery.run())
    env.process(fish.run())
    env.process(gh.run())

    env.run(until=SHIFT_MIN)

    b = bakery.end_of_day()
    f = fish.end_of_day()
    g = gh.end_of_day()

    return {
        "mode": "realtime",
        "bakery": b,
        "fish_farm": f,
        "greenhouse": g,
        "grid_energy_level": grid.energy.level,
        "grid_water_level": grid.water.level,
        "grid_produce_level": grid.produce.level,
        "island_gdp": b["profit"] + f["profit"] + g["profit"],
    }


# ══════════════════════════════════════════════════════════════════════════
#  MONTE CARLO & GRID SEARCH
# ══════════════════════════════════════════════════════════════════════════

def island_monte_carlo(config: Optional[Dict[str, Any]] = None,
                       n_trials: int = 500,
                       circular: bool = True) -> Dict[str, Any]:
    results = [
        simulate_island_day(config, seed=i, circular=circular)
        for i in range(n_trials)
    ]
    island = [r["island_gdp"] for r in results]
    bsl = [r["bakery_sl"] for r in results]
    fsl = [r["fish_sl"] for r in results]
    gsl = [r["greenhouse_sl"] for r in results]

    return {
        "config": _default_config(config),
        "trials": n_trials,
        "gdp_mean": float(np.mean(island)),
        "gdp_std": float(np.std(island)),
        "gdp_p5": float(np.percentile(island, 5)),
        "gdp_p95": float(np.percentile(island, 95)),
        "bakery_profit_mean": float(np.mean([r["bakery_profit"] for r in results])),
        "fish_profit_mean": float(np.mean([r["fish_profit"] for r in results])),
        "greenhouse_profit_mean": float(np.mean([r["greenhouse_profit"] for r in results])),
        "total_waste_tax_mean": float(np.mean([r["total_waste_tax"] for r in results])),
        "bakery_sl_mean": float(np.mean(bsl)),
        "fish_sl_mean": float(np.mean(fsl)),
        "greenhouse_sl_mean": float(np.mean(gsl)),
        "bakery_sl_ge90": float(np.mean([1 if x >= 90 else 0 for x in bsl]) * 100),
        "fish_sl_ge90": float(np.mean([1 if x >= 90 else 0 for x in fsl]) * 100),
        "greenhouse_sl_ge90": float(np.mean([1 if x >= 90 else 0 for x in gsl]) * 100),
        "basil_import_mean": float(np.mean([r["basil_import"] for r in results])),
        "heat_surplus_mean": float(np.mean([r["heat_surplus"] for r in results])),
        "total_credits_mean": float(np.mean([r["total_credits"] for r in results])),
    }


def island_grid_search(n_trials: int = 500, verbose: bool = True) -> list:
    oven_range = range(2, 8)
    tank_range = range(2, 7)
    unit_range = range(3, 9)
    combos = [
        (o, t, u) for o in oven_range for t in tank_range for u in unit_range
    ]
    summaries = []

    t0 = time.time()
    for idx, (ovens, tanks, units) in enumerate(combos, 1):
        cfg = dict(
            num_ovens=ovens, bakery_stagger=20, bakery_demand=400,
            num_tanks=tanks, fish_mode="optimal", fish_stagger=35, fish_demand=200,
            num_units=units, gh_mode="high", gh_stagger=14, gh_demand=500,
            use_grid_basil=True, use_grid_heat=True, use_grid_nutrients=True,
        )
        s = island_monte_carlo(cfg, n_trials=n_trials, circular=True)
        summaries.append(s)
        if verbose and idx % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / idx * (len(combos) - idx)
            print(f"  [{idx:4d}/{len(combos)}] ovens={ovens} tanks={tanks} units={units}"
                  f"  GDP={s['gdp_mean']:>8.2f}  ETA {eta:,.0f}s")

    summaries.sort(key=lambda x: x["gdp_mean"], reverse=True)
    if verbose:
        print(f"\nGrid search complete in {time.time() - t0:.1f}s")
    return summaries


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _default_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    defaults = dict(
        phase="PHASE_2",
        num_ovens=3, bakery_stagger=20, bakery_demand=400,
        use_grid_basil=True,
        num_tanks=6, fish_mode="optimal", fish_stagger=35, fish_demand=200,
        use_grid_heat=True,
        num_units=3, gh_mode="high", gh_stagger=14, gh_demand=500,
        use_grid_nutrients=True,
    )
    merged = dict(defaults)
    if config:
        merged.update(config)
    return merged


def _print_day_summary(cfg: Optional[Dict[str, Any]] = None,
                       seed: int = 42, circular: bool = True):
    r = simulate_island_day(cfg, seed=seed, circular=circular)
    label = "CIRCULAR + CTA" if circular else "STANDALONE (no CTA)"
    print(f"\n  [{label}]")
    print(f"  Island GDP:       ${r['island_gdp']:,.2f}")
    print(f"  Bakery:           ${r['bakery_profit']:,.2f} (SL={r['bakery_sl']:.1f}%)"
          f"  credits=${r['bakery_credits']:,.2f}")
    print(f"  Fish Farm:        ${r['fish_profit']:,.2f} (SL={r['fish_sl']:.1f}%)"
          f"  credits=${r['fish_credits']:,.2f}")
    print(f"  Greenhouse:       ${r['greenhouse_profit']:,.2f} (SL={r['greenhouse_sl']:.1f}%)"
          f"  credits=${r['greenhouse_credits']:,.2f}")
    print(f"  Basil: {r['basil_to_bakery']:.0f} to bakery / "
          f"{r['basil_to_market']:.0f} to market / "
          f"{r['basil_import']:.0f} imported")
    print(f"  Heat: {r['heat_supplied']:.0f} supplied / "
          f"{r['heat_deficit']:.0f} deficit / "
          f"{r['heat_surplus']:.0f} surplus")
    print(f"  Waste tax:        ${r['total_waste_tax']:,.2f}")
    print(f"  CTA credits:      ${r['total_credits']:,.2f}")


def _print_realtime_summary(cfg: Optional[Dict[str, Any]] = None,
                            seed: int = 42):
    r = simulate_island_day_realtime(cfg, seed=seed)
    b, f, g = r["bakery"], r["fish_farm"], r["greenhouse"]
    print(f"[Real-time] Island GDP: ${r['island_gdp']:,.2f}")
    print(f"  Bakery:     ${b['profit']:,.2f}  (SL={b['service_level']:.1f}%)")
    print(f"  Fish Farm:  ${f['profit']:,.2f}  (SL={f['service_level']:.1f}%)")
    print(f"  Greenhouse: ${g['profit']:,.2f}  (SL={g['service_level']:.1f}%)")
    print(f"  Grid levels — energy: {r['grid_energy_level']:.1f}"
          f"  water: {r['grid_water_level']:.1f}"
          f"  produce: {r['grid_produce_level']:.1f}")


# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════

from typing import Optional, Sequence
def _print_mc_summary(cfg: Dict[str, Any], n_trials: int, seed0: int = 0, circular: bool = True):
    # island_monte_carlo already loops seeds from 0..n_trials-1
    # so seed0 is optional (keep it simple)
    s = island_monte_carlo(cfg, n_trials=n_trials, circular=circular)

    label = "CIRCULAR + CTA" if circular else "STANDALONE (no CTA)"
    print(f"\n  [MONTE CARLO | {label} | N={n_trials}]")
    print(f"  E[Island GDP]:    ${s['gdp_mean']:,.2f}  (σ=${s['gdp_std']:,.2f})")
    print(f"  GDP p5–p95:       ${s['gdp_p5']:,.2f}  →  ${s['gdp_p95']:,.2f}")
    print(f"  E[Bakery Profit]: ${s['bakery_profit_mean']:,.2f}")
    print(f"  E[Fish Profit]:   ${s['fish_profit_mean']:,.2f}")
    print(f"  E[GH Profit]:     ${s['greenhouse_profit_mean']:,.2f}")
    print(f"  E[Waste Tax]:     ${s['total_waste_tax_mean']:,.2f}")
    print(f"  E[Credits]:       ${s['total_credits_mean']:,.2f}")
    print(f"  SL means:         bakery={s['bakery_sl_mean']:.1f}% | fish={s['fish_sl_mean']:.1f}% | gh={s['greenhouse_sl_mean']:.1f}%")

def run_cli(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Phase B Island runner")
    parser.add_argument("--ovens", type=int, default=3)
    parser.add_argument("--tanks", type=int, default=6)
    parser.add_argument("--units", type=int, default=3)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search", action="store_true",
                        help="run grid search over topology")
    parser.add_argument("--standalone", action="store_true",
                        help="disable circular exchange")
    parser.add_argument("--realtime", action="store_true",
                        help="use shared SimPy env + IslandGrid (Option A)")
    parser.add_argument("--mc", action="store_true",
                    help="run Monte Carlo simulation (uses --trials, default 500 recommended)")
    # ✅ Jupyter-safe: ignore unknown args like -f kernel.json
    args, _unknown = parser.parse_known_args(args=argv)

    cfg = dict(
        num_ovens=args.ovens, num_tanks=args.tanks, num_units=args.units,
        use_grid_basil=True, use_grid_heat=True, use_grid_nutrients=True,
        fish_mode="optimal", gh_mode="high", phase="PHASE_2",
    )

    if args.search:
        top = island_grid_search(n_trials=args.trials, verbose=True)
        best = top[0]
        c = best["config"]
        print(f"\nBest config: ovens={c['num_ovens']} tanks={c['num_tanks']} units={c['num_units']}")
        print(f"E[GDP] = ${best['gdp_mean']:,.2f} (+/- ${best['gdp_std']:,.2f})")
    elif args.mc:
        summary = island_monte_carlo(
            cfg,
            n_trials=args.trials,
            circular=(not args.standalone)
        )

        print("\n--- Monte Carlo Summary ---")
        print(f"Trials:              {summary['trials']}")
        print(f"Avg Island GDP:      ${summary['gdp_mean']:,.2f}")
        print(f"GDP Std Dev:         ${summary['gdp_std']:,.2f}")
        print(f"GDP 5th Percentile:  ${summary['gdp_p5']:,.2f}")
        print(f"GDP 95th Percentile: ${summary['gdp_p95']:,.2f}")
        print(f"Avg Bakery Profit:   ${summary['bakery_profit_mean']:,.2f}")
        print(f"Avg Fish Profit:     ${summary['fish_profit_mean']:,.2f}")
        print(f"Avg Greenhouse Profit:${summary['greenhouse_profit_mean']:,.2f}")
        print(f"Avg Waste Tax:       ${summary['total_waste_tax_mean']:,.2f}")
        print(f"Avg CTA Credits:     ${summary['total_credits_mean']:,.2f}")

    elif args.realtime:
        _print_realtime_summary(cfg, seed=args.seed)
    else:
        # Default report: show both single-run and Monte Carlo for
        # circular+CTA and standalone modes side-by-side.
        _print_day_summary(cfg, seed=args.seed, circular=True)
        _print_day_summary(cfg, seed=args.seed, circular=False)
        _print_mc_summary(cfg, n_trials=args.trials, circular=True)
        _print_mc_summary(cfg, n_trials=args.trials, circular=False)

def main(argv: Optional[Sequence[str]] = None):
    print("--- [ISLAND RUNNER] PHASE B INTEGRATED SIMULATION ---")
    run_cli(argv)

if __name__ == "__main__":
    # Notebook-safe: ignore Jupyter's injected args
    if "ipykernel" in sys.modules:
        main([])     # ✅ force empty argv in Jupyter
    else:
        main(None)   # ✅ normal CLI behavior (uses sys.argv implicitly)
