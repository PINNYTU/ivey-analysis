"""
BakeryDay — Flourish & Crumb (Phase B), Team A.

Full operational model: SimPy machine process, market, cost accounting.
All constants imported from island_grid_B (single source of truth).

Production: 40 loaves/batch, 20 kg flour + 2 kg basil per batch.
Single baker: 10 min load + 60 min bake + 10 min unload.
15-min grace period after bake or batch ruined.
1 unit gas/hr while baking -> 1 unit heat + 1 unit CO2.
"""

import sys
import random
from pathlib import Path

import numpy as np
import simpy

from enterprise_base import EnterpriseDay

_GRID_DIR = Path(__file__).resolve().parents[1]
if str(_GRID_DIR) not in sys.path:
    sys.path.insert(0, str(_GRID_DIR))

from island_grid_B import (
    SHIFT_MIN, LEASE_PER_UNIT, ADMIN_OVERHEAD,
    FAILURE_RATE, REPAIR_COST, REPAIR_RANGE, LIQUIDATION,
)


class BakeryDay(EnterpriseDay):
    """One 480-minute shift of Flourish & Crumb (Phase B)."""

    FLOUR_BATCH = 20.0   # kg flour per batch
    BASIL_BATCH = 2.0    # kg basil per batch
    GRACE_PERIOD = 15.0  # minutes: worker must arrive or batch ruined

    # ── Contract Properties ──────────────────────────────────────────────

    @property
    def enterprise_name(self):
        return "bakery"

    @property
    def num_machines(self):
        return self._num_ovens

    @property
    def stagger_minutes(self):
        return self._stagger

    @property
    def demand_mean(self):
        return self._demand_mean

    @property
    def setup_time(self):
        return 10.0

    @property
    def production_time(self):
        return 60.0

    @property
    def teardown_time(self):
        return 10.0

    @property
    def yield_per_batch(self):
        return 40.0

    @property
    def sale_price(self):
        return self.prices["BREAD_PRICE"]

    @property
    def gas_per_batch(self):
        return 1.0  # GAS_PER_HR * BAKE_MIN/60 = 1.0 * 60/60

    @property
    def output_key(self):
        return "loaves"

    # ── Constructor ──────────────────────────────────────────────────────

    def __init__(self, config, env=None, grid=None, prices=None):
        super().__init__(config, env, grid, prices)
        self._num_ovens = config["num_ovens"]
        self._stagger = config.get("bakery_stagger", 15)
        self._demand_mean = config.get("bakery_demand", 400)
        self._use_grid_basil = config.get("use_grid_basil", True)

    # ── Operations: State ────────────────────────────────────────────────

    def _make_state(self):
        return {
            self.output_key: 0.0,
            "batches_started": 0,
            "batches_completed": 0,
            "batches_failed": 0,
            "gas_used": 0.0,
            "repair_cost": 0.0,
            "downtime": 0.0,
            "worker_busy": 0.0,
            "heat_produced": 0.0,
            "co2_produced": 0.0,
        }

    # ── Operations: SimPy Machine Process ────────────────────────────────

    def _machine_process(self, env, worker, state, machine_id, delay):
        yield env.timeout(delay)

        while True:
            setup_t = self.setup_time
            prod_t = self.production_time
            tear_t = self.teardown_time

            if env.now > SHIFT_MIN - (setup_t + prod_t + tear_t):
                break

            # SETUP (worker required)
            with worker.request() as req:
                yield req
                if env.now > SHIFT_MIN - (prod_t + tear_t):
                    break
                state["worker_busy"] += setup_t
                yield env.timeout(setup_t)

            state["batches_started"] += 1

            # AUTONOMOUS PRODUCTION
            yield env.timeout(prod_t)
            state["gas_used"] += self.gas_per_batch
            state["heat_produced"] += self.gas_per_batch   # 1:1 gas -> heat
            state["co2_produced"] += self.gas_per_batch    # 1:1 gas -> CO2

            # Grid hook (shared-env mode)
            if self.grid is not None:
                yield self.grid.energy.put(self.gas_per_batch)

            # FAILURE CHECK
            if random.random() < FAILURE_RATE:
                state["batches_failed"] += 1
                state["repair_cost"] += REPAIR_COST
                repair_t = random.uniform(*REPAIR_RANGE)
                state["downtime"] += repair_t
                yield env.timeout(min(repair_t, max(0, SHIFT_MIN - env.now)))
                continue

            # TEARDOWN with grace period (bakery-specific)
            if env.now >= SHIFT_MIN:
                break
            with worker.request() as req:
                result = yield req | env.timeout(self.GRACE_PERIOD)
                if req in result:
                    state["worker_busy"] += tear_t
                    yield env.timeout(tear_t)
                    state["batches_completed"] += 1
                    state[self.output_key] += self.yield_per_batch
                else:
                    state["batches_failed"] += 1
                    continue

    # ── Operations: Market ───────────────────────────────────────────────

    def _compute_market(self, state):
        demand = max(0, np.random.poisson(self.demand_mean))
        produced = state[self.output_key]
        sold = min(produced, demand)
        unsold = max(produced - demand, 0)
        lost = max(demand - produced, 0)

        revenue = sold * self.sale_price + unsold * self.sale_price * LIQUIDATION
        sl = (sold / demand * 100) if demand > 0 else 100.0

        return dict(
            demand=demand, sold=sold, unsold=unsold,
            lost_sales=lost, revenue=revenue, service_level=sl,
        )

    # ── Operations: Cost Accounting ──────────────────────────────────────

    def _compute_base_costs(self, state):
        return dict(
            lease_cost=self.num_machines * LEASE_PER_UNIT,
            gas_cost=state["gas_used"] * self.prices["GAS"],
            overhead=ADMIN_OVERHEAD,
            repair_cost=state["repair_cost"],
        )

    def _compute_enterprise_costs(self, state):
        flour_cost = state["batches_started"] * self.FLOUR_BATCH * self.prices["FLOUR"]

        basil_kg = state["batches_started"] * self.BASIL_BATCH
        if self._use_grid_basil:
            basil_cost = 0.0
        else:
            basil_cost = basil_kg * self.prices["BASIL_IMPORT"]
            if basil_kg > 0:
                eoq = max(1, np.sqrt(
                    2 * basil_kg * self.prices["ORDER_FEE_BASIL"]
                    / self.prices["STORAGE_FEE_BASIL"]
                ))
                basil_cost += (basil_kg / eoq) * self.prices["ORDER_FEE_BASIL"]
                basil_cost += (eoq / 2) * self.prices["STORAGE_FEE_BASIL"]

        co2_tax = state["co2_produced"] * self.prices["WASTE_TAX"]

        return dict(
            flour_cost=flour_cost,
            basil_cost=basil_cost,
            basil_kg_needed=basil_kg,
            co2_tax=co2_tax,
        )

    # ── Operations: Results Assembly ─────────────────────────────────────

    def _build_results(self, state, market, base_costs, enterprise_costs):
        total_cost = (
            base_costs["lease_cost"] + base_costs["gas_cost"]
            + base_costs["overhead"] + base_costs["repair_cost"]
            + enterprise_costs["flour_cost"]
            + enterprise_costs["basil_cost"]
            + enterprise_costs["co2_tax"]
        )
        profit = market["revenue"] - total_cost

        return dict(
            enterprise="bakery",
            profit=profit,
            revenue=market["revenue"],
            total_cost=total_cost,
            service_level=market["service_level"],
            demand=market["demand"],
            loaves=state["loaves"],
            sold=market["sold"],
            unsold=market["unsold"],
            lost_sales=market["lost_sales"],
            batches_started=state["batches_started"],
            batches_completed=state["batches_completed"],
            batches_failed=state["batches_failed"],
            gas_used=state["gas_used"],
            gas_cost=base_costs["gas_cost"],
            heat_produced=state["heat_produced"],
            co2_produced=state["co2_produced"],
            co2_tax=enterprise_costs["co2_tax"],
            lease_cost=base_costs["lease_cost"],
            flour_cost=enterprise_costs["flour_cost"],
            basil_cost=enterprise_costs["basil_cost"],
            basil_kg_needed=enterprise_costs["basil_kg_needed"],
            repair_cost=state["repair_cost"],
            overhead=base_costs["overhead"],
            baker_util=state["worker_busy"] / SHIFT_MIN * 100,
        )

    # ── Operations: Simulate (Option B — standalone) ─────────────────────

    def simulate(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**31))

        env = simpy.Environment()
        worker = simpy.Resource(env, capacity=1)
        state = self._make_state()

        for i in range(self.num_machines):
            env.process(self._machine_process(
                env, worker, state, i + 1,
                delay=i * self.stagger_minutes,
            ))
        env.run(until=SHIFT_MIN)

        market = self._compute_market(state)
        base_costs = self._compute_base_costs(state)
        enterprise_costs = self._compute_enterprise_costs(state)

        self._state = state
        return self._build_results(state, market, base_costs, enterprise_costs)

    # ── Operations: Run (Option A — shared env) ──────────────────────────

    def run(self):
        worker = simpy.Resource(self.env, capacity=1)
        state = self._make_state()
        self._state = state

        for i in range(self.num_machines):
            self.env.process(self._machine_process(
                self.env, worker, state, i + 1,
                delay=i * self.stagger_minutes,
            ))
        yield self.env.timeout(0)

    def end_of_day(self):
        state = self._state
        market = self._compute_market(state)
        base_costs = self._compute_base_costs(state)
        enterprise_costs = self._compute_enterprise_costs(state)
        return self._build_results(state, market, base_costs, enterprise_costs)
# ─────────────────────────────────────────────────────────────
# Pretty Print Helper
# ─────────────────────────────────────────────────────────────

def pretty_print(res):
    print("\n=== BakeryDay (Phase B – Team A) Results ===")
    print(f"Profit:            ${res['profit']:,.2f}")
    print(f"Revenue:           ${res['revenue']:,.2f}")
    print(f"Total Cost:        ${res['total_cost']:,.2f}")
    print(f"Service Level:     {res['service_level']:.2f}%")
    print(f"Demand:            {res['demand']}")
    print(f"Loaves Produced:   {res['loaves']:.0f}")
    print(f"Loaves Sold:       {res['sold']:.0f}")
    print(f"Unsold:            {res['unsold']:.0f}")
    print(f"Lost Sales:        {res['lost_sales']:.0f}")
    print("\n--- Operations ---")
    print(f"Batches Started:   {res['batches_started']}")
    print(f"Batches Completed: {res['batches_completed']}")
    print(f"Batches Failed:    {res['batches_failed']}")
    print(f"Gas Used:          {res['gas_used']:.2f}")
    print(f"Heat Produced:     {res['heat_produced']:.2f}")
    print(f"CO2 Produced:      {res['co2_produced']:.2f}")
    print("\n--- Costs ---")
    print(f"Lease Cost:        ${res['lease_cost']:,.2f}")
    print(f"Gas Cost:          ${res['gas_cost']:,.2f}")
    print(f"Flour Cost:        ${res['flour_cost']:,.2f}")
    print(f"Basil Cost:        ${res['basil_cost']:,.2f}")
    print(f"CO2 Tax:           ${res['co2_tax']:,.2f}")
    print(f"Repair Cost:       ${res['repair_cost']:,.2f}")
    print(f"Overhead:          ${res['overhead']:,.2f}")
    print(f"Baker Utilization: {res['baker_util']:.2f}%")

# ─────────────────────────────────────────────────────────────
# Standalone run (no manual config needed)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Pull everything from the single source of truth
    from island_grid_B import PHASE_2_PRICES, CTA_DIRECTIVES

    # Minimal “internal defaults” built from grid/policy (so you don’t edit anything)
    cfg = {
        # CTA cap is max ovens per baker (Case B policy)
        "num_ovens": int(CTA_DIRECTIVES.get("MAX_OVENS_PER_BAKER", 3)),
        # these already have defaults in your __init__, but we’ll be explicit
        "bakery_stagger": 15,
        "bakery_demand": 400,
        "use_grid_basil": True,
    }

    bakery = BakeryDay(config=cfg, prices=PHASE_2_PRICES)
    results = bakery.simulate(seed=42)
    pretty_print(results)