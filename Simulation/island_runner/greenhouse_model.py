"""
GreenhouseDay — The Verdant Canopy (Phase B), Team C.

Full operational model: SimPy machine process, market, cost accounting.
All constants imported from island_grid_B (single source of truth).

High-Yield: 1 unit gas/hr, 25 kg basil, 2-hr cycle.
Low-Intensity: 0.5 unit/hr, 15 kg, 2-hr cycle.
Single tech: 10 min seed + 10 min harvest.
CO2 self-consumed by plants -> NO waste tax.
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


class GreenhouseDay(EnterpriseDay):
    """One 480-minute shift of The Verdant Canopy (Phase B)."""

    # ── Contract Properties ──────────────────────────────────────────────

    @property
    def enterprise_name(self):
        return "greenhouse"

    @property
    def num_machines(self):
        return self._num_units

    @property
    def stagger_minutes(self):
        return self._stagger

    @property
    def demand_mean(self):
        return self._demand_mean

    @property
    def setup_time(self):
        if self._stochastic_tech:
            return max(1.0, random.expovariate(1.0 / 10.0))
        return 10.0

    @property
    def production_time(self):
        return 120.0

    @property
    def teardown_time(self):
        if self._stochastic_tech:
            return max(1.0, random.expovariate(1.0 / 10.0))
        return 10.0

    @property
    def yield_per_batch(self):
        return self._yield_kg

    @property
    def sale_price(self):
        return self.prices["BASIL_PRICE"]

    @property
    def gas_per_batch(self):
        return self._gas_per_batch

    @property
    def output_key(self):
        return "basil_kg"

    # ── Constructor ──────────────────────────────────────────────────────

    def __init__(self, config, env=None, grid=None, prices=None):
        super().__init__(config, env, grid, prices)

        self._num_units = config.get("num_units", 3)
        self._mode = config.get("gh_mode", "high")
        self._stagger = config.get("gh_stagger", 14)
        self._demand_mean = config.get("gh_demand", 500)

        self._stochastic_tech = config.get("stochastic_tech", False)
        self._use_grid_nutrients = config.get("use_grid_nutrients", False)
        self._nutrient_per_batch = config.get("nutrient_per_batch", 0.0)
        self._water_per_batch = config.get("water_per_batch", 0.0)
        self._nutrient_order_qty = config.get("nutrient_order_qty", 50)

        if self._mode == "high":
            self._yield_kg = 25.0
            gas_per_hr = 1.0
        else:
            self._yield_kg = 15.0
            gas_per_hr = 0.5
        self._gas_per_batch = gas_per_hr * (120.0 / 60.0)

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
            # CO2 self-consumed by plants — no waste recorded

            # Grid hook (shared-env mode)
            if self.grid is not None:
                yield self.grid.produce.put(self.yield_per_batch)

            # FAILURE CHECK
            if random.random() < FAILURE_RATE:
                state["batches_failed"] += 1
                state["repair_cost"] += REPAIR_COST
                repair_t = random.uniform(*REPAIR_RANGE)
                state["downtime"] += repair_t
                yield env.timeout(min(repair_t, max(0, SHIFT_MIN - env.now)))
                continue

            # TEARDOWN (standard — no grace period)
            if env.now >= SHIFT_MIN:
                break
            with worker.request() as req:
                yield req
                if env.now >= SHIFT_MIN:
                    break
                state["worker_busy"] += tear_t
                yield env.timeout(tear_t)
            state["batches_completed"] += 1
            state[self.output_key] += self.yield_per_batch

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
        n_bat = state["batches_started"]
        daily_nut = n_bat * self._nutrient_per_batch
        nut_cost = daily_nut * self.prices["NUTRIENTS"]
        wat_cost = n_bat * self._water_per_batch * self.prices["WATER"]

        if daily_nut > 0 and not self._use_grid_nutrients:
            ord_fee = (daily_nut / self._nutrient_order_qty) * self.prices["ORDER_FEE_NUTRIENTS"]
            stor_fee = (self._nutrient_order_qty / 2) * self.prices["STORAGE_FEE_NUTRIENTS"]
        else:
            ord_fee = 0.0
            stor_fee = 0.0

        co2_tax = 0.0  # self-consumed

        return dict(
            nutrient_cost=nut_cost + ord_fee + stor_fee,
            water_cost=wat_cost,
            co2_tax=co2_tax,
        )

    # ── Operations: Results Assembly ─────────────────────────────────────

    def _build_results(self, state, market, base_costs, enterprise_costs):
        total_cost = (
            base_costs["lease_cost"] + base_costs["gas_cost"]
            + base_costs["overhead"] + base_costs["repair_cost"]
            + enterprise_costs["nutrient_cost"]
            + enterprise_costs["water_cost"]
            + enterprise_costs["co2_tax"]
        )
        profit = market["revenue"] - total_cost

        return dict(
            enterprise="greenhouse",
            profit=profit,
            revenue=market["revenue"],
            total_cost=total_cost,
            service_level=market["service_level"],
            demand=market["demand"],
            basil_kg=state["basil_kg"],
            sold=market["sold"],
            unsold=market["unsold"],
            lost_sales=market["lost_sales"],
            batches_started=state["batches_started"],
            batches_completed=state["batches_completed"],
            batches_failed=state["batches_failed"],
            gas_used=state["gas_used"],
            gas_cost=base_costs["gas_cost"],
            co2_tax=enterprise_costs["co2_tax"],
            lease_cost=base_costs["lease_cost"],
            nutrient_cost=enterprise_costs["nutrient_cost"],
            water_cost=enterprise_costs["water_cost"],
            repair_cost=state["repair_cost"],
            overhead=base_costs["overhead"],
            tech_util=state["worker_busy"] / SHIFT_MIN * 100,
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
