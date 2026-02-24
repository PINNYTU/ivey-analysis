"""
BakeryDay — Flourish & Crumb (PHASE_2 / Case B) — SUBMITTABLE VERSION (ADB-compliant)

This version fixes the EnterpriseDay requirement:
✅ prices dict MUST be passed into the constructor (no internal fetching).

Case B compliance implemented:
- Grid-first basil procurement from grid.produce (non-blocking; partial gets allowed)
- Mainland basil fallback with EOQ-style ordering + storage fees (ORDER_FEE_BASIL, STORAGE_FEE_BASIL)
- Tracks heat + CO2 produced (1 gas -> 1 heat + 1 CO2)
- Heat capture to grid.energy (non-blocking; overflow vents)
- Waste tax applied to TOTAL vented waste (heat + CO2) using compute_tiered_waste_tax()
- Splits tiered tax proportionally into heat vs CO2 components for reporting
- Applies CTA credits for captured heat and captured CO2 (CO2 capture is a policy proxy; no CO2 container in grid)

Standalone demo (__main__) runs shared-env mode and prints results.

pip install simpy numpy
"""

import sys
import random
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import simpy

from enterprise_base import EnterpriseDay

_GRID_DIR = Path(__file__).resolve().parents[1]
if str(_GRID_DIR) not in sys.path:
    sys.path.insert(0, str(_GRID_DIR))

from island_grid_B import (
    SHIFT_MIN, LEASE_PER_UNIT, ADMIN_OVERHEAD,
    FAILURE_RATE, REPAIR_COST, REPAIR_RANGE, LIQUIDATION,
    CTA_DIRECTIVES, compute_tiered_waste_tax,
    PHASE_2_PRICES, IslandGrid,
)


class BakeryDay(EnterpriseDay):
    """One 480-minute shift of Flourish & Crumb (PHASE_2 / Case B)."""

    FLOUR_BATCH = 20.0   # kg flour per batch
    BASIL_BATCH = 2.0    # kg basil per batch
    GRACE_PERIOD = 15.0  # minutes: worker must arrive or batch ruined

    # Case statement: 1 gas -> 1 heat + 1 CO2
    HEAT_PER_GAS = 1.0
    CO2_PER_GAS = 1.0

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
        return float(self.prices["BREAD_PRICE"])

    @property
    def gas_per_batch(self):
        return 1.0  # bake 60 min at 1 unit/hr

    @property
    def output_key(self):
        return "loaves"

    # ── Constructor ──────────────────────────────────────────────────────
    def __init__(self, config: Dict[str, Any], env=None, grid=None, prices=None):
        super().__init__(config, env, grid, prices)

        # EnterpriseDay contract: prices MUST be provided
        if prices is None:
            raise ValueError(
                "prices dict is required. Import PHASE_2_PRICES from island_grid_B "
                "and pass it to the enterprise constructor."
            )

        self._num_ovens = int(config["num_ovens"])
        self._stagger = int(config.get("bakery_stagger", 15))
        self._demand_mean = float(config.get("bakery_demand", 400))

        # Grid/circularity policies
        self._use_grid_basil = bool(config.get("use_grid_basil", True))
        self._capture_heat = bool(config.get("capture_heat", True))
        # No CO2 container exists in IslandGrid; we use this as a policy proxy
        self._capture_co2 = bool(config.get("capture_co2", False))

        # Optional internal transfer price for grid basil
        self._grid_basil_unit_price = config.get("grid_basil_unit_price", None)

        self._state: Optional[Dict[str, Any]] = None

    # ── Operations: State ────────────────────────────────────────────────
    def _make_state(self) -> Dict[str, Any]:
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

            # Case B tracking
            "basil_needed_kg": 0.0,
            "basil_from_grid_kg": 0.0,
            "basil_from_mainland_kg": 0.0,

            "heat_captured_kg": 0.0,
            "heat_vented_kg": 0.0,

            "co2_captured_kg": 0.0,
            "co2_vented_kg": 0.0,
        }

    # ── Helper: Non-blocking grid get/put ────────────────────────────────
    def _try_get_from_container(self, container: simpy.Container, amount: float):
        available = float(container.level)
        take = min(available, amount)
        if take > 0:
            yield container.get(take)
        return take

    def _try_put_to_container(self, container: simpy.Container, amount: float):
        free = float(container.capacity - container.level)
        put_amt = min(free, amount)
        if put_amt > 0:
            yield container.put(put_amt)
        return put_amt

    # ── SimPy Machine Process ────────────────────────────────────────────
    def _machine_process(self, env, worker, state: Dict[str, Any], machine_id: int, delay: float):
        yield env.timeout(delay)

        while True:
            setup_t = self.setup_time
            prod_t = self.production_time
            tear_t = self.teardown_time

            if env.now > SHIFT_MIN - (setup_t + prod_t + tear_t):
                break

            # ── Grid-first basil procurement (per batch) ──
            basil_needed = self.BASIL_BATCH
            state["basil_needed_kg"] += basil_needed

            basil_from_grid = 0.0
            basil_from_mainland = basil_needed

            if self.grid is not None and self._use_grid_basil:
                taken = yield from self._try_get_from_container(self.grid.produce, basil_needed)
                basil_from_grid = float(taken)
                basil_from_mainland = float(basil_needed - basil_from_grid)

            state["basil_from_grid_kg"] += basil_from_grid
            state["basil_from_mainland_kg"] += basil_from_mainland

            # SETUP (worker required)
            with worker.request() as req:
                yield req
                if env.now > SHIFT_MIN - (prod_t + tear_t):
                    break
                state["worker_busy"] += setup_t
                yield env.timeout(setup_t)

            state["batches_started"] += 1

            # PRODUCTION (autonomous)
            yield env.timeout(prod_t)

            state["gas_used"] += self.gas_per_batch
            heat = self.gas_per_batch * self.HEAT_PER_GAS
            co2 = self.gas_per_batch * self.CO2_PER_GAS

            state["heat_produced"] += heat
            state["co2_produced"] += co2

            # Heat capture to grid.energy (non-blocking; overflow vents)
            heat_captured = 0.0
            if self.grid is not None and self._capture_heat:
                heat_captured = float((yield from self._try_put_to_container(self.grid.energy, heat)))
            heat_vented = float(heat - heat_captured)

            state["heat_captured_kg"] += heat_captured
            state["heat_vented_kg"] += heat_vented

            # CO2 capture proxy (no CO2 container in IslandGrid)
            co2_captured = float(co2 if self._capture_co2 else 0.0)
            co2_vented = float(co2 - co2_captured)
            state["co2_captured_kg"] += co2_captured
            state["co2_vented_kg"] += co2_vented

            # FAILURE
            if random.random() < FAILURE_RATE:
                state["batches_failed"] += 1
                state["repair_cost"] += REPAIR_COST
                repair_t = random.uniform(*REPAIR_RANGE)
                state["downtime"] += repair_t
                yield env.timeout(min(repair_t, max(0, SHIFT_MIN - env.now)))
                continue

            # TEARDOWN with grace period
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

    # ── Market ───────────────────────────────────────────────────────────
    def _compute_market(self, state: Dict[str, Any]) -> Dict[str, Any]:
        demand = max(0, int(np.random.poisson(self.demand_mean)))
        produced = float(state[self.output_key])
        sold = min(produced, demand)
        unsold = max(produced - demand, 0.0)
        lost = max(demand - produced, 0.0)

        revenue = sold * self.sale_price + unsold * self.sale_price * LIQUIDATION
        sl = (sold / demand * 100.0) if demand > 0 else 100.0

        return dict(
            demand=demand, sold=sold, unsold=unsold,
            lost_sales=lost, revenue=revenue, service_level=sl,
        )

    # ── Cost Accounting ──────────────────────────────────────────────────
    def _compute_base_costs(self, state: Dict[str, Any]) -> Dict[str, float]:
        return dict(
            lease_cost=float(self.num_machines) * float(LEASE_PER_UNIT),
            gas_cost=float(state["gas_used"]) * float(self.prices["GAS"]),
            overhead=float(ADMIN_OVERHEAD),
            repair_cost=float(state["repair_cost"]),
        )

    def _compute_enterprise_costs(self, state: Dict[str, Any]) -> Dict[str, float]:
        flour_cost = float(state["batches_started"]) * self.FLOUR_BATCH * float(self.prices["FLOUR"])

        basil_grid = float(state["basil_from_grid_kg"])
        basil_mainland = float(state["basil_from_mainland_kg"])

        # Default internal basil transfer price = discounted BASIL_PRICE (optional)
        grid_unit_price = self._grid_basil_unit_price
        if grid_unit_price is None:
            grid_unit_price = float(self.prices.get("BASIL_PRICE", 0.0)) * (
                1.0 - float(CTA_DIRECTIVES.get("BASIL_PARTNER_DISCOUNT", 0.0))
            )

        basil_grid_cost = basil_grid * float(grid_unit_price)

        # Mainland basil: EOQ approximation for ordering + storage
        basil_mainland_cost = 0.0
        if basil_mainland > 0:
            basil_mainland_cost += basil_mainland * float(self.prices["BASIL_IMPORT"])
            order_fee = float(self.prices["ORDER_FEE_BASIL"])
            storage_fee = float(self.prices["STORAGE_FEE_BASIL"])
            eoq = max(1.0, float(np.sqrt(2.0 * basil_mainland * order_fee / max(storage_fee, 1e-9))))
            basil_mainland_cost += (basil_mainland / eoq) * order_fee
            basil_mainland_cost += (eoq / 2.0) * storage_fee

        basil_cost = basil_grid_cost + basil_mainland_cost

        # Waste tax: tiered on vented (heat + CO2)
        vented_heat = float(state["heat_vented_kg"])
        vented_co2 = float(state["co2_vented_kg"])
        vented_total = vented_heat + vented_co2

        waste_tax_total = float(compute_tiered_waste_tax(vented_total, CTA_DIRECTIVES))

        # Split tax proportionally (for reporting)
        if vented_total > 0:
            waste_tax_heat = waste_tax_total * (vented_heat / vented_total)
            waste_tax_co2 = waste_tax_total * (vented_co2 / vented_total)
        else:
            waste_tax_heat = 0.0
            waste_tax_co2 = 0.0

        # CTA credits
        heat_credit = float(state["heat_captured_kg"]) * float(CTA_DIRECTIVES["HEAT_CREDIT_PER_UNIT"])
        co2_credit = float(state["co2_captured_kg"]) * float(CTA_DIRECTIVES["CO2_CAPTURE_CREDIT_PER_KG"])

        return dict(
            flour_cost=flour_cost,
            basil_cost=basil_cost,
            basil_grid_cost=basil_grid_cost,
            basil_mainland_cost=basil_mainland_cost,
            basil_grid_kg=basil_grid,
            basil_mainland_kg=basil_mainland,
            waste_tax_heat=waste_tax_heat,
            waste_tax_co2=waste_tax_co2,
            waste_tax_total=waste_tax_total,
            heat_credit=heat_credit,
            co2_credit=co2_credit,
        )

    # ── Results Assembly ─────────────────────────────────────────────────
    def _build_results(self, state, market, base_costs, enterprise_costs) -> Dict[str, Any]:
        total_cost = (
            base_costs["lease_cost"] + base_costs["gas_cost"]
            + base_costs["overhead"] + base_costs["repair_cost"]
            + enterprise_costs["flour_cost"]
            + enterprise_costs["basil_cost"]
            + enterprise_costs["waste_tax_total"]
            - enterprise_costs["heat_credit"]
            - enterprise_costs["co2_credit"]
        )
        profit = float(market["revenue"]) - float(total_cost)

        basil_total = float(enterprise_costs["basil_grid_kg"] + enterprise_costs["basil_mainland_kg"])
        basil_grid_pct = (enterprise_costs["basil_grid_kg"] / basil_total * 100.0) if basil_total > 0 else 0.0
        basil_mainland_pct = 100.0 - basil_grid_pct if basil_total > 0 else 0.0
        

        return dict(
            enterprise="bakery",
            profit=float(profit),
            revenue=float(market["revenue"]),
            total_cost=float(total_cost),
            service_level=float(market["service_level"]),
            demand=int(market["demand"]),

            loaves=float(state["loaves"]),
            sold=float(market["sold"]),
            unsold=float(market["unsold"]),
            lost_sales=float(market["lost_sales"]),

            batches_started=int(state["batches_started"]),
            batches_completed=int(state["batches_completed"]),
            batches_failed=int(state["batches_failed"]),

            gas_used=float(state["gas_used"]),
            gas_cost=float(base_costs["gas_cost"]),

            heat_produced=float(state["heat_produced"]),
            heat_captured=float(state["heat_captured_kg"]),
            heat_vented=float(state["heat_vented_kg"]),

            co2_produced=float(state["co2_produced"]),
            co2_captured=float(state["co2_captured_kg"]),
            co2_vented=float(state["co2_vented_kg"]),

            waste_tax_heat=float(enterprise_costs["waste_tax_heat"]),
            waste_tax_co2=float(enterprise_costs["waste_tax_co2"]),
            waste_tax_total=float(enterprise_costs["waste_tax_total"]),

            heat_credit=float(enterprise_costs["heat_credit"]),
            co2_credit=float(enterprise_costs["co2_credit"]),

            lease_cost=float(base_costs["lease_cost"]),
            flour_cost=float(enterprise_costs["flour_cost"]),

            basil_cost=float(enterprise_costs["basil_cost"]),
            basil_grid_cost=float(enterprise_costs["basil_grid_cost"]),
            basil_mainland_cost=float(enterprise_costs["basil_mainland_cost"]),
            basil_grid_kg=float(enterprise_costs["basil_grid_kg"]),
            basil_mainland_kg=float(enterprise_costs["basil_mainland_kg"]),
            basil_grid_pct=float(basil_grid_pct),
            basil_mainland_pct=float(basil_mainland_pct),
            basil_kg_needed=float(state["basil_needed_kg"]),

            repair_cost=float(state["repair_cost"]),
            overhead=float(base_costs["overhead"]),
            baker_util=float(state["worker_busy"] / SHIFT_MIN * 100.0),
        )

    # ── Integrated runner mode ────────────────────────────────────────────
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

    def end_of_day(self) -> Dict[str, Any]:
        if self._state is None:
            raise RuntimeError("No state found. Did you call run() before end_of_day()?")

        state = self._state
        market = self._compute_market(state)
        base_costs = self._compute_base_costs(state)
        enterprise_costs = self._compute_enterprise_costs(state)
        return self._build_results(state, market, base_costs, enterprise_costs)

    # ── Standalone convenience wrapper ───────────────────────────────────
    def simulate(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**31))

        # Require injected prices (EnterpriseDay contract)
        if self.prices is None:
            raise ValueError(
                "prices dict is required. Import PHASE_2_PRICES from island_grid_B "
                "and pass it to the enterprise constructor."
            )

        # If no env/grid provided, build them
        if self.env is None:
            self.env = simpy.Environment()
        if self.grid is None:
            self.grid = IslandGrid(self.env)

        self.env.process(self.run())
        self.env.run(until=SHIFT_MIN)
        return self.end_of_day()


def pretty_print(res: Dict[str, Any]) -> None:
    money = {
        "profit", "revenue", "total_cost",
        "gas_cost", "lease_cost", "flour_cost", "basil_cost",
        "basil_grid_cost", "basil_mainland_cost",
        "repair_cost", "overhead",
        "waste_tax_heat", "waste_tax_co2", "waste_tax_total",
        "heat_credit", "co2_credit",
    }
    percent = {"service_level", "baker_util", "basil_grid_pct", "basil_mainland_pct"}

    print("\n=== BakeryDay (Case B) Results ===")
    for k in sorted(res.keys()):
        v = res[k]
        if k in money:
            print(f"{k:18s}: ${float(v):,.2f}")
        elif k in percent:
            print(f"{k:18s}: {float(v):.2f}%")
        else:
            if isinstance(v, float):
                print(f"{k:18s}: {v:,.2f}")
            else:
                print(f"{k:18s}: {v}")


if __name__ == "__main__":
    # ✅ This demo matches your EnterpriseDay rule by injecting PHASE_2_PRICES.
    env = simpy.Environment()
    grid = IslandGrid(env)

    config = {
        "num_ovens": True,
        "bakery_stagger": True,
        "bakery_demand": True,

        "use_grid_basil": True,
        "capture_heat": True,
        "capture_co2": True,

        # Optional: internal basil transfer price. If omitted,
        # defaults to BASIL_PRICE * (1 - BASIL_PARTNER_DISCOUNT).
        # "grid_basil_unit_price": 0.0,
    }

    bakery = BakeryDay(
        config=config,
        env=env,
        grid=grid,
        prices=PHASE_2_PRICES,  # REQUIRED
    )

    env.process(bakery.run())
    env.run(until=SHIFT_MIN)

    results = bakery.end_of_day()
    pretty_print(results)
