# bakery_refine_4ovens_optimize_stagger.py
# Fix ovens=4, search stagger values to maximize avg profit (and report max observed profit day),
# and plot profit per day for the best stagger.
#
# pip install simpy matplotlib

from typing import Dict, List, Optional
import simpy
import random
import math
import csv
import os
import sys
import matplotlib.pyplot as plt

# If you're in a notebook/web environment and island_grid.py is in /mnt/data, uncomment:
# sys.path.append("/mnt/data")

try:
    from island_grid import IslandGrid  # type: ignore
except Exception:
    IslandGrid = None


def default_prices() -> Dict[str, float]:
    return {
        "BREAD_PRICE": 8.0,
        "FLOUR": 1.10,
        "BASIL_ORDER_FEE": 50.0,
        "BASIL_IMPORT": 5.0,
        "BASIL_STORAGE": 1.0,  # $/kg/day
        "GAS": 0.05,
    }


def poisson_sample(lam: float, rng: random.Random) -> int:
    if lam >= 50:
        x = int(round(rng.gauss(lam, math.sqrt(lam))))
        return max(0, x)
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


class BakerySim:
    SHIFT_MINUTES = 480
    LOAD_TIME = 10
    BAKE_TIME = 60
    UNLOAD_TIME = 10
    UNLOAD_GRACE = 15

    BATCH_LOAVES = 40
    FLOUR_PER_BATCH = 20.0
    BASIL_PER_BATCH = 2.0

    OVEN_LEASE_PER_SHIFT = 400.0
    ADMIN_OVERHEAD = 100.0

    FAILURE_RATE_DEFAULT = 0.05
    REPAIR_FEE = 300.0
    REPAIR_DOWNTIME_MIN = 120
    REPAIR_DOWNTIME_MAX = 240

    DEMAND_LAMBDA = 400.0

    FLOUR_REPLENISH_QTY = 200.0
    BASIL_REPLENISH_QTY = 20.0

    def __init__(
        self,
        num_ovens: int,
        stagger_minutes: int,
        seed: int,
        failure_rate: float = FAILURE_RATE_DEFAULT,
        use_island_grid: bool = True,
        cap_downtime_to_shift: bool = True,
    ):
        self.num_ovens = int(num_ovens)
        self.stagger_minutes = max(0, int(stagger_minutes))
        self.failure_rate = float(failure_rate)
        self.rng = random.Random(seed)
        self.use_island_grid = use_island_grid
        self.cap_downtime_to_shift = cap_downtime_to_shift

    def _init_day(self):
        self.env = simpy.Environment()
        self.baker = simpy.Resource(self.env, capacity=1)

        if self.use_island_grid and IslandGrid is not None:
            grid = IslandGrid(self.env)
            self.prices = grid.get_prices()
        else:
            self.prices = default_prices()

        self.flour_stock = 100.0
        self.basil_stock = 10.0
        self.bread_inventory = 0

        self.total_revenue = 0.0
        self.total_costs = self.ADMIN_OVERHEAD + self.num_ovens * self.OVEN_LEASE_PER_SHIFT

        self.completed_batches = 0
        self.ruined_batches = 0
        self.failed_batches = 0

        self.demand = 0
        self.sold_full = 0
        self.liquidated = 0

    def _buy_flour_if_needed(self):
        if self.flour_stock < self.FLOUR_PER_BATCH:
            qty = self.FLOUR_REPLENISH_QTY
            self.flour_stock += qty
            self.total_costs += qty * float(self.prices["FLOUR"])

    def _buy_basil_if_needed(self):
        if self.basil_stock < self.BASIL_PER_BATCH:
            qty = self.BASIL_REPLENISH_QTY
            self.basil_stock += qty
            self.total_costs += float(self.prices["BASIL_ORDER_FEE"]) + qty * float(self.prices["BASIL_IMPORT"])

    def _consume_ingredients_for_batch(self):
        self._buy_flour_if_needed()
        self._buy_basil_if_needed()
        self.flour_stock -= self.FLOUR_PER_BATCH
        self.basil_stock -= self.BASIL_PER_BATCH

    def bake_cycle(self, oven_id: int):
        initial_delay = (oven_id - 1) * self.stagger_minutes
        if initial_delay > 0:
            yield self.env.timeout(initial_delay)

        core = self.LOAD_TIME + self.BAKE_TIME + self.UNLOAD_TIME

        while True:
            if self.env.now + core > self.SHIFT_MINUTES:
                return

            self._consume_ingredients_for_batch()

            with self.baker.request() as req:
                yield req
                yield self.env.timeout(self.LOAD_TIME)

            # gas for one-hour bake
            self.total_costs += 1.0 * float(self.prices["GAS"])
            yield self.env.timeout(self.BAKE_TIME)

            # failure
            if self.rng.random() < self.failure_rate:
                self.failed_batches += 1
                self.total_costs += self.REPAIR_FEE
                downtime = self.rng.randint(self.REPAIR_DOWNTIME_MIN, self.REPAIR_DOWNTIME_MAX)
                if self.cap_downtime_to_shift:
                    remaining = max(0, self.SHIFT_MINUTES - self.env.now)
                    yield self.env.timeout(min(downtime, remaining))
                else:
                    yield self.env.timeout(downtime)
                continue

            # unload must START within grace
            with self.baker.request() as req2:
                result = yield req2 | self.env.timeout(self.UNLOAD_GRACE)
                if req2 not in result:
                    self.ruined_batches += 1
                    continue
                yield self.env.timeout(self.UNLOAD_TIME)

            self.bread_inventory += self.BATCH_LOAVES
            self.completed_batches += 1

    def _end_of_day_sales(self):
        self.demand = poisson_sample(self.DEMAND_LAMBDA, self.rng)

        self.sold_full = min(self.bread_inventory, self.demand)
        self.total_revenue += self.sold_full * float(self.prices["BREAD_PRICE"])
        self.bread_inventory -= self.sold_full

        if self.bread_inventory > 0:
            self.liquidated = self.bread_inventory
            self.total_revenue += self.liquidated * (0.5 * float(self.prices["BREAD_PRICE"]))
            self.bread_inventory = 0

        basil_storage = float(self.prices.get("BASIL_STORAGE", 1.0))
        self.total_costs += self.basil_stock * basil_storage

    def run_one_day(self) -> Dict[str, float]:
        self._init_day()
        for i in range(self.num_ovens):
            self.env.process(self.bake_cycle(i + 1))
        self.env.run(until=self.SHIFT_MINUTES)
        self._end_of_day_sales()

        profit = self.total_revenue - self.total_costs
        sl = (self.sold_full / self.demand) if self.demand > 0 else 1.0

        return {
            "profit": float(profit),
            "service_level": float(sl),
            "demand": float(self.demand),
            "sold": float(self.sold_full),
            "liquidated": float(self.liquidated),
            "completed_batches": float(self.completed_batches),
            "ruined_batches": float(self.ruined_batches),
            "failed_batches": float(self.failed_batches),
            "revenue": float(self.total_revenue),
            "costs": float(self.total_costs),
        }


def evaluate_stagger(
    ovens: int,
    stagger: int,
    days: int,
    seed0: int,
    failure_rate: float,
    use_island_grid: bool,
    save_csv: bool = False,
    csv_path: str = "runs.csv",
) -> Dict[str, float]:
    profits: List[float] = []
    sls: List[float] = []

    best_day: Optional[int] = None
    best_profit: float = -1e18
    best_day_snapshot: Optional[Dict[str, float]] = None

    rows: List[Dict[str, float]] = []

    for d in range(days):
        sim = BakerySim(
            num_ovens=ovens,
            stagger_minutes=stagger,
            seed=seed0 + d,
            failure_rate=failure_rate,
            use_island_grid=use_island_grid,
            cap_downtime_to_shift=True,
        )
        out = sim.run_one_day()

        profits.append(out["profit"])
        sls.append(out["service_level"])

        if out["profit"] > best_profit:
            best_profit = out["profit"]
            best_day = d + 1
            best_day_snapshot = out

        if save_csv:
            rows.append({"day": d + 1, "ovens": ovens, "stagger": stagger, "failure_rate": failure_rate, **out})

    profits_sorted = sorted(profits)
    avg_profit = sum(profits) / days
    avg_sl = sum(sls) / days
    p10 = profits_sorted[max(0, int(0.10 * days) - 1)]

    if save_csv and rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    return {
        "ovens": float(ovens),
        "stagger": float(stagger),
        "failure_rate": float(failure_rate),
        "days": float(days),
        "avg_profit": float(avg_profit),
        "avg_sl": float(avg_sl),
        "p10_profit": float(p10),
        "max_profit_observed": float(best_profit),
        "max_profit_day": float(best_day or 0),
        "max_profit_sl": float(best_day_snapshot["service_level"]) if best_day_snapshot else 0.0,
        "max_profit_demand": float(best_day_snapshot["demand"]) if best_day_snapshot else 0.0,
        "max_profit_sold": float(best_day_snapshot["sold"]) if best_day_snapshot else 0.0,
        "max_profit_ruined": float(best_day_snapshot["ruined_batches"]) if best_day_snapshot else 0.0,
        "max_profit_failed": float(best_day_snapshot["failed_batches"]) if best_day_snapshot else 0.0,
    }


def plot_profit_per_day(
    ovens: int,
    stagger: int,
    days: int,
    seed0: int,
    failure_rate: float,
    use_island_grid: bool,
    out_png: str = "profit_per_day.png",
):
    profits: List[float] = []
    best_day = 0
    best_profit = -1e18

    for d in range(days):
        sim = BakerySim(
            num_ovens=ovens,
            stagger_minutes=stagger,
            seed=seed0 + d,
            failure_rate=failure_rate,
            use_island_grid=use_island_grid,
            cap_downtime_to_shift=True,
        )
        out = sim.run_one_day()
        p = float(out["profit"])
        profits.append(p)
        if p > best_profit:
            best_profit = p
            best_day = d + 1

    plt.figure()
    plt.plot(range(1, days + 1), profits)
    plt.xlabel("Day")
    plt.ylabel("Profit ($)")
    plt.title(f"Profit per day — ovens={ovens}, stagger={stagger} min, fail={failure_rate}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"\nSaved plot: {os.path.abspath(out_png)}")
    print(f"Max observed profit day (in this plot run): day {best_day} with profit ${best_profit:,.2f}")


if __name__ == "__main__":
    # ---- Your refined choice ----
    OVENS = 4

    # Search stagger to maximize profit (common ranges)
    STAGGERS = [0, 5, 10, 15, 20, 25, 30]

    # Case default failure rate
    FAILURE_RATE = 0.05

    # Monte Carlo reps
    DAYS = 2000       # bump this up for more stable averages (e.g., 5000)
    SEED0 = 1000

    # If island_grid import fails, set False
    USE_ISLAND_GRID = True

    # Optional: save a CSV for the best stagger only (set True)
    SAVE_BEST_CSV = False

    # Plot profit-per-day for best stagger (set True)
    PLOT_BEST = True

    results: List[Dict[str, float]] = []
    for s in STAGGERS:
        res = evaluate_stagger(
            ovens=OVENS,
            stagger=s,
            days=DAYS,
            seed0=SEED0,
            failure_rate=FAILURE_RATE,
            use_island_grid=USE_ISLAND_GRID,
            save_csv=False,
        )
        results.append(res)

    # pick best by average profit (recommended decision metric)
    best_avg = max(results, key=lambda r: r["avg_profit"])

    print("\n=== Fixed ovens=4 | Stagger search ===")
    print("stagger | avg_profit | avg_SL | p10_profit | max_profit_observed (day, SL)")
    for r in sorted(results, key=lambda x: x["avg_profit"], reverse=True):
        print(
            f"{int(r['stagger']):7d} | "
            f"${r['avg_profit']:9,.2f} | "
            f"{r['avg_sl']*100:6.2f}% | "
            f"${r['p10_profit']:9,.2f} | "
            f"${r['max_profit_observed']:9,.2f} (day {int(r['max_profit_day'])}, SL {r['max_profit_sl']*100:.1f}%)"
        )

    print("\n=== Best by AVG profit (recommended) ===")
    print(
        f"ovens=4, stagger={int(best_avg['stagger'])} | "
        f"avg_profit=${best_avg['avg_profit']:,.2f} | avg_SL={best_avg['avg_sl']*100:.2f}% | "
        f"p10=${best_avg['p10_profit']:,.2f}"
    )

    print("\n=== Highest single-day profit observed (within each stagger) ===")
    print(
        f"Best-avg stagger {int(best_avg['stagger'])}: max_profit=${best_avg['max_profit_observed']:,.2f} "
        f"on day {int(best_avg['max_profit_day'])} (SL {best_avg['max_profit_sl']*100:.1f}%, "
        f"demand {int(best_avg['max_profit_demand'])}, sold {int(best_avg['max_profit_sold'])}, "
        f"ruined {int(best_avg['max_profit_ruined'])}, failed {int(best_avg['max_profit_failed'])})"
    )

    # Optionally save runs for the best stagger
    if SAVE_BEST_CSV:
        best_stagger = int(best_avg["stagger"])
        out_path = f"runs_ovens4_beststagger{best_stagger}_days{DAYS}.csv"
        evaluate_stagger(
            ovens=OVENS,
            stagger=best_stagger,
            days=DAYS,
            seed0=SEED0,
            failure_rate=FAILURE_RATE,
            use_island_grid=USE_ISLAND_GRID,
            save_csv=True,
            csv_path=out_path,
        )
        print(f"\nSaved detailed runs for best stagger to: {os.path.abspath(out_path)}")

    # Plot profit/day for the best-by-average stagger
    if PLOT_BEST:
        best_stagger = int(best_avg["stagger"])
        plot_profit_per_day(
            ovens=OVENS,
            stagger=best_stagger,
            days=DAYS,
            seed0=SEED0,
            failure_rate=FAILURE_RATE,
            use_island_grid=USE_ISLAND_GRID,
            out_png=f"profit_per_day_ovens{OVENS}_stagger{best_stagger}_days{DAYS}.png",
        )
