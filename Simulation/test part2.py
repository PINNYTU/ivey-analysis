#!/usr/bin/env python
# coding: utf-8
"""
Restaurant Kazu – Combined Simulation (SimPy)

Features:
- FIFO multi-seat seating queue (strict FIFO)
- Explicit kitchen FIFO queue (SimPy Resource) with prep time
- NO calibration
- Leave-rate as an INPUT in the flow:
    * target_left_over_served (e.g., 1/3)
    * model converts to leaver share among arrivals (left/total = t/(1+t))
    * leavers have patience ~ Exp(mean=1 min) capped at 20 min
    * stayers effectively never renege
    * optional: leavers can BALK immediately if queue exists (prob)
- Event logs:
    Served log: Arrival, Table, Order Taken, Order Delivered, Payment, Departure
    Balk log
    Renege log
- Single rep + 100 rep summary
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

import simpy
import numpy as np


# ============================================================
# PARAMETERS
# ============================================================
@dataclass(frozen=True)
class Params:
    # --- simulation window ---
    sim_minutes: float = 240.0
    closeout_minutes: float = 180.0  # let in-flight groups finish/leave

    # --- ARRIVALS ---
    mean_interarrival_min: float = 3.0

    # --- SEATS ---
    seats: int = 27

    # --- KITCHEN FIFO capacity ---
    kitchen_servers: int = 2

    # --- group size distribution (1–5) ---
    group_size_probs: Tuple[Tuple[int, float], ...] = (
        (1, 0.12),
        (2, 0.58),
        (3, 0.18),
        (4, 0.09),
        (5, 0.03),
    )

    # --- explicit leave KPI input ---
    target_left_over_served: float = 1.0 / 3.0  # desired KPI input

    # leaver-type patience (your requirement)
    patience_leave_mean_min: float = 1.0
    patience_leave_cap_min: float = 20.0

    # knobs to keep realized left/served in ~[0.2, 0.4]
    leave_share_multiplier: float = 1.35          # increase => more leavers exist
    leaver_balk_if_queue_prob: float = 0.20       # increase => more balk when queue exists

    # --- flow stage timings ---
    # seated -> order taken
    mean_order_take_min: float = 3.0
    order_take_spread: float = 0.5

    # kitchen prep
    mean_prep_min: float = 8.0
    prep_spread: float = 0.5

    # delivered -> payment
    mean_to_payment_min: float = 35.0
    to_payment_spread: float = 0.5

    # payment -> departure
    mean_payment_to_depart_min: float = 1.0
    payment_to_depart_spread: float = 0.5

    # --- revenue per diner ---
    avg_main_price: float = 20.0
    prob_appetizer: float = 0.45
    avg_appetizer_price: float = 8.0
    prob_dressing: float = 0.08
    dressing_price: float = 6.0

    variable_cost_rate: float = 0.0

    # --- logging ---
    seed: int = 42
    validation_log: bool = False
    validation_log_limit: int = 60

    # clock start for logs (HH:MM)
    log_clock_start_hhmm: str = "17:30"


# ============================================================
# RECORDS / RESULTS
# ============================================================
@dataclass
class GroupRec:
    gid: int
    size: int
    arrival_t: float
    status: str  # "SERVED" | "BALK" | "RENEGE"
    revenue: float

    # served timeline stamps
    table_t: Optional[float] = None
    table_number: Optional[str] = None
    order_taken_t: Optional[float] = None
    order_delivered_t: Optional[float] = None
    payment_t: Optional[float] = None
    depart_t: Optional[float] = None

    # waits
    seat_wait_t: Optional[float] = None
    kitchen_wait_t: Optional[float] = None
    prep_t: Optional[float] = None

    # leaving metadata
    leaver_type: Optional[bool] = None
    patience_draw: Optional[float] = None
    leave_reason: Optional[str] = None  # "BALK"/"RENEGE"


@dataclass
class Results:
    params: Dict
    groups_arrived: int
    groups_served: int
    groups_balked: int
    groups_reneged: int
    diners_served: int
    diners_missed: int
    avg_seat_wait_min: float
    p95_seat_wait_min: float
    avg_kitchen_wait_min: float
    p95_kitchen_wait_min: float
    seat_util: float
    revenue_served: float
    revenue_lost: float
    revenue_lost_pct: float
    revenue_lost_balk: float
    revenue_lost_renege: float
    operating_cost: float
    profit: float
    left_over_served: float


# ============================================================
# HELPERS
# ============================================================
def exp_time(mean: float) -> float:
    return random.expovariate(1.0 / max(1e-9, mean))

def tri_time(mean: float, spread: float = 0.5) -> float:
    low = max(0.01, mean * (1.0 - spread))
    high = mean * (1.0 + spread)
    mode = mean
    return random.triangular(low, high, mode)

def weighted_choice(items: Tuple[Tuple[int, float], ...]) -> int:
    r = random.random()
    cum = 0.0
    for val, p in items:
        cum += p
        if r <= cum:
            return val
    return items[-1][0]

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] * (c - k) + ys[c] * (k - f)

def _parse_hhmm(s: str) -> Tuple[int, int]:
    hh, mm = s.split(":")
    return int(hh), int(mm)

def minutes_to_clock(t_min: float, start_hhmm: str) -> str:
    hh0, mm0 = _parse_hhmm(start_hhmm)
    total = hh0 * 60 + mm0 + int(round(t_min))
    hh = (total // 60) % 24
    mm = total % 60
    return f"{hh:02d}:{mm:02d}"


# ============================================================
# FIFO Seat Manager (strict FIFO, multi-seat)
# ============================================================
class SeatManager:
    def __init__(self, env: simpy.Environment, capacity: int):
        self.env = env
        self.capacity = capacity
        self.available = capacity
        self.queue: Deque[Tuple[simpy.Event, int]] = deque()

        self._last_t = 0.0
        self._area_busy = 0.0

    @property
    def busy(self) -> int:
        return self.capacity - self.available

    def queue_seat_demand(self) -> int:
        return int(sum(gs for (_, gs) in self.queue))

    def _update_util_area(self):
        now = self.env.now
        dt = now - self._last_t
        if dt > 0:
            self._area_busy += self.busy * dt
            self._last_t = now

    def utilization(self, horizon: float) -> float:
        self._update_util_area()
        return self._area_busy / (self.capacity * horizon) if horizon > 0 else 0.0

    def request(self, group_size: int) -> simpy.Event:
        evt = self.env.event()
        self.queue.append((evt, group_size))
        self._try_allocate()
        return evt

    def cancel(self, evt: simpy.Event) -> bool:
        for i, (e, _) in enumerate(self.queue):
            if e is evt:
                del self.queue[i]
                return True
        return False

    def release(self, group_size: int):
        self._update_util_area()
        self.available += group_size
        if self.available > self.capacity:
            raise RuntimeError("SeatManager release overflow.")
        self._try_allocate()

    def _try_allocate(self):
        while self.queue:
            evt, gs = self.queue[0]
            if evt.triggered:
                self.queue.popleft()
                continue
            if gs <= self.available:
                self._update_util_area()
                self.available -= gs
                self.queue.popleft()
                evt.succeed()
            else:
                break


# ============================================================
# MODEL
# ============================================================
class KazuModel:
    def __init__(self, env: simpy.Environment, P: Params):
        self.env = env
        self.P = P
        self.seats = SeatManager(env, P.seats)
        self.kitchen = simpy.Resource(env, capacity=P.kitchen_servers)
        self.recs: List[GroupRec] = []
        self._log_lines = 0
        self._table_counter = 0

    def log(self, msg: str):
        if not self.P.validation_log:
            return
        if self._log_lines >= self.P.validation_log_limit:
            return
        print(f"Time {self.env.now:7.2f}: {msg}")
        self._log_lines += 1

    # revenue per diner
    def diner_spend(self) -> float:
        spend = self.P.avg_main_price
        if random.random() < self.P.prob_appetizer:
            spend += self.P.avg_appetizer_price
        if random.random() < self.P.prob_dressing:
            spend += self.P.dressing_price
        return spend

    def group_revenue(self, size: int) -> float:
        return sum(self.diner_spend() for _ in range(size))

    # leaver share from target left/served
    def leave_fraction_total(self) -> float:
        t = self.P.target_left_over_served
        base = t / (1.0 + t) if t > 0 else 0.0  # e.g. 1/3 -> 0.25
        f = base * self.P.leave_share_multiplier
        return clamp(f, 0.0, 0.95)

    def sample_is_leaver(self) -> bool:
        return random.random() < self.leave_fraction_total()

    def patience_time(self, is_leaver: bool) -> float:
        if is_leaver:
            t = exp_time(self.P.patience_leave_mean_min)
            return min(t, self.P.patience_leave_cap_min)
        # stayers effectively never renege
        return self.P.sim_minutes + self.P.closeout_minutes + 1.0

    # stage times
    def order_take_time(self) -> float:
        return tri_time(self.P.mean_order_take_min, self.P.order_take_spread)

    def prep_time(self) -> float:
        return tri_time(self.P.mean_prep_min, self.P.prep_spread)

    def to_payment_time(self) -> float:
        return tri_time(self.P.mean_to_payment_min, self.P.to_payment_spread)

    def payment_to_depart_time(self) -> float:
        return tri_time(self.P.mean_payment_to_depart_min, self.P.payment_to_depart_spread)

    def assign_table_number(self) -> str:
        self._table_counter += 1
        n = self._table_counter
        return f"T{n}" if n <= 99 else f"B{n-99}"

    def group(self, gid: int):
        size = weighted_choice(self.P.group_size_probs)
        t_arr = self.env.now
        rev = self.group_revenue(size)

        is_leaver = self.sample_is_leaver()

        # leaver balk if queue exists
        if is_leaver and self.seats.queue_seat_demand() > 0:
            if random.random() < self.P.leaver_balk_if_queue_prob:
                self.log(f"BALK  G{gid} size={size} leaver=True")
                self.recs.append(
                    GroupRec(
                        gid=gid, size=size, arrival_t=t_arr,
                        status="BALK", leave_reason="BALK",
                        revenue=rev,
                        leaver_type=True,
                        patience_draw=0.0,
                    )
                )
                return

        patience = self.patience_time(is_leaver)
        self.log(f"Arrive G{gid} size={size} leaver={is_leaver} patience={patience:.2f} q_seats={self.seats.queue_seat_demand()}")

        # join seat FIFO
        seat_evt = self.seats.request(size)

        outcome = yield simpy.events.AnyOf(self.env, [seat_evt, self.env.timeout(patience)])

        # renege if not seated
        if seat_evt not in outcome.events:
            removed = self.seats.cancel(seat_evt)
            self.log(f"RENEGE G{gid} after={patience:.2f} removed={removed}")
            self.recs.append(
                GroupRec(
                    gid=gid, size=size, arrival_t=t_arr,
                    status="RENEGE", leave_reason="RENEGE",
                    revenue=rev,
                    seat_wait_t=patience,
                    leaver_type=is_leaver,
                    patience_draw=patience,
                )
            )
            return

        # seated
        t_table = self.env.now
        seat_wait = t_table - t_arr
        table_no = self.assign_table_number()
        self.log(f"SEATED G{gid} wait={seat_wait:.2f} table={table_no}")

        # order taken
        yield self.env.timeout(self.order_take_time())
        t_order = self.env.now

        # kitchen FIFO
        kitchen_req_t = self.env.now
        with self.kitchen.request() as req:
            yield req
            k_start = self.env.now
            k_wait = k_start - kitchen_req_t
            prep = self.prep_time()
            self.log(f"KITCHEN G{gid} k_wait={k_wait:.2f} prep={prep:.2f}")
            yield self.env.timeout(prep)
        t_delivered = self.env.now

        # payment
        yield self.env.timeout(self.to_payment_time())
        t_payment = self.env.now

        # depart
        yield self.env.timeout(self.payment_to_depart_time())
        t_dep = self.env.now
        self.seats.release(size)

        self.recs.append(
            GroupRec(
                gid=gid, size=size, arrival_t=t_arr,
                status="SERVED", revenue=rev,
                table_t=t_table, table_number=table_no,
                order_taken_t=t_order,
                order_delivered_t=t_delivered,
                payment_t=t_payment,
                depart_t=t_dep,
                seat_wait_t=seat_wait,
                kitchen_wait_t=k_wait,
                prep_t=prep,
                leaver_type=is_leaver,
                patience_draw=patience,
            )
        )

    def arrivals(self):
        gid = 0
        while self.env.now < self.P.sim_minutes:
            ia = exp_time(self.P.mean_interarrival_min)
            yield self.env.timeout(ia)
            gid += 1
            self.env.process(self.group(gid))


# ============================================================
# RUNNERS
# ============================================================
def run_rep(P: Params) -> tuple[Results, List[GroupRec]]:
    random.seed(P.seed)
    env = simpy.Environment()
    model = KazuModel(env, P)
    env.process(model.arrivals())
    env.run(until=P.sim_minutes + P.closeout_minutes)

    recs = model.recs
    served = [r for r in recs if r.status == "SERVED"]
    balked = [r for r in recs if r.status == "BALK"]
    reneged = [r for r in recs if r.status == "RENEGE"]

    diners_served = sum(r.size for r in served)
    diners_missed = sum(r.size for r in balked) + sum(r.size for r in reneged)

    seat_waits = [float(r.seat_wait_t) for r in served if r.seat_wait_t is not None]
    avg_seat_wait = float(np.mean(seat_waits)) if seat_waits else 0.0
    p95_seat_wait = percentile(seat_waits, 0.95) if seat_waits else 0.0

    k_waits = [float(r.kitchen_wait_t) for r in served if r.kitchen_wait_t is not None]
    avg_k_wait = float(np.mean(k_waits)) if k_waits else 0.0
    p95_k_wait = percentile(k_waits, 0.95) if k_waits else 0.0

    rev_served = sum(r.revenue for r in served)
    rev_lost_balk = sum(r.revenue for r in balked)
    rev_lost_renege = sum(r.revenue for r in reneged)
    rev_lost = rev_lost_balk + rev_lost_renege
    pot = rev_served + rev_lost
    rev_lost_pct = (rev_lost / pot) if pot > 0 else 0.0

    util = model.seats.utilization(P.sim_minutes + P.closeout_minutes)

    operating_cost = P.variable_cost_rate * rev_served
    profit = rev_served - operating_cost

    left_over_served = (diners_missed / diners_served) if diners_served > 0 else 0.0

    res = Results(
        params=asdict(P),
        groups_arrived=len(recs),
        groups_served=len(served),
        groups_balked=len(balked),
        groups_reneged=len(reneged),
        diners_served=diners_served,
        diners_missed=diners_missed,
        avg_seat_wait_min=avg_seat_wait,
        p95_seat_wait_min=p95_seat_wait,
        avg_kitchen_wait_min=avg_k_wait,
        p95_kitchen_wait_min=p95_k_wait,
        seat_util=util,
        revenue_served=rev_served,
        revenue_lost=rev_lost,
        revenue_lost_pct=rev_lost_pct,
        revenue_lost_balk=rev_lost_balk,
        revenue_lost_renege=rev_lost_renege,
        operating_cost=operating_cost,
        profit=profit,
        left_over_served=left_over_served,
    )
    return res, recs


def run_many(P: Params, n_reps: int = 100, seed0: int = 2000) -> Results:
    reps: List[Results] = []
    for i in range(n_reps):
        Pi = Params(**{**asdict(P), "seed": seed0 + i, "validation_log": False})
        r, _ = run_rep(Pi)
        reps.append(r)

    def mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    return Results(
        params={**asdict(P), "n_reps": n_reps, "seed0": seed0, "validation_log": False},
        groups_arrived=int(mean([r.groups_arrived for r in reps])),
        groups_served=int(mean([r.groups_served for r in reps])),
        groups_balked=int(mean([r.groups_balked for r in reps])),
        groups_reneged=int(mean([r.groups_reneged for r in reps])),
        diners_served=int(mean([r.diners_served for r in reps])),
        diners_missed=int(mean([r.diners_missed for r in reps])),
        avg_seat_wait_min=mean([r.avg_seat_wait_min for r in reps]),
        p95_seat_wait_min=mean([r.p95_seat_wait_min for r in reps]),
        avg_kitchen_wait_min=mean([r.avg_kitchen_wait_min for r in reps]),
        p95_kitchen_wait_min=mean([r.p95_kitchen_wait_min for r in reps]),
        seat_util=mean([r.seat_util for r in reps]),
        revenue_served=mean([r.revenue_served for r in reps]),
        revenue_lost=mean([r.revenue_lost for r in reps]),
        revenue_lost_pct=mean([r.revenue_lost_pct for r in reps]),
        revenue_lost_balk=mean([r.revenue_lost_balk for r in reps]),
        revenue_lost_renege=mean([r.revenue_lost_renege for r in reps]),
        operating_cost=mean([r.operating_cost for r in reps]),
        profit=mean([r.profit for r in reps]),
        left_over_served=mean([r.left_over_served for r in reps]),
    )


# ============================================================
# LOG PRINTING
# ============================================================
def print_event_log_table(recs: List[GroupRec], start_hhmm: str, limit: int = 50):
    served = [r for r in sorted(recs, key=lambda x: x.gid) if r.status == "SERVED"][:limit]
    print("Index\tGroup Size\tArrival\tMenu\tTable\tTable Number\tOrder Taken\tOrder Delivered\tPayment\tDeparture")
    for i, r in enumerate(served, 1):
        arrival = minutes_to_clock(r.arrival_t, start_hhmm)
        table = minutes_to_clock(r.table_t or r.arrival_t, start_hhmm)
        order_taken = minutes_to_clock(r.order_taken_t or r.arrival_t, start_hhmm)
        delivered = minutes_to_clock(r.order_delivered_t or r.arrival_t, start_hhmm)
        payment = minutes_to_clock(r.payment_t or r.arrival_t, start_hhmm)
        depart = minutes_to_clock(r.depart_t or r.arrival_t, start_hhmm)
        menu = ""
        print(
            f"{i}\t{r.size}\t{arrival}\t{menu}\t{table}\t{r.table_number or ''}\t"
            f"{order_taken}\t{delivered}\t{payment}\t{depart}"
        )

def print_leave_logs(recs: List[GroupRec], start_hhmm: str, limit: int = 80):
    balked = [r for r in sorted(recs, key=lambda x: x.gid) if r.status == "BALK"][:limit]
    reneged = [r for r in sorted(recs, key=lambda x: x.gid) if r.status == "RENEGE"][:limit]

    print("\n=== EVENT LOG (BALKED groups) ===")
    print("Index\tGroup Size\tArrival\tReason")
    for i, r in enumerate(balked, 1):
        print(f"{i}\t{r.size}\t\t{minutes_to_clock(r.arrival_t, start_hhmm)}\tBALK")

    print("\n=== EVENT LOG (RENEGED groups) ===")
    print("Index\tGroup Size\tArrival\tWaited(min)\tLeft At\tReason")
    for i, r in enumerate(reneged, 1):
        waited = float(r.seat_wait_t or 0.0)
        left_at = r.arrival_t + waited
        print(
            f"{i}\t{r.size}\t\t{minutes_to_clock(r.arrival_t, start_hhmm)}\t"
            f"{waited:5.1f}\t\t{minutes_to_clock(left_at, start_hhmm)}\tRENEGE"
        )


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    P = Params(
        sim_minutes=240,
        closeout_minutes=180,
        mean_interarrival_min=3.0,
        seats=27,
        kitchen_servers=2,

        target_left_over_served=1/3,

        # REQUIRED leaver patience:
        patience_leave_mean_min=1.0,
        patience_leave_cap_min=20.0,

        # band tuning knobs:
        leave_share_multiplier=1.35,
        leaver_balk_if_queue_prob=0.20,

        # flow times
        mean_order_take_min=3.0,
        mean_prep_min=8.0,
        mean_to_payment_min=35.0,
        mean_payment_to_depart_min=1.0,

        log_clock_start_hhmm="17:30",

        validation_log=False,
        seed=42,
    )

    # single rep
    r, recs = run_rep(P)

    print("\n=== BASE CASE SUMMARY (single replication) ===")
    print(f"Diners served:               {r.diners_served}")
    print(f"Diners left (balk+renege):   {r.diners_missed}")
    print(f"Left/served:                {r.left_over_served:.3f}  (input={P.target_left_over_served:.3f})")
    print(f"Avg seat-wait (min):         {r.avg_seat_wait_min:.2f}")
    print(f"P95 seat-wait (min):         {r.p95_seat_wait_min:.2f}")
    print(f"Avg kitchen-wait (min):      {r.avg_kitchen_wait_min:.2f}")
    print(f"P95 kitchen-wait (min):      {r.p95_kitchen_wait_min:.2f}")
    print(f"Seat util:                   {r.seat_util:.3f}")
    print(f"% Revenue lost:              {100*r.revenue_lost_pct:.2f}%")

    # logs
    print("\n=== EVENT LOG (served groups) ===")
    print_event_log_table(recs, start_hhmm=P.log_clock_start_hhmm, limit=50)

    print_leave_logs(recs, start_hhmm=P.log_clock_start_hhmm, limit=80)

    # 100 reps
    agg = run_many(Params(**{**asdict(P), "validation_log": False}), n_reps=100, seed0=2000)

    print("\n=== BASE CASE (100 reps) AVERAGES ===")
    print(f"Avg diners served:           {agg.diners_served}")
    print(f"Avg diners left:             {agg.diners_missed}")
    print(f"Avg left/served:             {agg.left_over_served:.3f}  (input={P.target_left_over_served:.3f})")
    print(f"Avg seat-wait (min):         {agg.avg_seat_wait_min:.2f}")
    print(f"Avg P95 seat-wait (min):     {agg.p95_seat_wait_min:.2f}")
    print(f"Avg kitchen-wait (min):      {agg.avg_kitchen_wait_min:.2f}")
    print(f"Avg P95 kitchen-wait (min):  {agg.p95_kitchen_wait_min:.2f}")
    print(f"Avg % revenue lost:          {100*agg.revenue_lost_pct:.2f}%")
