#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
"""

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

Tri3 = Tuple[float, float, float]  # (min, mode, max)

@dataclass(frozen=True)
class Params:
    # --- simulation window ---
    sim_minutes: float = 240.0
    closeout_minutes: float = 180.0  # let in-flight groups finish/leave

    # --- ARRIVALS ---
    mean_interarrival_min: float = 3.0

    # --- SEATS ---
    seats: int = 27
    seats_cap: int = 30

    servers_weekday: int = 6
    servers_weekend: int = 7

    chef_count: int = 1
    chef_helpers_lunch: int = 2
    chef_helpers_dinner: int = 3

    # --- KITCHEN FIFO capacity ---
    servers: int = 6
    chef_count: int = 1
    chef_helpers: int = 2

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

    # leaver-type patience
    patience_leave_mean_min: float = 1.0
    patience_leave_cap_min: float = 20.0

    # knobs to keep realized left/served in ~[0.2, 0.4]
    leave_share_multiplier: float = 1.35          # increase => more leavers exist
    leaver_balk_if_queue_prob: float = 0.20       # increase => more balk when queue exists

    # ========================================================
    # FLOW STAGE TIMINGS (Triangular: min, mode, max)
    # IMPORTANT: Replace these with your Excel min/mode/max.
    # ========================================================

    # seated -> order taken
    tri_order_take_min_mode_max: Tri3 = (1.5, 3.0, 4.5)

    # kitchen prep
    tri_prep_min_mode_max: Tri3 = (4.0, 8.0, 12.0)

    # delivered -> payment
    tri_to_payment_min_mode_max: Tri3 = (17.5, 35.0, 52.5)

    # payment -> departure
    tri_payment_to_depart_min_mode_max: Tri3 = (0.5, 1.0, 1.5)

    # --- revenue per diner ---
    avg_main_price: float = 20.0
    prob_appetizer: float = 0.45
    avg_appetizer_price: float = 8.0
    prob_dressing: float = 0.08
    dressing_price: float = 6.0

    variable_cost_rate: float = 0.0
    # --- scenario ---
    patio_cost_per_daypart: float = 120.0 
    extra_staff_cost_per_daypart: float = 180.0

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

def tri_time_min_mode_max(tri: Tri3) -> float:
    lo, mode, hi = tri
    # safety
    lo = max(0.01, float(lo))
    hi = max(lo, float(hi))
    mode = float(mode)
    mode = max(lo, min(mode, hi))
    return random.triangular(lo, hi, mode)

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
         # NEW: front-of-house servers
        self.servers = simpy.Resource(env, capacity=P.kitchen_servers)    # placeholder; set later via Params-derived daypart

        # NEW: kitchen resources (chef + helpers)
        self.chef = simpy.Resource(env, capacity=1)        # placeholder; set later via Params-derived daypart
        self.helpers = simpy.Resource(env, capacity=1)     # placeholder; set later via Params-derived daypart


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

    # stage times (Triangular MIN/MODE/MAX)
    def order_take_time(self) -> float:
        return tri_time_min_mode_max(self.P.tri_order_take_min_mode_max)

    def prep_time(self) -> float:
        return tri_time_min_mode_max(self.P.tri_prep_min_mode_max)

    def to_payment_time(self) -> float:
        return tri_time_min_mode_max(self.P.tri_to_payment_min_mode_max)

    def payment_to_depart_time(self) -> float:
        return tri_time_min_mode_max(self.P.tri_payment_to_depart_min_mode_max)

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
        self.log(
            f"Arrive G{gid} size={size} leaver={is_leaver} patience={patience:.2f} q_seats={self.seats.queue_seat_demand()}"
        )

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
# COMBINED LOG (SERVED + BALK + RENEG)  ✅
# ============================================================

def print_combined_log(recs: List[GroupRec], start_hhmm: str, limit: int = 200):
    """
    One table that includes ALL groups (served, balk, renege).
    Columns are designed so SERVED has full timeline, leavers show left_at.
    """
    rows = sorted(recs, key=lambda x: x.gid)[:limit]

    print("\n=== COMBINED EVENT LOG (SERVED + BALK + RENEG) ===")
    print(
        "Idx\tGID\tStatus\tSize\tLeaver?\tArrival\t"
        "Table\tTable#\tOrderTaken\tDelivered\tPayment\tDepart\t"
        "SeatWait\tKitchenWait\tPrep\tLeftAt\tReason"
    )

    for i, r in enumerate(rows, 1):
        arrival = minutes_to_clock(r.arrival_t, start_hhmm)

        # served timeline
        table = minutes_to_clock(r.table_t, start_hhmm) if r.table_t is not None else ""
        order_taken = minutes_to_clock(r.order_taken_t, start_hhmm) if r.order_taken_t is not None else ""
        delivered = minutes_to_clock(r.order_delivered_t, start_hhmm) if r.order_delivered_t is not None else ""
        payment = minutes_to_clock(r.payment_t, start_hhmm) if r.payment_t is not None else ""
        depart = minutes_to_clock(r.depart_t, start_hhmm) if r.depart_t is not None else ""

        # left at (for balk/renege)
        left_at = ""
        if r.status == "BALK":
            left_at = minutes_to_clock(r.arrival_t, start_hhmm)
        elif r.status == "RENEGE":
            waited = float(r.seat_wait_t or 0.0)
            left_at = minutes_to_clock(r.arrival_t + waited, start_hhmm)

        seat_wait = f"{float(r.seat_wait_t):.1f}" if r.seat_wait_t is not None else ""
        k_wait = f"{float(r.kitchen_wait_t):.1f}" if r.kitchen_wait_t is not None else ""
        prep = f"{float(r.prep_t):.1f}" if r.prep_t is not None else ""

        leaver_flag = "Y" if r.leaver_type else "N"

        print(
            f"{i}\t{r.gid}\t{r.status}\t{r.size}\t{leaver_flag}\t{arrival}\t"
            f"{table}\t{r.table_number or ''}\t{order_taken}\t{delivered}\t{payment}\t{depart}\t"
            f"{seat_wait}\t{k_wait}\t{prep}\t{left_at}\t{r.leave_reason or ''}"
        )

 # ============================================================
# CI 95
# ============================================================
       
def ci95(xs: List[float]) -> Tuple[float, float, float]:
    xs = np.asarray(xs, dtype=float)
    xs = xs[~np.isnan(xs)]
    n = len(xs)
    if n == 0:
        return (0.0, 0.0, 0.0)
    m = float(xs.mean())
    if n == 1:
        return (m, m, m)
    s = float(xs.std(ddof=1))
    half = 1.96 * s / math.sqrt(n)
    return (m, m - half, m + half)

def prob_gt(xs: List[float], thr: float) -> float:
    xs = np.asarray(xs, dtype=float)
    xs = xs[~np.isnan(xs)]
    return float(np.mean(xs > thr)) if len(xs) else 0.0

def p95_nan(xs: List[float]) -> float:
    xs = np.asarray(xs, dtype=float)
    xs = xs[~np.isnan(xs)]
    return float(np.percentile(xs, 95)) if len(xs) else 0.0

# ============================================================
# SCENARIOS + ANNUAL EXPERIMENT (back in)
# ============================================================

MON, TUE, WED, THU, FRI, SAT, SUN = 0, 1, 2, 3, 4, 5, 6
SUMMER_MONTHS = {6, 7, 8}

@dataclass(frozen=True)
class DaypartConfig:
    sim_minutes: float
    mean_interarrival_min: float

@dataclass(frozen=True)
class AnnualConfig:
    reps: int = 50
    days: int = 365
    year_start_weekday: int = MON
    base_seed: int = 2000
    lunch: DaypartConfig = DaypartConfig(sim_minutes=180, mean_interarrival_min=5.0)
    dinner: DaypartConfig = DaypartConfig(sim_minutes=240, mean_interarrival_min=3.0)

def is_weekend(dow: int) -> bool:
    return dow in {SAT, SUN}

def staff_for_daypart(P: Params, dow: int, is_lunch: bool) -> tuple[int, int, int, int]:
    """
    Returns: (seats, servers, chef_count, helper_count)
    """
    # seats always capped
    seats = clamp(P.seats_base, 1, P.seats_cap)

    servers = P.servers_weekend if is_weekend(dow) else P.servers_weekday
    chef = P.chef_count
    helpers = P.chef_helpers_lunch if is_lunch else P.chef_helpers_dinner
    return int(seats), int(servers), int(chef), int(helpers)

def day_to_month(day_index: int) -> int:
    # simple mapping: 30-day months (good enough for scenario seasonality)
    return min(12, (day_index // 30) + 1)

def baseline_open(day_of_week: int) -> Tuple[bool, bool]:
    # same pattern as earlier:
    # Open Mon/Thu/Fri/Sat dinner; lunch same except Sat lunch closed; Tue/Wed/Sun closed
    dinner_open = day_of_week in {MON, THU, FRI, SAT}
    lunch_open  = day_of_week in {MON, THU, FRI, SAT}
    if day_of_week == SAT:
        lunch_open = False
    return lunch_open, dinner_open

def scenario_open_rules(scenario_name: str, day_of_week: int) -> Tuple[bool, bool]:
    lunch_open, dinner_open = baseline_open(day_of_week)

    # Example: open Wed + Sat lunch for scenario 2 if you want it
    if scenario_name == "scenario2_more_chefs":
        # keep open days identical by default (only staffing changes)
        return lunch_open, dinner_open

    if scenario_name == "scenario1_patio":
        # patio doesn’t change hours
        return lunch_open, dinner_open

    if scenario_name == "scenario3_price_adjust":
        # price doesn’t change hours
        return lunch_open, dinner_open

    return lunch_open, dinner_open

def build_params_for_daypart(base_P: Params, dp: DaypartConfig, seed: int) -> Params:
    return Params(**{
        **asdict(base_P),
        "sim_minutes": dp.sim_minutes,
        "mean_interarrival_min": dp.mean_interarrival_min,
        "seed": seed,
        "validation_log": False,
    })

def scenario_apply_params(P: Params, scenario_name: str, month: int) -> Params:
    """
    Returns a NEW Params with scenario adjustments.
    Important: we do NOT calibrate. We only change the system inputs.
    """
    if scenario_name == "baseline":
        return P

    if scenario_name == "scenario1_patio":
        # add seats only in summer months
        if month in SUMMER_MONTHS:
            return Params(**{**asdict(P), "seats": P.seats + 20})
        return P

    if scenario_name == "scenario2_more_chefs":
        # add kitchen capacity
        return Params(**{**asdict(P), "kitchen_servers": P.kitchen_servers + 1})

    if scenario_name == "scenario3_price_adjust":
        # price + elasticity reduces demand
        price_increase = 0.10
        elasticity = -0.5  # demand multiplier < 1 when price rises
        demand_multiplier = (1.0 + price_increase) ** (elasticity)

        new_interarrival = P.mean_interarrival_min / demand_multiplier
        new_price = P.avg_main_price * (1.0 + price_increase)

        return Params(**{
            **asdict(P),
            "avg_main_price": new_price,
            "mean_interarrival_min": new_interarrival,
        })

    raise ValueError(f"Unknown scenario: {scenario_name}")
    
def print_annual_scenario_results(rep_outputs: Dict[str, List[AnnualRepResult]], threshold: float = 0.333):
    print("\n=== ANNUAL SCENARIO RESULTS (mean + 95% CI over reps) ===")

    for scen, reps in rep_outputs.items():
        print(f"\n--- {scen} ---")

        # helpers
        def fmt_ci(xs, digits=3):
            m, lo, hi = ci95(xs)
            return f"{m:.{digits}f}  (95% CI: {lo:.{digits}f}–{hi:.{digits}f})"

        def fmt_ci_int(xs):
            m, lo, hi = ci95(xs)
            return f"{int(round(m))}  (95% CI: {int(round(lo))}–{int(round(hi))})"

        # --- core metrics ---
        annual_los = [rr.avg_daily_left * 365 / max(1, rr.avg_daily_served) for rr in reps]
        served_year = [rr.avg_daily_served * 365 for rr in reps]
        left_year = [rr.avg_daily_left * 365 for rr in reps]
        seat_wait = [rr.avg_wait for rr in reps]

        pct_rev_lost = [
            100.0 * rr.avg_revenue_lost_per_day / max(1e-9, rr.avg_revenue_total_per_day)
            for rr in reps
        ]

        profit_year = [
            (rr.avg_revenue_per_day - rr.avg_cost_per_day) * 365
            for rr in reps
        ]

        p_bad = [
            prob_gt(rr.daily_left_over_served, threshold)
            for rr in reps
        ]

        avg_daily_los = [
            float(np.nanmean(rr.daily_left_over_served))
            for rr in reps
        ]

        p95_daily_los = [
            np.nanpercentile(rr.daily_left_over_served, 95)
            for rr in reps
        ]

        # --- printing ---
        print(f"Annual left/served:      {fmt_ci(annual_los)}")
        print(f"Served/year:             {fmt_ci_int(served_year)}")
        print(f"Left/year:               {fmt_ci_int(left_year)}")
        print(f"Avg seat-wait (min):     {fmt_ci(seat_wait, 2)}")
        print(f"% revenue lost:          {fmt_ci(pct_rev_lost, 2)}%")
        print(f"Profit/year ($):         {fmt_ci(profit_year, 2)}")
        print(f"P(daily left/served > {threshold:.3f}): {fmt_ci(p_bad, 3)}")
        print(f"Avg daily left/served:   {fmt_ci(avg_daily_los, 3)}")
        print(f"P95 daily left/served:   {fmt_ci(p95_daily_los, 3)}")

def scenario_incremental_cost(P: Params, scenario_name: str, month: int, daypart_open: bool) -> float:
    if not daypart_open:
        return 0.0
    extra = 0.0
    if scenario_name == "scenario1_patio" and month in SUMMER_MONTHS:
        extra += P.patio_cost_per_daypart
    if scenario_name == "scenario2_more_chefs":
        extra += P.extra_staff_cost_per_daypart
    return extra


SCENARIOS = [
    "baseline",
    "scenario1_patio",
    "scenario2_more_chefs",
    "scenario3_price_adjust",
]


# ============================================================
# Annual replication results for dashboard metrics
# ============================================================

MIN_SERVED_FOR_RATIO = 30
DAILY_RATE_FLOOR = 0.20
DAILY_RATE_CAP = 0.40

@dataclass
class AnnualRepResult:
    scenario: str
    rep_seed: int
    daily_left_over_served: List[float]  # bounded for reporting
    revenue_lost_pct: float              # annual
    avg_wait_min: float                  # annual weighted
    left_over_served_year: float         # annual totals
    served_year: int
    left_year: int
    profit_year: float

def run_one_year_for_scenario(base_P: Params, cfg: AnnualConfig, scenario_name: str, rep_seed: int) -> AnnualRepResult:
    total_served = 0
    total_left = 0
    total_rev_served = 0.0
    total_rev_lost = 0.0
    total_profit = 0.0
    total_cost = 0.0

    # weighted wait
    wait_weighted_sum = 0.0
    wait_weight = 0

    daily_rates: List[float] = []

    for day in range(cfg.days):
        month = day_to_month(day)
        dow = (cfg.year_start_weekday + day) % 7
        lunch_open, dinner_open = scenario_open_rules(scenario_name, dow)

        day_served = 0
        day_left = 0
        day_rev_served = 0.0
        day_rev_lost = 0.0
        day_profit = 0.0

        def run_daypart(dp: DaypartConfig, seed_offset: int):
            nonlocal day_served, day_left, day_rev_served, day_rev_lost, day_profit
            nonlocal total_cost, total_profit, wait_weighted_sum, wait_weight

            P_day = build_params_for_daypart(base_P, dp, seed=rep_seed + day * 10 + seed_offset)
            P_day = scenario_apply_params(P_day, scenario_name, month)

            r, _ = run_rep(P_day)

            inc_cost = scenario_incremental_cost(P_day, scenario_name, month, True)

            day_served += r.diners_served
            day_left += r.diners_missed
            day_rev_served += r.revenue_served
            day_rev_lost += r.revenue_lost

            # r.profit already includes variable costs (if any)
            day_profit += (r.profit - inc_cost)

            total_cost += (r.operating_cost + inc_cost)

            # annual weighted wait: weight by served diners
            w = max(1, r.diners_served)
            wait_weighted_sum += r.avg_seat_wait_min * w
            wait_weight += w

        if lunch_open:
            run_daypart(cfg.lunch, seed_offset=1)
        if dinner_open:
            run_daypart(cfg.dinner, seed_offset=2)

        total_served += day_served
        total_left += day_left
        total_rev_served += day_rev_served
        total_rev_lost += day_rev_lost
        total_profit += day_profit

        # bounded daily KPI for dashboard only
        if day_served >= MIN_SERVED_FOR_RATIO:
            rate = day_left / max(1, day_served)
            rate = clamp(rate, DAILY_RATE_FLOOR, DAILY_RATE_CAP)
            daily_rates.append(rate)
        else:
            daily_rates.append(np.nan)

    pot = total_rev_served + total_rev_lost
    rev_lost_pct = (total_rev_lost / pot) if pot > 0 else 0.0
    avg_wait_year = (wait_weighted_sum / wait_weight) if wait_weight > 0 else 0.0
    los_year = (total_left / total_served) if total_served > 0 else 0.0

    return AnnualRepResult(
        scenario=scenario_name,
        rep_seed=rep_seed,
        daily_left_over_served=daily_rates,
        revenue_lost_pct=rev_lost_pct,
        avg_wait_min=avg_wait_year,
        left_over_served_year=los_year,
        served_year=int(total_served),
        left_year=int(total_left),
        profit_year=float(total_profit),
    )

def run_annual_experiment_with_reps(base_P: Params, cfg: AnnualConfig) -> Dict[str, List[AnnualRepResult]]:
    rep_outputs: Dict[str, List[AnnualRepResult]] = {}
    for scen in SCENARIOS:
        reps: List[AnnualRepResult] = []
        for i in range(cfg.reps):
            rep_seed = cfg.base_seed + i * 100_000
            reps.append(run_one_year_for_scenario(base_P, cfg, scen, rep_seed))
        rep_outputs[scen] = reps
    return rep_outputs

def print_scenario_annual_ci(rep_outputs: Dict[str, List[AnnualRepResult]], threshold: float):
    print("\n=== ANNUAL SCENARIO RESULTS (mean + 95% CI over reps) ===")
    for scen, reps in rep_outputs.items():
        los_year = [rr.left_over_served_year for rr in reps]
        served_year = [rr.served_year for rr in reps]
        left_year = [rr.left_year for rr in reps]
        wait_year = [rr.avg_wait_min for rr in reps]
        rev_lost_pct = [100.0 * rr.revenue_lost_pct for rr in reps]  # %
        profit_year = [rr.profit_year for rr in reps]

        # dashboard-style daily stats per rep
        prob_bad = [prob_gt(rr.daily_left_over_served, threshold) for rr in reps]
        avg_daily = [float(np.nanmean(rr.daily_left_over_served)) for rr in reps]
        p95_daily = [p95_nan(rr.daily_left_over_served) for rr in reps]

        m_los, lo_los, hi_los = ci95(los_year)
        m_serv, lo_serv, hi_serv = ci95(served_year)
        m_left, lo_left, hi_left = ci95(left_year)
        m_wait, lo_wait, hi_wait = ci95(wait_year)
        m_rev, lo_rev, hi_rev = ci95(rev_lost_pct)
        m_prof, lo_prof, hi_prof = ci95(profit_year)

        m_pb, lo_pb, hi_pb = ci95(prob_bad)
        m_ad, lo_ad, hi_ad = ci95(avg_daily)
        m_p95, lo_p95, hi_p95 = ci95(p95_daily)

        print(f"\n--- {scen} ---")
        print(f"Annual left/served:      {m_los:.3f}  (95% CI: {lo_los:.3f}–{hi_los:.3f})")
        print(f"Served/year:             {m_serv:.0f}  (95% CI: {lo_serv:.0f}–{hi_serv:.0f})")
        print(f"Left/year:               {m_left:.0f}  (95% CI: {lo_left:.0f}–{hi_left:.0f})")
        print(f"Avg seat-wait (min):     {m_wait:.2f}  (95% CI: {lo_wait:.2f}–{hi_wait:.2f})")
        print(f"% revenue lost:          {m_rev:.2f}% (95% CI: {lo_rev:.2f}–{hi_rev:.2f})")
        print(f"Profit/year ($):         {m_prof:.2f} (95% CI: {lo_prof:.2f}–{hi_prof:.2f})")

        print(f"P(daily left/served > {threshold:.3f}): {m_pb:.3f}  (95% CI: {lo_pb:.3f}–{hi_pb:.3f})")
        print(f"Avg daily left/served:   {m_ad:.3f}  (95% CI: {lo_ad:.3f}–{hi_ad:.3f})")
        print(f"P95 daily left/served:   {m_p95:.3f}  (95% CI: {lo_p95:.3f}–{hi_p95:.3f})")

# ============================================================
# SUMMARY TABLE
# ============================================================
def print_scenario_summary_table(outputs, rep_outputs, days=365, threshold=0.33):
    """
    Prints the scenario comparison table exactly as requested.
    """

    headers = [
        "Scenario",
        "P(left >0.33)",
        "Avg. Daily left",
        "Avg. Daily serve",
        "Avg time wait in line",
        "Avg time order food",
        "Avg time cook",
        "Avg time to pay",
        "Avg time to depart",
        "Avg. cost",
        "Avg. revenue",
    ]

    print("\t".join(headers))
    print("-" * 160)

    for scen in SCENARIOS:
        reps = rep_outputs[scen]
        summaries = [rr.summary for rr in reps]

        # --- probability daily left/served > threshold ---
        p_bad = np.mean([
            np.mean(np.array(rr.daily_left_over_served) > threshold)
            for rr in reps
        ])

        # --- daily volumes ---
        avg_left = np.mean([s.diners_missed for s in summaries]) / days
        avg_served = np.mean([s.diners_served for s in summaries]) / days

        # --- time metrics (served customers only) ---
        def mean_attr(attr):
            xs = []
            for rr in reps:
                for g in getattr(rr, "groups", []):
                    if g.status == "SERVED" and getattr(g, attr) is not None:
                        xs.append(getattr(g, attr))
            return float(np.mean(xs)) if xs else 0.0

        avg_wait = np.mean([s.avg_seat_wait_min for s in summaries])
        avg_order = mean_attr("order_taken_t") - mean_attr("table_t")
        avg_cook = mean_attr("prep_t")
        avg_pay = mean_attr("payment_t") - mean_attr("order_delivered_t")
        avg_depart = mean_attr("depart_t") - mean_attr("payment_t")

        # --- financials ---
        avg_cost = np.mean([s.operating_cost for s in summaries]) / days
        avg_revenue = np.mean([s.revenue_served for s in summaries]) / days

        print(
            f"{scen}\t"
            f"{p_bad:.3f}\t"
            f"{avg_left:.1f}\t"
            f"{avg_served:.1f}\t"
            f"{avg_wait:.2f}\t"
            f"{avg_order:.2f}\t"
            f"{avg_cook:.2f}\t"
            f"{avg_pay:.2f}\t"
            f"{avg_depart:.2f}\t"
            f"{avg_cost:.2f}\t"
            f"{avg_revenue:.2f}"
        )

    outputs, rep_outputs = run_annual_experiment_with_reps(base_P, cfg)

    print_scenario_summary_table(
        outputs,
        rep_outputs,
        days=365,
        threshold=0.33
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

        # ✅ Put Excel min/mode/max here:
        tri_order_take_min_mode_max=(1.5, 3.0, 4.5),
        tri_prep_min_mode_max=(4.0, 8.0, 12.0),
        tri_to_payment_min_mode_max=(17.5, 35.0, 52.5),
        tri_payment_to_depart_min_mode_max=(0.5, 1.0, 1.5),

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

    # ✅ combined log
    print_combined_log(recs, start_hhmm=P.log_clock_start_hhmm, limit=200)

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

    # ----------------------------
    # Annual scenarios (back in)
    # ----------------------------
    cfg = AnnualConfig(reps=30, days=365, year_start_weekday=MON, base_seed=2000)

    rep_outputs = run_annual_experiment_with_reps(
        base_P=Params(**{**asdict(P), "validation_log": False}),
        cfg=cfg
    )

    # choose threshold = your input KPI (or 0.333, or 0.4 etc)
    print_scenario_annual_ci(rep_outputs, threshold=P.target_left_over_served)

    cfg = AnnualConfig(
        reps=50,
        days=365,
        base_seed=2000
    )

    # Run ALL scenarios from P
    outputs, rep_outputs = run_annual_experiment_with_reps(P, cfg)

    # Print scenario comparison table
    print_scenario_summary_table(
        outputs,
        rep_outputs,
        days=365,
        threshold=0.33
    )


# In[ ]:




