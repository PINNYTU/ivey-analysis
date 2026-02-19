#!/usr/bin/env python
# coding: utf-8
"""
Restaurant Kazu – FULL Combined Simulation (SimPy) + Layout (T1–T6 + BAR B1–B10)

INCLUDES:
✅  seating queue with *physical layout*:
   - Tables: T1..T6 (fixed capacities)
   - Bar: BAR (B1..B10 pooled anyone)
   - Total seats NEVER exceed seats_cap (<=30). Extra seats (if any) are added to BAR.
✅ Store hours -> sim_minutes (Lunch closes 15:00, Dinner closes 21:30)
✅ No one can be seated after close:
   - If a party hasn't been seated by closing time, they leave as "RENEGE" with reason "CLOSE"
✅ FOH servers resource:
   - 6 servers weekdays
   - 7 servers weekends
✅ Kitchen resources:
   - 1 chef
   - 2 helpers lunch
   - 3 helpers dinner
   - Cooking requires BOTH: 1 chef + 1 helper simultaneously
✅ Leave model:
   - target_left_over_served sets a leaver-share pool
   - some balk immediately if queue exists
   - patience Exp(mean) capped
✅ Logs + annual scenarios + summary table
✅ IMPORTANT FIX per your requirement:
   - Leaver column means: Y if BALK/RENEGE, N if SERVED
     (not the sampled “leaver type”)

NOTES:
- Times are in minutes.
- "Avg time order food" = Table -> Order Taken (server time)
- "Avg time cook"       = kitchen wait + prep (chef+helper)
- "Avg time to pay"     = Delivered -> Payment
- "Avg time to depart"  = Payment -> Depart (server time)

✅ FIX INCLUDED (your request):
All "per day" metrics are now computed per **OPEN CALENDAR DAY** (not per daypart, not per 365).
- avg_daily_served/left/cost/revenue = totals / open_days
- annual totals (revenue/cost/profit) = per-open-day * open_days
- plots also use open_days
- overhead calibration is aligned to open_days (not calendar days)

ADDED (your latest request):
✅ Point-estimate scenario blocks (NO CI)
✅ Exhibit table (NO CI) built from annual scenario outputs
"""


from __future__ import annotations
from dataclasses import replace
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

import random
import simpy
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# TYPES
# ============================================================
Tri3 = Tuple[float, float, float]  # (min, mode, max)

# ============================================================
# LAYOUT TYPES
# ============================================================
@dataclass(frozen=True)
class TableSpec:
    name: str
    capacity: int
    kind: str  # "TABLE" or "BAR"

# ============================================================
# PARAMETERS
# ============================================================
@dataclass(frozen=True)
class Params:
    sim_minutes: float = 240.0
    closeout_minutes: float = 180.0

    lunch_open_hhmm: str = "12:00"
    lunch_close_hhmm: str = "15:00"
    dinner_open_hhmm: str = "15:00"
    dinner_close_hhmm: str = "21:30"

    mean_interarrival_min: float = 3.2
    seats_cap: int = 30

    layout_tables: Tuple[TableSpec, ...] = (
        TableSpec("T1", 4, "TABLE"),
        TableSpec("T2", 2, "TABLE"),
        TableSpec("T3", 3, "TABLE"),
        TableSpec("T4", 2, "TABLE"),
        TableSpec("T5", 2, "TABLE"),
        TableSpec("T6", 4, "TABLE"),
    )
    bar_base_seats: int = 10
    bar_extra_seats: int = 2

    servers_weekday: int = 6
    servers_weekend: int = 7

    chef_count: int = 1
    chef_helpers_lunch: int = 2
    chef_helpers_dinner: int = 3

    group_size_probs: Tuple[Tuple[int, float], ...] = (
        (1, 0.0471),
        (2, 0.6588),
        (3, 0.1647),
        (4, 0.0941),
        (5, 0.0353),
    )

    target_left_over_served: float = 1.0 / 3.0
    patience_leave_mean_min: float = 8.0
    patience_leave_cap_min: float = 16.0

    balk_if_queue_demand_ge: int = 14
    balk_prob_if_threshold: float = 0.05

    leave_share_multiplier: float = 0.9
    prob_chef_required: float = 0.3

    # stage-time controls
    tri_order_take_min_mode_max: Tri3 = (1.0, 3.0, 6.0)          # Table -> Order Taken
    tri_prep_helper_min_mode_max: Tri3 = (1.0, 3.5, 6.5)         # kitchen prep if "no chef"
    tri_prep_chef_min_mode_max: Tri3 = (2.0, 4.0, 7.0)           # kitchen prep if chef required
    tri_to_payment_min_mode_max: Tri3 = (22.0, 32.0, 50.0)       # Delivered -> Payment
    tri_payment_to_depart_min_mode_max: Tri3 = (0.0, 1.0, 3.0)   # Payment -> Depart
    tri_host_delay_min_mode_max: Tri3 = (0.0, 2.5, 4.0)          # Arrival -> host delay (off)

    avg_main_price: float = 20.0
    prob_appetizer: float = 0.45
    avg_appetizer_price: float = 8.0
    prob_dressing: float = 0.08
    dressing_price: float = 6.0

    patio_cost_per_daypart: float = 200.0

    seed: int = 42
    validation_log: bool = False
    validation_log_limit: int = 60
    log_clock_start_hhmm: str = "17:30"

    food_cost_rate: float = 0.20

    wage_server_per_hr: float = 18.0
    wage_helper_per_hr: float = 17.0
    wage_chef_per_hr: float = 25.0

    overhead_per_open_day: float = 900.0
    closeout_labor_factor: float = 0.50

# ============================================================
# RECORDS
# ============================================================
@dataclass
class GroupRec:
    gid: int
    size: int
    arrival_t: float
    status: str
    revenue: float

    table_t: Optional[float] = None
    table_number: Optional[str] = None
    order_taken_t: Optional[float] = None
    order_delivered_t: Optional[float] = None
    payment_t: Optional[float] = None
    depart_t: Optional[float] = None

    seat_wait_t: Optional[float] = None
    kitchen_wait_t: Optional[float] = None
    prep_t: Optional[float] = None

    leaver_type: Optional[bool] = None
    patience_draw: Optional[float] = None
    leave_reason: Optional[str] = None

@dataclass
class RepStats:
    diners_served: int
    diners_left: int
    left_over_served: float

    avg_wait: float
    avg_order: float
    avg_cook: float
    avg_pay: float
    avg_depart: float

    revenue_served: float
    revenue_lost: float
    operating_cost: float
    profit: float
    revenue_total: float
    pct_revenue_lost: float

# ============================================================
# HELPERS
# ============================================================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def exp_time(mean: float) -> float:
    return random.expovariate(1.0 / max(1e-9, mean))

def tri_time_min_mode_max(tri: Tri3) -> float:
    lo, mode, hi = tri
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

def _parse_hhmm(s: str) -> Tuple[int, int]:
    hh, mm = s.split(":")
    return int(hh), int(mm)

def minutes_to_clock(t_min: float, start_hhmm: str) -> str:
    hh0, mm0 = _parse_hhmm(start_hhmm)
    total = hh0 * 60 + mm0 + int(round(t_min))
    hh = (total // 60) % 24
    mm = total % 60
    return f"{hh:02d}:{mm:02d}"

def hhmm_to_minutes(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)

def minutes_between(start_hhmm: str, end_hhmm: str) -> int:
    a = hhmm_to_minutes(start_hhmm)
    b = hhmm_to_minutes(end_hhmm)
    if b < a:
        b += 24 * 60
    return b - a

def build_daypart_params(base_P: Params, *, seed: int, is_lunch: bool, mean_interarrival: float) -> Params:
    if is_lunch:
        open_hhmm = base_P.lunch_open_hhmm
        close_hhmm = base_P.lunch_close_hhmm
    else:
        open_hhmm = base_P.dinner_open_hhmm
        close_hhmm = base_P.dinner_close_hhmm

    sim_minutes = float(minutes_between(open_hhmm, close_hhmm))

    return replace(
        base_P,
        seed=seed,
        sim_minutes=sim_minutes,
        mean_interarrival_min=mean_interarrival,
        log_clock_start_hhmm=open_hhmm,
        validation_log=base_P.validation_log,
        validation_log_limit=base_P.validation_log_limit,
    )

# ============================================================
# STAFFING
# ============================================================
MON, TUE, WED, THU, FRI, SAT, SUN = 0, 1, 2, 3, 4, 5, 6
SUMMER_MONTHS = {6, 7, 8, 9}

def is_weekend(dow: int) -> bool:
    return dow in {SAT, SUN}

def servers_for_day(P: Params, dow: int) -> int:
    return int(P.servers_weekend if is_weekend(dow) else P.servers_weekday)

def helpers_for_daypart(P: Params, is_lunch: bool) -> int:
    return int(P.chef_helpers_lunch if is_lunch else P.chef_helpers_dinner)

# ============================================================
# LAYOUT
# ============================================================
class LayoutManager:
    def __init__(self, env: simpy.Environment, P: Params):
        self.env = env
        self.P = P

        tables: List[TableSpec] = []
        for t in list(P.layout_tables):
            if isinstance(t, TableSpec):
                tables.append(t)
            elif isinstance(t, dict):
                tables.append(TableSpec(name=t["name"], capacity=int(t["capacity"]), kind=t.get("kind", "TABLE")))
            else:
                raise TypeError(f"layout_tables items must be TableSpec or dict, got: {type(t)}")
        self.tables = tables

        table_sum = sum(t.capacity for t in self.tables)
        bar_target = P.bar_base_seats + max(0, int(P.bar_extra_seats))
        bar_capacity = min(bar_target, max(0, int(P.seats_cap) - table_sum))
        self.bar = TableSpec("BAR", int(bar_capacity), "BAR")

        self.free_tables: Dict[str, bool] = {t.name: True for t in self.tables}
        self.table_borrowed: Dict[str, int] = {t.name: 0 for t in self.tables}
        # Borrow rules: table -> {group_size: seats_to_add_from_bar}
        self.borrow_rules: Dict[str, Dict[int, int]] = {
            "T6": {5: 1},       # +1 seat to fit 5
            "T5": {3: 1, 4: 2}, # +1 to fit 3, +2 to fit 4
        }
        self.bar_free = self.bar.capacity

        # NEW (evt, group_size, allow_bar, allowed_tables or None)
        # was: Deque[Tuple[simpy.Event, int, bool]]
        self.queue: Deque[Tuple[simpy.Event, int, bool, Optional[set]]] = deque()

    def queue_demand(self) -> int:
        return int(sum(gs for (_, gs, _, _) in self.queue))


    def request(self, group_size: int, allow_bar: bool, allowed_tables: Optional[set] = None) -> simpy.Event:
        evt = self.env.event()
        self.queue.append((evt, group_size, allow_bar, allowed_tables))
        self._try_allocate()
        return evt



    def cancel(self, evt: simpy.Event) -> bool:
        for i, (e, _, _, _) in enumerate(self.queue):
            if e is evt:
                del self.queue[i]
                return True
        return False


    def release_table(self, table_name: str):
        borrowed = self.table_borrowed.get(table_name, 0)
        if borrowed > 0:
            self.bar_free += borrowed
            self.bar_free = min(self.bar_free, self.bar.capacity)
            self.table_borrowed[table_name] = 0

        self.free_tables[table_name] = True
        self._try_allocate()

    def release_bar(self, seats: int):
        self.bar_free += seats
        self.bar_free = min(self.bar_free, self.bar.capacity)
        self._try_allocate()

    def _try_allocate(self):
        def borrow_needed(table_name: str, cap: int, gs: int) -> int:
            """Seats to borrow from BAR to make gs fit at this table; 0 if none."""
            if gs <= cap:
                return 0
            rules = self.borrow_rules.get(table_name, {})
            need = rules.get(gs, 0)
            # Only valid if it exactly enables the fit
            if need > 0 and gs <= cap + need:
                return need
            return 0

        def can_fit(table_name: str, cap: int, gs: int) -> bool:
            if gs <= cap:
                return True
            need = borrow_needed(table_name, cap, gs)
            return (need > 0 and self.bar_free >= need)

        made_progress = True
        while made_progress:
            made_progress = False

            free_table_specs = sorted(
                [t for t in self.tables if self.free_tables.get(t.name, False)],
                key=lambda t: (t.capacity, t.name)
            )

            # ---------- TABLE ASSIGNMENT ----------
            for t in free_table_specs:
                best_idx = None
                best_score = None  # lower is better (waste, then borrow)

                for i, (evt, gs, allow_bar, allowed_tables) in enumerate(self.queue):
                    if evt.triggered:
                        continue

                    # restriction: e.g. size-5 only allowed at T6
                    if allowed_tables is not None and t.name not in allowed_tables:
                        continue

                    if not can_fit(t.name, t.capacity, gs):
                        continue

                    need = borrow_needed(t.name, t.capacity, gs)
                    waste = (t.capacity + need) - gs  # prioritize exact/matching

                    score = (waste, need, -gs)  # prefer less waste, less borrow, larger group
                    if best_score is None or score < best_score:
                        best_score = score
                        best_idx = i

                if best_idx is not None:
                    evt, gs, allow_bar, allowed_tables = self.queue[best_idx]
                    del self.queue[best_idx]

                    self.free_tables[t.name] = False

                    need = borrow_needed(t.name, t.capacity, gs)
                    if need > 0:
                        self.bar_free -= need
                        self.bar_free = max(0, self.bar_free)
                        self.table_borrowed[t.name] = need

                    evt.succeed(value=("TABLE", t.name))
                    made_progress = True

            # ---------- BAR ASSIGNMENT ----------
            # Per your rule:
            # - 1/2 can sit anywhere (bar OK)
            # - 3 can sit table or bar (bar OK)
            # - 4 can sit matching tables and bars (bar OK if enough bar seats)
            while self.bar_free > 0:
                best_idx = None
                best_score = None

                for i, (evt, gs, allow_bar, allowed_tables) in enumerate(self.queue):
                    if evt.triggered:
                        continue
                    if not allow_bar:
                        continue
                    if allowed_tables is not None:
                        continue  # if restricted-to-table, don't seat at bar
                    if gs > self.bar_free:
                        continue

                    # "prioritize matching bar seats" = minimize leftover bar seats after seating
                    waste = self.bar_free - gs
                    score = (waste, -gs)

                    if best_score is None or score < best_score:
                        best_score = score
                        best_idx = i

                if best_idx is None:
                    break

                evt, gs, allow_bar, allowed_tables = self.queue[best_idx]
                del self.queue[best_idx]

                self.bar_free -= gs
                evt.succeed(value=("BAR", "BAR"))
                made_progress = True

# ============================================================
# FINANCE
# ============================================================
def daypart_labor_cost(P: Params, dow: int, is_lunch: bool) -> float:
    servers = servers_for_day(P, dow)
    helpers = helpers_for_daypart(P, is_lunch)
    chefs = P.chef_count

    paid_minutes = P.sim_minutes + P.closeout_labor_factor * P.closeout_minutes
    hours = paid_minutes / 60.0

    return hours * (
        servers * P.wage_server_per_hr +
        helpers * P.wage_helper_per_hr +
        chefs * P.wage_chef_per_hr
    )

# ============================================================
# MODEL
# ============================================================
class KazuModel:
    def __init__(self, env: simpy.Environment, P: Params, dow: int, is_lunch: bool):
        self.env = env
        self.P = P
        self.dow = dow
        self.is_lunch = is_lunch

        self.layout = LayoutManager(env, P)

        self.servers = simpy.Resource(env, capacity=servers_for_day(P, dow))
        self.chef = simpy.Resource(env, capacity=int(P.chef_count))
        self.helpers = simpy.Resource(env, capacity=helpers_for_daypart(P, is_lunch))

        self.recs: List[GroupRec] = []
        self._log_lines = 0

    def log(self, msg: str):
        if not self.P.validation_log:
            return
        if self._log_lines >= self.P.validation_log_limit:
            return
        print(f"Time {self.env.now:7.2f}: {msg}")
        self._log_lines += 1

    def diner_spend(self) -> float:
        spend = self.P.avg_main_price
        if random.random() < self.P.prob_appetizer:
            spend += self.P.avg_appetizer_price
        if random.random() < self.P.prob_dressing:
            spend += self.P.dressing_price
        return spend

    def group_revenue(self, size: int) -> float:
        return sum(self.diner_spend() for _ in range(size))

    def leave_fraction_total(self) -> float:
        t = self.P.target_left_over_served
        base = t / (1.0 + t) if t > 0 else 0.0
        f = base * self.P.leave_share_multiplier
        return clamp(f, 0.0, 0.95)

    def sample_is_leaver(self) -> bool:
        return random.random() < self.leave_fraction_total()

    def patience_time(self, is_leaver: bool) -> float:
        if is_leaver:
            t = exp_time(self.P.patience_leave_mean_min)
            return min(t, self.P.patience_leave_cap_min)
        return self.P.sim_minutes + self.P.closeout_minutes + 1.0

    def group(self, gid: int):
        size = weighted_choice(self.P.group_size_probs)
        t_arr = self.env.now

        host_delay = self.host_delay_time()
        if host_delay > 0:
            yield self.env.timeout(host_delay)

        rev = self.group_revenue(size)
        sampled_leaver = self.sample_is_leaver()

        queue_people_waiting = self.layout.queue_demand()
        if queue_people_waiting >= self.P.balk_if_queue_demand_ge and random.random() < self.P.balk_prob_if_threshold:
            self.recs.append(GroupRec(
                gid=gid, size=size, arrival_t=t_arr,
                status="BALK", leave_reason="BALK_LINE_TOO_LONG",
                revenue=rev, seat_wait_t=0.0, leaver_type=True, patience_draw=0.0,
            ))
            return

        # ---- Scenario seating rules ----
        # 1-2: anywhere (tables or bar)
        # 3: tables or bar
        # 4: matching tables and bars (bar allowed if space)
        # 5: only T6 (can borrow +1 seat via LayoutManager rules)

        allowed_tables = None
        if size == 5:
            allow_bar = False
            allowed_tables = {"T6"}   # ONLY T6
        elif size in (1, 2, 3, 4):
            allow_bar = True          # bar allowed for 1-4
        else:
            allow_bar = False
            allowed_tables = None


        # Hard feasibility check based on your scenario rules
        if size == 5:
            # must have T6 available in layout (borrowing is handled inside LayoutManager)
            has_t6 = any(t.name == "T6" for t in self.layout.tables)
            if not has_t6:
                self.recs.append(GroupRec(
                    gid=gid, size=size, arrival_t=t_arr,
                    status="BALK", leave_reason="NO_FIT",
                    revenue=rev, seat_wait_t=0.0, leaver_type=True, patience_draw=0.0,
                ))
                return
        else:
            # sizes 1-4 always feasible in principle (table or bar), so do NOT balk here
            pass


        patience = self.patience_time(sampled_leaver)

        seat_evt = self.layout.request(size, allow_bar, allowed_tables=allowed_tables)

        time_to_close = max(0.0, self.P.sim_minutes - self.env.now)
        close_evt = self.env.timeout(time_to_close)

        outcome = yield simpy.events.AnyOf(self.env, [seat_evt, self.env.timeout(patience), close_evt])

        if close_evt in outcome.events and seat_evt not in outcome.events:
            self.layout.cancel(seat_evt)
            self.recs.append(GroupRec(
                gid=gid, size=size, arrival_t=t_arr,
                status="RENEGE", leave_reason="CLOSE",
                revenue=rev, seat_wait_t=(self.env.now - t_arr),
                leaver_type=True, patience_draw=patience,
            ))
            return

        if seat_evt not in outcome.events:
            self.layout.cancel(seat_evt)
            self.recs.append(GroupRec(
                gid=gid, size=size, arrival_t=t_arr,
                status="RENEGE", leave_reason="RENEGE",
                revenue=rev, seat_wait_t=(self.env.now - t_arr),
                leaver_type=True, patience_draw=patience,
            ))
            return

        kind, loc = seat_evt.value
        t_table = self.env.now
        seat_wait = t_table - t_arr

        with self.servers.request() as sreq:
            yield sreq
            yield self.env.timeout(tri_time_min_mode_max(self.P.tri_order_take_min_mode_max))
        t_order = self.env.now

        kitchen_req_t = self.env.now
        needs_chef = (random.random() < self.P.prob_chef_required)

        if needs_chef:
            with self.chef.request() as creq, self.helpers.request() as hreq:
                yield creq
                yield hreq
                k_start = self.env.now
                k_wait = k_start - kitchen_req_t
                prep = tri_time_min_mode_max(self.P.tri_prep_chef_min_mode_max)
                yield self.env.timeout(prep)
        else:
            with self.helpers.request() as hreq:
                yield hreq
                k_start = self.env.now
                k_wait = k_start - kitchen_req_t
                prep = tri_time_min_mode_max(self.P.tri_prep_helper_min_mode_max)
                yield self.env.timeout(prep)

        t_delivered = self.env.now
        yield self.env.timeout(tri_time_min_mode_max(self.P.tri_to_payment_min_mode_max))
        t_payment = self.env.now

        with self.servers.request() as sreq:
            yield sreq
            yield self.env.timeout(tri_time_min_mode_max(self.P.tri_payment_to_depart_min_mode_max))
        t_dep = self.env.now

        if kind == "TABLE":
            self.layout.release_table(loc)
        else:
            self.layout.release_bar(size)

        self.recs.append(GroupRec(
            gid=gid, size=size, arrival_t=t_arr,
            status="SERVED", revenue=rev,
            table_t=t_table, table_number=loc,
            order_taken_t=t_order, order_delivered_t=t_delivered,
            payment_t=t_payment, depart_t=t_dep,
            seat_wait_t=seat_wait, kitchen_wait_t=k_wait, prep_t=prep,
            leaver_type=False, patience_draw=patience,
        ))

    def arrivals(self):
        gid = 0
        while self.env.now < self.P.sim_minutes:
            ia = exp_time(self.P.mean_interarrival_min)
            yield self.env.timeout(ia)
            if self.env.now >= self.P.sim_minutes:
                break
            gid += 1
            self.env.process(self.group(gid))

    def host_delay_time(self) -> float:
        return tri_time_min_mode_max(self.P.tri_host_delay_min_mode_max)

# ============================================================
# RUN DAYPART
# ============================================================
def run_rep_daypart(P: Params, seed: int, dow: int, is_lunch: bool) -> Tuple[RepStats, List[GroupRec]]:
    random.seed(seed)
    env = simpy.Environment()
    model = KazuModel(env, P, dow=dow, is_lunch=is_lunch)
    env.process(model.arrivals())
    env.run(until=P.sim_minutes + P.closeout_minutes)

    recs = model.recs
    served = [r for r in recs if r.status == "SERVED"]
    balked = [r for r in recs if r.status == "BALK"]
    reneged = [r for r in recs if r.status == "RENEGE"]

    rev_served = sum(r.revenue for r in served)
    rev_lost   = sum(r.revenue for r in balked) + sum(r.revenue for r in reneged)

    food_cost = P.food_cost_rate * rev_served
    labor_cost = daypart_labor_cost(P, dow=dow, is_lunch=is_lunch)
    overhead_cost = P.overhead_per_open_day / 2.0

    operating_cost = food_cost + labor_cost + overhead_cost
    profit = rev_served - operating_cost

    diners_served = sum(r.size for r in served)
    diners_left = sum(r.size for r in balked) + sum(r.size for r in reneged)
    los = (diners_left / diners_served) if diners_served > 0 else 0.0

    wait_sum = order_sum = cook_sum = pay_sum = depart_sum = 0.0
    for r in served:
        n = r.size
        wait_sum += n * float(r.seat_wait_t or 0.0)
        if r.table_t is not None and r.order_taken_t is not None:
            order_sum += n * float(r.order_taken_t - r.table_t)
        cook_sum += n * float((r.kitchen_wait_t or 0.0) + (r.prep_t or 0.0))
        if r.order_delivered_t is not None and r.payment_t is not None:
            pay_sum += n * float(r.payment_t - r.order_delivered_t)
        if r.payment_t is not None and r.depart_t is not None:
            depart_sum += n * float(r.depart_t - r.payment_t)

    denom = max(1, diners_served)
    avg_wait = wait_sum / denom
    avg_order = order_sum / denom
    avg_cook = cook_sum / denom
    avg_pay = pay_sum / denom
    avg_depart = depart_sum / denom

    rev_total = rev_served + rev_lost
    pct_rev_lost = (rev_lost / rev_total) if rev_total > 0 else 0.0

    stats = RepStats(
        diners_served=diners_served,
        diners_left=diners_left,
        left_over_served=los,
        avg_wait=avg_wait,
        avg_order=avg_order,
        avg_cook=avg_cook,
        avg_pay=avg_pay,
        avg_depart=avg_depart,
        revenue_served=rev_served,
        revenue_lost=rev_lost,
        revenue_total=rev_total,
        pct_revenue_lost=pct_rev_lost,
        operating_cost=operating_cost,
        profit=profit,
    )
    return stats, recs

# ============================================================
# COMBINED EVENT LOG (the table you showed)
# ============================================================
def print_combined_log(recs: List[GroupRec], start_hhmm: str, limit: int = 200):
    rows = sorted(recs, key=lambda x: x.gid)[:limit]

    print("\n=== COMBINED EVENT LOG (SERVED + BALK + RENEG) ===")
    print(
        "Idx\tGID\tStatus\tSize\tLeaver?\tArrival\t"
        "Table\tTable#\tOrderTaken\tDelivered\tPayment\tDepart\t"
        "SeatWait\tKitchenWait\tPrep\tLeftAt\tReason"
    )

    for i, r in enumerate(rows, 1):
        arrival = minutes_to_clock(r.arrival_t, start_hhmm)

        table = minutes_to_clock(r.table_t, start_hhmm) if r.table_t is not None else ""
        order_taken = minutes_to_clock(r.order_taken_t, start_hhmm) if r.order_taken_t is not None else ""
        delivered = minutes_to_clock(r.order_delivered_t, start_hhmm) if r.order_delivered_t is not None else ""
        payment = minutes_to_clock(r.payment_t, start_hhmm) if r.payment_t is not None else ""
        depart = minutes_to_clock(r.depart_t, start_hhmm) if r.depart_t is not None else ""

        left_at = ""
        if r.status == "BALK":
            left_at = minutes_to_clock(r.arrival_t, start_hhmm)
        elif r.status == "RENEGE":
            waited = float(r.seat_wait_t or 0.0)
            left_at = minutes_to_clock(r.arrival_t + waited, start_hhmm)

        seat_wait = f"{float(r.seat_wait_t):.1f}" if r.seat_wait_t is not None else ""
        k_wait = f"{float(r.kitchen_wait_t):.1f}" if r.kitchen_wait_t is not None else ""
        prep = f"{float(r.prep_t):.1f}" if r.prep_t is not None else ""

        leaver_flag = "Y" if r.status in {"BALK", "RENEGE"} else "N"

        print(
            f"{i}\t{r.gid}\t{r.status}\t{r.size}\t{leaver_flag}\t{arrival}\t"
            f"{table}\t{r.table_number or ''}\t{order_taken}\t{delivered}\t{payment}\t{depart}\t"
            f"{seat_wait}\t{k_wait}\t{prep}\t{left_at}\t{r.leave_reason or ''}"
        )

# ============================================================
# ANNUAL SCENARIOS (OPEN DAY FIX kept)
# ============================================================
@dataclass(frozen=True)
class DaypartConfig:
    name: str
    mean_interarrival_min: float
    is_lunch: bool

@dataclass(frozen=True)
class AnnualConfig:
    reps: int = 5
    days: int = 365
    year_start_weekday: int = MON
    base_seed: int = 2000
    lunch: DaypartConfig = DaypartConfig(name="lunch", mean_interarrival_min=4.0, is_lunch=True)
    dinner: DaypartConfig = DaypartConfig(name="dinner", mean_interarrival_min=3.3, is_lunch=False)

def day_to_month(day_index: int) -> int:
    return min(12, (day_index // 30) + 1)

def baseline_open(dow: int) -> Tuple[bool, bool]:
    dinner_open = dow in {MON, THU, FRI, SAT}
    lunch_open  = dow in {MON, THU, FRI, SAT}
    if dow == SAT:
        lunch_open = False
    return lunch_open, dinner_open

def scenario_open_rules(scenario_name: str, dow: int) -> Tuple[bool, bool]:
    lunch_open, dinner_open = baseline_open(dow)
    if scenario_name == "scenario2_more_chefs_more_day (wed+sat)":
        if dow == WED:
            lunch_open = True
            dinner_open = True
        elif dow == SAT:
            lunch_open = True
    return lunch_open, dinner_open

def scenario_apply_params(P: Params, scenario_name: str, month: int, dp: DaypartConfig, dow: int, dp_open: bool) -> Params:
    # -------------------------
    # 1) Demand by day-of-week (applies to ALL scenarios incl baseline)
    # smaller interarrival => MORE arrivals
    # -------------------------
    if dow not in {SAT, SUN}:
        P = replace(P, mean_interarrival_min=P.mean_interarrival_min * 1.15)  # 15% less demand

    if scenario_name == "baseline":
        return P

    if scenario_name == "scenario1_patio_add11":
        if month in SUMMER_MONTHS:
            patio_tables = (
                TableSpec("P1", 5, "TABLE"),
                TableSpec("P3", 4, "TABLE"),
                TableSpec("P5", 2, "TABLE"),
            )
            patio_add = sum(t.capacity for t in patio_tables)

            demand_lift = 0.85   # 15% MORE demand (smaller = more arrivals)

            return replace(
                P,
                seats_cap=P.seats_cap + patio_add,
                layout_tables=tuple(P.layout_tables) + patio_tables,

                # ✅ monetize extra seats
                mean_interarrival_min=P.mean_interarrival_min * demand_lift,

                # staffing (keep as you had for now)
                servers_weekday=P.servers_weekday + 1,
                servers_weekend=P.servers_weekend + 1,
            )
        return P


    if scenario_name == "scenario2_more_chefs_more_day (wed+sat)":
        if not dp_open:
            return P

        base_lunch_open, base_dinner_open = baseline_open(dow)
        is_new_open = (dp.is_lunch and not base_lunch_open) or ((not dp.is_lunch) and not base_dinner_open)
        if not is_new_open:
            return P

        if dp.is_lunch:
            # ✅ keep lunch staffing lean (2 helpers)
            return replace(
                P,
                mean_interarrival_min=P.mean_interarrival_min ,
                chef_helpers_lunch=2,
            )
        else:
            # ✅ add helper only for dinner if needed
            return replace(
                P,
                mean_interarrival_min=P.mean_interarrival_min ,
                chef_helpers_dinner=3,
            )


    if scenario_name == "scenario3_price_adjust":
        price_increase = 0.10
        elasticity = -0.5
        demand_multiplier = (1.0 + price_increase) ** elasticity
        new_interarrival = P.mean_interarrival_min / demand_multiplier
        new_price = P.avg_main_price * (1.0 + price_increase)
        return replace(P, avg_main_price=new_price, mean_interarrival_min=new_interarrival)

    raise ValueError(f"Unknown scenario: {scenario_name}")

def scenario_incremental_cost(P: Params, scenario_name: str, month: int, dp_open: bool) -> float:
    if not dp_open:
        return 0.0
    if scenario_name == "scenario1_patio_add11" and month in SUMMER_MONTHS:
        return P.patio_cost_per_daypart
    return 0.0

SCENARIOS = [
    "baseline",
    "scenario1_patio_add11",
    "scenario2_more_chefs_more_day (wed+sat)",
    "scenario3_price_adjust",
]

@dataclass
class AnnualRepResult:
    scenario: str
    rep_seed: int
    days: int
    open_lunch_days: int
    open_dinner_days: int
    open_dayparts: int
    open_days: int

    # ✅ daily congestion ratios
    daily_left_over_served: List[float]   # combined lunch+dinner per calendar day
    daily_los_lunch: List[float]          # lunch-only per calendar day (nan if closed)
    daily_los_dinner: List[float]         # dinner-only per calendar day (nan if closed)
    daily_los_weekday: List[float]
    daily_los_weekend: List[float]

    avg_daily_left: float
    avg_daily_served: float
    avg_wait: float
    avg_order: float
    avg_cook: float
    avg_pay: float
    avg_depart: float
    avg_cost_per_day: float
    avg_revenue_per_day: float
    avg_revenue_lost_per_day: float
    avg_revenue_total_per_day: float

    lunch_avg_daily_left: float
    lunch_avg_daily_served: float
    lunch_avg_wait: float
    lunch_avg_order: float
    lunch_avg_cook: float
    lunch_avg_pay: float
    lunch_avg_depart: float
    lunch_avg_cost_per_day: float
    lunch_avg_rev_per_day: float
    lunch_avg_rev_lost_per_day: float
    lunch_avg_rev_total_per_day: float

    dinner_avg_daily_left: float
    dinner_avg_daily_served: float
    dinner_avg_wait: float
    dinner_avg_order: float
    dinner_avg_cook: float
    dinner_avg_pay: float
    dinner_avg_depart: float
    dinner_avg_cost_per_day: float
    dinner_avg_rev_per_day: float
    dinner_avg_rev_lost_per_day: float
    dinner_avg_rev_total_per_day: float

def run_one_year_for_scenario(base_P: Params, cfg: AnnualConfig, scenario_name: str, rep_seed: int) -> AnnualRepResult:
    days = cfg.days
    daily_los: List[float] = []
    daily_los_lunch: List[float] = []   # lunch-only
    daily_los_dinner: List[float] = []  # dinner-only
    daily_los_weekday: List[float] = []
    daily_los_weekend: List[float] = []

    open_lunch_days = 0
    open_dinner_days = 0
    open_days = 0

    L_left = L_served = 0
    L_wait_sum = L_order_sum = L_cook_sum = L_pay_sum = L_depart_sum = 0.0
    L_rev_served = L_rev_lost = L_cost = 0.0

    D_left = D_served = 0
    D_wait_sum = D_order_sum = D_cook_sum = D_pay_sum = D_depart_sum = 0.0
    D_rev_served = D_rev_lost = D_cost = 0.0

    total_left = total_served = 0
    wait_sum = order_sum = cook_sum = pay_sum = depart_sum = 0.0
    total_rev_served = total_rev_lost = total_cost = 0.0

    base_lunch  = build_daypart_params(base_P, seed=1, is_lunch=True,  mean_interarrival=cfg.lunch.mean_interarrival_min)
    base_dinner = build_daypart_params(base_P, seed=1, is_lunch=False, mean_interarrival=cfg.dinner.mean_interarrival_min)

    for day in range(days):
        month = day_to_month(day)
        dow = (cfg.year_start_weekday + day) % 7

        lunch_open, dinner_open = scenario_open_rules(scenario_name, dow)

        if lunch_open:
            open_lunch_days += 1
        if dinner_open:
            open_dinner_days += 1
        if lunch_open or dinner_open:
            open_days += 1

        # ✅ per calendar day split by daypart
        l_served = l_left = 0
        d_served = d_left = 0

        if lunch_open:
            seed = rep_seed + day * 10 + 1
            P_day = replace(base_lunch, seed=seed)
            P_day = scenario_apply_params(P_day, scenario_name, month, cfg.lunch, dow, dp_open=True)
            stats, _ = run_rep_daypart(P_day, seed=P_day.seed, dow=dow, is_lunch=True)
            inc_cost = scenario_incremental_cost(P_day, scenario_name, month, dp_open=True)

            ds, dl = stats.diners_served, stats.diners_left
            l_served = ds
            l_left = dl

            L_served += ds; L_left += dl
            L_wait_sum += stats.avg_wait * ds
            L_order_sum += stats.avg_order * ds
            L_cook_sum += stats.avg_cook * ds
            L_pay_sum += stats.avg_pay * ds
            L_depart_sum += stats.avg_depart * ds
            L_rev_served += stats.revenue_served
            L_rev_lost += stats.revenue_lost
            L_cost += (stats.operating_cost + inc_cost)

            total_served += ds; total_left += dl
            wait_sum += stats.avg_wait * ds
            order_sum += stats.avg_order * ds
            cook_sum += stats.avg_cook * ds
            pay_sum += stats.avg_pay * ds
            depart_sum += stats.avg_depart * ds
            total_rev_served += stats.revenue_served
            total_rev_lost += stats.revenue_lost
            total_cost += (stats.operating_cost + inc_cost)

        if dinner_open:
            seed = rep_seed + day * 10 + 2
            P_day = replace(base_dinner, seed=seed)
            P_day = scenario_apply_params(P_day, scenario_name, month, cfg.dinner, dow, dp_open=True)
            stats, _ = run_rep_daypart(P_day, seed=P_day.seed, dow=dow, is_lunch=False)
            inc_cost = scenario_incremental_cost(P_day, scenario_name, month, dp_open=True)

            ds, dl = stats.diners_served, stats.diners_left
            d_served = ds
            d_left = dl

            D_served += ds; D_left += dl
            D_wait_sum += stats.avg_wait * ds
            D_order_sum += stats.avg_order * ds
            D_cook_sum += stats.avg_cook * ds
            D_pay_sum += stats.avg_pay * ds
            D_depart_sum += stats.avg_depart * ds
            D_rev_served += stats.revenue_served
            D_rev_lost += stats.revenue_lost
            D_cost += (stats.operating_cost + inc_cost)

            total_served += ds; total_left += dl
            wait_sum += stats.avg_wait * ds
            order_sum += stats.avg_order * ds
            cook_sum += stats.avg_cook * ds
            pay_sum += stats.avg_pay * ds
            depart_sum += stats.avg_depart * ds
            total_rev_served += stats.revenue_served
            total_rev_lost += stats.revenue_lost
            total_cost += (stats.operating_cost + inc_cost)

        # ✅ lunch-only + dinner-only LOS (nan if that daypart closed or served=0)
        daily_los_lunch.append(l_left / l_served if (lunch_open and l_served > 0) else np.nan)
        daily_los_dinner.append(d_left / d_served if (dinner_open and d_served > 0) else np.nan)

        # ✅ combined LOS for that calendar day
        day_served = l_served + d_served
        day_left = l_left + d_left
        los_day = (day_left / max(1, day_served)) if day_served > 0 else np.nan
        daily_los.append(los_day)

        if not np.isnan(los_day):
            if dow in {SAT, SUN}:
                daily_los_weekend.append(los_day)
            else:
                daily_los_weekday.append(los_day)

    open_dayparts = open_lunch_days + open_dinner_days
    if open_dayparts == 0:
        raise ValueError("open_dayparts=0. Check scenario_open_rules() and weekday mapping.")

    served_denom = max(1, total_served)
    avg_wait   = wait_sum / served_denom
    avg_order  = order_sum / served_denom
    avg_cook   = cook_sum / served_denom
    avg_pay    = pay_sum / served_denom
    avg_depart = depart_sum / served_denom

    open_days_safe = max(1, open_days)
    avg_daily_left   = total_left / open_days_safe
    avg_daily_served = total_served / open_days_safe
    avg_cost_per_day = total_cost / open_days_safe
    avg_revenue_per_day = total_rev_served / open_days_safe
    avg_revenue_lost_per_day = total_rev_lost / open_days_safe
    avg_revenue_total_per_day = (total_rev_served + total_rev_lost) / open_days_safe

    L_open = max(1, open_lunch_days)
    L_denom = max(1, L_served)
    lunch_avg_wait   = L_wait_sum / L_denom
    lunch_avg_order  = L_order_sum / L_denom
    lunch_avg_cook   = L_cook_sum / L_denom
    lunch_avg_pay    = L_pay_sum / L_denom
    lunch_avg_depart = L_depart_sum / L_denom
    lunch_avg_daily_left   = L_left / L_open
    lunch_avg_daily_served = L_served / L_open
    lunch_avg_cost_per_day = L_cost / L_open
    lunch_avg_rev_per_day  = L_rev_served / L_open
    lunch_avg_rev_lost_per_day  = L_rev_lost / L_open
    lunch_avg_rev_total_per_day = (L_rev_served + L_rev_lost) / L_open

    D_open = max(1, open_dinner_days)
    D_denom = max(1, D_served)
    dinner_avg_wait   = D_wait_sum / D_denom
    dinner_avg_order  = D_order_sum / D_denom
    dinner_avg_cook   = D_cook_sum / D_denom
    dinner_avg_pay    = D_pay_sum / D_denom
    dinner_avg_depart = D_depart_sum / D_denom
    dinner_avg_daily_left   = D_left / D_open
    dinner_avg_daily_served = D_served / D_open
    dinner_avg_cost_per_day = D_cost / D_open
    dinner_avg_rev_per_day  = D_rev_served / D_open
    dinner_avg_rev_lost_per_day  = D_rev_lost / D_open
    dinner_avg_rev_total_per_day = (D_rev_served + D_rev_lost) / D_open

    return AnnualRepResult(
        scenario=scenario_name,
        rep_seed=rep_seed,
        days=days,
        open_lunch_days=open_lunch_days,
        open_dinner_days=open_dinner_days,
        open_dayparts=open_dayparts,
        open_days=open_days,
        daily_left_over_served=daily_los,
        daily_los_weekday=daily_los_weekday,
        daily_los_weekend=daily_los_weekend,
        daily_los_lunch=daily_los_lunch,
        daily_los_dinner=daily_los_dinner,
        avg_daily_left=float(avg_daily_left),
        avg_daily_served=float(avg_daily_served),
        avg_wait=float(avg_wait),
        avg_order=float(avg_order),
        avg_cook=float(avg_cook),
        avg_pay=float(avg_pay),
        avg_depart=float(avg_depart),
        avg_cost_per_day=float(avg_cost_per_day),
        avg_revenue_per_day=float(avg_revenue_per_day),
        avg_revenue_lost_per_day=float(avg_revenue_lost_per_day),
        avg_revenue_total_per_day=float(avg_revenue_total_per_day),
        lunch_avg_daily_left=float(lunch_avg_daily_left),
        lunch_avg_daily_served=float(lunch_avg_daily_served),
        lunch_avg_wait=float(lunch_avg_wait),
        lunch_avg_order=float(lunch_avg_order),
        lunch_avg_cook=float(lunch_avg_cook),
        lunch_avg_pay=float(lunch_avg_pay),
        lunch_avg_depart=float(lunch_avg_depart),
        lunch_avg_cost_per_day=float(lunch_avg_cost_per_day),
        lunch_avg_rev_per_day=float(lunch_avg_rev_per_day),
        lunch_avg_rev_lost_per_day=float(lunch_avg_rev_lost_per_day),
        lunch_avg_rev_total_per_day=float(lunch_avg_rev_total_per_day),
        dinner_avg_daily_left=float(dinner_avg_daily_left),
        dinner_avg_daily_served=float(dinner_avg_daily_served),
        dinner_avg_wait=float(dinner_avg_wait),
        dinner_avg_order=float(dinner_avg_order),
        dinner_avg_cook=float(dinner_avg_cook),
        dinner_avg_pay=float(dinner_avg_pay),
        dinner_avg_depart=float(dinner_avg_depart),
        dinner_avg_cost_per_day=float(dinner_avg_cost_per_day),
        dinner_avg_rev_per_day=float(dinner_avg_rev_per_day),
        dinner_avg_rev_lost_per_day=float(dinner_avg_rev_lost_per_day),
        dinner_avg_rev_total_per_day=float(dinner_avg_rev_total_per_day),
    )

def run_annual_experiment_with_reps(base_P: Params, cfg: AnnualConfig) -> Dict[str, List[AnnualRepResult]]:
    rep_outputs: Dict[str, List[AnnualRepResult]] = {scen: [] for scen in SCENARIOS}
    for scen in SCENARIOS:
        for i in range(cfg.reps):
            rep_seed = cfg.base_seed + i * 100_000
            res = run_one_year_for_scenario(base_P, cfg, scen, rep_seed)
            rep_outputs[res.scenario].append(res)
    return rep_outputs

def calibrate_overhead_per_open_day(base_P: Params, cfg: AnnualConfig,
                                   target_margin: float = 0.026,
                                   calib_days: int = 30,
                                   calib_reps: int = 3) -> Params:
    cfg_cal = AnnualConfig(
        reps=calib_reps,
        days=calib_days,
        year_start_weekday=cfg.year_start_weekday,
        base_seed=cfg.base_seed,
        lunch=cfg.lunch,
        dinner=cfg.dinner,
    )

    tmp_P = replace(base_P, overhead_per_open_day=0.0)

    reps = [
        run_one_year_for_scenario(tmp_P, cfg_cal, "baseline", cfg_cal.base_seed + i * 100_000)
        for i in range(cfg_cal.reps)
    ]

    avg_rev_day = float(np.mean([r.avg_revenue_per_day for r in reps]))
    avg_food_day = tmp_P.food_cost_rate * avg_rev_day

    labor_total = 0.0
    open_days_cal = 0
    for day in range(cfg_cal.days):
        dow = (cfg_cal.year_start_weekday + day) % 7
        lunch_open, dinner_open = baseline_open(dow)
        if lunch_open or dinner_open:
            open_days_cal += 1
        if lunch_open:
            P_l = build_daypart_params(tmp_P, seed=1, is_lunch=True, mean_interarrival=cfg_cal.lunch.mean_interarrival_min)
            labor_total += daypart_labor_cost(P_l, dow, True)
        if dinner_open:
            P_d = build_daypart_params(tmp_P, seed=1, is_lunch=False, mean_interarrival=cfg_cal.dinner.mean_interarrival_min)
            labor_total += daypart_labor_cost(P_d, dow, False)

    avg_labor_day = labor_total / max(1, open_days_cal)
    target_cost_day = (1.0 - target_margin) * avg_rev_day
    overhead_day = max(0.0, target_cost_day - avg_food_day - avg_labor_day)
    return replace(base_P, overhead_per_open_day=overhead_day)

# ============================================================
# KPI PRINT (annualization uses open_days ✅)
# ============================================================
def print_scenario_kpis_short(rep_outputs: Dict[str, List[AnnualRepResult]], threshold: float = 0.33):
    def fmt_ci(xs: List[float], digits: int):
        m, lo, hi = ci95(xs)
        return f"{m:.{digits}f} ({lo:.{digits}f}–{hi:.{digits}f})"

    def fmt_ci_year(xs: List[float], digits: int = 2):
        m, lo, hi = ci95(xs)
        return f"{m:.{digits}f} ({lo:.{digits}f}–{hi:.{digits}f})"

    for scen in SCENARIOS:
        reps = rep_outputs[scen]
        print(f"\n--- {scen} ---")

        p_bad = [prob_gt(rr.daily_left_over_served, threshold) for rr in reps]
        daily_left = [rr.avg_daily_left for rr in reps]
        daily_served = [rr.avg_daily_served for rr in reps]

        wait = [rr.avg_wait for rr in reps]
        order = [rr.avg_order for rr in reps]
        cook = [rr.avg_cook for rr in reps]
        pay = [rr.avg_pay for rr in reps]
        depart = [rr.avg_depart for rr in reps]

        cost_day = [rr.avg_cost_per_day for rr in reps]
        rev_day = [rr.avg_revenue_per_day for rr in reps]

        cost_year = [rr.avg_cost_per_day * rr.open_days for rr in reps]
        rev_year  = [rr.avg_revenue_per_day * rr.open_days for rr in reps]

        cost_year_p95 = float(np.nanpercentile(cost_year, 95)) if len(cost_year) else float("nan")
        rev_year_p95  = float(np.nanpercentile(rev_year, 95)) if len(rev_year) else float("nan")

        pct_rev_lost = [
            100.0 * rr.avg_revenue_lost_per_day / max(1e-9, rr.avg_revenue_total_per_day)
            for rr in reps
        ]

        profit_year = [(rr.avg_revenue_per_day - rr.avg_cost_per_day) * rr.open_days for rr in reps]

        daily_series_list = [rr.daily_left_over_served for rr in reps]
        p95_daily_los = [float(np.nanpercentile(ds, 95)) for ds in daily_series_list]
        mean_los_by_rep = [np.nanmean(rr.daily_left_over_served) for rr in reps]

        print(f"% revenue lost (queue friction): {fmt_ci(pct_rev_lost, 2)}%")
        print(f"Risk: P(left >0.33):            {fmt_ci(p_bad, 3)}")
        print(f"Daily left/served:          {fmt_ci(mean_los_by_rep, 3)}")
        print(f"P95 daily left/served:    {fmt_ci(p95_daily_los, 3)}")
        print(f"Avg Daily left:           {fmt_ci(daily_left, 1)}")
        print(f"Avg Daily served:         {fmt_ci(daily_served, 1)}")
        print(f"Avg wait:                 {fmt_ci(wait, 2)}")
        print(f"Avg order:                {fmt_ci(order, 2)}")
        print(f"Avg cook:                 {fmt_ci(cook, 2)}")
        print(f"Avg pay:                  {fmt_ci(pay, 2)}")
        print(f"Avg depart:               {fmt_ci(depart, 2)}")
        print(f"Avg cost/day:             {fmt_ci(cost_day, 2)}")
        print(f"cost per year:            {fmt_ci_year(cost_year, 2)}")
        print(f"P95 cost per year:        {cost_year_p95:.2f}")
        print(f"Avg revenue/day:          {fmt_ci(rev_day, 2)}")
        print(f"revenue per year:         {fmt_ci_year(rev_year, 2)}")
        print(f"P95 revenue per year:     {rev_year_p95:.2f}")
        print(f"% revenue lost:           {fmt_ci(pct_rev_lost, 2)}%")
        print(f"Profit/year ($):          {fmt_ci(profit_year, 2)}")

        print(f"Open days/year (mean):    {np.mean([r.open_days for r in reps]):.1f}")

        print("--[LUNCH]--")
        print(f"Avg Daily left:   {fmt_ci([r.lunch_avg_daily_left for r in reps], 1)}")
        print(f"Avg Daily served: {fmt_ci([r.lunch_avg_daily_served for r in reps], 1)}")
        print(f"Avg wait:         {fmt_ci([r.lunch_avg_wait for r in reps], 2)}")
        print(f"Avg cook:         {fmt_ci([r.lunch_avg_cook for r in reps], 2)}")
        print(f"Avg cost/day:     {fmt_ci([r.lunch_avg_cost_per_day for r in reps], 2)}")
        print(f"Avg rev/day:      {fmt_ci([r.lunch_avg_rev_per_day for r in reps], 2)}")

        print("--[DINNER]--")
        print(f"Avg Daily left:   {fmt_ci([r.dinner_avg_daily_left for r in reps], 1)}")
        print(f"Avg Daily served: {fmt_ci([r.dinner_avg_daily_served for r in reps], 1)}")
        print(f"Avg wait:         {fmt_ci([r.dinner_avg_wait for r in reps], 2)}")
        print(f"Avg cook:         {fmt_ci([r.dinner_avg_cook for r in reps], 2)}")
        print(f"Avg cost/day:     {fmt_ci([r.dinner_avg_cost_per_day for r in reps], 2)}")
        print(f"Avg rev/day:      {fmt_ci([r.dinner_avg_rev_per_day for r in reps], 2)}")

# ============================================================
# NEW: POINT ESTIMATES (NO CI) SUMMARY + EXHIBIT TABLE
# ============================================================
def summarize_scenario_point_estimates(rep_outputs: Dict[str, List[AnnualRepResult]], threshold: float = 0.33):
    out: Dict[str, Dict[str, float]] = {}

    for scen in SCENARIOS:
        reps = rep_outputs.get(scen, [])
        if not reps:
            continue

        p_bad = [prob_gt(rr.daily_left_over_served, threshold) for rr in reps]

        mean_los_by_rep = [float(np.nanmean(rr.daily_left_over_served)) for rr in reps]
        p95_daily_los_by_rep = [float(np.nanpercentile(rr.daily_left_over_served, 95)) for rr in reps]

        rev_year = [rr.avg_revenue_per_day * rr.open_days for rr in reps]
        cost_year = [rr.avg_cost_per_day * rr.open_days for rr in reps]
        profit_year = [(rr.avg_revenue_per_day - rr.avg_cost_per_day) * rr.open_days for rr in reps]

        pct_rev_lost = [
            100.0 * rr.avg_revenue_lost_per_day / max(1e-9, rr.avg_revenue_total_per_day)
            for rr in reps
        ]

        out[scen] = {
            "open_days": float(np.mean([r.open_days for r in reps])),

            "avg_rev_day": float(np.mean([r.avg_revenue_per_day for r in reps])),
            "avg_cost_day": float(np.mean([r.avg_cost_per_day for r in reps])),

            "avg_served_day": float(np.mean([r.avg_daily_served for r in reps])),
            "avg_left_day": float(np.mean([r.avg_daily_left for r in reps])),

            "daily_los": float(np.mean(mean_los_by_rep)),
            "p95_daily_los": float(np.mean(p95_daily_los_by_rep)),

            "risk_p_gt_thr": float(np.mean(p_bad)),
            "pct_rev_lost": float(np.mean(pct_rev_lost)),

            "avg_wait": float(np.mean([r.avg_wait for r in reps])),
            "avg_order": float(np.mean([r.avg_order for r in reps])),
            "avg_cook": float(np.mean([r.avg_cook for r in reps])),
            "avg_pay": float(np.mean([r.avg_pay for r in reps])),
            "avg_depart": float(np.mean([r.avg_depart for r in reps])),

            "rev_year": float(np.mean(rev_year)),
            "cost_year": float(np.mean(cost_year)),
            "profit_year": float(np.mean(profit_year)),
        }

    return out


def print_scenario_block_point_estimates(name: str, s: Dict[str, float], threshold: float = 0.33):
    print(f"\n--- {name} (NO CI; point estimates over reps) ---")
    print(f"% revenue lost (queue friction): {s['pct_rev_lost']:.2f}%")
    print(f"Risk: P(left/served > {threshold:.2f}):       {s['risk_p_gt_thr']:.3f}")
    print(f"Daily left/served:               {s['daily_los']:.3f}")
    print(f"P95 daily left/served:           {s['p95_daily_los']:.3f}")
    print(f"Avg Daily left:                  {s['avg_left_day']:.1f}")
    print(f"Avg Daily served:                {s['avg_served_day']:.1f}")
    print(f"Avg wait:                        {s['avg_wait']:.2f}")
    print(f"Avg order:                       {s['avg_order']:.2f}")
    print(f"Avg cook:                        {s['avg_cook']:.2f}")
    print(f"Avg pay:                         {s['avg_pay']:.2f}")
    print(f"Avg depart:                      {s['avg_depart']:.2f}")
    print(f"Avg cost/day:                    {s['avg_cost_day']:.2f}")
    print(f"Avg revenue/day:                 {s['avg_rev_day']:.2f}")
    print(f"Profit/year ($):                 {s['profit_year']:.2f}")
    print(f"Open days/year (mean):           {s['open_days']:.1f}")


def print_exhibit_table_no_ci(summary: Dict[str, Dict[str, float]]):
    scen_order = [s for s in SCENARIOS if s in summary]

    print("\n================= EXHIBIT: SIMULATION RESULTS (NO CI) =================")
    header = ["Section", "Key metric"] + scen_order
    print("\t".join(header))

    rows = [
        ("Context", "Open days / year", "open_days", "{:.1f}"),
        ("Context", "Avg revenue / day", "avg_rev_day", "{:,.2f}"),
        ("Context", "Avg cost / day", "avg_cost_day", "{:,.2f}"),

        ("Customer outcomes", "Avg daily served", "avg_served_day", "{:.1f}"),
        ("Customer outcomes", "Avg daily lost (balk+renege)", "avg_left_day", "{:.1f}"),
        ("Customer outcomes", "Daily left/served ratio", "daily_los", "{:.3f}"),
        ("Customer outcomes", "P95 daily left/served", "p95_daily_los", "{:.3f}"),

        ("Congestion risk", "P(left/served > 0.33)", "risk_p_gt_thr", "{:.3f}"),
        ("Congestion risk", "% revenue lost (queue friction)", "pct_rev_lost", "{:.2f}%"),

        ("Time by stage (min)", "Wait (line/seat)", "avg_wait", "{:.2f}"),
        ("Time by stage (min)", "Order", "avg_order", "{:.2f}"),
        ("Time by stage (min)", "Cook", "avg_cook", "{:.2f}"),
        ("Time by stage (min)", "Pay", "avg_pay", "{:.2f}"),
        ("Time by stage (min)", "Depart", "avg_depart", "{:.2f}"),

        ("Financial outcomes (annual)", "Revenue / year", "rev_year", "{:,.2f}"),
        ("Financial outcomes (annual)", "Cost / year", "cost_year", "{:,.2f}"),
        ("Financial outcomes (annual)", "Profit / year ($)", "profit_year", "{:,.2f}"),
    ]

    for section, metric, key, fmt in rows:
        line = [section, metric]
        for scen in scen_order:
            line.append(fmt.format(summary[scen][key]))
        print("\t".join(line))

# ============================================================
# PLOTS (annual totals use open_days ✅)
# ============================================================
def prob_gt_daily(daily_series_list, thr=0.33):
    out = []
    for ds in daily_series_list:
        a = np.asarray(ds, dtype=float)
        a = a[~np.isnan(a)]
        out.append(float(np.mean(a > thr)) if len(a) else np.nan)
    return out

def scenario_metrics(rep_outputs, threshold=0.33):
    scen_names = list(rep_outputs.keys())
    M = {}
    for scen in scen_names:
        reps = rep_outputs[scen]

        daily_los_list = [r.daily_left_over_served for r in reps]
        p_bad = prob_gt_daily(daily_los_list, thr=threshold)

        avg_rev = [r.avg_revenue_per_day for r in reps]
        avg_cost = [r.avg_cost_per_day for r in reps]
        avg_profit = [rv - c for rv, c in zip(avg_rev, avg_cost)]

        M[scen] = dict(
            p_bad=p_bad,
            rev=avg_rev,
            cost=avg_cost,
            profit=avg_profit,
            dinner_wait=[r.dinner_avg_wait for r in reps],
            dinner_cook=[r.dinner_avg_cook for r in reps],
            dinner_order=[r.dinner_avg_order for r in reps],
            dinner_pay=[r.dinner_avg_pay for r in reps],
            dinner_depart=[r.dinner_avg_depart for r in reps],
            daily_los_list=daily_los_list,
        )
    return scen_names, M


def plot_ci_bars(title, scen_names, values_by_scen, ylabel, fmt=None):
    means, lows, highs = [], [], []
    for s in scen_names:
        m, lo, hi = ci95(values_by_scen[s])
        means.append(m); lows.append(lo); highs.append(hi)

    means = np.array(means)
    yerr = np.vstack([means - np.array(lows), np.array(highs) - means])

    x = np.arange(len(scen_names))
    bars = plt.bar(x, means)
    plt.errorbar(x, means, yerr=yerr, fmt="none", capsize=4)

    # ---- LABELS ON BARS ----
    offset = 0.02 * np.max(means)
    for i, m in enumerate(means):
        label = fmt.format(m) if fmt else f"{m:.2f}"
        plt.text(x[i], m + offset, label, ha="center", va="bottom", fontsize=9)

    plt.xticks(x, scen_names, rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def split_lunch_dinner_weekday_weekend(rep_outputs, cfg, threshold):
    out = {}

    for scen, reps in rep_outputs.items():
        out[scen] = {
            "lunch_weekday": [],
            "lunch_weekend": [],
            "dinner_weekday": [],
            "dinner_weekend": [],
        }

        for r in reps:
            # ---- LUNCH: use lunch-only daily series ----
            for day_idx, los in enumerate(r.daily_los_lunch):
                if np.isnan(los):
                    continue
                dow = (cfg.year_start_weekday + day_idx) % 7
                weekend = dow in {SAT, SUN}
                key = "lunch_weekend" if weekend else "lunch_weekday"
                out[scen][key].append(los > threshold)

            # ---- DINNER: use dinner-only daily series ----
            for day_idx, los in enumerate(r.daily_los_dinner):
                if np.isnan(los):
                    continue
                dow = (cfg.year_start_weekday + day_idx) % 7
                weekend = dow in {SAT, SUN}
                key = "dinner_weekend" if weekend else "dinner_weekday"
                out[scen][key].append(los > threshold)

        for k in out[scen]:
            vals = np.asarray(out[scen][k], dtype=float)
            out[scen][k] = vals if len(vals) else np.array([np.nan])

    return out


def plot_risk_small_multiples(segmented, scen_names, threshold):
    panels = [
        ("Lunch – Weekday", "lunch_weekday"),
        ("Lunch – Weekend", "lunch_weekend"),
        ("Dinner – Weekday", "dinner_weekday"),
        ("Dinner – Weekend", "dinner_weekend"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    axes = axes.flatten()

    for ax, (title, key) in zip(axes, panels):
        means, lows, highs = [], [], []

        for s in scen_names:
            m, lo, hi = ci95(segmented[s][key])
            means.append(m)
            lows.append(lo)
            highs.append(hi)

        means = np.array(means)
        yerr = np.vstack([means - np.array(lows), np.array(highs) - means])
        x = np.arange(len(scen_names))

        bars = ax.bar(x, means, yerr=yerr, capsize=4)

        # labels on bars
        for i, m in enumerate(means):
            ax.text(i, m + 0.01, f"{m:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(scen_names, rotation=20, ha="right")
        ax.axhline(threshold, linestyle="--", linewidth=1)
        ax.set_ylim(0, max(0.15, np.nanmax(means) * 1.3))

    fig.supylabel(f"P(daily left/served > {threshold})")
    fig.suptitle("Congestion Risk by Daypart and Day Type (95% CI)", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_los_box_by_scenario_weekday_weekend(rep_outputs, scen_names, threshold=0.33):
    import numpy as np
    import matplotlib.pyplot as plt

    # Flatten per scenario
    wk_box_data = []
    we_box_data = []

    for s in scen_names:
        wk_flat = []
        we_flat = []

        for rr in rep_outputs[s]:
            wk_flat.extend([x for x in rr.daily_los_weekday if not np.isnan(x)])
            we_flat.extend([x for x in rr.daily_los_weekend if not np.isnan(x)])

        wk_box_data.append(wk_flat)
        we_box_data.append(we_flat)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # --- Weekday ---
    axes[0].boxplot(wk_box_data, labels=scen_names, showfliers=False)
    axes[0].axhline(threshold, linestyle="--", linewidth=1.5, label=f"Threshold = {threshold:.2f}")
    axes[0].set_title("Weekday")
    axes[0].set_ylabel("Daily left/served")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(loc="upper right")

    # --- Weekend ---
    axes[1].boxplot(we_box_data, labels=scen_names, showfliers=False)
    axes[1].axhline(threshold, linestyle="--", linewidth=1.5, label=f"Threshold = {threshold:.2f}")
    axes[1].set_title("Weekend")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(loc="upper right")

    fig.suptitle("Distribution of daily left/served by scenario\nSplit: weekday vs weekend", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_stage_times_by_stage(
    title,
    scen_names,
    stage_by_scen,   # dict: stage -> {scenario -> list of values}
):
    stages = list(stage_by_scen.keys())
    n_stage = len(stages)
    n_scen = len(scen_names)

    x = np.arange(n_stage)
    width = 0.8 / n_scen

    plt.figure(figsize=(10, 5))

    for i, scen in enumerate(scen_names):
        means = []
        lows = []
        highs = []

        for stage in stages:
            m, lo, hi = ci95(stage_by_scen[stage][scen])
            means.append(m)
            lows.append(lo)
            highs.append(hi)

        means = np.array(means)
        yerr = np.vstack([means - np.array(lows), np.array(highs) - means])

        plt.bar(
            x + (i - n_scen/2) * width + width/2,
            means,
            width,
            yerr=yerr,
            capsize=3,
            label=scen
        )

    plt.xticks(x, stages)
    plt.ylabel("Minutes per served diner")
    plt.title(title)
    plt.legend(title="Scenario")
    plt.tight_layout()
    plt.show()




# ============================================================
# MAIN (✅ includes SINGLE DAY + COMBINED EVENT LOG)
# ============================================================
if __name__ == "__main__":
    P = Params()
    
    # 1) One dinner rep log + combined event log
    P_log = replace(P, seed=999, validation_log=True, log_clock_start_hhmm=P.dinner_open_hhmm)
    stats, recs = run_rep_daypart(P_log, seed=P_log.seed, dow=SAT, is_lunch=False)
    print_combined_log(recs, start_hhmm=P_log.log_clock_start_hhmm, limit=200)

    # 2) Single-day dinner (Saturday) - many reps, 17:00–21:30
    P_dinner = replace(
        P,
        dinner_open_hhmm="17:00",
        dinner_close_hhmm="21:30",
        log_clock_start_hhmm="17:00",   # <-- IMPORTANT for clock printing
    )

    P_single = build_daypart_params(P_dinner, seed=42, is_lunch=False, mean_interarrival=3.2)

    print("DEBUG dinner_open:", P_single.dinner_open_hhmm)
    print("DEBUG dinner_close:", P_single.dinner_close_hhmm)
    print("DEBUG sim_minutes:", P_single.sim_minutes)   # should be 270.0
    print("DEBUG log start:", P_single.log_clock_start_hhmm)

    N_REP = 100
    stats_list = []
    for i in range(N_REP):
        seed_i = 2000 + i
        P_i = replace(P_single, seed=seed_i)
        st, _ = run_rep_daypart(P_i, seed=seed_i, dow=SAT, is_lunch=False)
        stats_list.append(st)

    print("\n=== SINGLE DAY (Dinner, Saturday; 100 reps, 17:00–21:30) ===")
    print("sim_minutes:", P_single.sim_minutes)
    print("Served (mean):", np.mean([s.diners_served for s in stats_list]))
    print("Left (mean):  ", np.mean([s.diners_left for s in stats_list]))
    print("LOS (mean):   ", np.mean([s.left_over_served for s in stats_list]))
    print("Wait (mean):  ", np.mean([s.avg_wait for s in stats_list]))
    print(f"Avg order:     {np.mean([s.avg_order for s in stats_list]):.2f} min")
    print(f"Avg cook:      {np.mean([s.avg_cook for s in stats_list]):.2f} min")
    print(f"Avg pay:       {np.mean([s.avg_pay for s in stats_list]):.2f} min")
    print(f"Avg depart:    {np.mean([s.avg_depart for s in stats_list]):.2f} min")


    # --- build "new" directly from code output (stats_list) ---
    new = {
        "Left Over Served":    float(np.mean([s.left_over_served for s in stats_list])),
        "Wait":   float(np.mean([s.avg_wait for s in stats_list])),
        "Order":  float(np.mean([s.avg_order for s in stats_list])),
        "Cook":   float(np.mean([s.avg_cook for s in stats_list])),
        "Pay":    float(np.mean([s.avg_pay for s in stats_list])),
        "Depart": float(np.mean([s.avg_depart for s in stats_list])),
    }

    # --- historical (keep hardcoded unless you also compute it from Excel in this script) ---
    hist = {
        "Left Over Served": 0.289,
        "Wait": 15.478,
        "Order": 3.396,
        "Cook": 4.421,
        "Pay": 36.233,
        "Depart": 1.434,
    }


    # --- plot: Historical vs New (+ change vs historical labels) ---
    metrics = ["Left Over Served", "Wait", "Order", "Cook", "Pay", "Depart"]
    hist_vals = np.array([hist[m] for m in metrics], dtype=float)
    new_vals  = np.array([new[m]  for m in metrics], dtype=float)

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, hist_vals, width, label="Historical", color="grey")
    plt.bar(x + width/2, new_vals,  width, label="New single-day (mean of reps)", color="green")

    for i, (h, n) in enumerate(zip(hist_vals, new_vals)):
        if abs(h) > 1e-12:
            pct = 100.0 * (n - h) / h
            label = f"{pct:+.1f}%"
        else:
            label = f"{n - h:+.2f}"

        y = max(h, n)
        plt.text(i, y, label, ha="center", va="bottom", fontsize=9)

    plt.xticks(x, metrics)
    plt.ylabel("Value")
    plt.title("Saturday Dinner: Historical vs Single-Day Simulation\n(+ change vs historical)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # ANNUAL (OPEN DAY metrics)
    # ----------------------------
    threshold = 0.33
    cfg = AnnualConfig(reps=10, days=365, year_start_weekday=MON, base_seed=2000)

    P = calibrate_overhead_per_open_day(P, cfg, target_margin=0.026)
    rep_outputs = run_annual_experiment_with_reps(P, cfg)

    print_scenario_kpis_short(rep_outputs, threshold=threshold)

    summary = summarize_scenario_point_estimates(rep_outputs, threshold=threshold)
    for scen in SCENARIOS:
        if scen in summary:
            print_scenario_block_point_estimates(scen, summary[scen], threshold=threshold)
    print_exhibit_table_no_ci(summary)

    # ---------- plots ----------
    scen_names, M = scenario_metrics(rep_outputs, threshold=threshold)

    segmented = split_lunch_dinner_weekday_weekend(rep_outputs, cfg, threshold=threshold)
    plot_risk_small_multiples(segmented, scen_names, threshold=threshold)

    plot_los_box_by_scenario_weekday_weekend(rep_outputs, scen_names, threshold=threshold)

    plot_stage_times_by_stage(
        title="Dinner service times by stage (95% CI)",
        scen_names=scen_names,
        stage_by_scen={
            "Wait":   {s: M[s]["dinner_wait"]   for s in scen_names},
            "Order":  {s: M[s]["dinner_order"]  for s in scen_names},
            "Cook":   {s: M[s]["dinner_cook"]   for s in scen_names},
            "Pay":    {s: M[s]["dinner_pay"]    for s in scen_names},
            "Depart": {s: M[s]["dinner_depart"] for s in scen_names},
        }
    )
    plot_ci_bars(
        title="Annual Profit (95% CI over reps)",
        scen_names=scen_names,
        values_by_scen={
            s: [(r.avg_revenue_per_day - r.avg_cost_per_day) * r.open_days for r in rep_outputs[s]]
            for s in scen_names
        },
        ylabel="Annual profit ($)"
    )

    plot_ci_bars(
        title="Annual Revenue (95% CI over reps)",
        scen_names=scen_names,
        values_by_scen={
            s: [r.avg_revenue_per_day * r.open_days for r in rep_outputs[s]]
            for s in scen_names
        },
        ylabel="Annual revenue ($)"
    )

    plot_ci_bars(
        title="Annual Cost (95% CI over reps)",
        scen_names=scen_names,
        values_by_scen={
            s: [r.avg_cost_per_day * r.open_days for r in rep_outputs[s]]
            for s in scen_names
        },
        ylabel="Annual cost ($)"
    )


    # Profit vs Risk scatter (profit per day)
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in scen_names:
        x = np.asarray(M[s]["p_bad"], dtype=float)
        y = np.asarray(M[s]["profit"], dtype=float)
        ax.scatter(x, y, label=s, alpha=0.8)

    ax.set_xlabel(f"Risk: P(daily left/served > {threshold})")
    ax.set_ylabel("Profit / open day")
    ax.set_title("Profit vs Risk (each dot = one rep)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.subplots_adjust(right=0.75)
    plt.show()


# ============================================================
# PLOT: % revenue lost from queue friction by scenario
# (% lost = revenue_lost_per_day / revenue_total_per_day)
# ============================================================
pct_loss_by_scen = {
    s: [
        100.0 * r.avg_revenue_lost_per_day / max(1e-9, r.avg_revenue_total_per_day)
        for r in rep_outputs[s]
    ]
    for s in scen_names
}

plot_ci_bars(
    title="% Revenue Lost from Queue Friction (95% CI over reps)",
    scen_names=scen_names,
    values_by_scen=pct_loss_by_scen,
    ylabel="% revenue lost",
    fmt="{:.2f}%"
)


# ============================================================
# PLOT: Bottleneck Effectiveness Matrix (pulled from rep_outputs)
# ============================================================

# --- build scenario table from AnnualRepResult ---
rows = []
for s in scen_names:
    reps = rep_outputs[s]

    open_days_mean = float(np.mean([r.open_days for r in reps]))

    avg_daily_lost   = float(np.mean([r.avg_daily_left for r in reps]))
    avg_daily_served = float(np.mean([r.avg_daily_served for r in reps]))
    avg_wait_min     = float(np.mean([r.avg_wait for r in reps]))  # <-- AnnualRepResult uses avg_wait

    rev_lost_pct = float(np.mean([
        100.0 * r.avg_revenue_lost_per_day / max(1e-9, r.avg_revenue_total_per_day)
        for r in reps
    ]))

    avg_rev_day  = float(np.mean([r.avg_revenue_per_day for r in reps]))
    avg_cost_day = float(np.mean([r.avg_cost_per_day for r in reps]))

    revenue_per_year = avg_rev_day * open_days_mean
    profit_per_year  = (avg_rev_day - avg_cost_day) * open_days_mean

    rows.append({
        "scenario": s,
        "open_days": open_days_mean,
        "avg_daily_lost": avg_daily_lost,
        "avg_daily_served": avg_daily_served,
        "avg_wait_min": avg_wait_min,
        "rev_lost_pct": rev_lost_pct,
        "revenue_per_year": revenue_per_year,
        "profit_per_year": profit_per_year,
    })

df = pd.DataFrame(rows).set_index("scenario")

# --- baseline row ---
if "baseline" not in df.index:
    raise ValueError(f"'baseline' not found in df.index. Available: {list(df.index)}")

base = df.loc["baseline"]


df["loss_reduction_pct"] = (base["avg_daily_lost"] - df["avg_daily_lost"]) / base["avg_daily_lost"] * 100
df["rev_lost_reduction_pct"] = (base["rev_lost_pct"] - df["rev_lost_pct"]) / base["rev_lost_pct"] * 100
df["wait_reduction_min"] = base["avg_wait_min"] - df["avg_wait_min"]
df["served_change_pct"] = (df["avg_daily_served"] - base["avg_daily_served"]) / base["avg_daily_served"] * 100
df["profit_change_pct"] = (df["profit_per_year"] - base["profit_per_year"]) / base["profit_per_year"] * 100

# only scenarios (exclude baseline) get scored
scenarios = df.drop(index="baseline").copy()


def minmax_0_100(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    if np.isclose(mx - mn, 0):
        return pd.Series(50.0, index=s.index)
    return (s - mn) / (mx - mn) * 100

loss_red_score = minmax_0_100(scenarios["loss_reduction_pct"])
rev_lost_red_score = minmax_0_100(scenarios["rev_lost_reduction_pct"])
wait_red_score = minmax_0_100(scenarios["wait_reduction_min"])
served_up_score = minmax_0_100(scenarios["served_change_pct"])
revenue_extract_score = minmax_0_100(scenarios["profit_change_pct"])

# weighted composite: congestion relief
scenarios["front_end_congestion_relief"] = (
    0.40 * loss_red_score +
    0.35 * rev_lost_red_score +
    0.25 * wait_red_score
)

scenarios["throughput_expansion"] = served_up_score
scenarios["revenue_extraction"] = revenue_extract_score


x = scenarios["throughput_expansion"]
y = scenarios["front_end_congestion_relief"]
size = scenarios["revenue_extraction"]

fig, ax = plt.subplots(figsize=(11, 7))

ax.scatter(x, y, s=(size + 10) * 20, alpha=0.6)

# baseline anchor at (0,0) in "score space"
ax.scatter(0, 0, marker="X", s=160, linewidths=2, label="Baseline")
ax.text(2, 2, "Baseline", fontsize=10, fontweight="bold")

for name in scenarios.index:
    ax.text(x.loc[name] + 1.2, y.loc[name] + 1.2, name.replace("scenario", "S"), fontsize=10)

ax.set_title("Bottleneck Effectiveness Matrix (Scores vs Baseline)")
ax.set_xlabel("Throughput Expansion Score (0–100)")
ax.set_ylabel("Front-End Congestion Relief Score (0–100)")

ax.axvline(50, linestyle="--", linewidth=1)
ax.axhline(50, linestyle="--", linewidth=1)

# bubble-size legend (move legend wherever you want)
for val in [0, 50, 100]:
    ax.scatter([], [], s=(val + 10) * 20, alpha=0.6, label=f"Revenue extraction score ~ {val}")

ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, title="Bubble size")
ax.set_xlim(-5, 105)
ax.set_ylim(-5, 105)
ax.grid(True, linestyle=":", linewidth=0.8)

plt.tight_layout()
plt.show()


print("\n=== Bottleneck Effectiveness Metrics (vs baseline) ===")
print(
    scenarios[[
        "loss_reduction_pct",
        "rev_lost_reduction_pct",
        "wait_reduction_min",
        "served_change_pct",
        "profit_change_pct",
        "front_end_congestion_relief",
        "throughput_expansion",
        "revenue_extraction",
    ]].round(2)
)
