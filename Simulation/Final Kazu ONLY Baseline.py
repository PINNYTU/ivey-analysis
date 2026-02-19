#!/usr/bin/env python
# coding: utf-8
"""
Restaurant Kazu – BASELINE ONLY (SimPy) + Layout (T1–T6 + BAR)
"""


from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

import random
import simpy
import numpy as np
import math
import matplotlib.pyplot as plt


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
    # set dynamically per daypart by build_daypart_params()
    sim_minutes: float = 240
    closeout_minutes: float = 180.0

    # hours (customer cutoff = close time)
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

    # leaving model
    target_left_over_served: float = 1.0 / 3.0
    patience_leave_mean_min: float = 8.0
    patience_leave_cap_min: float = 16.0

    balk_if_queue_demand_ge: int = 14
    balk_prob_if_threshold: float = 0.05

    leave_share_multiplier: float = 0.9
    prob_chef_required: float = 0.3

    # stage-time controls
    tri_order_take_min_mode_max: Tri3 = (1.0, 3.0, 6.0)
    tri_prep_helper_min_mode_max: Tri3 = (1.0, 3.5, 6.5)
    tri_prep_chef_min_mode_max: Tri3 = (2.0, 4.0, 7.0)
    tri_to_payment_min_mode_max: Tri3 = (22.0, 32.0, 50.0)
    tri_payment_to_depart_min_mode_max: Tri3 = (0.0, 1.0, 3.0)
    tri_host_delay_min_mode_max: Tri3 = (0.0, 2.5, 4.0)

    # revenue model
    avg_main_price: float = 20.0
    prob_appetizer: float = 0.45
    avg_appetizer_price: float = 8.0
    prob_dressing: float = 0.08
    dressing_price: float = 6.0

    # costing
    food_cost_rate: float = 0.20
    wage_server_per_hr: float = 18.0
    wage_helper_per_hr: float = 17.0
    wage_chef_per_hr: float = 25.0

    overhead_per_open_day: float = 900.0
    closeout_labor_factor: float = 0.50

    # logging
    seed: int = 42
    validation_log: bool = False
    validation_log_limit: int = 60
    log_clock_start_hhmm: str = "15:00"


# ============================================================
# RECORDS
# ============================================================
@dataclass
class GroupRec:
    gid: int
    size: int
    arrival_t: float
    status: str  # SERVED/BALK/RENEGE
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


def tri_time(tri: Tri3) -> float:
    lo, mode, hi = tri
    lo = max(0.01, float(lo))
    hi = max(lo, float(hi))
    mode = max(lo, min(float(mode), hi))
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
    xs = xs[np.isfinite(xs)]
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    m = float(xs.mean())
    if n == 1:
        return (m, m, m)
    s = float(xs.std(ddof=1))
    half = 1.96 * s / math.sqrt(n)
    return (m, m - half, m + half)


def hhmm_to_minutes(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)


def minutes_between(start_hhmm: str, end_hhmm: str) -> int:
    a = hhmm_to_minutes(start_hhmm)
    b = hhmm_to_minutes(end_hhmm)
    if b < a:
        b += 24 * 60
    return b - a


def minutes_to_clock(t_min: float, start_hhmm: str) -> str:
    hh0, mm0 = map(int, start_hhmm.split(":"))
    total = hh0 * 60 + mm0 + int(round(t_min))
    hh = (total // 60) % 24
    mm = total % 60
    return f"{hh:02d}:{mm:02d}"


def build_daypart_params(base_P: Params, *, seed: int, is_lunch: bool, mean_interarrival: float) -> Params:
    open_hhmm = base_P.lunch_open_hhmm if is_lunch else base_P.dinner_open_hhmm
    close_hhmm = base_P.lunch_close_hhmm if is_lunch else base_P.dinner_close_hhmm
    sim_minutes = float(minutes_between(open_hhmm, close_hhmm))

    return replace(
        base_P,
        seed=seed,
        sim_minutes=sim_minutes,
        mean_interarrival_min=mean_interarrival,
        log_clock_start_hhmm=open_hhmm,
    )


# ============================================================
# STAFFING + OPERATING DAYS
# ============================================================
MON, TUE, WED, THU, FRI, SAT, SUN = 0, 1, 2, 3, 4, 5, 6


def is_weekend(dow: int) -> bool:
    return dow in {SAT, SUN}


def servers_for_day(P: Params, dow: int) -> int:
    return int(P.servers_weekend if is_weekend(dow) else P.servers_weekday)


def helpers_for_daypart(P: Params, is_lunch: bool) -> int:
    return int(P.chef_helpers_lunch if is_lunch else P.chef_helpers_dinner)


def baseline_open(dow: int) -> Tuple[bool, bool]:
    """
    Operating Days: Monday, Thursday, Friday, Saturday
    Saturday lunch: CLOSED
    """
    open_days = {MON, THU, FRI, SAT}
    dinner_open = dow in open_days
    lunch_open = dow in open_days and dow != SAT
    return lunch_open, dinner_open


# ============================================================
# LAYOUT
# ============================================================
class LayoutManager:
    def __init__(self, env: simpy.Environment, P: Params):
        self.env = env
        self.P = P

        self.tables = list(P.layout_tables)
        table_sum = sum(t.capacity for t in self.tables)

        bar_target = P.bar_base_seats + max(0, int(P.bar_extra_seats))
        bar_capacity = min(bar_target, max(0, int(P.seats_cap) - table_sum))
        self.bar = TableSpec("BAR", int(bar_capacity), "BAR")

        self.free_tables: Dict[str, bool] = {t.name: True for t in self.tables}
        self.table_borrowed: Dict[str, int] = {t.name: 0 for t in self.tables}

        # Borrow rules: table -> {group_size: seats_to_add_from_bar}
        self.borrow_rules: Dict[str, Dict[int, int]] = {
            "T6": {5: 1},
            "T5": {3: 1, 4: 2},
        }
        self.bar_free = self.bar.capacity

        # queue items: (evt, group_size, allow_bar, allowed_tables or None)
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
            self.bar_free = min(self.bar.capacity, self.bar_free + borrowed)
            self.table_borrowed[table_name] = 0

        self.free_tables[table_name] = True
        self._try_allocate()

    def release_bar(self, seats: int):
        self.bar_free = min(self.bar.capacity, self.bar_free + seats)
        self._try_allocate()

    def _try_allocate(self):
        def borrow_needed(table_name: str, cap: int, gs: int) -> int:
            if gs <= cap:
                return 0
            need = self.borrow_rules.get(table_name, {}).get(gs, 0)
            return need if (need > 0 and gs <= cap + need) else 0

        def can_fit(table_name: str, cap: int, gs: int) -> bool:
            if gs <= cap:
                return True
            need = borrow_needed(table_name, cap, gs)
            return need > 0 and self.bar_free >= need

        made_progress = True
        while made_progress:
            made_progress = False

            free_table_specs = sorted(
                [t for t in self.tables if self.free_tables.get(t.name, False)],
                key=lambda t: (t.capacity, t.name),
            )

            # TABLE assignment
            for t in free_table_specs:
                best_idx = None
                best_score = None  # (waste, borrow, -gs)

                for i, (evt, gs, _allow_bar, allowed_tables) in enumerate(self.queue):
                    if evt.triggered:
                        continue
                    if allowed_tables is not None and t.name not in allowed_tables:
                        continue
                    if not can_fit(t.name, t.capacity, gs):
                        continue

                    need = borrow_needed(t.name, t.capacity, gs)
                    waste = (t.capacity + need) - gs
                    score = (waste, need, -gs)

                    if best_score is None or score < best_score:
                        best_score = score
                        best_idx = i

                if best_idx is not None:
                    evt, gs, allow_bar, allowed_tables = self.queue[best_idx]
                    del self.queue[best_idx]

                    self.free_tables[t.name] = False

                    need = borrow_needed(t.name, t.capacity, gs)
                    if need > 0:
                        self.bar_free = max(0, self.bar_free - need)
                        self.table_borrowed[t.name] = need

                    evt.succeed(value=("TABLE", t.name))
                    made_progress = True

            # BAR assignment
            while self.bar_free > 0:
                best_idx = None
                best_score = None  # (waste, -gs)

                for i, (evt, gs, allow_bar, allowed_tables) in enumerate(self.queue):
                    if evt.triggered:
                        continue
                    if not allow_bar:
                        continue
                    if allowed_tables is not None:
                        continue
                    if gs > self.bar_free:
                        continue

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
    chefs = int(P.chef_count)

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
        return clamp(base * self.P.leave_share_multiplier, 0.0, 0.95)

    def sample_is_leaver(self) -> bool:
        return random.random() < self.leave_fraction_total()

    def patience_time(self, is_leaver: bool) -> float:
        if is_leaver:
            return min(exp_time(self.P.patience_leave_mean_min), self.P.patience_leave_cap_min)
        return self.P.sim_minutes + self.P.closeout_minutes + 1.0

    def group(self, gid: int):
        size = weighted_choice(self.P.group_size_probs)
        t_arr = self.env.now

        host_delay = tri_time(self.P.tri_host_delay_min_mode_max)
        if host_delay > 0:
            yield self.env.timeout(host_delay)

        rev = self.group_revenue(size)
        sampled_leaver = self.sample_is_leaver()

        # balk if line is long
        if self.layout.queue_demand() >= self.P.balk_if_queue_demand_ge and random.random() < self.P.balk_prob_if_threshold:
            self.recs.append(GroupRec(
                gid=gid, size=size, arrival_t=t_arr,
                status="BALK", leave_reason="BALK_LINE_TOO_LONG",
                revenue=rev, seat_wait_t=0.0, leaver_type=True, patience_draw=0.0,
            ))
            return

        # seating policy
        allowed_tables = None
        if size == 5:
            allow_bar = False
            allowed_tables = {"T6"}
        else:
            allow_bar = True

        patience = self.patience_time(sampled_leaver)
        seat_evt = self.layout.request(size, allow_bar, allowed_tables=allowed_tables)

        # closing rule (customer cutoff): if not seated by close, leave as CLOSE
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

        # order taking (server)
        with self.servers.request() as sreq:
            yield sreq
            yield self.env.timeout(tri_time(self.P.tri_order_take_min_mode_max))
        t_order = self.env.now

        # kitchen
        kitchen_req_t = self.env.now
        needs_chef = (random.random() < self.P.prob_chef_required)

        if needs_chef:
            with self.chef.request() as creq, self.helpers.request() as hreq:
                yield creq
                yield hreq
                k_start = self.env.now
                k_wait = k_start - kitchen_req_t
                prep = tri_time(self.P.tri_prep_chef_min_mode_max)
                yield self.env.timeout(prep)
        else:
            with self.helpers.request() as hreq:
                yield hreq
                k_start = self.env.now
                k_wait = k_start - kitchen_req_t
                prep = tri_time(self.P.tri_prep_helper_min_mode_max)
                yield self.env.timeout(prep)

        t_delivered = self.env.now

        # pay
        yield self.env.timeout(tri_time(self.P.tri_to_payment_min_mode_max))
        t_payment = self.env.now

        # depart (server)
        with self.servers.request() as sreq:
            yield sreq
            yield self.env.timeout(tri_time(self.P.tri_payment_to_depart_min_mode_max))
        t_dep = self.env.now

        # release seat
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
            yield self.env.timeout(exp_time(self.P.mean_interarrival_min))
            if self.env.now >= self.P.sim_minutes:
                break
            gid += 1
            self.env.process(self.group(gid))


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
    rev_lost = sum(r.revenue for r in balked) + sum(r.revenue for r in reneged)

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

    return RepStats(
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
    ), recs


# ============================================================
# COMBINED EVENT LOG
# ============================================================
def print_combined_log(recs: List[GroupRec], start_hhmm: str, limit: int = 200):
    rows = sorted(recs, key=lambda x: x.gid)[:limit]
    print("\n=== COMBINED EVENT LOG (SERVED + BALK + RENEG) ===")
    print(
        "Idx\tGID\tStatus\tSize\tArrival\t"
        "Table#\tOrderTaken\tDelivered\tPayment\tDepart\t"
        "SeatWait\tKitchenWait\tPrep\tLeftAt\tReason"
    )

    for i, r in enumerate(rows, 1):
        arrival = minutes_to_clock(r.arrival_t, start_hhmm)
        order_taken = minutes_to_clock(r.order_taken_t, start_hhmm) if r.order_taken_t is not None else ""
        delivered = minutes_to_clock(r.order_delivered_t, start_hhmm) if r.order_delivered_t is not None else ""
        payment = minutes_to_clock(r.payment_t, start_hhmm) if r.payment_t is not None else ""
        depart = minutes_to_clock(r.depart_t, start_hhmm) if r.depart_t is not None else ""

        left_at = ""
        if r.status == "BALK":
            left_at = arrival
        elif r.status == "RENEGE":
            waited = float(r.seat_wait_t or 0.0)
            left_at = minutes_to_clock(r.arrival_t + waited, start_hhmm)

        seat_wait = f"{float(r.seat_wait_t):.1f}" if r.seat_wait_t is not None else ""
        k_wait = f"{float(r.kitchen_wait_t):.1f}" if r.kitchen_wait_t is not None else ""
        prep = f"{float(r.prep_t):.1f}" if r.prep_t is not None else ""

        print(
            f"{i}\t{r.gid}\t{r.status}\t{r.size}\t{arrival}\t"
            f"{r.table_number or ''}\t{order_taken}\t{delivered}\t{payment}\t{depart}\t"
            f"{seat_wait}\t{k_wait}\t{prep}\t{left_at}\t{r.leave_reason or ''}"
        )


# ============================================================
# ANNUAL BASELINE
# ============================================================
@dataclass(frozen=True)
class DaypartConfig:
    name: str
    mean_interarrival_min: float
    is_lunch: bool


@dataclass(frozen=True)
class AnnualConfig:
    reps: int = 10
    days: int = 365
    year_start_weekday: int = MON
    base_seed: int = 2000
    lunch: DaypartConfig = DaypartConfig(name="lunch", mean_interarrival_min=4.0, is_lunch=True)
    dinner: DaypartConfig = DaypartConfig(name="dinner", mean_interarrival_min=3.3, is_lunch=False)


@dataclass
class AnnualRepResult:
    rep_seed: int
    days: int
    open_lunch_days: int
    open_dinner_days: int
    open_dayparts: int
    open_days: int

    daily_left_over_served: List[float]
    daily_los_lunch: List[float]
    daily_los_dinner: List[float]
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

        # ---- BREAKDOWNS ----
    # Lunch (per open lunch day)
    lunch_avg_daily_left: float
    lunch_avg_daily_served: float
    lunch_avg_wait: float
    lunch_avg_order: float
    lunch_avg_cook: float
    lunch_avg_pay: float
    lunch_avg_depart: float
    lunch_avg_cost_per_day: float
    lunch_avg_revenue_per_day: float
    lunch_avg_revenue_lost_per_day: float
    lunch_avg_revenue_total_per_day: float

    # Dinner (per open dinner day)
    dinner_avg_daily_left: float
    dinner_avg_daily_served: float
    dinner_avg_wait: float
    dinner_avg_order: float
    dinner_avg_cook: float
    dinner_avg_pay: float
    dinner_avg_depart: float
    dinner_avg_cost_per_day: float
    dinner_avg_revenue_per_day: float
    dinner_avg_revenue_lost_per_day: float
    dinner_avg_revenue_total_per_day: float

    # Weekday / Weekend (per open calendar day of that type; lunch+dinner combined)
    weekday_open_days: int
    weekend_open_days: int
    weekday_avg_daily_left: float
    weekday_avg_daily_served: float
    weekday_avg_cost_per_day: float
    weekday_avg_revenue_per_day: float
    weekday_avg_revenue_lost_per_day: float
    weekday_avg_revenue_total_per_day: float

    weekend_avg_daily_left: float
    weekend_avg_daily_served: float
    weekend_avg_cost_per_day: float
    weekend_avg_revenue_per_day: float
    weekend_avg_revenue_lost_per_day: float
    weekend_avg_revenue_total_per_day: float



@dataclass
class AnnualRepDinnerStages:
    dinner_avg_wait: float
    dinner_avg_order: float
    dinner_avg_cook: float
    dinner_avg_pay: float
    dinner_avg_depart: float


def run_one_year_baseline(
    base_P: Params,
    cfg: AnnualConfig,
    rep_seed: int
) -> Tuple[AnnualRepResult, AnnualRepDinnerStages]:

    daily_los: List[float] = []
    daily_los_lunch: List[float] = []
    daily_los_dinner: List[float] = []
    daily_los_weekday: List[float] = []
    daily_los_weekend: List[float] = []

    open_lunch_days = open_dinner_days = open_days = 0

    # --- Lunch accumulators (for lunch-only averages) ---
    L_served = 0
    L_left = 0
    L_wait_sum = 0.0
    L_order_sum = 0.0
    L_cook_sum = 0.0
    L_pay_sum = 0.0
    L_depart_sum = 0.0
    L_rev_served = 0.0
    L_rev_lost = 0.0
    L_cost = 0.0

    # --- Dinner accumulators (for dinner-only averages) ---
    D_served = 0
    D_left = 0
    D_wait_sum = 0.0
    D_order_sum = 0.0
    D_cook_sum = 0.0
    D_pay_sum = 0.0
    D_depart_sum = 0.0
    D_rev_served = 0.0
    D_rev_lost = 0.0
    D_cost = 0.0

    # --- Weekday / Weekend accumulators (open days only) ---
    WD_open_days = 0
    WE_open_days = 0

    WD_served = 0
    WD_left = 0
    WD_rev_served = 0.0
    WD_rev_lost = 0.0
    WD_cost = 0.0

    WE_served = 0
    WE_left = 0
    WE_rev_served = 0.0
    WE_rev_lost = 0.0
    WE_cost = 0.0

    # --- Overall accumulators (lunch + dinner combined) ---
    total_left = total_served = 0
    wait_sum = order_sum = cook_sum = pay_sum = depart_sum = 0.0
    total_rev_served = total_rev_lost = total_cost = 0.0

    base_lunch = build_daypart_params(
        base_P, seed=1, is_lunch=True, mean_interarrival=cfg.lunch.mean_interarrival_min
    )
    base_dinner = build_daypart_params(
        base_P, seed=1, is_lunch=False, mean_interarrival=cfg.dinner.mean_interarrival_min
    )

    for day in range(cfg.days):
        dow = (cfg.year_start_weekday + day) % 7
        lunch_open, dinner_open = baseline_open(dow)

        # per-calendar-day totals (optional; not used later)
        day_rev_served = 0.0
        day_rev_lost = 0.0
        day_cost = 0.0

        if lunch_open:
            open_lunch_days += 1
        if dinner_open:
            open_dinner_days += 1

        if lunch_open or dinner_open:
            open_days += 1
            if dow in {SAT, SUN}:
                WE_open_days += 1
            else:
                WD_open_days += 1
        else:
            daily_los.append(np.nan)
            daily_los_lunch.append(np.nan)
            daily_los_dinner.append(np.nan)
            continue

        l_served = l_left = 0
        d_served = d_left = 0

        # ----------------
        # LUNCH
        # ----------------
        if lunch_open:
            seed = rep_seed + day * 10 + 1
            P_day = replace(base_lunch, seed=seed)
            if dow not in {SAT, SUN}:
                P_day = replace(P_day, mean_interarrival_min=P_day.mean_interarrival_min * 1.15)

            stats, _ = run_rep_daypart(P_day, seed=P_day.seed, dow=dow, is_lunch=True)

            # lunch totals
            L_served += stats.diners_served
            L_left += stats.diners_left
            L_wait_sum += stats.avg_wait * stats.diners_served
            L_order_sum += stats.avg_order * stats.diners_served
            L_cook_sum += stats.avg_cook * stats.diners_served
            L_pay_sum += stats.avg_pay * stats.diners_served
            L_depart_sum += stats.avg_depart * stats.diners_served
            L_rev_served += stats.revenue_served
            L_rev_lost += stats.revenue_lost
            L_cost += stats.operating_cost

            # weekday/weekend totals (open day)
            if dow in {SAT, SUN}:
                WE_served += stats.diners_served
                WE_left += stats.diners_left
                WE_rev_served += stats.revenue_served
                WE_rev_lost += stats.revenue_lost
                WE_cost += stats.operating_cost
            else:
                WD_served += stats.diners_served
                WD_left += stats.diners_left
                WD_rev_served += stats.revenue_served
                WD_rev_lost += stats.revenue_lost
                WD_cost += stats.operating_cost

            # optional per-day totals
            day_rev_served += stats.revenue_served
            day_rev_lost += stats.revenue_lost
            day_cost += stats.operating_cost

            # daypart outputs
            l_served, l_left = stats.diners_served, stats.diners_left

            # overall totals
            total_served += stats.diners_served
            total_left += stats.diners_left
            wait_sum += stats.avg_wait * stats.diners_served
            order_sum += stats.avg_order * stats.diners_served
            cook_sum += stats.avg_cook * stats.diners_served
            pay_sum += stats.avg_pay * stats.diners_served
            depart_sum += stats.avg_depart * stats.diners_served
            total_rev_served += stats.revenue_served
            total_rev_lost += stats.revenue_lost
            total_cost += stats.operating_cost

        # ----------------
        # DINNER
        # ----------------
        if dinner_open:
            seed = rep_seed + day * 10 + 2
            P_day = replace(base_dinner, seed=seed)
            if dow not in {SAT, SUN}:
                P_day = replace(P_day, mean_interarrival_min=P_day.mean_interarrival_min * 1.15)

            stats, _ = run_rep_daypart(P_day, seed=P_day.seed, dow=dow, is_lunch=False)

            # dinner totals (ONLY ONCE)
            D_served += stats.diners_served
            D_left += stats.diners_left
            D_wait_sum += stats.avg_wait * stats.diners_served
            D_order_sum += stats.avg_order * stats.diners_served
            D_cook_sum += stats.avg_cook * stats.diners_served
            D_pay_sum += stats.avg_pay * stats.diners_served
            D_depart_sum += stats.avg_depart * stats.diners_served
            D_rev_served += stats.revenue_served
            D_rev_lost += stats.revenue_lost
            D_cost += stats.operating_cost

            # weekday/weekend totals (open day)
            if dow in {SAT, SUN}:
                WE_served += stats.diners_served
                WE_left += stats.diners_left
                WE_rev_served += stats.revenue_served
                WE_rev_lost += stats.revenue_lost
                WE_cost += stats.operating_cost
            else:
                WD_served += stats.diners_served
                WD_left += stats.diners_left
                WD_rev_served += stats.revenue_served
                WD_rev_lost += stats.revenue_lost
                WD_cost += stats.operating_cost

            # optional per-day totals
            day_rev_served += stats.revenue_served
            day_rev_lost += stats.revenue_lost
            day_cost += stats.operating_cost

            # daypart outputs
            d_served, d_left = stats.diners_served, stats.diners_left

            # overall totals
            total_served += stats.diners_served
            total_left += stats.diners_left
            wait_sum += stats.avg_wait * stats.diners_served
            order_sum += stats.avg_order * stats.diners_served
            cook_sum += stats.avg_cook * stats.diners_served
            pay_sum += stats.avg_pay * stats.diners_served
            depart_sum += stats.avg_depart * stats.diners_served
            total_rev_served += stats.revenue_served
            total_rev_lost += stats.revenue_lost
            total_cost += stats.operating_cost

        # daily LOS series
        daily_los_lunch.append(l_left / l_served if (lunch_open and l_served > 0) else np.nan)
        daily_los_dinner.append(d_left / d_served if (dinner_open and d_served > 0) else np.nan)

        day_served = l_served + d_served
        day_left = l_left + d_left
        los_day = (day_left / day_served) if day_served > 0 else np.nan
        daily_los.append(los_day)

        if not np.isnan(los_day):
            (daily_los_weekend if dow in {SAT, SUN} else daily_los_weekday).append(los_day)

    # ---- final aggregation ----
    open_dayparts = open_lunch_days + open_dinner_days
    if open_dayparts == 0:
        raise ValueError("open_dayparts=0. Check baseline_open().")

    served_denom = max(1, total_served)
    avg_wait = wait_sum / served_denom
    avg_order = order_sum / served_denom
    avg_cook = cook_sum / served_denom
    avg_pay = pay_sum / served_denom
    avg_depart = depart_sum / served_denom

    open_days_safe = max(1, open_days)
    avg_daily_left = total_left / open_days_safe
    avg_daily_served = total_served / open_days_safe
    avg_cost_per_day = total_cost / open_days_safe
    avg_revenue_per_day = total_rev_served / open_days_safe
    avg_revenue_lost_per_day = total_rev_lost / open_days_safe
    avg_revenue_total_per_day = (total_rev_served + total_rev_lost) / open_days_safe

    # dinner stage summary (weighted by diners served at dinner)
    D_denom = max(1, D_served)
    dinner = AnnualRepDinnerStages(
        dinner_avg_wait=float(D_wait_sum / D_denom),
        dinner_avg_order=float(D_order_sum / D_denom),
        dinner_avg_cook=float(D_cook_sum / D_denom),
        dinner_avg_pay=float(D_pay_sum / D_denom),
        dinner_avg_depart=float(D_depart_sum / D_denom),
    )

    # -----------------------------
    # NEW: Lunch averages
    # -----------------------------
    lunch_days = max(1, open_lunch_days)
    lunch_served_denom = max(1, L_served)

    lunch_avg_daily_left = L_left / lunch_days
    lunch_avg_daily_served = L_served / lunch_days
    lunch_avg_wait = L_wait_sum / lunch_served_denom
    lunch_avg_order = L_order_sum / lunch_served_denom
    lunch_avg_cook = L_cook_sum / lunch_served_denom
    lunch_avg_pay = L_pay_sum / lunch_served_denom
    lunch_avg_depart = L_depart_sum / lunch_served_denom

    lunch_avg_cost_per_day = L_cost / lunch_days
    lunch_avg_rev_per_day = L_rev_served / lunch_days
    lunch_avg_rev_lost_per_day = L_rev_lost / lunch_days
    lunch_avg_rev_total_per_day = (L_rev_served + L_rev_lost) / lunch_days

    # -----------------------------
    # NEW: Dinner averages
    # -----------------------------
    dinner_days = max(1, open_dinner_days)
    dinner_served_denom = max(1, D_served)

    dinner_avg_daily_left = D_left / dinner_days
    dinner_avg_daily_served = D_served / dinner_days
    dinner_avg_wait2 = D_wait_sum / dinner_served_denom
    dinner_avg_order2 = D_order_sum / dinner_served_denom
    dinner_avg_cook2 = D_cook_sum / dinner_served_denom
    dinner_avg_pay2 = D_pay_sum / dinner_served_denom
    dinner_avg_depart2 = D_depart_sum / dinner_served_denom

    dinner_avg_cost_per_day = D_cost / dinner_days
    dinner_avg_rev_per_day = D_rev_served / dinner_days
    dinner_avg_rev_lost_per_day = D_rev_lost / dinner_days
    dinner_avg_rev_total_per_day = (D_rev_served + D_rev_lost) / dinner_days

    # -----------------------------
    # NEW: Weekday vs Weekend averages (per OPEN day)
    # -----------------------------
    WD_days = max(1, WD_open_days)
    WE_days = max(1, WE_open_days)

    weekday_avg_daily_left = WD_left / WD_days
    weekday_avg_daily_served = WD_served / WD_days
    weekday_avg_cost_per_day = WD_cost / WD_days
    weekday_avg_rev_per_day = WD_rev_served / WD_days
    weekday_avg_rev_lost_per_day = WD_rev_lost / WD_days
    weekday_avg_rev_total_per_day = (WD_rev_served + WD_rev_lost) / WD_days

    weekend_avg_daily_left = WE_left / WE_days
    weekend_avg_daily_served = WE_served / WE_days
    weekend_avg_cost_per_day = WE_cost / WE_days
    weekend_avg_rev_per_day = WE_rev_served / WE_days
    weekend_avg_rev_lost_per_day = WE_rev_lost / WE_days
    weekend_avg_rev_total_per_day = (WE_rev_served + WE_rev_lost) / WE_days

    annual = AnnualRepResult(
        rep_seed=rep_seed,
        days=cfg.days,
        open_lunch_days=open_lunch_days,
        open_dinner_days=open_dinner_days,
        open_dayparts=open_dayparts,
        open_days=open_days,
        daily_left_over_served=daily_los,
        daily_los_lunch=daily_los_lunch,
        daily_los_dinner=daily_los_dinner,
        daily_los_weekday=daily_los_weekday,
        daily_los_weekend=daily_los_weekend,
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
        lunch_avg_revenue_per_day=float(lunch_avg_rev_per_day),
        lunch_avg_revenue_lost_per_day=float(lunch_avg_rev_lost_per_day),
        lunch_avg_revenue_total_per_day=float(lunch_avg_rev_total_per_day),

        dinner_avg_daily_left=float(dinner_avg_daily_left),
        dinner_avg_daily_served=float(dinner_avg_daily_served),
        dinner_avg_wait=float(dinner_avg_wait2),
        dinner_avg_order=float(dinner_avg_order2),
        dinner_avg_cook=float(dinner_avg_cook2),
        dinner_avg_pay=float(dinner_avg_pay2),
        dinner_avg_depart=float(dinner_avg_depart2),
        dinner_avg_cost_per_day=float(dinner_avg_cost_per_day),
        dinner_avg_revenue_per_day=float(dinner_avg_rev_per_day),
        dinner_avg_revenue_lost_per_day=float(dinner_avg_rev_lost_per_day),
        dinner_avg_revenue_total_per_day=float(dinner_avg_rev_total_per_day),

        weekday_open_days=int(WD_open_days),
        weekend_open_days=int(WE_open_days),
        weekday_avg_daily_left=float(weekday_avg_daily_left),
        weekday_avg_daily_served=float(weekday_avg_daily_served),
        weekday_avg_cost_per_day=float(weekday_avg_cost_per_day),
        weekday_avg_revenue_per_day=float(weekday_avg_rev_per_day),
        weekday_avg_revenue_lost_per_day=float(weekday_avg_rev_lost_per_day),
        weekday_avg_revenue_total_per_day=float(weekday_avg_rev_total_per_day),

        weekend_avg_daily_left=float(weekend_avg_daily_left),
        weekend_avg_daily_served=float(weekend_avg_daily_served),
        weekend_avg_cost_per_day=float(weekend_avg_cost_per_day),
        weekend_avg_revenue_per_day=float(weekend_avg_rev_per_day),
        weekend_avg_revenue_lost_per_day=float(weekend_avg_rev_lost_per_day),
        weekend_avg_revenue_total_per_day=float(weekend_avg_rev_total_per_day),

    )
    return annual, dinner


def run_annual_baseline_reps(base_P: Params, cfg: AnnualConfig) -> Tuple[List[AnnualRepResult], List[AnnualRepDinnerStages]]:
    reps: List[AnnualRepResult] = []
    dinners: List[AnnualRepDinnerStages] = []
    for i in range(cfg.reps):
        rep_seed = cfg.base_seed + i * 100_000
        annual, dinner = run_one_year_baseline(base_P, cfg, rep_seed)
        reps.append(annual)
        dinners.append(dinner)
    return reps, dinners


def calibrate_overhead_per_open_day(
    base_P: Params,
    cfg: AnnualConfig,
    target_margin: float = 0.026,
    calib_days: int = 30,
    calib_reps: int = 3
) -> Params:
    cfg_cal = replace(cfg, reps=calib_reps, days=calib_days)
    tmp_P = replace(base_P, overhead_per_open_day=0.0)

    reps = [run_one_year_baseline(tmp_P, cfg_cal, cfg_cal.base_seed + i * 100_000)[0] for i in range(cfg_cal.reps)]
    avg_rev_day = float(np.mean([r.avg_revenue_per_day for r in reps]))
    avg_food_day = tmp_P.food_cost_rate * avg_rev_day

    labor_total = 0.0
    open_days_cal = 0
    P_l = build_daypart_params(tmp_P, seed=1, is_lunch=True, mean_interarrival=cfg_cal.lunch.mean_interarrival_min)
    P_d = build_daypart_params(tmp_P, seed=1, is_lunch=False, mean_interarrival=cfg_cal.dinner.mean_interarrival_min)

    for day in range(cfg_cal.days):
        dow = (cfg_cal.year_start_weekday + day) % 7
        lunch_open, dinner_open = baseline_open(dow)
        if lunch_open or dinner_open:
            open_days_cal += 1
        if lunch_open:
            labor_total += daypart_labor_cost(P_l, dow, True)
        if dinner_open:
            labor_total += daypart_labor_cost(P_d, dow, False)

    avg_labor_day = labor_total / max(1, open_days_cal)
    target_cost_day = (1.0 - target_margin) * avg_rev_day
    overhead_day = max(0.0, target_cost_day - avg_food_day - avg_labor_day)
    return replace(base_P, overhead_per_open_day=overhead_day)

def prob_gt(series, thr: float) -> float:
    """Probability that values in `series` exceed `thr`, ignoring NaNs."""
    a = np.asarray(series, dtype=float)
    a = a[~np.isnan(a)]
    return float(np.mean(a > thr)) if a.size else float("nan")

def print_baseline_summary_with_breakdowns(baseline_reps: List[AnnualRepResult], threshold: float = 0.33):
    def fmt_ci(xs: List[float], digits: int):
        m, lo, hi = ci95(xs)
        return f"{m:.{digits}f} ({lo:.{digits}f}–{hi:.{digits}f})"

    # overall (existing logic)
    pct_rev_lost = [
        100.0 * r.avg_revenue_lost_per_day / max(1e-9, r.avg_revenue_total_per_day)
        for r in baseline_reps
    ]
    p_bad = [prob_gt(r.daily_left_over_served, threshold) for r in baseline_reps]
    mean_los = [float(np.nanmean(r.daily_left_over_served)) for r in baseline_reps]
    p95_los = [float(np.nanpercentile(r.daily_left_over_served, 95)) for r in baseline_reps]

    cost_year = [r.avg_cost_per_day * r.open_days for r in baseline_reps]
    rev_year  = [r.avg_revenue_per_day * r.open_days for r in baseline_reps]
    profit_year = [(r.avg_revenue_per_day - r.avg_cost_per_day) * r.open_days for r in baseline_reps]

    print("\n--- baseline ---")
    print(f"% revenue lost (queue friction): {fmt_ci(pct_rev_lost, 2)}%")
    print(f"Risk: P(left >{threshold:.2f}):            {fmt_ci(p_bad, 3)}")
    print(f"Daily left/served:          {fmt_ci(mean_los, 3)}")
    print(f"P95 daily left/served:    {fmt_ci(p95_los, 3)}")
    print(f"Avg Daily left:           {fmt_ci([r.avg_daily_left for r in baseline_reps], 1)}")
    print(f"Avg Daily served:         {fmt_ci([r.avg_daily_served for r in baseline_reps], 1)}")
    print(f"Avg wait:                 {fmt_ci([r.avg_wait for r in baseline_reps], 2)}")
    print(f"Avg order:                {fmt_ci([r.avg_order for r in baseline_reps], 2)}")
    print(f"Avg cook:                 {fmt_ci([r.avg_cook for r in baseline_reps], 2)}")
    print(f"Avg pay:                  {fmt_ci([r.avg_pay for r in baseline_reps], 2)}")
    print(f"Avg depart:               {fmt_ci([r.avg_depart for r in baseline_reps], 2)}")
    print(f"Avg cost/day:             {fmt_ci([r.avg_cost_per_day for r in baseline_reps], 2)}")
    print(f"cost per year:            {fmt_ci(cost_year, 2)}")
    print(f"P95 cost per year:        {np.percentile(cost_year, 95):.2f}")
    print(f"Avg revenue/day:          {fmt_ci([r.avg_revenue_per_day for r in baseline_reps], 2)}")
    print(f"revenue per year:         {fmt_ci(rev_year, 2)}")
    print(f"P95 revenue per year:     {np.percentile(rev_year, 95):.2f}")
    print(f"% revenue lost:           {fmt_ci(pct_rev_lost, 2)}%")
    print(f"Profit/year ($):          {fmt_ci(profit_year, 2)}")
    print(f"Open days/year (mean):    {np.mean([r.open_days for r in baseline_reps]):.1f}")

    # ---- Lunch ----
    print("--[LUNCH]--")
    print(f"Avg Daily left:   {fmt_ci([r.lunch_avg_daily_left for r in baseline_reps], 1)}")
    print(f"Avg Daily served: {fmt_ci([r.lunch_avg_daily_served for r in baseline_reps], 1)}")
    print(f"Avg wait:         {fmt_ci([r.lunch_avg_wait for r in baseline_reps], 2)}")
    print(f"Avg order:        {fmt_ci([r.lunch_avg_order for r in baseline_reps], 2)}")
    print(f"Avg cook:         {fmt_ci([r.lunch_avg_cook for r in baseline_reps], 2)}")
    print(f"Avg pay:          {fmt_ci([r.lunch_avg_pay for r in baseline_reps], 2)}")
    print(f"Avg depart:       {fmt_ci([r.lunch_avg_depart for r in baseline_reps], 2)}")
    print(f"Avg cost/day:     {fmt_ci([r.lunch_avg_cost_per_day for r in baseline_reps], 2)}")
    print(f"Avg rev/day:      {fmt_ci([r.lunch_avg_revenue_per_day for r in baseline_reps], 2)}")

    # ---- Dinner ----
    print("--[DINNER]--")
    print(f"Avg Daily left:   {fmt_ci([r.dinner_avg_daily_left for r in baseline_reps], 1)}")
    print(f"Avg Daily served: {fmt_ci([r.dinner_avg_daily_served for r in baseline_reps], 1)}")
    print(f"Avg wait:         {fmt_ci([r.dinner_avg_wait for r in baseline_reps], 2)}")
    print(f"Avg order:        {fmt_ci([r.dinner_avg_order for r in baseline_reps], 2)}")
    print(f"Avg cook:         {fmt_ci([r.dinner_avg_cook for r in baseline_reps], 2)}")
    print(f"Avg pay:          {fmt_ci([r.dinner_avg_pay for r in baseline_reps], 2)}")
    print(f"Avg depart:       {fmt_ci([r.dinner_avg_depart for r in baseline_reps], 2)}")
    print(f"Avg cost/day:     {fmt_ci([r.dinner_avg_cost_per_day for r in baseline_reps], 2)}")
    print(f"Avg rev/day:      {fmt_ci([r.dinner_avg_revenue_per_day for r in baseline_reps], 2)}")

    # ---- Weekday vs Weekend ----
    print("--[WEEKDAY]--")
    print(f"Open days (mean): {np.mean([r.weekday_open_days for r in baseline_reps]):.1f}")
    print(f"Avg Daily left:   {fmt_ci([r.weekday_avg_daily_left for r in baseline_reps], 1)}")
    print(f"Avg Daily served: {fmt_ci([r.weekday_avg_daily_served for r in baseline_reps], 1)}")
    print(f"Avg cost/day:     {fmt_ci([r.weekday_avg_cost_per_day for r in baseline_reps], 2)}")
    print(f"Avg rev/day:      {fmt_ci([r.weekday_avg_revenue_per_day for r in baseline_reps], 2)}")

    print("--[WEEKEND]--")
    print(f"Open days (mean): {np.mean([r.weekend_open_days for r in baseline_reps]):.1f}")
    print(f"Avg Daily left:   {fmt_ci([r.weekend_avg_daily_left for r in baseline_reps], 1)}")
    print(f"Avg Daily served: {fmt_ci([r.weekend_avg_daily_served for r in baseline_reps], 1)}")
    print(f"Avg cost/day:     {fmt_ci([r.weekend_avg_cost_per_day for r in baseline_reps], 2)}")
    print(f"Avg rev/day:      {fmt_ci([r.weekend_avg_revenue_per_day for r in baseline_reps], 2)}")


# ============================================================
# PLOTS
# ============================================================
def clean(xs):
    a = np.asarray(xs, dtype=float)
    return a[~np.isnan(a)]

def plot_ci_bars(title, label, xs):
    m, lo, hi = ci95(xs)
    plt.figure(figsize=(6, 4))
    plt.bar([0], [m])
    plt.errorbar([0], [m], yerr=[[m - lo], [hi - m]], fmt="none", capsize=5)
    plt.xticks([0], [label])
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_risk_one_chart_baseline(baseline_reps: List[AnnualRepResult], cfg: AnnualConfig, threshold=0.33):
    seg = {
        "Lunch – Weekday": [],
        "Lunch – Weekend": [],
        "Dinner – Weekday": [],
        "Dinner – Weekend": [],
    }

    for r in baseline_reps:
        for day_idx, los in enumerate(r.daily_los_lunch):
            if np.isnan(los):
                continue
            dow = (cfg.year_start_weekday + day_idx) % 7
            key = "Lunch – Weekend" if dow in {SAT, SUN} else "Lunch – Weekday"
            seg[key].append(float(los > threshold))

        for day_idx, los in enumerate(r.daily_los_dinner):
            if np.isnan(los):
                continue
            dow = (cfg.year_start_weekday + day_idx) % 7
            key = "Dinner – Weekend" if dow in {SAT, SUN} else "Dinner – Weekday"
            seg[key].append(float(los > threshold))

    labels = list(seg.keys())
    means, lows, highs = [], [], []
    for k in labels:
        m, lo, hi = ci95(seg[k])
        means.append(m); lows.append(lo); highs.append(hi)

    means = np.array(means)
    yerr = np.vstack([means - np.array(lows), np.array(highs) - means])

    x = np.arange(len(labels))
    plt.figure(figsize=(10, 5))
    plt.bar(x, means)
    plt.errorbar(x, means, yerr=yerr, fmt="none", capsize=5)
    for i, m in enumerate(means):
        if np.isfinite(m):
            plt.text(i, m, f"{m:,.0f}", ha="center", va="bottom", fontsize=10)

    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylim(0, max(0.15, float(np.nanmax(means) * 1.3)))
    plt.axhline(threshold, linestyle="--", linewidth=1)
    plt.ylabel(f"P(daily left/served > {threshold})")
    plt.title("Baseline Congestion Risk by Segment (95% CI)")
    plt.tight_layout()
    plt.show()


def plot_los_box_weekday_weekend(baseline_reps: List[AnnualRepResult], threshold=0.33):
    wk = []
    we = []
    for r in baseline_reps:
        wk.extend([x for x in r.daily_los_weekday if not np.isnan(x)])
        we.extend([x for x in r.daily_los_weekend if not np.isnan(x)])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].boxplot([wk], labels=["baseline"], showfliers=False)
    axes[0].axhline(threshold, linestyle="--", linewidth=1.5)
    axes[0].set_title("Weekday")
    axes[0].set_ylabel("Daily left/served")

    axes[1].boxplot([we], labels=["baseline"], showfliers=False)
    axes[1].axhline(threshold, linestyle="--", linewidth=1.5)
    axes[1].set_title("Weekend")

    fig.suptitle("Distribution of daily left/served (baseline)\nSplit: weekday vs weekend", y=1.02)
    plt.tight_layout()
    plt.show()


# ============================================================
# ADDITIONAL BASELINE PLOTS:
#   1) % revenue lost (balk+renege) with 95% CI
#   2) Distribution (hist) of balked diners + reneged diners (per open day)
#   3) Pie chart of balk vs renege share (by diners, aggregated across reps)
# ============================================================

def _collect_open_day_totals_from_annual_rep(
    base_P: Params,
    cfg: AnnualConfig,
    rep_seed: int
) -> Tuple[List[int], List[int], List[float], List[float]]:
    """
    Re-run one rep-year (same logic as run_one_year_baseline) but ALSO
    collect per-open-day:
      - balked diners
      - reneged diners
      - revenue served
      - revenue lost (balk+renege)
    Returns:
      balk_diners_per_open_day, renege_diners_per_open_day,
      rev_served_per_open_day, rev_lost_per_open_day
    """
    days = cfg.days

    balk_diners_open: List[int] = []
    renege_diners_open: List[int] = []
    rev_served_open: List[float] = []
    rev_lost_open: List[float] = []

    base_lunch = build_daypart_params(base_P, seed=1, is_lunch=True, mean_interarrival=cfg.lunch.mean_interarrival_min)
    base_dinner = build_daypart_params(base_P, seed=1, is_lunch=False, mean_interarrival=cfg.dinner.mean_interarrival_min)

    for day in range(days):
        dow = (cfg.year_start_weekday + day) % 7
        lunch_open, dinner_open = baseline_open(dow)

        if not (lunch_open or dinner_open):
            continue  # not an open day

        day_balk = 0
        day_renege = 0
        day_rev_served = 0.0
        day_rev_lost = 0.0

        # ---- lunch ----
        if lunch_open:
            seed = rep_seed + day * 10 + 1
            P_day = replace(base_lunch, seed=seed)

            # weekday softer demand (keep consistent with your annual baseline)
            if dow not in {SAT, SUN}:
                P_day = replace(P_day, mean_interarrival_min=P_day.mean_interarrival_min * 1.15)

            stats, recs = run_rep_daypart(P_day, seed=P_day.seed, dow=dow, is_lunch=True)

            day_rev_served += stats.revenue_served
            day_rev_lost += stats.revenue_lost

            day_balk += sum(r.size for r in recs if r.status == "BALK")
            day_renege += sum(r.size for r in recs if r.status == "RENEGE")

        # ---- dinner ----
        if dinner_open:
            seed = rep_seed + day * 10 + 2
            P_day = replace(base_dinner, seed=seed)

            if dow not in {SAT, SUN}:
                P_day = replace(P_day, mean_interarrival_min=P_day.mean_interarrival_min * 1.15)

            stats, recs = run_rep_daypart(P_day, seed=P_day.seed, dow=dow, is_lunch=False)

            day_rev_served += stats.revenue_served
            day_rev_lost += stats.revenue_lost

            day_balk += sum(r.size for r in recs if r.status == "BALK")
            day_renege += sum(r.size for r in recs if r.status == "RENEGE")

        balk_diners_open.append(day_balk)
        renege_diners_open.append(day_renege)
        rev_served_open.append(day_rev_served)
        rev_lost_open.append(day_rev_lost)

    return balk_diners_open, renege_diners_open, rev_served_open, rev_lost_open


def plot_balk_renege_distributions(
    base_P: Params,
    cfg: AnnualConfig,
    baseline_reps: List[AnnualRepResult],
):
    """
    Re-runs each rep-year (using the same rep_seed values embedded in AnnualRepResult)
    to capture daily balk vs renege counts and plot their distributions.
    """
    all_balk = []
    all_renege = []

    for r in baseline_reps:
        b, g, _, _ = _collect_open_day_totals_from_annual_rep(base_P, cfg, r.rep_seed)
        all_balk.extend(b)
        all_renege.extend(g)

    all_balk = np.asarray(all_balk, dtype=float)
    all_renege = np.asarray(all_renege, dtype=float)

    plt.figure(figsize=(7, 4))
    plt.hist(clean(all_balk), bins=20)
    plt.title("Baseline: Distribution of Balked Diners per Open Day")
    plt.xlabel("Balked diners (count)")
    plt.ylabel("Days (pooled across reps)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.hist(clean(all_renege), bins=20)
    plt.title("Baseline: Distribution of Reneged Diners per Open Day")
    plt.xlabel("Reneged diners (count)")
    plt.ylabel("Days (pooled across reps)")
    plt.tight_layout()
    plt.show()



# ============================================================
# DISTRIBUTION: DAILY % REVENUE LOST (balk + renege) PER OPEN DAY
# (pooled across ALL reps and all open days)
# ============================================================

def plot_daily_pct_revenue_lost_distribution(
    base_P: Params,
    cfg: AnnualConfig,
    baseline_reps: List[AnnualRepResult],
    bins: int = 30
):
    """
    Builds a per-open-day series:
      pct_lost_day = rev_lost_day / (rev_served_day + rev_lost_day)
    and plots its distribution pooled across all reps + open days.
    """
    daily_pct = []

    for r in baseline_reps:
        _, _, rev_served_open, rev_lost_open = _collect_open_day_totals_from_annual_rep(
            base_P=base_P,
            cfg=cfg,
            rep_seed=r.rep_seed
        )

        for rs, rl in zip(rev_served_open, rev_lost_open):
            total = rs + rl
            if total <= 1e-9:
                continue
            daily_pct.append(100.0 * rl / total)

    daily_pct = np.asarray(daily_pct, dtype=float)

    if len(daily_pct) == 0:
        print("No daily % revenue lost values found (check open-day collection).")
        return

    plt.figure(figsize=(8, 4))
    plt.hist(daily_pct, bins=bins)
    plt.title("Baseline: Distribution of Daily % Revenue Lost (Queue Friction)")
    plt.xlabel("% revenue lost per open day")
    plt.ylabel("Open days (pooled across reps)")

    # helpful reference lines
    p50 = float(np.percentile(daily_pct, 50))
    p95 = float(np.percentile(daily_pct, 95))
    mean = float(np.mean(daily_pct))

    plt.axvline(mean, linestyle="--", linewidth=1.5, label=f"Mean = {mean:.2f}%")
    plt.axvline(p50, linestyle="--", linewidth=1.5, label=f"Median = {p50:.2f}%")
    plt.axvline(p95, linestyle="--", linewidth=1.5, label=f"P95 = {p95:.2f}%")

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_annual_financial_bar_baseline(baseline_reps, show_ci=True):
    """
    One bar chart (baseline):
      - Annual revenue (served)
      - Annual revenue lost (balk+renege)
      - Annual cost
      - Annual profit (served - cost)
    Values are computed per-rep-year, then averaged across reps.
    """

    # per-rep annual totals
    annual_rev_served = np.array([r.avg_revenue_per_day * r.open_days for r in baseline_reps], dtype=float)
    annual_rev_lost   = np.array([r.avg_revenue_lost_per_day * r.open_days for r in baseline_reps], dtype=float)
    annual_cost       = np.array([r.avg_cost_per_day * r.open_days for r in baseline_reps], dtype=float)
    annual_profit     = annual_rev_served - annual_cost

    labels = [
        "Revenue (served)",
        "Revenue lost\n(balk+renege)",
        "Cost",
        "Profit",
    ]
    series = [annual_rev_served, annual_rev_lost, annual_cost, annual_profit]

    means = []
    yerr_low = []
    yerr_high = []

    for xs in series:
        m, lo, hi = ci95(list(xs))   # uses your existing ci95()
        means.append(m)
        yerr_low.append(m - lo)
        yerr_high.append(hi - m)

    means = np.array(means, dtype=float)

    x = np.arange(len(labels))
    plt.figure(figsize=(10, 5))
    plt.bar(x, means)

    if show_ci:
        yerr = np.vstack([np.array(yerr_low), np.array(yerr_high)])
        plt.errorbar(x, means, yerr=yerr, fmt="none", capsize=5)

    # labels on bars
    offset = 0.02 * (np.nanmax(means) if np.isfinite(np.nanmax(means)) else 1.0)
    for i, m in enumerate(means):
        if np.isfinite(m):
            plt.text(i, m + offset, f"{m:,.0f}", ha="center", va="bottom", fontsize=10)

    plt.xticks(x, labels)
    plt.ylabel("Annual $ (avg over reps)")
    plt.title("Baseline: Annual Financial Summary (avg over reps)")
    plt.tight_layout()
    plt.show()
# ============================================================
# BAR CHART: DAILY BALK vs RENEGE (baseline only)
# ============================================================
def plot_daily_balk_renege_bar(
    base_P: Params,
    cfg: AnnualConfig,
    baseline_reps: List[AnnualRepResult],
):
    """
    Bar chart of mean daily balk vs renege diners per open day.
    Error bars = 95% CI across rep-years.
    """
    balk_means = []
    renege_means = []

    for r in baseline_reps:
        balk_open, renege_open, _, _ = _collect_open_day_totals_from_annual_rep(
            base_P=base_P, cfg=cfg, rep_seed=r.rep_seed
        )
        balk_means.append(float(np.mean(balk_open)) if len(balk_open) else np.nan)
        renege_means.append(float(np.mean(renege_open)) if len(renege_open) else np.nan)

    balk_m, balk_lo, balk_hi = ci95(balk_means)
    ren_m,  ren_lo,  ren_hi  = ci95(renege_means)

    labels = ["Balk", "Renege"]
    means = np.array([balk_m, ren_m], dtype=float)
    yerr = np.array([
        [balk_m - balk_lo, ren_m - ren_lo],
        [balk_hi - balk_m, ren_hi - ren_m],
    ], dtype=float)

    x = np.arange(len(labels))
    plt.figure(figsize=(6, 4))
    plt.bar(x, means)
    plt.errorbar(x, means, yerr=yerr, fmt="none", capsize=6)

    for i, v in enumerate(means):
        if np.isfinite(v):
            plt.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10)

    plt.xticks(x, labels)
    plt.ylabel("Diners per open day")
    plt.title("Baseline: Mean Daily Walk-Aways by Type (95% CI)")
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    P = Params()

    # 1) One dinner rep log + combined event log (Saturday 17:00–21:30)
    P_log_base = replace(
        P,
        dinner_open_hhmm="17:00",
        dinner_close_hhmm="21:30",
    )

    P_log = build_daypart_params(
        P_log_base,
        seed=999,
        is_lunch=False,
        mean_interarrival=P.mean_interarrival_min
    )

    # ensure clock prints from 17:00
    P_log = replace(
        P_log,
        validation_log=True,
        log_clock_start_hhmm="17:00"
    )

    # sanity check
    print("LOG sim_minutes:", P_log.sim_minutes)  # should be 270.0
    print("LOG start time:", P_log.log_clock_start_hhmm)

    stats, recs = run_rep_daypart(
        P_log,
        seed=P_log.seed,
        dow=SAT,
        is_lunch=False
    )

    print_combined_log(
        recs,
        start_hhmm=P_log.log_clock_start_hhmm,
        limit=200
    )


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

    # labels: SHOW % CHANGE ONLY (New vs Historical)
    for i, (h, n) in enumerate(zip(hist_vals, new_vals)):
        if h > 0:
            pct = 100.0 * (n - h) / h
            label = f"{pct:+.1f}%"
        else:
            label = "n/a"

        y = max(h, n)
        plt.text(
            i,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=9
        )


    plt.xticks(x, metrics)
    plt.ylabel("Value")
    plt.title("Saturday Dinner: Historical vs Single-Day Simulation\n(+ change vs historical)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) Annual baseline (calibrate overhead once, then run reps once)
    threshold = 0.33
    cfg = AnnualConfig(reps=10, days=365, year_start_weekday=MON, base_seed=2000)

    P = calibrate_overhead_per_open_day(P, cfg, target_margin=0.026)
    baseline_reps, baseline_dinner = run_annual_baseline_reps(P, cfg)
    print_baseline_summary_with_breakdowns(baseline_reps, threshold=threshold)

    # 4) Plots (all baseline)
    plot_risk_one_chart_baseline(baseline_reps, cfg, threshold=threshold)
    plot_los_box_weekday_weekend(baseline_reps, threshold=threshold)

   
    plot_annual_financial_bar_baseline(baseline_reps, show_ci=False)

    plot_daily_balk_renege_bar(P, cfg, baseline_reps)
    plot_balk_renege_distributions(P, cfg, baseline_reps)
    plot_daily_pct_revenue_lost_distribution(P, cfg, baseline_reps, bins=30)






