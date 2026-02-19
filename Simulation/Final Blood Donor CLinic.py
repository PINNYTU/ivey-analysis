import random
import math
import simpy
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# =========================================================
# GLOBAL PARAMETERS
# =========================================================
REPS = 1000
OPEN_HOURS = 5
OPEN_MIN = OPEN_HOURS * 60

DAY_RUN_BUFFER = 480  # allow clinic to finish serving after close

# Initial line-up every day: triangular 5–10 people (integers)
INITIAL_LINEUP_TRI = (5, 7.5, 10)  # (low, mode, high)

# Interarrival intensity per day: random between 25 and 30 arrivals/hour (fixed for that day)
ARRIVALS_PER_HR_RANGE = (25, 30)

# Capacities (fixed downstream)
CAP_CUBICLE = 6
CAP_POD = 8

# Branching probabilities
P_REGISTER = [("returning", 0.50), ("first", 0.30), ("issue", 0.20)]
P_SURVEY   = [("no_travel", 0.75), ("travel", 0.25)]
P_HEALTH   = [("regular", 0.95), ("extreme", 0.05)]
P_DONATE   = [("regular", 0.97), ("extreme", 0.03)]

# Register: enforce "register_in <= 300" (cannot start registration after close)
# Optional: also keep max wait balking (can disable by setting to None)
MAX_REGISTER_WAIT = 300.0
AVG_REGISTER_SERVICE = 3.75  # used only for quick backlog estimate

QUEUED_STAGES = ["register", "survey", "health", "donate"]

# Surge scenarios
SCENARIOS = {
    "baseline": 0.00,
    "+5%": 0.05,
    "+15%": 0.15,
    "+20%": 0.20,
}

# (registration desks, survey kiosks)
CONFIGS = [
    (1, 3),
    (2, 3),
    (3, 3),
    (3, 4),   # recommended
    (3, 5),
]

# =========================================================
# HELPERS
# =========================================================
def choose(rng, options):
    r = rng.random()
    s = 0.0
    for name, p in options:
        s += p
        if r <= s:
            return name
    return options[-1][0]

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def ci95_over_samples(samples):
    vals = [v for v in samples if v is not None and not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = sum(vals) / n
    if n == 1:
        return m, m, m, 1
    s = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))
    se = s / math.sqrt(n)
    return m, m - 1.96 * se, m + 1.96 * se, n

def initial_lineup_size(rng):
    x = rng.triangular(*INITIAL_LINEUP_TRI)
    return max(0, int(round(x)))

def mean_interarrival_minutes_for_day(rng, surge=0.0):
    arrivals_per_hour = rng.uniform(*ARRIVALS_PER_HR_RANGE) * (1.0 + surge)
    return 60.0 / arrivals_per_hour

def exp_interarrival(rng, mean_minutes):
    return -mean_minutes * math.log(1.0 - rng.random())

# Service times (minutes)
def t_register(rng, kind):
    if kind == "returning":
        return 1.0
    if kind == "first":
        return rng.triangular(5.0, 10.0, 7.5)
    return 5.0

def t_survey(rng, kind):
    if kind == "no_travel":
        return rng.triangular(5.5, 6.0, 5.75)
    return 10.0

def t_health(rng, kind):
    if kind == "regular":
        return rng.triangular(10.0, 12.0, 11.0)
    return rng.triangular(12.0, 30.0, 18.0)

def t_donate(rng, kind):
    if kind == "regular":
        return rng.triangular(14.0, 15.0, 14.5)
    return rng.triangular(15.0, 30.0, 20.0)

def t_refresh(rng):
    return rng.triangular(5.0, 10.0, 7.5)

# =========================================================
# SIMPY MODEL
# =========================================================
class Clinic:
    def __init__(self, env, rng, mean_interarrival, cap_reg, cap_survey):
        self.env = env
        self.rng = rng
        self.mean_interarrival = mean_interarrival

        # Configurable resources
        self.reg = simpy.Resource(env, capacity=cap_reg)       # registration desks
        self.survey = simpy.Resource(env, capacity=cap_survey) # survey kiosks

        # Fixed resources
        self.cubicle = simpy.Resource(env, capacity=CAP_CUBICLE)
        self.pod = simpy.Resource(env, capacity=CAP_POD)

        # Wait times at queued stages (donor-level samples)
        self.wait = defaultdict(list)

        # Per-donor totals (completed donors only)
        self.client_cycle_times = []
        self.client_total_waits = []
        self.client_total_services = []

        # Counts
        self.donors_completed = 0
        self.balked_register = 0

    def use_resource(self, res, stage_name, service_time):
        t_arrive = self.env.now
        with res.request() as req:
            yield req
            t_start = self.env.now
            wait = t_start - t_arrive
            self.wait[stage_name].append(wait)

            yield self.env.timeout(service_time)
            return wait, service_time

    def register_step(self):
        """
        Enforce: register must START before close (register_in <= OPEN_MIN).
        If they don't get a desk before closing => balk.
        Optional: keep max-wait balking as an extra guard.
        """
        time_at_lining = self.env.now
        kind = choose(self.rng, P_REGISTER)
        service_time = t_register(self.rng, kind)

        # can't start registration after close
        time_to_close = OPEN_MIN - time_at_lining
        if time_to_close <= 0:
            self.balked_register += 1
            return False, 0.0, 0.0

        # optional backlog-based balking (can disable by setting MAX_REGISTER_WAIT=None)
        if MAX_REGISTER_WAIT is not None:
            est_wait = len(self.reg.queue) * AVG_REGISTER_SERVICE
            if est_wait > MAX_REGISTER_WAIT:
                self.balked_register += 1
                return False, 0.0, 0.0

        with self.reg.request() as req:
            result = yield req | self.env.timeout(time_to_close)
            if req not in result:
                # did not get a reg desk before close
                self.balked_register += 1
                return False, 0.0, 0.0

            register_in = self.env.now
            wait = register_in - time_at_lining

            # hard safety (should already be guaranteed)
            if register_in > OPEN_MIN:
                self.balked_register += 1
                return False, 0.0, 0.0

            # also enforce max-wait if enabled
            if MAX_REGISTER_WAIT is not None and wait > MAX_REGISTER_WAIT:
                self.balked_register += 1
                return False, 0.0, 0.0

            yield self.env.timeout(service_time)

        # record wait for register stage (donor-level)
        self.wait["register"].append(wait)
        return True, wait, service_time

    def donor(self):
        # cycle begins when they join register line
        t_cycle_start = self.env.now

        ok, w_reg, s_reg = yield from self.register_step()
        if not ok:
            return

        total_wait = w_reg
        total_service = s_reg

        s_kind = choose(self.rng, P_SURVEY)
        w, s = yield from self.use_resource(self.survey, "survey", t_survey(self.rng, s_kind))
        total_wait += w
        total_service += s

        h_kind = choose(self.rng, P_HEALTH)
        w, s = yield from self.use_resource(self.cubicle, "health", t_health(self.rng, h_kind))
        total_wait += w
        total_service += s

        d_kind = choose(self.rng, P_DONATE)
        w, s = yield from self.use_resource(self.pod, "donate", t_donate(self.rng, d_kind))
        total_wait += w
        total_service += s

        ref_t = t_refresh(self.rng)
        yield self.env.timeout(ref_t)
        total_service += ref_t

        cycle_time = self.env.now - t_cycle_start
        self.client_cycle_times.append(cycle_time)
        self.client_total_waits.append(total_wait)
        self.client_total_services.append(total_service)

        self.donors_completed += 1

    def initial_lineup(self):
        n0 = initial_lineup_size(self.rng)
        for _ in range(n0):
            self.env.process(self.donor())

    def arrivals(self):
        while True:
            ia = exp_interarrival(self.rng, self.mean_interarrival)
            yield self.env.timeout(ia)
            if self.env.now > OPEN_MIN:
                break
            self.env.process(self.donor())

# =========================================================
# RUN ONE CONFIG + ONE SCENARIO (donor-weighted pooling)
# =========================================================
def run_config_scenario(reg_cap, survey_cap, surge, reps=REPS):
    pooled_waits = {st: [] for st in QUEUED_STAGES}
    pooled_cycle = []
    pooled_total_wait = []
    pooled_total_service = []

    pooled_balked_per_day = []
    pooled_completed_per_day = []
    pooled_arrivals_per_hour_per_day = []

    for _ in range(reps):
        rng = random.Random()
        mean_ia = mean_interarrival_minutes_for_day(rng, surge=surge)
        arrivals_per_hour = 60.0 / mean_ia

        env = simpy.Environment()
        clinic = Clinic(env, rng, mean_ia, cap_reg=reg_cap, cap_survey=survey_cap)

        clinic.initial_lineup()
        env.process(clinic.arrivals())
        env.run(until=OPEN_MIN + DAY_RUN_BUFFER)

        # day-level counts (still per day, but reporting mean/CI is fine)
        pooled_balked_per_day.append(clinic.balked_register)
        pooled_completed_per_day.append(clinic.donors_completed)
        pooled_arrivals_per_hour_per_day.append(arrivals_per_hour)

        # donor-level pooling (DONOR-WEIGHTED)
        pooled_cycle.extend(clinic.client_cycle_times)
        pooled_total_wait.extend(clinic.client_total_waits)
        pooled_total_service.extend(clinic.client_total_services)

        for st in QUEUED_STAGES:
            pooled_waits[st].extend(clinic.wait.get(st, []))

    # Donor-weighted KPI: mean waits, cycle, etc.
    kpi = {}

    kpi["n_donors"] = len(pooled_cycle)
    kpi["cycle_mean"], _, _, _ = ci95_over_samples(pooled_cycle)
    kpi["wait_total_mean"], _, _, _ = ci95_over_samples(pooled_total_wait)
    kpi["service_total_mean"], _, _, _ = ci95_over_samples(pooled_total_service)
    kpi["pooled_waits"] = pooled_waits              # <-- add this
    kpi["pooled_total_wait"] = pooled_total_wait    # <-- add this (for P(cycle wait <10) if needed)
    kpi["pooled_cycle"] = pooled_cycle              # <-- add this

    # total waiting to minimize (you can change objective)
    kpi["objective_mean_wait_total"] = kpi["wait_total_mean"]

    # stage means
    for st in QUEUED_STAGES:
        m, _, _, _ = ci95_over_samples(pooled_waits[st])
        kpi[f"wait_{st}_mean"] = m

        # donor-weighted P(wait < 10)
        ws = pooled_waits[st]
        kpi[f"p_wait_lt10_{st}"] = (sum(1 for w in ws if w < 10.0) / len(ws)) if ws else float("nan")

    # day-level for context
    kpi["arrivals_per_hour_mean"], _, _, _ = ci95_over_samples(pooled_arrivals_per_hour_per_day)
    kpi["completed_per_day_mean"], _, _, _ = ci95_over_samples(pooled_completed_per_day)
    kpi["balked_per_day_mean"], _, _, _ = ci95_over_samples(pooled_balked_per_day)

    kpi["p_cycle_wait_lt10"] = (
    sum(1 for w in pooled_total_wait if w < 10.0) / len(pooled_total_wait)
    ) if pooled_total_wait else float("nan")
    # --- % wait out of whole cycle by stage ---
    mean_cycle = kpi["cycle_mean"]

    wait_pct_by_stage = {}
    for st in QUEUED_STAGES:
        mw = kpi[f"wait_{st}_mean"]
        wait_pct_by_stage[st] = (mw / mean_cycle * 100.0) if (mean_cycle > 0 and not math.isnan(mw)) else float("nan")

    kpi["wait_pct_by_stage"] = wait_pct_by_stage

    return kpi

# =========================================================
# GRID SEARCH: find lowest wait time by config (per scenario)
# =========================================================
def evaluate_all_configs():
    all_results = {}  # all_results[scenario_label][(reg,survey)] = kpi

    for sc_label, surge in SCENARIOS.items():
        all_results[sc_label] = {}

        for reg_cap, survey_cap in CONFIGS:
            kpi = run_config_scenario(reg_cap, survey_cap, surge=surge, reps=REPS)
            all_results[sc_label][(reg_cap, survey_cap)] = kpi

    return all_results

def find_best_configs(all_results, objective_key="objective_mean_wait_total"):
    best = {}  # best[scenario] = (config, kpi)

    for sc_label, config_map in all_results.items():
        best_cfg = None
        best_kpi = None
        best_val = float("inf")

        for cfg, kpi in config_map.items():
            val = kpi.get(objective_key, float("inf"))
            if not math.isnan(val) and val < best_val:
                best_val = val
                best_cfg = cfg
                best_kpi = kpi

        best[sc_label] = (best_cfg, best_kpi)

    return best

# =========================================================
# RUN + PRINT
# =========================================================
all_results = evaluate_all_configs()
SELECTED_CONFIG = (3, 4) ##change to see result

for sc_label in SCENARIOS.keys():
    kpi = all_results[sc_label][SELECTED_CONFIG]
    reg_cap, survey_cap = SELECTED_CONFIG

    print("\n" + "=" * 80)
    print(f"CONFIG: REG={reg_cap}, SURVEY={survey_cap} | Scenario: {sc_label}")
    print("=" * 80)

    print("\n[Context]")
    print(f"  Mean arrivals/hour (day avg):   {kpi['arrivals_per_hour_mean']:.2f}")
    print(f"  Mean completed/day:             {kpi['completed_per_day_mean']:.2f}")
    print(f"  Mean balked at register/day:    {kpi['balked_per_day_mean']:.2f}")

    print("\n[Donor-weighted totals]")
    print(f"  Mean total wait per donor:      {kpi['wait_total_mean']:.2f} min")
    print(f"  Mean total service per donor:   {kpi['service_total_mean']:.2f} min")
    print(f"  Mean cycle time per donor:      {kpi['cycle_mean']:.2f} min")

    print("\n[Donor-weighted wait by stage + P(wait < 10)]")
    for st in QUEUED_STAGES:
        print(
            f"  {st.capitalize():9s}: "
            f"wait_mean={kpi[f'wait_{st}_mean']:.2f} min | "
            f"P(wait<10)={kpi[f'p_wait_lt10_{st}']:.3f}"
        )

# Optional: print full table for all configs (baseline only)

# 1) Box plot chart of wait time by stage (use BASELINE scenario, best config there)
base_label = "baseline"
kpi_base = all_results[base_label][SELECTED_CONFIG]

box_data = [kpi_base["pooled_waits"][st] for st in QUEUED_STAGES]

plt.figure(figsize=(9, 5))
plt.boxplot(
    box_data,
    labels=[st.capitalize() for st in QUEUED_STAGES],
    showfliers=True
)
plt.title(
    f"Wait Time by Stage (Boxplot)\n"
    f"Config: REG={SELECTED_CONFIG[0]}, SURVEY={SELECTED_CONFIG[1]} | Baseline"
)
plt.xlabel("Stage")
plt.ylabel("Wait time (minutes)")
plt.tight_layout()
plt.show()


# 2) Bar chart of P(total cycle wait < 10 min) by scenario (best config per scenario)
stages = QUEUED_STAGES
scenarios = list(SCENARIOS.keys())

x = np.arange(len(stages))
width = 0.18

plt.figure(figsize=(11, 4))

for i, sc in enumerate(scenarios):
    pvals = [
        all_results[sc][SELECTED_CONFIG][f"p_wait_lt10_{st}"]
        for st in stages
    ]

    bars = plt.bar(x + i * width, pvals, width, label=sc)

    # labels on bars
    for b in bars:
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            f"{b.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.xticks(
    x + width * (len(scenarios) - 1) / 2,
    [st.capitalize() for st in stages]
)
plt.ylim(0, 1.05)
plt.xlabel("Stage")
plt.ylabel("P(wait < 10 min)")
plt.title(
    f"P(wait < 10 min) by Stage and Scenario\n"
    f"Config: REG={SELECTED_CONFIG[0]}, SURVEY={SELECTED_CONFIG[1]}"
)
plt.legend(title="Scenario")
plt.tight_layout()
plt.show()


#3️ Bar chart: % of cycle spent waiting by stage & scenario
plt.figure(figsize=(12, 8))

for i, sc in enumerate(scenarios, start=1):
    kpi = all_results[sc][SELECTED_CONFIG]

    waits = np.array([kpi[f"wait_{st}_mean"] for st in stages])
    total_wait = waits.sum()
    pct = (waits / total_wait * 100) if total_wait > 0 else np.zeros_like(waits)

    plt.subplot(2, 2, i)
    plt.pie(
        pct,
        autopct="%1.1f%%",
        startangle=90,
        labeldistance=1.15,
        pctdistance=0.7
    )
    plt.title(f"{sc} scenario")

plt.legend(
    [st.capitalize() for st in stages],
    loc="center right",
    title="Stage"
)

plt.suptitle(
    f"%Waiting Time Spent by Stage\n"
    f"Config: REG={SELECTED_CONFIG[0]}, SURVEY={SELECTED_CONFIG[1]}",
    fontsize=14
)
plt.tight_layout()
plt.show()
