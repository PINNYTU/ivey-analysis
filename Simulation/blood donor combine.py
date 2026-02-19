import simpy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# GLOBAL PARAMETERS
# =========================================================
DAYS = 1000
DAY_LENGTH = 300  # minutes per day

SCENARIOS = {
    "Baseline": 1.00,
    "+5%": 1.05,
    "+10%": 1.10,
    "+15%": 1.15,
    "+20%": 1.20,
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
# UPDATED ASSUMPTIONS (REPLACED)
# =========================================================
INITIAL_LINEUP_TRI = (5, 7.5, 10)  # (low, mode, high)

# Interarrival intensity per day: random between 25 and 30 arrivals/hour (fixed for that day)
ARRIVALS_PER_HR_RANGE = (25, 30)

# Branching probabilities
P_REGISTER = [("returning", 0.50), ("first", 0.30), ("issue", 0.20)]
P_SURVEY   = [("no_travel", 0.75), ("travel", 0.25)]
P_HEALTH   = [("regular", 0.95), ("extreme", 0.05)]
P_DONATE   = [("regular", 0.97), ("extreme", 0.03)]

# =========================================================
# UTILS
# =========================================================
def draw_branch(dist):
    r = random.random()
    cum = 0.0
    for label, p in dist:
        cum += p
        if r <= cum:
            return label
    return dist[-1][0]  # safe fallback

# =========================================================
# SERVICE TIME FUNCTIONS (MINUTES)
# =========================================================
def registration_time():
    reg_type = draw_branch(P_REGISTER)
    if reg_type == "returning":
        return 1
    elif reg_type == "first":
        return 5
    return random.uniform(5, 10)  # issue

def survey_time():
    return 10 if draw_branch(P_SURVEY) == "travel" else random.uniform(5.5, 6)

def screening_time():
    return random.uniform(13, 30) if draw_branch(P_HEALTH) == "extreme" else random.uniform(10, 12)

def donation_time():
    return random.uniform(16, 30) if draw_branch(P_DONATE) == "extreme" else random.uniform(14, 15)

def refresh_time():
    return random.uniform(5, 10)

# =========================================================
# DONOR PROCESS (logs raw registration events)
# =========================================================
def donor(env, clinic, stats, donor_id, reg_log):

    # -------------------------
    # Registration (RAW LOG + CLOSE-TIME BALKING)
    # -------------------------
    time_at_lining = env.now
    remaining_to_close = max(DAY_LENGTH - time_at_lining, 0)

    with clinic['reg'].request() as req:
        outcome = yield req | env.timeout(remaining_to_close)

        # If we hit closing before getting a desk -> donor leaves (balks)
        if req not in outcome:
            stats['balked_reg'] += 1
            reg_log.append({
                "donor": donor_id,
                "time_at_lining": round(time_at_lining, 2),
                "register_in": np.nan,
                "register_out": np.nan,
                "total_wait_to_register": round(remaining_to_close, 2),  # waited until close
                "status": "balked_at_close",
            })
            return  # donor exits system

        # Got a desk before close
        register_in = env.now
        w_reg = register_in - time_at_lining

        s_reg = registration_time()
        yield env.timeout(s_reg)
        register_out = env.now

    reg_log.append({
        "donor": donor_id,
        "time_at_lining": round(time_at_lining, 2),
        "register_in": round(register_in, 2),
        "register_out": round(register_out, 2),
        "total_wait_to_register": round(w_reg, 2),
        "status": "served",
    })


    # Survey
    arrival = env.now
    with clinic['survey'].request() as req:
        yield req
        w_sur = env.now - arrival
        s_sur = survey_time()
        yield env.timeout(s_sur)

    # Screening
    arrival = env.now
    with clinic['screen'].request() as req:
        yield req
        w_scr = env.now - arrival
        s_scr = screening_time()
        yield env.timeout(s_scr)

    # Donation
    arrival = env.now
    with clinic['donation'].request() as req:
        yield req
        w_don = env.now - arrival
        s_don = donation_time()
        yield env.timeout(s_don)

    # Refresh (no waiting)
    s_ref = refresh_time()
    yield env.timeout(s_ref)

    total_wait = w_reg + w_sur + w_scr + w_don
    total_service = s_reg + s_sur + s_scr + s_don + s_ref
    cycle = total_wait + total_service

    # Record stats
    stats['cycle'].append(cycle)
    stats['wait_total'].append(total_wait)
    stats['service_total'].append(total_service)

    stats['wait_reg'].append(w_reg)
    stats['wait_survey'].append(w_sur)
    stats['wait_screen'].append(w_scr)
    stats['wait_donation'].append(w_don)

# =========================================================
# ARRIVALS (arr_# donor IDs)
# =========================================================
def arrivals(env, clinic, stats, rate_per_hr, id_state, reg_log):
    mean_iat = 60 / rate_per_hr  # minutes
    while env.now < DAY_LENGTH:
        yield env.timeout(random.expovariate(1 / mean_iat))
        id_state["arr"] += 1
        donor_id = f"arr_{id_state['arr']}"
        env.process(donor(env, clinic, stats, donor_id, reg_log))

# =========================================================
# RUN ONE DAY
# =========================================================
def run_day(multiplier, reg_cap, survey_cap, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = simpy.Environment()

    # fixed for the day
    daily_rate_per_hr = random.uniform(*ARRIVALS_PER_HR_RANGE) * multiplier

    clinic = {
        'reg': simpy.Resource(env, reg_cap),
        'survey': simpy.Resource(env, survey_cap),
        'screen': simpy.Resource(env, 6),
        'donation': simpy.Resource(env, 8),
    }

    stats = {
        'cycle': [],
        'wait_total': [],
        'service_total': [],
        'wait_reg': [],
        'wait_survey': [],
        'wait_screen': [],
        'wait_donation': [],
        'balked_reg': 0,   
    }

    reg_log = []
    id_state = {"init": 0, "arr": 0}

    # Initial lineup at opening (triangular)
    init_n = int(round(random.triangular(*INITIAL_LINEUP_TRI)))
    init_n = max(init_n, 0)

    for _ in range(init_n):
        id_state["init"] += 1
        donor_id = f"init_{id_state['init']}"
        env.process(donor(env, clinic, stats, donor_id, reg_log))

    env.process(arrivals(env, clinic, stats, daily_rate_per_hr, id_state, reg_log))
    env.run(until=DAY_LENGTH)

    reg_raw_df = pd.DataFrame(reg_log)
    return stats, reg_raw_df

# =========================================================
# RUN ALL SIMULATIONS (KPIs)
# =========================================================
results = {}

for reg, survey in CONFIGS:
    key = f"REG={reg}, SURVEY={survey}"
    results[key] = {}

    for sc, mult in SCENARIOS.items():
        agg = {k: [] for k in [
            'cycle', 'wait_total', 'service_total',
            'wait_reg', 'wait_survey', 'wait_screen', 'wait_donation'
        ]}

        for _ in range(DAYS):
            day_stats, _ = run_day(mult, reg, survey)
            for k in agg:
                agg[k].extend(day_stats[k])

        results[key][sc] = {
            "Cycle(min)": np.mean(agg['cycle']) if agg['cycle'] else np.nan,
            "Wait(min)": np.mean(agg['wait_total']) if agg['wait_total'] else np.nan,
            "Service(min)": np.mean(agg['service_total']) if agg['service_total'] else np.nan,
            "Wait%": (np.mean(agg['wait_total']) / np.mean(agg['cycle']) * 100)
                     if agg['wait_total'] and agg['cycle'] else np.nan,
            "RegWait": np.mean(agg['wait_reg']) if agg['wait_reg'] else np.nan,
            "SurveyWait": np.mean(agg['wait_survey']) if agg['wait_survey'] else np.nan,
            "ScreenWait": np.mean(agg['wait_screen']) if agg['wait_screen'] else np.nan,
            "DonationWait": np.mean(agg['wait_donation']) if agg['wait_donation'] else np.nan,
        }

# =========================================================
# PRINT KPI TABLES (ALL KEY VARIABLES)
# =========================================================
for cfg in results:
    print(f"\n=== KPIs: {cfg} ===")
    print(
        "Scenario | Cycle(min) | Wait(min) | Service(min) | Wait% | "
        "RegWait | SurveyWait | ScreenWait | DonationWait"
    )
    print("-" * 100)

    for sc in SCENARIOS:
        v = results[cfg][sc]
        print(
            f"{sc:8s} | "
            f"{v['Cycle(min)']:10.1f} | "
            f"{v['Wait(min)']:9.1f} | "
            f"{v['Service(min)']:12.1f} | "
            f"{v['Wait%']:5.1f}% | "
            f"{v['RegWait']:7.1f} | "
            f"{v['SurveyWait']:10.2f} | "
            f"{v['ScreenWait']:10.2f} | "
            f"{v['DonationWait']:12.2f}"
        )

# =========================================================
# SAVE ONE-DAY RAW REGISTER DATA (RECOMMENDED CONFIG)
# =========================================================
CONFIG_KEY = "REG=3, SURVEY=4"
SCENARIO_NAME = "Baseline"

reg_cap, survey_cap = 1, 3
mult = SCENARIOS[SCENARIO_NAME]

day_stats, reg_raw_day = run_day(mult, reg_cap, survey_cap, seed=42)

print("\n--- Raw registration data sample (first 20 rows) ---")
print(reg_raw_day.head(20))

out_csv = "register_raw_day_final.csv"
reg_raw_day.to_csv(out_csv, index=False)
print(f"\nSaved one-day raw register data to: {out_csv}")

print("Max reg wait:", reg_raw_day["total_wait_to_register"].max())
print("Balked at close:", (reg_raw_day["status"] == "balked_at_close").sum())

# =========================================================
# FINAL PLOTS (RECOMMENDED CONFIG ONLY)
# =========================================================
scenarios = list(SCENARIOS.keys())
x = np.arange(len(scenarios))

cycle_vals = [results[CONFIG_KEY][s]["Cycle(min)"] for s in scenarios]
wait_pct = [results[CONFIG_KEY][s]["Wait%"] for s in scenarios]

def annotate(ax, bars, fmt):
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.6,
            fmt.format(b.get_height()),
            ha="center",
            va="bottom",
            fontsize=10
        )

# --- Cycle Time Plot ---
plt.figure(figsize=(10, 5))
ax = plt.gca()
bars = ax.bar(x, cycle_vals, edgecolor="black", linewidth=1.2)
annotate(ax, bars, "{:.1f}")
ax.set_xticks(x)
ax.set_xticklabels(scenarios)
ax.set_ylabel("Average Cycle Time (minutes)")
ax.set_title("Cycle Time by Demand Scenario\n(3 Registration, 4 Survey)")
plt.tight_layout()
plt.show()

# --- Waiting % Plot ---
plt.figure(figsize=(10, 5))
ax = plt.gca()
bars = ax.bar(x, wait_pct, edgecolor="black", linewidth=1.2)
annotate(ax, bars, "{:.1f}%")
ax.set_xticks(x)
ax.set_xticklabels(scenarios)
ax.set_ylabel("Waiting Time (% of Total)")
ax.set_title("Waiting Time Percentage by Scenario\n(3 Registration, 4 Survey)")
plt.tight_layout()
plt.show()
