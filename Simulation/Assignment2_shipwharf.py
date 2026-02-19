# ============================================================
# Scenarios (3 wharfs, one-season distributions):
#   Baseline: 135 ships, load 2–3
#   Scenario 1: ships +10–20% (per season), load 2–3
#   Scenario 2: 135 ships, load 2–5 (delayed)
#   Scenario 3: ships +10–20% (per season), load 2–5 (delayed)
#
# Outputs:
#   - Baseline (Year 1): mean wait + total cost (95% CI)
#   - 5-year analyses (year-by-year): baseline + scenario1 + scenario2 + scenario3
#   - Annual (one-season) distributions (3 wharfs): cost + utilization + risk(wait>3d)
#   - Plots:
#       * cumulative net cash flow (investment in Year 1) for each 5-yr scenario
#       * box plot: annual costs (3W) for the 4 scenarios
#       * bar chart: utilization mean + 95% CI (3W) for the 4 scenarios
#       * bar chart: operational risk P(wait>3 days) mean + 95% CI (3W)
# ============================================================

import simpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, PercentFormatter

# ============================================================
# Core ship + arrival processes (Excel-style reservation)
# ============================================================

def ship_process(env, wharf_out, lock, load_min, load_max, rng, tracker, wait_cost_per_day):
    arrival_time = env.now

    # Reservation lock = spreadsheet row-by-row behavior
    with lock.request() as req:
        yield req

        # choose earliest-available wharf
        i = int(np.argmin(wharf_out))
        earliest_out = wharf_out[i]

        # waiting time
        wait_time = max(0.0, earliest_out - arrival_time)

        # loading time distribution (Triangular)
        loading_time = rng.triangular(left=load_min, mode=(load_min + load_max) / 2.0, right=load_max)

        # reserve chosen wharf immediately
        start_service = arrival_time + wait_time
        finish_time = start_service + loading_time
        wharf_out[i] = finish_time

        # track
        tracker["ships_arrived"] += 1
        tracker["total_wait_time"] += wait_time
        tracker["total_cost"] += wait_time * wait_cost_per_day
        tracker["wait_samples"].append(wait_time)
        tracker["loading_samples"].append(loading_time)

    # execute timeline
    if wait_time > 0:
        yield env.timeout(wait_time)
    yield env.timeout(loading_time)


def arrival_generator(env, days, mean_interarrival, wharf_out, lock, load_min, load_max, rng, tracker, wait_cost_per_day):
    while True:
        interarrival = rng.exponential(mean_interarrival)
        tracker["interarrival_samples"].append(interarrival)

        yield env.timeout(interarrival)
        if env.now > days:
            break

        env.process(
            ship_process(env, wharf_out, lock, load_min, load_max, rng, tracker, wait_cost_per_day)
        )

# ============================================================
# Single simulation run
# ============================================================

def simulate_one_run(
    n_wharf=2,
    days=240.0,
    ships=135,
    load_min=2.0,
    load_max=3.0,
    wait_cost_per_day=2000.0,
    seed=None
):
    rng = np.random.default_rng(seed)
    mean_interarrival = days / ships

    env = simpy.Environment()
    wharf_out = np.zeros(n_wharf, dtype=float)
    lock = simpy.Resource(env, capacity=1)

    tracker = {
        "ships_arrived": 0,
        "total_wait_time": 0.0,
        "total_cost": 0.0,
        "interarrival_samples": [],
        "loading_samples": [],
        "wait_samples": []
    }

    env.process(arrival_generator(
        env, days, mean_interarrival, wharf_out, lock,
        load_min, load_max, rng, tracker, wait_cost_per_day
    ))
    env.run(until=days)

    ships_arr = tracker["ships_arrived"]
    tracker["mean_wait_per_ship"] = tracker["total_wait_time"] / ships_arr if ships_arr else 0.0
    tracker["mean_load_per_ship"] = float(np.mean(tracker["loading_samples"])) if tracker["loading_samples"] else 0.0
    tracker["utilization"] = (float(np.sum(tracker["loading_samples"])) / (n_wharf * days)) if days > 0 else 0.0

    return tracker

# ============================================================
# Monte Carlo helpers
# ============================================================

def mean_ci(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = x.mean()
    std = x.std(ddof=1)
    se = std / np.sqrt(n)
    z = 1.96
    moe = z * se
    return mean, (mean - moe, mean + moe), moe


def mean_ci_95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    s = x.std(ddof=1)
    se = s / np.sqrt(n)
    z = 1.96
    return m, (m - z * se), (m + z * se)


def p95(x):
    return float(np.percentile(np.asarray(x, dtype=float), 95))


def prob_wait_gt(wait_samples, threshold=3.0):
    wait_samples = np.asarray(wait_samples, dtype=float)
    if wait_samples.size == 0:
        return 0.0
    return float(np.mean(wait_samples > threshold))


def run_replications(replications, base_seed, n_wharf, days, ships, load_min, load_max, wait_cost_per_day):
    """
    Returns arrays per replication:
      - mean wait per ship
      - total cost (season)
      - mean loading time per ship
      - utilization
      - all wait samples (concatenated) for risk calc
    """
    mean_waits = np.empty(replications, dtype=float)
    total_costs = np.empty(replications, dtype=float)
    mean_loads = np.empty(replications, dtype=float)
    utils = np.empty(replications, dtype=float)
    all_wait_samples = []

    for i in range(replications):
        tr = simulate_one_run(
            n_wharf=n_wharf,
            days=days,
            ships=ships,
            load_min=load_min,
            load_max=load_max,
            wait_cost_per_day=wait_cost_per_day,
            seed=base_seed + i
        )
        mean_waits[i] = tr["mean_wait_per_ship"]
        total_costs[i] = tr["total_cost"]
        mean_loads[i] = tr["mean_load_per_ship"]
        utils[i] = tr["utilization"]
        all_wait_samples.extend(tr["wait_samples"])

    return mean_waits, total_costs, mean_loads, utils, np.array(all_wait_samples, dtype=float)


def compare_2_vs_3(replications, seed_block, days, ships,
                   load_min_2, load_max_2,
                   load_min_3, load_max_3,
                   wait_cost_per_day):
    # paired seeds inside a year/season block
    w2, c2, l2, u2, waits2 = run_replications(replications, seed_block, 2, days, ships, load_min_2, load_max_2, wait_cost_per_day)
    w3, c3, l3, u3, waits3 = run_replications(replications, seed_block, 3, days, ships, load_min_3, load_max_3, wait_cost_per_day)

    cost_saved = c2 - c3
    time_saved = w2 - w3

    return (w2, c2, l2, u2, waits2), (w3, c3, l3, u3, waits3), time_saved, cost_saved

# ============================================================
# Ship schedule
# ============================================================

def ships_schedule_random_growth_non_compounded(base_ships, years, rng, low=0.10, high=0.20, round_to_int=True):
    """
    Each year independently: ships_y = base_ships*(1 + U(low,high)), NOT compounded.
    """
    ships_list = []
    for _ in range(years):
        g = rng.uniform(low, high)
        ships = base_ships * (1.0 + g)
        ships_list.append(int(round(ships)) if round_to_int else ships)
    return ships_list

# ============================================================
# 5-year analyses
# ============================================================

def contract_analysis_fixed_volume(
    base_ships=135,
    years=5,
    wharf_cost=1_500_000,
    replications=1000,
    base_seed=42,
    days=240.0,
    load_min=2.0,
    load_max=3.0,
    wait_cost_per_day=2000.0
):
    rows = []
    cum_savings = 0.0

    for year in range(1, years + 1):
        ships_y = int(base_ships)
        seed_block = base_seed + year * 100_000

        (w2, c2, l2, u2, waits2), (w3, c3, l3, u3, waits3), _, cost_saved = compare_2_vs_3(
            replications, seed_block, days, ships_y,
            load_min, load_max,
            load_min, load_max,
            wait_cost_per_day
        )

        w2_m, w2_ci, _ = mean_ci(w2)
        w3_m, w3_ci, _ = mean_ci(w3)

        c2_m, c2_ci, _ = mean_ci(c2)
        c3_m, c3_ci, _ = mean_ci(c3)

        s_m, s_ci, _ = mean_ci(cost_saved)

        l2_m, l2_ci, _ = mean_ci(l2)
        l3_m, l3_ci, _ = mean_ci(l3)

        u2_m, u2_ci, _ = mean_ci(u2)
        u3_m, u3_ci, _ = mean_ci(u3)

        risk3 = float(np.mean(waits3 > 3.0)) if waits3.size else 0.0

        cum_savings += s_m

        rows.append({
            "year": year,
            "ships": ships_y,

            "mean_wait_2": w2_m, "wait2_ci_low": w2_ci[0], "wait2_ci_high": w2_ci[1],
            "mean_wait_3": w3_m, "wait3_ci_low": w3_ci[0], "wait3_ci_high": w3_ci[1],

            "mean_load_2": l2_m, "load2_ci_low": l2_ci[0], "load2_ci_high": l2_ci[1],
            "mean_load_3": l3_m, "load3_ci_low": l3_ci[0], "load3_ci_high": l3_ci[1],

            "mean_util_2": u2_m, "util2_ci_low": u2_ci[0], "util2_ci_high": u2_ci[1],
            "mean_util_3": u3_m, "util3_ci_low": u3_ci[0], "util3_ci_high": u3_ci[1],

            "risk_wait_gt_3_3wharf": risk3,

            "mean_cost_2": c2_m, "cost2_ci_low": c2_ci[0], "cost2_ci_high": c2_ci[1],
            "mean_cost_3": c3_m, "cost3_ci_low": c3_ci[0], "cost3_ci_high": c3_ci[1],

            "mean_savings": s_m, "savings_ci_low": s_ci[0], "savings_ci_high": s_ci[1],
            "cum_savings": cum_savings
        })

    total_savings = cum_savings
    total_net = total_savings - wharf_cost
    return rows, total_savings, total_net


def contract_analysis_with_scenarios(
    ships_by_year,
    wharf_cost=1_500_000,
    replications=1000,
    base_seed=42,
    days=240.0,
    load_min_2=2.0, load_max_2=3.0,
    load_min_3=2.0, load_max_3=3.0,
    wait_cost_per_day=2000.0
):
    years = len(ships_by_year)
    rows = []
    cum_savings = 0.0

    for year in range(1, years + 1):
        ships_y = int(ships_by_year[year - 1])
        seed_block = base_seed + year * 100_000

        (w2, c2, l2, u2, waits2), (w3, c3, l3, u3, waits3), _, cost_saved = compare_2_vs_3(
            replications, seed_block, days, ships_y,
            load_min_2, load_max_2,
            load_min_3, load_max_3,
            wait_cost_per_day
        )

        w2_m, w2_ci, _ = mean_ci(w2)
        w3_m, w3_ci, _ = mean_ci(w3)

        c2_m, c2_ci, _ = mean_ci(c2)
        c3_m, c3_ci, _ = mean_ci(c3)

        s_m, s_ci, _ = mean_ci(cost_saved)

        l2_m, l2_ci, _ = mean_ci(l2)
        l3_m, l3_ci, _ = mean_ci(l3)

        u2_m, u2_ci, _ = mean_ci(u2)
        u3_m, u3_ci, _ = mean_ci(u3)

        risk3 = float(np.mean(waits3 > 3.0)) if waits3.size else 0.0

        cum_savings += s_m

        rows.append({
            "year": year,
            "ships": ships_y,

            "mean_wait_2": w2_m, "wait2_ci_low": w2_ci[0], "wait2_ci_high": w2_ci[1],
            "mean_wait_3": w3_m, "wait3_ci_low": w3_ci[0], "wait3_ci_high": w3_ci[1],

            "mean_load_2": l2_m, "load2_ci_low": l2_ci[0], "load2_ci_high": l2_ci[1],
            "mean_load_3": l3_m, "load3_ci_low": l3_ci[0], "load3_ci_high": l3_ci[1],

            "mean_util_2": u2_m, "util2_ci_low": u2_ci[0], "util2_ci_high": u2_ci[1],
            "mean_util_3": u3_m, "util3_ci_low": u3_ci[0], "util3_ci_high": u3_ci[1],

            "risk_wait_gt_3_3wharf": risk3,

            "mean_cost_2": c2_m, "cost2_ci_low": c2_ci[0], "cost2_ci_high": c2_ci[1],
            "mean_cost_3": c3_m, "cost3_ci_low": c3_ci[0], "cost3_ci_high": c3_ci[1],

            "mean_savings": s_m, "savings_ci_low": s_ci[0], "savings_ci_high": s_ci[1],
            "cum_savings": cum_savings
        })

    total_savings = cum_savings
    total_net = total_savings - wharf_cost
    return rows, total_savings, total_net

# ============================================================
# Annual (one-season) collectors for distributions (3 wharfs)
# ============================================================

def collect_annual_metrics_fixed_ships(replications, base_seed, n_wharf, days, ships, load_min, load_max, wait_cost_per_day):
    costs = np.empty(replications, dtype=float)
    utils = np.empty(replications, dtype=float)
    risks = np.empty(replications, dtype=float)

    for i in range(replications):
        tr = simulate_one_run(
            n_wharf=n_wharf, days=days, ships=ships,
            load_min=load_min, load_max=load_max,
            wait_cost_per_day=wait_cost_per_day,
            seed=base_seed + i
        )
        costs[i] = tr["total_cost"]
        utils[i] = tr["utilization"]
        waits = np.asarray(tr["wait_samples"], dtype=float)
        risks[i] = float(np.mean(waits > 3.0)) if waits.size else 0.0
    
    return costs, utils, risks


def collect_annual_metrics_growth_non_cumulative(replications, base_seed, ship_seed,
                                                n_wharf, days, base_ships, low, high,
                                                load_min, load_max, wait_cost_per_day):
    rng = np.random.default_rng(ship_seed)
    costs = np.empty(replications, dtype=float)
    utils = np.empty(replications, dtype=float)
    risks = np.empty(replications, dtype=float)
    
    for i in range(replications):
        ships_i = int(round(base_ships * (1.0 + rng.uniform(low, high))))

        tr = simulate_one_run(
            n_wharf=n_wharf, days=days, ships=ships_i,
            load_min=load_min, load_max=load_max,
            wait_cost_per_day=wait_cost_per_day,
            seed=base_seed + i
        )
        costs[i] = tr["total_cost"]
        utils[i] = tr["utilization"]
        waits = np.asarray(tr["wait_samples"], dtype=float)
        risks[i] = float(np.mean(waits > 3.0)) if waits.size else 0.0

    return costs, utils, risks

# ============================================================
# Summaries
# ============================================================

def print_scenario_summary(name, costs, utils):
    c_m, c_lo, c_hi = mean_ci_95(costs)
    u_m, u_lo, u_hi = mean_ci_95(utils)

    print(f"\n=== {name} ===")
    print(f"Annual cost: mean=${c_m:,.0f} | 95% CI (${c_lo:,.0f}, ${c_hi:,.0f}) | P95=${p95(costs):,.0f}")
    print(f"Utilization: mean={u_m:.1%} | 95% CI ({u_lo:.1%}, {u_hi:.1%}) | P95={p95(utils):.1%}")

# ============================================================
# Plots
# ============================================================

def plot_cumulative_net_year1_investment(yearly_rows, wharf_cost, title):
    years = [row["year"] for row in yearly_rows]
    cum_net = []
    running = 0.0

    for i, row in enumerate(yearly_rows):
        running += row["mean_savings"]
        if i == 0:
            running -= wharf_cost
        cum_net.append(running)

    colors = ["red" if v < 0 else "green" for v in cum_net]
    plt.figure()
    plt.bar(years, cum_net, color=colors)
    plt.axhline(0)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Cumulative net cash flow ($)")
    plt.xticks(years)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


def plot_cost_boxplot(costs_list, labels, title):
    plt.figure(figsize=(10, 5))
    plt.boxplot(costs_list, labels=labels, showfliers=True)
    plt.title(title)
    plt.ylabel("Annual cost ($)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
    plt.tight_layout()
    plt.show()


def plot_utilization_barchart(utils_list, labels, title):
    means, yerr_low, yerr_high = [], [], []

    for u in utils_list:
        m, lo, hi = mean_ci_95(u)
        means.append(m)
        yerr_low.append(m - lo)
        yerr_high.append(hi - m)

    x = np.arange(len(labels))
    plt.figure(figsize=(10, 5))
    plt.bar(x, means, yerr=[yerr_low, yerr_high], capsize=6)
    plt.xticks(x, labels)
    plt.title(title)
    plt.ylabel("Utilization")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_operational_risk_barchart(risk_list, labels, title):
    means, yerr_low, yerr_high = [], [], []

    for r in risk_list:
        m, lo, hi = mean_ci_95(r)
        means.append(m)
        yerr_low.append(m - lo)
        yerr_high.append(hi - m)

    x = np.arange(len(labels))
    plt.figure(figsize=(10, 5))
    plt.bar(x, means, yerr=[yerr_low, yerr_high], capsize=6)
    plt.xticks(x, labels)
    plt.title(title)
    plt.ylabel("P(wait > 3 days)")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_savings_barchart(savings_list, labels, title):
    means, yerr_low, yerr_high = [], [], []

    for s in savings_list:
        m, lo, hi = mean_ci_95(s)
        means.append(m)
        yerr_low.append(m - lo)
        yerr_high.append(hi - m)

    x = np.arange(len(labels))
    plt.figure(figsize=(10, 5))
    plt.bar(x, means, yerr=[yerr_low, yerr_high], capsize=6)
    plt.xticks(x, labels)
    plt.title(title)
    plt.ylabel("Mean annual savings vs Baseline ($)")
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================
# Run it
# ============================================================

if __name__ == "__main__":
    # Base inputs
    DAYS = 240.0
    BASE_SHIPS = 135
    LOAD_MIN = 2.0
    LOAD_MAX = 3.0
    WAIT_COST_PER_DAY = 2000.0

    # Scenario 2/3 delayed loading (3 wharfs)
    DELAY_LOAD_MIN_3 = 2.0
    DELAY_LOAD_MAX_3 = 5.0

    # Contract inputs
    YEARS = 5
    WHARF_COST = 1_500_000

    # Simulation settings
    REPS = 1000
    SEED = 42

    labels = [
        "Baseline\n(3W,135,2–3)",
        "Scenario 1\n(+10–20%)",
        "Scenario 2\n(3W,2–5)",
        "Scenario 3\n(3W,2–5,+10–20%)",
    ]

    # -----------------------------
    # Baseline (Year 1 only) wait+cost with CI (2 vs 3 wharfs)
    # -----------------------------
    (w2, c2, *_), (w3, c3, *_), time_saved, cost_saved = compare_2_vs_3(
        REPS, SEED, DAYS, BASE_SHIPS,
        LOAD_MIN, LOAD_MAX,
        LOAD_MIN, LOAD_MAX,
        WAIT_COST_PER_DAY
    )

    w2_m, w2_ci, _ = mean_ci(w2)
    w3_m, w3_ci, _ = mean_ci(w3)
    ws_m, ws_ci, _ = mean_ci(time_saved)

    c2_m, c2_ci, _ = mean_ci(c2)
    c3_m, c3_ci, _ = mean_ci(c3)
    cs_m, cs_ci, _ = mean_ci(cost_saved)

    print("\n=================")
    print("\n=== BASELINE (Year 1) MEAN WAIT PER SHIP (days) ===")
    print(f"2 wharfs: {w2_m:,.2f} | 95% CI ({w2_ci[0]:,.2f}, {w2_ci[1]:,.2f})")
    print(f"3 wharfs: {w3_m:,.2f} | 95% CI ({w3_ci[0]:,.2f}, {w3_ci[1]:,.2f})")
    print(f"Time saved (2-3): {ws_m:,.2f} | 95% CI ({ws_ci[0]:,.2f}, {ws_ci[1]:,.2f})")

    print("\n=== BASELINE (Year 1) TOTAL WAITING COST ($ over 240 days) ===")
    print(f"2 wharfs: ${c2_m:,.0f} | 95% CI (${c2_ci[0]:,.0f}, ${c2_ci[1]:,.0f})")
    print(f"3 wharfs: ${c3_m:,.0f} | 95% CI (${c3_ci[0]:,.0f}, ${c3_ci[1]:,.0f})")
    print(f"Cost saved (2-3): ${cs_m:,.0f} | 95% CI (${cs_ci[0]:,.0f}, ${cs_ci[1]:,.0f})")

    # -----------------------------
    # 5-year analyses (2 vs 3 wharfs)
    # -----------------------------
    yearly_base, base_total_savings, base_total_net = contract_analysis_fixed_volume(
        base_ships=BASE_SHIPS,
        years=YEARS,
        wharf_cost=WHARF_COST,
        replications=REPS,
        base_seed=SEED,
        days=DAYS,
        load_min=LOAD_MIN,
        load_max=LOAD_MAX,
        wait_cost_per_day=WAIT_COST_PER_DAY
    )

    rng_year = np.random.default_rng(999)
    ships_random_years = ships_schedule_random_growth_non_compounded(BASE_SHIPS, YEARS, rng_year, low=0.10, high=0.20)

    yearly_s1, s1_total_savings, s1_total_net = contract_analysis_with_scenarios(
        ships_by_year=ships_random_years,
        wharf_cost=WHARF_COST,
        replications=REPS,
        base_seed=SEED,
        days=DAYS,
        load_min_2=LOAD_MIN, load_max_2=LOAD_MAX,
        load_min_3=LOAD_MIN, load_max_3=LOAD_MAX,
        wait_cost_per_day=WAIT_COST_PER_DAY
    )

    ships_fixed_years = [BASE_SHIPS for _ in range(YEARS)]
    yearly_s2, s2_total_savings, s2_total_net = contract_analysis_with_scenarios(
        ships_by_year=ships_fixed_years,
        wharf_cost=WHARF_COST,
        replications=REPS,
        base_seed=SEED,
        days=DAYS,
        load_min_2=LOAD_MIN, load_max_2=LOAD_MAX,
        load_min_3=DELAY_LOAD_MIN_3, load_max_3=DELAY_LOAD_MAX_3,
        wait_cost_per_day=WAIT_COST_PER_DAY
    )

    # Scenario 3: growth + delayed
    rng_year2 = np.random.default_rng(999)
    ships_random_years_2 = ships_schedule_random_growth_non_compounded(BASE_SHIPS, YEARS, rng_year2, low=0.10, high=0.20)
    yearly_s3, s3_total_savings, s3_total_net = contract_analysis_with_scenarios(
        ships_by_year=ships_random_years_2,
        wharf_cost=WHARF_COST,
        replications=REPS,
        base_seed=SEED,
        days=DAYS,
        load_min_2=LOAD_MIN, load_max_2=LOAD_MAX,
        load_min_3=DELAY_LOAD_MIN_3, load_max_3=DELAY_LOAD_MAX_3,
        wait_cost_per_day=WAIT_COST_PER_DAY
    )

    # -----------------------------
    # One-season (annual) distributions for 3 wharfs
    # -----------------------------
    # --- Pre-construction (2W, 115 ships, load 2–3) one-season distributions ---
    costs_pre, utils_pre, risks_pre = collect_annual_metrics_fixed_ships(
    replications=REPS,
    base_seed=SEED,
    n_wharf=2,
    days=DAYS,
    ships=115,
    load_min=2.0,
    load_max=3.0,
    wait_cost_per_day=WAIT_COST_PER_DAY
    )

    costs_base, utils_base, risks_base = collect_annual_metrics_fixed_ships(
        REPS, SEED, 3, DAYS, BASE_SHIPS, 2.0, 3.0, WAIT_COST_PER_DAY
    )
    costs_s1, utils_s1, risks_s1 = collect_annual_metrics_growth_non_cumulative(
        REPS, SEED, 999, 3, DAYS, BASE_SHIPS, 0.10, 0.20, 2.0, 3.0, WAIT_COST_PER_DAY
    )
    costs_s2, utils_s2, risks_s2 = collect_annual_metrics_fixed_ships(
        REPS, SEED, 3, DAYS, BASE_SHIPS, DELAY_LOAD_MIN_3, DELAY_LOAD_MAX_3, WAIT_COST_PER_DAY
    )
    costs_s3, utils_s3, risks_s3 = collect_annual_metrics_growth_non_cumulative(
        REPS, SEED, 999, 3, DAYS, BASE_SHIPS, 0.10, 0.20, DELAY_LOAD_MIN_3, DELAY_LOAD_MAX_3, WAIT_COST_PER_DAY
    )
    print_scenario_summary("Preconstruc (2W, 115 ships, load 2–3)", costs_pre, utils_pre)
    print_scenario_summary("Baseline (3W, 135 ships, load 2–3)", costs_base, utils_base)
    print_scenario_summary("Scenario 1 (3W, ships +10–20%, load 2–3)", costs_s1, utils_s1)
    print_scenario_summary("Scenario 2 (3W, 135 ships, load 2–5)", costs_s2, utils_s2)
    print_scenario_summary("Scenario 3 (3W, ships +10–20%, load 2–5)", costs_s3, utils_s3)

    # -----------------------------
    # Cost + utilization + risk plots
    # -----------------------------
    plot_cost_boxplot(
        [costs_base, costs_s1, costs_s2, costs_s3],
        labels,
        "Annual Demurrage Cost — Box Plot (3 wharfs)"
    )

    plot_utilization_barchart(
        [utils_base, utils_s1, utils_s2, utils_s3],
        labels,
        "Wharf Utilization — Mean with 95% CI (3 wharfs)"
    )


    # (Optional) Print summary for pre-construction risk
    r_m, r_lo, r_hi = mean_ci_95(risks_pre)
    print(
    f"\n=== Pre-construction (2W,115,2–3) Operational Risk ===\n"
    f"P(wait>3d): mean={r_m:.1%} | 95% CI ({r_lo:.1%}, {r_hi:.1%}) | "
    f"P95={np.percentile(risks_pre,95):.1%}"
    )

    # --- Update your labels to include Pre-construction ---
    labels_risk = [
    "Pre-contract\n(2W,115,2–3)",
    "Baseline\n(3W,135,2–3)",
    "S1\n(+10–20%,2–3)",
    "S2\n(3W,2–5)",
    "S3\n(+10–20%,2–5)",
    ]

    # --- Plot risk bars (mean + 95% CI) including pre-construction ---
    plot_operational_risk_barchart(
        [risks_pre, risks_base, risks_s1, risks_s2, risks_s3],
        labels_risk,
        "Operational Risk — Probability a ship waits > 3 days"
    )

# -----------------------------
# 5-year AVERAGE ANNUAL COST plot (NOT savings)
# Groups: 115 ships pre-contract, Baseline, S1, S2, S3
# -----------------------------

def avg_annual_cost_from_yearly_rows(yearly_rows, cost_key="mean_cost_3"):
    """Average annual cost across years from the yearly summary rows."""
    return float(np.mean([r[cost_key] for r in yearly_rows]))

def plot_5yr_avg_cost_barchart(avg_costs, labels, title):
    x = np.arange(len(labels))
    plt.figure(figsize=(10, 5))
    plt.bar(x, avg_costs)
    plt.xticks(x, labels)
    plt.title(title)
    plt.ylabel("Average annual demurrage cost over 5 years ($)")
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

# ---- 1) Build a 5-year "pre-contract" set: 115 ships, OLD system (2 wharfs, load 2–3) ----
# We only need costs, so we'll run 5 years with the same ship count (115) and same loading.
# Use the same year-seed blocks for consistency.

ships_115 = [115] * YEARS

# Make a "yearly" style list for pre-contract (2 wharfs only) by reusing your simulator per year
yearly_pre = []
for year in range(1, YEARS + 1):
    seed_block = SEED + year * 100_000
    _, c2, _, _, _ = run_replications(
        replications=REPS,
        base_seed=seed_block,
        n_wharf=2,                 # <-- old system
        days=DAYS,
        ships=115,
        load_min=2.0,
        load_max=3.0,
        wait_cost_per_day=WAIT_COST_PER_DAY
    )
    c2_m, _, _ = mean_ci(c2)
    yearly_pre.append({"year": year, "mean_cost_2": c2_m})

# ---- 2) Compute 5-year average annual costs ----
avg_cost_pre  = float(np.mean([r["mean_cost_2"] for r in yearly_pre]))       # 2 wharfs, 115 ships
avg_cost_base = avg_annual_cost_from_yearly_rows(yearly_base,    cost_key="mean_cost_3")   # baseline 3 wharfs
avg_cost_s1   = avg_annual_cost_from_yearly_rows(yearly_s1, cost_key="mean_cost_3")
avg_cost_s2   = avg_annual_cost_from_yearly_rows(yearly_s2, cost_key="mean_cost_3")
avg_cost_s3   = avg_annual_cost_from_yearly_rows(yearly_s3, cost_key="mean_cost_3")
YEARS = 5

total_cost_pre  = avg_cost_pre  * YEARS
total_cost_base = avg_cost_base * YEARS
total_cost_s1   = avg_cost_s1   * YEARS
total_cost_s2   = avg_cost_s2   * YEARS
total_cost_s3   = avg_cost_s3   * YEARS

labels_cost = [
    "Pre-contract\n(2W,115 ship,2–3 load)",
    "Baseline\n(3W,135 ship)",
    "S1\n(3W,+10–20%)",
    "S2\n(3W,2–5 load)",
    "S3\n(3W,+10–20% & 2–5)",
]

avg_costs = [total_cost_pre, total_cost_base, total_cost_s1, total_cost_s2, total_cost_s3]

plot_5yr_avg_cost_barchart(
    avg_costs,
    labels_cost,
    "5-Year TOTAL Demurrage Cost"
)

print("\n--- 5-year TOTAL Cost---")
print(f"Pre-contract (2W,115,2–3): ${total_cost_pre:,.0f}")
print(f"Baseline (3W):            ${total_cost_base:,.0f}")
print(f"Scenario 1 (3W,+10–20%):  ${total_cost_s1:,.0f}")
print(f"Scenario 2 (3W,2–5):      ${total_cost_s2:,.0f}")
print(f"Scenario 3 (3W,+10–20%,2–5): ${total_cost_s3:,.0f}")

# ============================================================

# ============================================================
def plot_cost_distributions( costs_baseline, costs_s1, costs_s2, costs_s3, title):
    all_costs = np.concatenate([ costs_baseline, costs_s1, costs_s2, costs_s3])
    bins = np.histogram_bin_edges(all_costs, bins=50)

    plt.figure(figsize=(12, 5))
    
    plt.hist(costs_baseline, bins=bins, density=True, alpha=0.45,
             label="Baseline (3W, 135 ships, load 2–3)")
    plt.hist(costs_s1,       bins=bins, density=True, alpha=0.45,
             label="Scenario 1 (3W, ships +10–20%, load 2–3)")
    plt.hist(costs_s2,       bins=bins, density=True, alpha=0.45,
             label="Scenario 2 (3W, 135 ships, load 2–5)")
    plt.hist(costs_s3,       bins=bins, density=True, alpha=0.45,
             label="Scenario 3 (3W, ships +10–20%, load 2–5)")

    plt.xlabel("Annual demurrage cost ($)")
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_cost_distributions(
     costs_base, costs_s1, costs_s2, costs_s3,
    "Annual Demurrage Cost Distribution (Baseline + 3-wharf scenarios)"
)
