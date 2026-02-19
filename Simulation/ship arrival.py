import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statistics as stats


#step1 see the distribution from historical data and identify the fit // calcualte some metrics like arrival rate/ service rate
#step2 random the numbers (why 135? ship) which aligns with the fit form step1
#step3 run the simulation with the set up resources and capacity multiple times (1000+) 
#STEPP 4 calculate 95% confidence interval


# -----------------------------
# Static inputs
# -----------------------------
random_seed = 2026
#-- update to 3 for case
N_warf = 2
anchor_cost = 2000  # per day
total_ships = 135 #20 more ship

arrival_rate =  0.56 #ship per day   
service_time = 0.8 #ship per day
# -----------------------------
# for checking histogram
# -----------------------------
#Days since last arrival
INTERARRIVAL_DAYS = [
    1.57, 1.05, 1.93, 0.74, 6.51, 7.56, 2.19, 4.18, 3.79,
    1.35, 0.93, 2.06, 0.67, 0.21, 0.09, 2.63, 0.28, 0.06,
    0.32, 0.96, 1.67, 0.02, 1.17, 5.82, 0.73, 2.53,
    4.64, 0.93, 1.03, 2.21, 2.80
]
#Loading time (days)
LOADING_DAYS = [
    2.18, 2.53, 2.24, 2.10, 2.24, 2.62, 2.36, 2.81, 2.10,
    2.06, 2.57, 2.86, 2.49, 2.56, 2.14, 2.52, 2.75, 2.45,
    2.71, 2.76, 2.87, 2.18, 2.91, 2.77, 2.60, 2.16, 2.91,
    2.27, 2.57, 2.50, 2.68
]

# -----------------------------
# setting arrival and service time
# -----------------------------

def sample_interarrival():
    return random.expovariate(1/arrival_rate)


def sample_loading_time():
    return random.uniform(2,3)

def compute_interarrivals_from_timestamps(arrival_timestamps):
    """
    arrival_timestamps: list of datetime (sorted or unsorted)
    returns interarrival times in same unit as timestamps (days here)
    """
    ts = sorted(arrival_timestamps)
    inter = []
    for a, b in zip(ts[:-1], ts[1:]):
        dt_days = (b - a).total_seconds() / (24*3600)
        if dt_days >= 0:
            inter.append(dt_days)
    return inter

def estimate_exponential_rate(interarrivals):
    """
    Exponential MLE: lambda = 1/mean
    """
    m = stats.mean(interarrivals)
    lam = 1.0 / m if m > 0 else float("inf")
    return lam, m  # (arrival rate per day), mean interarrival (days)

def estimate_service_uniform(service_times):
    """
    Simple uniform fit using min/max.
    If you prefer, replace with triangular/normal/gamma, etc.
    """

    mean_s = stats.mean(service_times)
    return mean_s

def print_step1_metrics(interarrivals, service_times, arrival_rate, mean_interarrival, service_a, service_b, mean_service):
    service_rate = 1.0 / mean_service if mean_service > 0 else float("inf")  # per day per wharf

    print("=== Step 1: Historical metrics ===")
    print(f"Interarrival samples: {len(interarrivals)}")
    print(f"Service samples:      {len(service_times)}")
    print(f"Mean interarrival (days): {mean_interarrival:.4f}")
    print(f"Arrival rate λ (ships/day): {arrival_rate:.4f}")
    print(f"Uniform service fit (days): a={service_a:.4f}, b={service_b:.4f}")
    print(f"Mean service (days): {mean_service:.4f}")
    print(f"Service rate μ (ships/day per wharf): {service_rate:.4f}")
    # Traffic intensity idea (rough): rho ≈ λ / (c*μ)
    print("NOTE: Utilization rho depends on wharf count c: rho ≈ λ / (c*μ)")
    print()

# Optional: a very light “fit check” without external libs.
# (If you want KS test + p-value, we can add scipy.stats.kstest version.)
def quick_exponential_fit_check(interarrivals, lam):
    """
    Quick sanity checks: compare empirical mean vs 1/lam, and tail probability at a few points.
    Not a formal goodness-of-fit test, but useful for debugging.
    """
    m = stats.mean(interarrivals)
    print("Exponential fit sanity check:")
    print(f"  Empirical mean: {m:.4f} vs Theoretical mean: {(1/lam):.4f}")
    for x in [0.25*m, 1*m, 2*m]:
        emp_tail = sum(1 for t in interarrivals if t > x) / len(interarrivals)
        theo_tail = math.exp(-lam*x)
        print(f"  P(T>x) at x={x:.4f}: empirical={emp_tail:.4f}, theoretical={theo_tail:.4f}")
    print()

# ============================================================
# STEP 2) Generate synthetic arrivals + service times
#         (why 135 ships? -> you choose N_SHIPS to match a period or dataset size)
# ============================================================

def generate_synthetic_samples(N_SHIPS, arrival_rate, service_a, seed=2026):
    """
    Generates:
      - interarrival times for N_SHIPS ships (first ship at time 0)
      - service times for N_SHIPS ships
    """
    random.seed(seed)
    interarrivals = [0.0]  # first ship at time 0
    for _ in range(N_SHIPS - 1):
        interarrivals.append(random.expovariate(arrival_rate))

    service_times = [random.uniform(service_a) for _ in range(N_SHIPS)]
    return interarrivals, service_times

# ============================================================
# STEP 3) Run simulation many times (1000+) for each scenario
# ============================================================

def simulate_once(interarrivals, service_times, N_warf, anchor_cost):
    """
    Simulates exactly len(service_times) ships arriving according to interarrivals.
    Returns summary stats for this replication.
    """
    env = simpy.Environment()
    wharf = simpy.Resource(env, capacity=N_warf)

    waits = []
    costs = []

    def ship_process(i):
        # arrive at current env.now (scheduled by arrival generator)
        req_t = env.now
        with wharf.request() as req:
            yield req
            w = env.now - req_t

            # service time predetermined for ship i
            s = service_times[i]
            yield env.timeout(s)

        waits.append(w)
        costs.append(w * anchor_cost)

    def arrival_generator():
        t = 0.0
        for i, ia in enumerate(interarrivals):
            t += ia
            yield env.timeout(ia)
            env.process(ship_process(i))

    env.process(arrival_generator())
    env.run()

    avg_wait = stats.mean(waits) if waits else 0.0
    max_wait = max(waits) if waits else 0.0
    total_cost = sum(costs)
    avg_cost = total_cost / len(costs) if costs else 0.0

    return {
        "ships": len(waits),
        "avg_wait": avg_wait,
        "max_wait": max_wait,
        "total_cost": total_cost,
        "avg_cost": avg_cost
    }

def run_replications(N_REPS, N_SHIPS, arrival_rate, service_a, N_warf, anchor_cost, base_seed=2026):
    """
    For each replication:
      - generate synthetic interarrivals & service times aligned with Step 1 fit
      - simulate
    Returns list of results dicts.
    """
    results = []
    for r in range(N_REPS):
        seed = base_seed + r
        inter, serv = generate_synthetic_samples(
            N_SHIPS=N_SHIPS,
            arrival_rate=arrival_rate,
            service_a=service_a,
            seed=seed
        )
        out = simulate_once(inter, serv, N_warf=N_warf, anchor_cost=anchor_cost)
        results.append(out)
    return results

def summarize_results(results):
    """
    Aggregates replication outputs (mean + percentile-ish view).
    """
    def col(name): return [x[name] for x in results]

    avg_waits = col("avg_wait")
    totals = col("total_cost")

    avg_wait_mean = stats.mean(avg_waits)
    avg_wait_p50 = stats.median(avg_waits)
    avg_wait_p95 = sorted(avg_waits)[int(0.95 * (len(avg_waits)-1))]

    total_mean = stats.mean(totals)
    total_p50 = stats.median(totals)
    total_p95 = sorted(totals)[int(0.95 * (len(totals)-1))]

    return {
        "reps": len(results),
        "avg_wait_mean": avg_wait_mean,
        "avg_wait_p50": avg_wait_p50,
        "avg_wait_p95": avg_wait_p95,
        "total_cost_mean": total_mean,
        "total_cost_p50": total_p50,
        "total_cost_p95": total_p95,
    }

# ---------- Step 3: Run many replications ----------
N_REPS = 1000
N_SHIPS = 135
service_a=
service_b=  

res_2 = run_replications(
    N_REPS=N_REPS,
    N_SHIPS=N_SHIPS,
    arrival_rate=arrival_rate,
    service_a=service_a,
    service_b=service_b,
    N_warf=2,
    anchor_cost=anchor_cost,
    base_seed=2026
)
res_3 = run_replications(
    N_REPS=N_REPS,
    N_SHIPS=N_SHIPS,
    arrival_rate=arrival_rate,
    service_a=service_a,
    service_b=service_b,
    N_warf=3,
    anchor_cost=anchor_cost,
    base_seed=3026
)

sum_2 = summarize_results(res_2)
sum_3 = summarize_results(res_3)

print("=== Comparison over replications ===")
print("2 wharfs:", sum_2)
print("3 wharfs:", sum_3)