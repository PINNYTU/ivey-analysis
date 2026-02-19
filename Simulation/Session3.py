import simpy
import random
import statistics as stats

# -----------------------------
# Settings
# -----------------------------
RANDOM_SEED = 2026
SIM_MIN = 5 * 60  # 5 hours

# Arrivals
INITIAL_MIN = 5
INITIAL_MAX = 10
ARRIVALS_PER_HOUR_MIN = 25
ARRIVALS_PER_HOUR_MAX = 30

# Resources (from your diagram)
REG_CAPACITY = 1          # Not mention in case
KIOSK_CAPACITY = 3        # questionnaire kiosks
SCREEN_CAPACITY = 6       # cubicles
DONATION_CAPACITY = 8     # pods

# Probabilities (you can tweak; not given explicitly)
P_RETURNING = 0.70
P_FIRST_TIMER = 0.25
P_ISSUE = 0.05

P_OTHER_COUNTRY = 0.10
P_UNUSUAL_SCREEN = 0.06
P_DONATION_ISSUE = 0.05


# -----------------------------
# Random time helpers (minutes)
# -----------------------------
def u(a, b):
    return random.uniform(a, b)

def tri(a, m, b):
    return random.triangular(a, b, m)

def interarrival_minutes():
    # busy day 25–30/hr => interarrival between 2.0 and 2.4 minutes
    return random.expovariate(1/2.18)

def reg_time(donor_type):
    if donor_type == "returning":
        return u(0.4, 0.9)      # <1 min
    if donor_type == "first":
        return u(5, 10)
    return tri(3, 5, 7)         # issue ~5

def questionnaire_time(other_country):
    return 10.0 if other_country else u(5.5, 6.0)

def screen_time(unusual):
    return 30.0 if unusual else u(10, 12)

def donation_time(issue):
    return u(18, 30) if issue else u(14, 15)


# -----------------------------
# Model
# -----------------------------
class Clinic:
    def __init__(self, env):
        self.env = env
        self.registration = simpy.Resource(env, capacity=REG_CAPACITY)
        self.kiosks = simpy.Resource(env, capacity=KIOSK_CAPACITY)
        self.screen = simpy.Resource(env, capacity=SCREEN_CAPACITY)
        self.donate = simpy.Resource(env, capacity=DONATION_CAPACITY)

        # Stats
        self.waits = {k: [] for k in ["registration", "kiosk", "screen", "donate"]}
        self.total_time = []
        self.completed = 0

def donor(env, clinic: Clinic, i: int):
    start = env.now

    # ---- donor attributes ----
    r = random.random()
    if r < P_RETURNING:
        donor_type = "returning"
    elif r < P_RETURNING + P_FIRST_TIMER:
        donor_type = "first"
    else:
        donor_type = "issue"

    other_country = (random.random() < P_OTHER_COUNTRY)
    unusual = (random.random() < P_UNUSUAL_SCREEN)
    donation_issue = (random.random() < P_DONATION_ISSUE)

    # ---- Registration (capacity = R) ----
    t_req = env.now
    with clinic.registration.request() as req:
        yield req
        clinic.waits["registration"].append(env.now - t_req)
        yield env.timeout(reg_time(donor_type))

    # ---- Questionnaire (capacity = 3 kiosks) ----
    t_req = env.now
    with clinic.kiosks.request() as req:
        yield req
        clinic.waits["kiosk"].append(env.now - t_req)
        yield env.timeout(questionnaire_time(other_country))

    # ---- wait for nurse 1–2 min ----
    yield env.timeout(u(1, 2))

    # ---- Health screen (6 cubicles) ----
    t_req = env.now
    with clinic.screen.request() as req:
        yield req
        clinic.waits["screen"].append(env.now - t_req)
        yield env.timeout(screen_time(unusual))

    # ---- Donate (8 pods) ----
    t_req = env.now
    with clinic.donate.request() as req:
        yield req
        clinic.waits["donate"].append(env.now - t_req)
        yield env.timeout(donation_time(donation_issue))

    # ---- sit & wait 5 min ----
    yield env.timeout(5)

    # ---- refreshment (unlimited) 5–10 min ----
    yield env.timeout(u(5, 10))

    clinic.completed += 1
    clinic.total_time.append(env.now - start)

def arrivals(env, clinic: Clinic):
    # initial line at opening
    initial = random.randint(INITIAL_MIN, INITIAL_MAX)
    for j in range(initial):
        env.process(donor(env, clinic, j + 1))

    i = initial
    while env.now < SIM_MIN:
        yield env.timeout(interarrival_minutes())
        i += 1
        env.process(donor(env, clinic, i))

def summarize(x):
    if not x:
        return "n=0"
    return f"n={len(x)} avg={stats.fmean(x):.2f} p90={sorted(x)[int(0.9*(len(x)-1))]:.2f} max={max(x):.2f}"

def run():
    random.seed(RANDOM_SEED)
    env = simpy.Environment()
    clinic = Clinic(env)
    env.process(arrivals(env, clinic))
    env.run(until=SIM_MIN)

    print("Completed donors:", clinic.completed)
    print("Total time in system (min):", summarize(clinic.total_time))
    print("\nWaits (min):")
    for k in ["registration", "kiosk", "screen", "donate"]:
        print(f"  {k:12s}: {summarize(clinic.waits[k])}")

if __name__ == "__main__":
    run()


# ============================================================
# Run the model 10,000 times
# ============================================================

N_RUNS = 10_000

completed = []
avg_total_time = []
avg_waits = {k: [] for k in ["registration", "kiosk", "screen", "donate"]}

def run_once(seed):
    random.seed(seed)
    env = simpy.Environment()
    clinic = Clinic(env)
    env.process(arrivals(env, clinic))
    env.run(until=SIM_MIN)

    completed.append(clinic.completed)
    avg_total_time.append(
        stats.fmean(clinic.total_time) if clinic.total_time else 0
    )

    for k in avg_waits:
        if clinic.waits[k]:
            avg_waits[k].append(stats.fmean(clinic.waits[k]))
        else:
            avg_waits[k].append(0)

# ---- run replications ----
for i in range(N_RUNS):
    run_once(RANDOM_SEED + i)

# ============================================================
# Summary across 10,000 runs
# ============================================================

def pctl(x, p):
    x = sorted(x)
    return x[int(p * (len(x) - 1))]

print("\n=== 10,000-run summary ===")
print(f"Completed donors: avg={stats.fmean(completed):.1f}, p90={pctl(completed,0.9):.0f}")

print("\nTotal time in system (min):")
print(
    f"  avg={stats.fmean(avg_total_time):.2f}, "
    f"p90={pctl(avg_total_time,0.9):.2f}, "
    f"p95={pctl(avg_total_time,0.95):.2f}"
)

print("\nAverage waits per stage (min):")
for k in ["registration", "kiosk", "screen", "donate"]:
    print(
        f"  {k:12s}: "
        f"avg={stats.fmean(avg_waits[k]):.2f}, "
        f"p90={pctl(avg_waits[k],0.9):.2f}, "
        f"p95={pctl(avg_waits[k],0.95):.2f}"
    )
