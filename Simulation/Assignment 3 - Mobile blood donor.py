import random
import math
import simpy
import matplotlib.pyplot as plt
from collections import defaultdict

# -------------------------
# CONFIG
# -------------------------

REPS = 1000
OPEN_HOURS = 5
OPEN_MIN = OPEN_HOURS * 60

CAP_REGISTER = 1  # adjust later
CAP_QUESTION = 3
CAP_HEALTH = 6
CAP_DONATE = 8

P_BUSY_DAY = 0.50
NORMAL_ARRIVALS_PER_HR = (5, 10)
BUSY_ARRIVALS_PER_HR = (25, 30)

P_REGISTER = [("returning", 0.6), ("first", 0.35), ("issue", 0.05)]
P_QUESTION = [("normal", 0.9), ("foreign", 0.1)]
P_HEALTH = [("normal", 0.9), ("unusual", 0.1)]
P_DONATE = [("good", 0.95), ("issue", 0.05)]

# Triangular parameter RANGES (low/mode/high are random within these ranges)
TRI = {
    ("register", "returning"): {"low": (0.1, 0.4), "mode": (0.4, 0.7), "high": (0.7, 1.2)},
    ("register", "first"):     {"low": (4.5, 5.5), "mode": (6.5, 8.5), "high": (9.5, 10.5)},
    ("register", "issue"):     {"low": (3.5, 4.5), "mode": (4.5, 5.5), "high": (5.5, 6.5)},

    ("question", "normal"):    {"low": (5.0, 5.5), "mode": (5.5, 5.9), "high": (5.9, 6.2)},
    ("question", "foreign"):   {"low": (8.5, 9.5), "mode": (9.5, 10.5), "high": (10.5, 11.5)},

    ("nurse", "wait"):         {"low": (0.8, 1.2), "mode": (1.2, 1.6), "high": (1.6, 2.2)},

    ("health", "normal"):      {"low": (9.0, 10.0), "mode": (10.5, 11.5), "high": (11.5, 12.5)},
    ("health", "unusual"):     {"low": (10.0, 14.0), "mode": (18.0, 22.0), "high": (25.0, 35.0)},

    ("donate", "good"):        {"low": (13.5, 14.0), "mode": (14.0, 14.7), "high": (14.7, 15.3)},
    ("donate", "issue"):       {"low": (14.0, 18.0), "mode": (20.0, 25.0), "high": (25.0, 35.0)},

    ("sit", "fixed"):          {"low": (5.0, 5.0), "mode": (5.0, 5.0), "high": (5.0, 5.0)},
    ("refresh", "any"):        {"low": (4.5, 5.5), "mode": (6.5, 8.5), "high": (8.5, 10.5)},
}

QUEUED_STAGES = ["register", "question", "health", "donate"]

# -------------------------
# Helpers (RNG-based)
# -------------------------

def choose(rng, opts):
    r = rng.random()
    s = 0.0
    for k, p in opts:
        s += p
        if r <= s:
            return k
    return opts[-1][0]

def exp_interarrival(rng, mean):
    return -mean * math.log(1.0 - rng.random())

def tri_random(rng, key):
    spec = TRI[key]
    low = rng.uniform(*spec["low"])
    high = rng.uniform(*spec["high"])
    if high < low:
        low, high = high, low

    mode_low = max(low, spec["mode"][0])
    mode_high = min(high, spec["mode"][1])
    mode = rng.uniform(mode_low, mode_high) if mode_low <= mode_high else (low + high) / 2.0

    return rng.triangular(low, high, mode)

def draw_mean_interarrival_for_day(rng, day_type):
    if day_type == "busy":
        rate_per_hr = rng.uniform(*BUSY_ARRIVALS_PER_HR)
    else:
        rate_per_hr = rng.uniform(*NORMAL_ARRIVALS_PER_HR)
    return 60.0 / rate_per_hr  # minutes

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def sample_std(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))

Z_95 = 1.96

def ci95_of_rep_means(rep_means):
    vals = [v for v in rep_means if not math.isnan(v)]
    n = len(vals)
    m = mean(vals)
    s = sample_std(vals)
    se = s / math.sqrt(n) if n > 0 else float("nan")
    return m, m - Z_95 * se, m + Z_95 * se, n

# -------------------------
# NEW: explicit capacity check helper
# -------------------------

def is_available(res: simpy.Resource) -> bool:
    # True if at least one server free right now
    return res.count < res.capacity

def seize(env, res: simpy.Resource, wait_store: list):
    """
    Explicitly:
    - if server free -> wait=0, proceed
    - else -> queue and measure actual wait
    Returns the request event (so caller can release via 'with').
    """
    if is_available(res):
        wait_store.append(0.0)
        req = res.request()
        # This will (effectively) succeed immediately
        yield req
        return req
    else:
        t0 = env.now
        req = res.request()
        yield req
        wait_store.append(env.now - t0)
        return req

# -------------------------
# SimPy Model
# -------------------------

class Clinic:
    def __init__(self, env, rng, mean_interarrival):
        self.env = env
        self.rng = rng
        self.mean_interarrival = mean_interarrival

        self.reg = simpy.Resource(env, CAP_REGISTER)
        self.q = simpy.Resource(env, CAP_QUESTION)
        self.h = simpy.Resource(env, CAP_HEALTH)
        self.d = simpy.Resource(env, CAP_DONATE)
        self.donors_completed = 0
        self.wait = defaultdict(list)

    def donor(self):
        # Register (explicit check)
        reg_kind = choose(self.rng, P_REGISTER)
        reg_time = tri_random(self.rng, ("register", reg_kind))
        with (yield from seize(self.env, self.reg, self.wait["register"])) as req:
            yield self.env.timeout(reg_time)

        # Questionnaire (explicit check)
        q_kind = choose(self.rng, P_QUESTION)
        q_time = tri_random(self.rng, ("question", q_kind))
        with (yield from seize(self.env, self.q, self.wait["question"])) as req:
            yield self.env.timeout(q_time)

        # Wait for nurse (no capacity)
        yield self.env.timeout(tri_random(self.rng, ("nurse", "wait")))

        # Health screen (explicit check)
        h_kind = choose(self.rng, P_HEALTH)
        h_time = tri_random(self.rng, ("health", h_kind))
        with (yield from seize(self.env, self.h, self.wait["health"])) as req:
            yield self.env.timeout(h_time)

        # Donate (explicit check)
        d_kind = choose(self.rng, P_DONATE)
        d_time = tri_random(self.rng, ("donate", d_kind))
        with (yield from seize(self.env, self.d, self.wait["donate"])) as req:
            yield self.env.timeout(d_time)

        # Sit + refresh (no capacity)
        yield self.env.timeout(tri_random(self.rng, ("sit", "fixed")))
        yield self.env.timeout(tri_random(self.rng, ("refresh", "any")))

        self.donors_completed += 1

    def arrivals(self):
        while True:
            ia = exp_interarrival(self.rng, self.mean_interarrival)
            yield self.env.timeout(ia)
            if self.env.now > OPEN_MIN:
                break
            self.env.process(self.donor())

# -------------------------
# Run Replications
# -------------------------

all_waits = {k: [] for k in QUEUED_STAGES}
rep_means = {k: [] for k in QUEUED_STAGES}
rep_donors_per_day = []
rep_day_types = {"normal": 0, "busy": 0}

for rep in range(REPS):
    rng = random.Random()  # random seed from OS entropy each rep

    day_type = "busy" if rng.random() < P_BUSY_DAY else "normal"
    rep_day_types[day_type] += 1
    mean_ia = draw_mean_interarrival_for_day(rng, day_type)

    env = simpy.Environment()
    clinic = Clinic(env, rng, mean_ia)
    env.process(clinic.arrivals())
    env.run(until=OPEN_MIN + 480)
    rep_donors_per_day.append(clinic.donors_completed)

    for stage in QUEUED_STAGES:
        waits = clinic.wait[stage]
        all_waits[stage].extend(waits)
        rep_means[stage].append(mean(waits))

print("Day types over reps:", rep_day_types)

# -------------------------
# Plot Distributions
# -------------------------

for stage, data in all_waits.items():
    plt.figure()
    plt.hist(data, bins=40, density=True)
    plt.title(f"Wait Time Distribution – {stage}")
    plt.xlabel("Minutes")
    plt.ylabel("Density")
    plt.tight_layout()

plt.show()

# -------------------------
# Confidence Intervals
# -------------------------

print("\n95% Confidence Intervals for MEAN wait time (minutes), across replications:")
for stage in QUEUED_STAGES:
    m, lo, hi, n = ci95_of_rep_means(rep_means[stage])
    print(f"{stage:12s} | reps={n:4d} | mean={m:7.3f} | 95% CI [{lo:7.3f}, {hi:7.3f}]")

mD, loD, hiD, nD = ci95_of_rep_means(rep_donors_per_day)
print(f"\nDonors per day | reps={nD:4d} | mean={mD:7.3f} | 95% CI [{loD:7.3f}, {hiD:7.3f}]")
