import simpy
import random
import statistics as stats

SIM_TIME = 60  # minutes
RANDOM_SEED = 42

# ----- Distributions from the slide -----
def interarrival_time():
    # Uniform integers from 1 to 8 minutes
    return random.randint(1, 8)

def service_time():
    # Discrete distribution for service times 1..6
    times = [1, 2, 3, 4, 5, 6]
    probs = [0.10, 0.20, 0.30, 0.25, 0.10, 0.05]
    return random.choices(times, weights=probs, k=1)[0]

# ----- Metrics container -----
class Metrics:
    def __init__(self):
        self.wait_times = []
        self.system_times = []
        self.num_served = 0
        self.busy_time = 0.0  # for utilization
        self.queue_lengths = []  # sampled over time

def customer(env, name, checkout: simpy.Resource, m: Metrics):
    arrive = env.now

    # Request the single checkout counter
    with checkout.request() as req:
        yield req
        start_service = env.now
        m.wait_times.append(start_service - arrive)

        st = service_time()
        m.busy_time += st
        yield env.timeout(st)

    depart = env.now
    m.system_times.append(depart - arrive)
    m.num_served += 1

def arrival_process(env, checkout, m: Metrics):
    i = 0
    while env.now < SIM_TIME:
        i += 1
        env.process(customer(env, f"C{i}", checkout, m))
        yield env.timeout(interarrival_time())

def queue_monitor(env, checkout, m: Metrics, sample_every=1):
    # record queue length every minute
    while env.now <= SIM_TIME:
        m.queue_lengths.append(len(checkout.queue))
        yield env.timeout(sample_every)

def run():
    random.seed(RANDOM_SEED)

    env = simpy.Environment()
    checkout = simpy.Resource(env, capacity=1)  # one checkout counter
    m = Metrics()

    env.process(arrival_process(env, checkout, m))
    env.process(queue_monitor(env, checkout, m))
    env.run(until=SIM_TIME)

    # ----- Report -----
    avg_wait = stats.mean(m.wait_times) if m.wait_times else 0
    avg_system = stats.mean(m.system_times) if m.system_times else 0
    max_queue = max(m.queue_lengths) if m.queue_lengths else 0
    avg_queue = stats.mean(m.queue_lengths) if m.queue_lengths else 0
    utilization = m.busy_time / SIM_TIME  # since capacity=1

    print("---- Grocery Checkout Simulation (60 mins) ----")
    print(f"Customers served:              {m.num_served}")
    print(f"Average wait time (mins):      {avg_wait:.2f}")
    print(f"Average time in system (mins): {avg_system:.2f}")
    print(f"Average queue length:          {avg_queue:.2f}")
    print(f"Max queue length:              {max_queue}")
    print(f"Checkout utilization:          {utilization:.2%}")

if __name__ == "__main__":
    run()
