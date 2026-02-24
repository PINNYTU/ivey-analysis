"""
Beer Distribution Game — FULL 4-ROLE SIMULATOR (Retailer / Wholesaler / Distributor / Factory)
Python 3.9 compatible.

What you get:
- All 4 roles modeled at once
- 2-week order processing delay + 2-week shipping delay between adjacent roles
- Factory has 2-week production delay (modeled like shipping delay)
- Backlog + inventory costs:
    inventory:  $0.25 per unit per week
    backlog:    $1.00 per unit per week
- Dynamic per-week order input:
    You can manually enter orders for any/all roles each week
    Or use "auto" (simple base-stock rule) per role

How the weekly flow works (per role):
1) Receive inbound shipment arriving this week (or finished production for factory)
2) Observe incoming order from downstream (or customer demand for retailer)
3) Ship as much as possible to satisfy backlog + incoming order
4) Accumulate holding/backlog cost
5) Place order upstream (factory places production order)
   - Orders arrive to upstream after 2 weeks (order delay)
   - Goods arrive to you after an additional 2 weeks (ship/production delay)
   => Total 4-week delay from ordering to receiving goods

NOTE:
This is a standard "beer game" structure. Different classrooms sometimes tweak initial pipelines;
you can edit initial inventories and pipelines in main().
"""

from collections import deque
import copy
from dataclasses import dataclass
from itertools import product
from typing import Deque, Dict, List, Optional, Tuple


INV_COST = 0.25
BACKLOG_COST = 1.00

ORDER_DELAY = 2
SHIP_DELAY = 2
PROD_DELAY = 2  # factory production time
ROLE_NAMES = ["Retailer", "Wholesaler", "Distributor", "Factory"]


def get_int_input(prompt: str, default: Optional[int] = None, min_value: int = 0) -> int:
    """Robust integer input for Python 3.9. Supports Enter for default; 'q' to quit."""
    while True:
        s = input(prompt).strip().lower()
        if s in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if s == "" and default is not None:
            return default
        try:
            val = int(s)
            if val < min_value:
                print(f"Please enter an integer >= {min_value}.")
                continue
            return val
        except ValueError:
            print("Please enter a valid integer (or press Enter for default).")


def get_order_input(role_name: str, last_order: int, suggested: int) -> int:
    """
    Reads an order decision for a role.
    - Enter integer
    - Enter 'same' -> repeats last_order
    - Enter 'auto' -> uses suggested
    - Enter empty -> repeats last_order
    """
    raw = input(
        f"{role_name} order (last={last_order}) [int/same/auto/Enter/q]: "
    ).strip().lower()

    if raw in {"q", "quit", "exit"}:
        raise KeyboardInterrupt
    if raw == "" or raw == "same":
        return last_order
    if raw == "auto":
        print(f"[auto] {role_name} suggested order = {suggested}")
        return max(0, int(suggested))

    try:
        q = int(raw)
        return max(0, q)
    except ValueError:
        print("Invalid input -> using last order.")
        return last_order


@dataclass
class WeekResult:
    week: int
    demand_or_order_in: int
    received: int
    shipped: int
    inv_end: int
    backlog_end: int
    inv_cost: float
    backlog_cost: float
    week_cost: float
    total_cost: float


class Role:
    """
    One supply chain role (Retailer, Wholesaler, Distributor, Factory).

    It maintains:
    - inventory
    - backlog (unfilled outgoing demand to downstream)
    - inbound_pipeline: goods arriving in SHIP_DELAY/PROD_DELAY weeks (index 0 arrives this week)
    - order_pipeline: orders arriving to upstream after ORDER_DELAY (index 0 arrives upstream this week)

    Note: order_pipeline is tracked here for convenience; in the system, those orders are delivered to the upstream role.
    """

    def __init__(
        self,
        name: str,
        ship_delay: int,
        order_delay: int,
        inv_cost: float = INV_COST,
        backlog_cost: float = BACKLOG_COST,
        initial_inventory: int = 100,
        initial_inbound_pipeline: Optional[List[int]] = None,
    ):
        self.name = name
        self.ship_delay = ship_delay
        self.order_delay = order_delay
        self.inv_cost = inv_cost
        self.backlog_cost = backlog_cost

        self.inventory = initial_inventory
        self.backlog = 0

        if initial_inbound_pipeline is None:
            # default: total 100 spread over next 2 weeks
            initial_inbound_pipeline = [50, 50] if ship_delay == 2 else [0] * ship_delay
        if len(initial_inbound_pipeline) != ship_delay:
            raise ValueError(f"{name}: initial_inbound_pipeline must have length == ship_delay ({ship_delay}).")

        # Goods arriving to this role in the next ship_delay weeks
        self.inbound_pipeline: Deque[int] = deque(initial_inbound_pipeline)

        # Orders placed by this role that will reach upstream in the next order_delay weeks
        self.order_pipeline_to_upstream: Deque[int] = deque([0] * order_delay)

        self.total_cost = 0.0
        self.last_order = 0

    def receive_goods(self) -> int:
        """Step 1: Receive goods arriving this week."""
        arriving = self.inbound_pipeline.popleft()
        self.inventory += arriving
        self.inbound_pipeline.append(0)  # keep length constant
        return arriving

    def fulfill(self, incoming_order: int) -> int:
        """
        Step 2-3: Fulfill backlog + incoming order from inventory.
        Returns shipped amount.
        """
        required = self.backlog + incoming_order
        shipped = min(self.inventory, required)
        self.inventory -= shipped
        self.backlog = required - shipped
        return shipped

    def charge_cost(self) -> Tuple[float, float, float]:
        """Step 4: Compute and accumulate end-of-week cost."""
        inv_c = self.inventory * self.inv_cost
        back_c = self.backlog * self.backlog_cost
        total_c = inv_c + back_c
        self.total_cost += total_c
        return inv_c, back_c, total_c

    def place_order(self, qty: int) -> None:
        """
        Step 5: Place an order upstream.
        The order will reach upstream after order_delay weeks.
        """
        qty = max(0, int(qty))
        # Schedule it to arrive upstream in order_delay weeks (at the end of deque)
        self.order_pipeline_to_upstream[-1] += qty
        self.last_order = qty

    def advance_order_pipeline(self) -> int:
        """
        Advances the order pipeline by 1 week.
        Returns the quantity that arrives to upstream THIS week.
        """
        arriving_to_upstream = self.order_pipeline_to_upstream.popleft()
        self.order_pipeline_to_upstream.append(0)
        return arriving_to_upstream

    def inventory_position(self) -> int:
        """
        Inventory position (common policy input):
        on-hand + inbound pipeline - backlog
        """
        return self.inventory + sum(self.inbound_pipeline) - self.backlog


class BeerGame4Role:
    """
    Full 4-role system:
    Customer -> Retailer -> Wholesaler -> Distributor -> Factory

    Key flows:
    - Orders flow upstream (with ORDER_DELAY)
    - Goods flow downstream (with SHIP_DELAY); factory uses PROD_DELAY as its 'ship_delay' of production completion
    """

    def __init__(
        self,
        weeks: int = 35,
        expected_demand: int = 100,
        initial_inventories: Optional[Dict[str, int]] = None,
        initial_pipelines: Optional[Dict[str, List[int]]] = None,
    ):
        self.weeks = weeks
        self.expected_demand = expected_demand

        if initial_inventories is None:
            initial_inventories = {
                "Retailer": 100,
                "Wholesaler": 100,
                "Distributor": 100,
                "Factory": 100,
            }
        if initial_pipelines is None:
            # default inbound pipeline: 50 arriving week+1, 50 arriving week+2 for everyone
            initial_pipelines = {
                "Retailer": [50, 50],
                "Wholesaler": [50, 50],
                "Distributor": [50, 50],
                "Factory": [50, 50],  # factory 'production completion pipeline'
            }

        self.retailer = Role("Retailer", ship_delay=SHIP_DELAY, order_delay=ORDER_DELAY,
                             initial_inventory=initial_inventories["Retailer"],
                             initial_inbound_pipeline=initial_pipelines["Retailer"])
        self.wholesaler = Role("Wholesaler", ship_delay=SHIP_DELAY, order_delay=ORDER_DELAY,
                               initial_inventory=initial_inventories["Wholesaler"],
                               initial_inbound_pipeline=initial_pipelines["Wholesaler"])
        self.distributor = Role("Distributor", ship_delay=SHIP_DELAY, order_delay=ORDER_DELAY,
                                initial_inventory=initial_inventories["Distributor"],
                                initial_inbound_pipeline=initial_pipelines["Distributor"])
        self.factory = Role("Factory", ship_delay=PROD_DELAY, order_delay=ORDER_DELAY,
                            initial_inventory=initial_inventories["Factory"],
                            initial_inbound_pipeline=initial_pipelines["Factory"])

        self.roles = [self.retailer, self.wholesaler, self.distributor, self.factory]

    def _suggest_order_base_stock(self, role: Role, target_weeks: int = 4) -> int:
        """
        Simple (not optimal) suggestion:
        Target inventory position = expected_demand * target_weeks
        Suggested order = max(0, target - inventory_position)
        """
        target = self.expected_demand * target_weeks
        ip = role.inventory_position()
        return max(0, target - ip)

    def step_week(
        self,
        week: int,
        customer_demand: int,
        orders: Dict[str, int],
    ) -> Dict[str, WeekResult]:
        """
        Runs one full week across all roles, with the classic sequence:
        A) Goods arrive to each role
        B) Incoming orders/demand arrive
        C) Each role ships to downstream (as much as possible)
        D) Each role pays cost
        E) Orders placed upstream (or production order at factory)
        F) Orders propagate to upstream order-queues
        G) Upstream shipments are scheduled into downstream inbound pipelines
        """
        results: Dict[str, WeekResult] = {}

        # A) Receive goods
        received = {r.name: r.receive_goods() for r in self.roles}

        # B) Determine incoming orders to each role this week:
        # Retailer sees customer demand; others see orders placed by downstream that arrive after ORDER_DELAY
        # We advance each downstream role's order pipeline to compute what upstream receives this week.
        retailer_in = customer_demand
        wholesaler_in = self.retailer.advance_order_pipeline()     # retailer's order reaches wholesaler now
        distributor_in = self.wholesaler.advance_order_pipeline()  # wholesaler's order reaches distributor now
        factory_in = self.distributor.advance_order_pipeline()     # distributor's order reaches factory now

        incoming = {
            "Retailer": retailer_in,
            "Wholesaler": wholesaler_in,
            "Distributor": distributor_in,
            "Factory": factory_in,
        }

        # C) Fulfillment + shipping downstream
        # Each role ships to its downstream node (or customer for retailer)
        shipped_to_downstream = {
            "Retailer": self.retailer.fulfill(incoming["Retailer"]),
            "Wholesaler": self.wholesaler.fulfill(incoming["Wholesaler"]),
            "Distributor": self.distributor.fulfill(incoming["Distributor"]),
            "Factory": self.factory.fulfill(incoming["Factory"]),
        }

        # D) Costs
        week_cost_parts = {r.name: r.charge_cost() for r in self.roles}

        # E) Place orders (decisions happen after observing this week)
        self.retailer.place_order(orders.get("Retailer", self.retailer.last_order))
        self.wholesaler.place_order(orders.get("Wholesaler", self.wholesaler.last_order))
        self.distributor.place_order(orders.get("Distributor", self.distributor.last_order))
        self.factory.place_order(orders.get("Factory", self.factory.last_order))  # production order

        # F) NOTE: We already advanced order pipelines earlier (B).
        # The newly placed orders are already scheduled at the end of each role.order_pipeline_to_upstream by place_order().

        # G) Schedule shipments into downstream inbound pipelines with SHIP_DELAY/PROD_DELAY
        # - Factory shipped units become inbound to Distributor after SHIP_DELAY (transport)
        #   BUT factory "ship_delay" represents production completion already for factory inbound;
        #   the physical transport from factory to distributor is still SHIP_DELAY in many beer game setups.
        #   To keep consistent with the typical 2+2 delay between stages, we treat factory shipment travel as SHIP_DELAY.
        #
        # For simplicity: every shipment from any role to downstream arrives in 2 weeks (SHIP_DELAY),
        # except factory "inbound_pipeline" is production completion (2 weeks) that feeds factory inventory.
        #
        # So: Factory shipping to distributor also uses SHIP_DELAY.
        #
        # We schedule shipped quantities into downstream inbound_pipeline[-1] (arrives in ship_delay weeks).
        self.retailer.inbound_pipeline[-1] += shipped_to_downstream["Wholesaler"]    # wholesaler -> retailer
        self.wholesaler.inbound_pipeline[-1] += shipped_to_downstream["Distributor"] # distributor -> wholesaler
        self.distributor.inbound_pipeline[-1] += shipped_to_downstream["Factory"]    # factory -> distributor (transport)
        # Factory has no upstream shipping here; its "inbound_pipeline" is production completion.
        # We model production completion as: production order placed => after PROD_DELAY, those units appear in factory inbound_pipeline.
        self.factory.inbound_pipeline[-1] += self.factory.last_order  # production completes in PROD_DELAY weeks

        # Pack results
        for r in self.roles:
            results[r.name] = WeekResult(
                week=week,
                demand_or_order_in=incoming[r.name],
                received=received[r.name],
                shipped=shipped_to_downstream[r.name],
                inv_end=r.inventory,
                backlog_end=r.backlog,
                inv_cost=round(week_cost_parts[r.name][0], 2),
                backlog_cost=round(week_cost_parts[r.name][1], 2),
                week_cost=round(week_cost_parts[r.name][2], 2),
                total_cost=round(r.total_cost, 2),
            )

        return results

    def total_system_cost(self) -> float:
        return sum(r.total_cost for r in self.roles)


def print_week_summary(week: int, res: Dict[str, WeekResult]) -> None:
    print("\n" + "=" * 90)
    print(f"Week {week} Summary")
    print("-" * 90)
    for role_name in ROLE_NAMES:
        r = res[role_name]
        print(
            f"{role_name:11s} | In:{r.demand_or_order_in:4d}  Recv:{r.received:4d}  "
            f"Ship:{r.shipped:4d}  InvEnd:{r.inv_end:4d}  BackEnd:{r.backlog_end:4d}  "
            f"InvCost:{r.inv_cost:7.2f}  BackCost:{r.backlog_cost:7.2f}  "
            f"WeekCost:{r.week_cost:8.2f}  Total:{r.total_cost:9.2f}"
        )
    print("=" * 90 + "\n")


def print_week_recommendations(week: int, orders: Dict[str, int], policy: Dict[str, int]) -> None:
    print(f"Week {week} Recommended Orders (optimized)")
    print("-" * 90)
    print(f"Retailer    -> Wholesaler : {orders['Retailer']:4d}   (target={policy['Retailer']} weeks)")
    print(f"Wholesaler  -> Distributor: {orders['Wholesaler']:4d}   (target={policy['Wholesaler']} weeks)")
    print(f"Distributor -> Factory    : {orders['Distributor']:4d}   (target={policy['Distributor']} weeks)")
    print(f"Factory production         : {orders['Factory']:4d}   (target={policy['Factory']} weeks)")
    print("=" * 90 + "\n")


def _mean_int(values: List[int], fallback: int = 100) -> int:
    if not values:
        return fallback
    return int(round(sum(values) / len(values)))


def get_customer_demand_plan(weeks: int, default: int = 100) -> List[int]:
    print("\nCustomer demand input mode:")
    print("  1) Constant demand every week")
    print("  2) Enter all weeks as comma-separated list")
    print("  3) Enter demand week-by-week")
    mode = input("Choose [1/2/3] (Enter=1): ").strip().lower()
    if mode in {"q", "quit", "exit"}:
        raise KeyboardInterrupt
    if mode == "" or mode == "1":
        d = get_int_input(f"Constant customer demand per week (Enter={default}): ", default=default, min_value=0)
        return [d] * weeks
    if mode == "2":
        raw = input(
            f"Enter {weeks} non-negative integers separated by commas/spaces:\n"
        ).strip().lower()
        if raw in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        tokens = [x for x in raw.replace(",", " ").split() if x]
        vals: List[int] = []
        for tok in tokens:
            try:
                vals.append(max(0, int(tok)))
            except ValueError:
                continue
        if not vals:
            print("No valid numbers found. Using default demand for all weeks.")
            return [default] * weeks
        if len(vals) < weeks:
            vals.extend([vals[-1]] * (weeks - len(vals)))
        return vals[:weeks]

    demands = []
    for w in range(1, weeks + 1):
        d = get_int_input(f"Customer demand week {w} (Enter={default}): ", default=default, min_value=0)
        demands.append(d)
    return demands


def run_base_stock_policy(
    demand_plan: List[int],
    target_weeks_by_role: Dict[str, int],
    base_game: Optional[BeerGame4Role] = None,
    show_weekly: bool = False,
) -> Tuple[float, BeerGame4Role]:
    if base_game is None:
        expected_demand = _mean_int(demand_plan)
        game = BeerGame4Role(weeks=len(demand_plan), expected_demand=expected_demand)
    else:
        game = copy.deepcopy(base_game)
        game.weeks = len(demand_plan)

    for week, demand in enumerate(demand_plan, start=1):
        orders: Dict[str, int] = {}
        for role_name, role in zip(ROLE_NAMES, game.roles):
            tw = target_weeks_by_role[role_name]
            orders[role_name] = game._suggest_order_base_stock(role, target_weeks=tw)
        res = game.step_week(week=week, customer_demand=demand, orders=orders)
        if show_weekly:
            print_week_summary(week, res)
    return game.total_system_cost(), game


def optimize_base_stock_policy(
    demand_plan: List[int],
    min_target_weeks: int = 1,
    max_target_weeks: int = 8,
    base_game: Optional[BeerGame4Role] = None,
) -> Tuple[Dict[str, int], float]:
    search_values = list(range(min_target_weeks, max_target_weeks + 1))
    best_policy: Dict[str, int] = {}
    best_cost = float("inf")

    for combo in product(search_values, repeat=4):
        policy = dict(zip(ROLE_NAMES, combo))
        cost, _ = run_base_stock_policy(
            demand_plan, policy, base_game=base_game, show_weekly=False
        )
        if cost < best_cost:
            best_cost = cost
            best_policy = policy

    return best_policy, best_cost


def run_manual_mode(weeks: int = 35, expected_demand: int = 100) -> None:
    game = BeerGame4Role(weeks=weeks, expected_demand=expected_demand)
    for week in range(1, game.weeks + 1):
        print(f"--- Week {week} ---")
        demand = get_int_input("Customer demand this week (Enter=100): ", default=100, min_value=0)
        sugg = {
            "Retailer": game._suggest_order_base_stock(game.retailer, target_weeks=4),
            "Wholesaler": game._suggest_order_base_stock(game.wholesaler, target_weeks=4),
            "Distributor": game._suggest_order_base_stock(game.distributor, target_weeks=4),
            "Factory": game._suggest_order_base_stock(game.factory, target_weeks=4),
        }
        orders = {
            "Retailer": get_order_input("Retailer", game.retailer.last_order, sugg["Retailer"]),
            "Wholesaler": get_order_input("Wholesaler", game.wholesaler.last_order, sugg["Wholesaler"]),
            "Distributor": get_order_input("Distributor", game.distributor.last_order, sugg["Distributor"]),
            "Factory": get_order_input("Factory(production)", game.factory.last_order, sugg["Factory"]),
        }
        res = game.step_week(week=week, customer_demand=demand, orders=orders)
        print_week_summary(week, res)
    print("GAME OVER")
    print(f"Total system cost (sum of all roles): ${game.total_system_cost():.2f}")
    for r in game.roles:
        print(f" - {r.name:11s}: ${r.total_cost:.2f}")


def run_optimization_mode() -> None:
    weeks = get_int_input("Number of weeks (Enter=35): ", default=35, min_value=1)
    min_tw = get_int_input("Min base-stock target (weeks of demand, Enter=1): ", default=1, min_value=1)
    max_tw = get_int_input("Max base-stock target (weeks of demand, Enter=6): ", default=6, min_value=min_tw)
    horizon = get_int_input("Look-ahead horizon in weeks (Enter=6): ", default=6, min_value=1)

    game = BeerGame4Role(weeks=weeks, expected_demand=100)
    observed_demands: List[int] = []

    print("\nRolling optimization mode:")
    print("Enter customer demand each week; the system optimizes orders for that week, then prints summary.\n")

    for week in range(1, weeks + 1):
        print(f"--- Week {week} ---")
        demand = get_int_input("Customer demand this week (Enter=100): ", default=100, min_value=0)
        observed_demands.append(demand)
        forecast = _mean_int(observed_demands, fallback=100)
        effective_horizon = min(horizon, weeks - week + 1)

        best_policy, _ = optimize_base_stock_policy(
            demand_plan=[forecast] * effective_horizon,
            min_target_weeks=min_tw,
            max_target_weeks=max_tw,
            base_game=game,
        )
        orders = {
            "Retailer": game._suggest_order_base_stock(game.retailer, target_weeks=best_policy["Retailer"]),
            "Wholesaler": game._suggest_order_base_stock(game.wholesaler, target_weeks=best_policy["Wholesaler"]),
            "Distributor": game._suggest_order_base_stock(game.distributor, target_weeks=best_policy["Distributor"]),
            "Factory": game._suggest_order_base_stock(game.factory, target_weeks=best_policy["Factory"]),
        }
        res = game.step_week(week=week, customer_demand=demand, orders=orders)
        print_week_summary(week, res)
        print_week_recommendations(week, orders, best_policy)

    print("ROLLING OPTIMIZATION COMPLETE")
    print(f"Total system cost (sum of all roles): ${game.total_system_cost():.2f}")
    for r in game.roles:
        print(f" - {r.name:11s}: ${r.total_cost:.2f}")


def main():
    print("Welcome to the FULL Beer Distribution Game Simulator (4 roles)!\n")
    print("Roles: Customer -> Retailer -> Wholesaler -> Distributor -> Factory\n")
    print("Delays: order processing 2 weeks, shipping/transport 2 weeks, factory production 2 weeks\n")
    print("Costs: inventory $0.25/unit/week, backlog $1.00/unit/week\n")
    print("Type 'q' anytime to quit.\n")
    print("Modes:")
    print("  1) Manual weekly orders (interactive)")
    print("  2) Enter weekly demand + optimize orders week by week")

    mode = input("Choose mode [1/2] (Enter=2): ").strip().lower()
    if mode in {"q", "quit", "exit"}:
        raise KeyboardInterrupt
    if mode == "1":
        run_manual_mode()
    else:
        run_optimization_mode()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
