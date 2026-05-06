What this model does
This model simulates a 4-stage beer supply chain:

Retailer -> Wholesaler -> Distributor -> Factory

The goal is to keep the whole supply chain cost as low as possible each week by balancing:

Inventory cost: $0.25 per unit per week
Backlog cost: $1.00 per unit per week
Weekly flow (what happens every week)
Each week, the model runs this sequence:

Each role receives any shipments arriving this week.
Each role receives incoming demand/orders:
Retailer gets customer demand directly.
Other roles get delayed orders from downstream.
Each role ships as much as possible using on-hand inventory to satisfy:
backlog + new incoming order
Unmet demand becomes backlog.
Weekly costs are calculated:
InvCost = inventory × 0.25
BackCost = backlog × 1.00
Each role places a new order upstream (Factory places a production order).
Delays (key concept)
Order information delay: 2 weeks
Shipping/production delay: 2 weeks
Total time from placing an order to receiving goods: about 4 weeks
Because of this delay, demand changes can cause overreaction and oscillations (the bullwhip effect).

Starting conditions
At the beginning:

Each role has 100 units on hand
Each role has 100 units in transit over the next 2 weeks (50, 50)
Expected demand is around 100 units/week
Script modes
Mode 1: Manual
You enter order decisions for each role every week.

Mode 2: Rolling Optimization
You enter customer demand each week, and the model automatically recommends orders that minimize cost.
------------------------------------------------
You set 3 optimization controls:

- Min base-stock target (weeks of demand) = 1
Lowest target the optimizer can choose for any role.

- Max base-stock target (weeks of demand) = 6
Highest target the optimizer can choose for any role.

- Look-ahead horizon (weeks) = 6
Each week, the optimizer simulates candidate policies over the next 6 weeks and picks the one with the lowest total inventory + backlog cost.
------------------------------------------------
Then the model:
    Computes the best near-term policy
    Prints recommended order quantities for all roles
    Applies those orders
    Prints the weekly summary
    How the optimizer works (simple)
    Tries many candidate base-stock policies
    Simulates each policy forward from the current state
    Selects the policy with the lowest total cost
    Converts that policy into this week’s recommended orders
-------------------------------------------------
How to read the weekly summary:
    In: demand/order arriving this week
    Recv: shipment/production received this week
    Ship: units shipped this week
    InvEnd: ending inventory
    BackEnd: ending backlog
    InvCost / BackCost / WeekCost / Total: weekly cost breakdown and cumulative cost