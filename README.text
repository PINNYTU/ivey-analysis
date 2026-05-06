What this model is
This model simulates the full Beer Distribution Game with four connected roles: Retailer -> Wholesaler -> Distributor -> Factory.
The objective is to minimize total weekly supply-chain cost, made up of:

Inventory holding cost = $0.25 per unit per week
Backlog cost = $1.00 per unit per week
How each week works (core logic)
Each simulated week follows this sequence:

Each role receives incoming goods scheduled to arrive that week.
Each role receives incoming demand/order:
Retailer receives customer demand.
Other roles receive delayed orders from their downstream role.
Each role ships from on-hand inventory to satisfy backlog + new incoming order.
Any unmet demand becomes new backlog.
Weekly costs are calculated:
InvCost = inventory * 0.25
BackCost = backlog * 1.00
Each role places a new order upstream (Factory places a production order).
Delays (most important concept)

Order information delay: 2 weeks
Shipping/production delay: 2 weeks
So, total delay from placing an order to receiving goods is about 4 weeks.
This is the main reason demand changes can create oscillations (bullwhip effect).
Initial conditions
At the start:

Each role has 100 units on hand
Each role has 100 units in transit across the next 2 weeks (50, 50)
Expected demand is around 100 units/week
Modes in the script

Mode 1 (Manual): You decide orders for each role each week.
Mode 2 (Rolling Optimization):
You enter weekly customer demand.
The model computes near-term best base-stock targets for all roles.
It prints recommended order quantities for each role.
It applies those orders and prints the weekly summary.
What the optimizer is doing (simple explanation)

It tests many candidate base-stock policies (target weeks of demand per role).
For each policy, it simulates forward from the current state.
It picks the policy with the lowest combined inventory + backlog cost.
It converts that policy into this week’s recommended orders.
How to read the weekly summary

In: demand/order arriving this week
Recv: shipment/production received this week
Ship: units shipped out this week
InvEnd: ending inventory
BackEnd: ending backlog
InvCost / BackCost / WeekCost / Total: weekly cost breakdown and cumulative total