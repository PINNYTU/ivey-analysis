# %% [markdown]
# # Session 1: The Walton Bookstore Simulation (Foundations Track)
#
# ## Executive Summary
# **The Business Problem:**
# Walton Bookstore must decide how many calendars to order for the upcoming year. This is a classic "Newsvendor Problem":
# * If we order too few, we lose potential sales (Opportunity Cost).
# * If we order too many, we are stuck with worthless inventory (Overage Cost).
#
# **The Flaw of Averages:**
# Management believes that since the **average demand** is 200, ordering 200 units is the safest bet.
# In this simulation, we will prove that ordering the average demand often leads to **less than average profit**.
#
# **Key Inputs:**
# * **Unit Cost:** $7.50 (What we pay)
# * **Unit Price:** $10.00 (What customers pay)
# * **Refund:** $2.50 (What we get back for unsold items)
# * **Demand:** Uncertain! Follows a Normal Distribution (Mean=200, SD=40).
# * **Decision:** We are currently ordering **200** calendars.

# %%
# ---------------------------------------------------------
# STEP 1: SETUP
# ---------------------------------------------------------
# We import 'numpy' for math operations (arrays) and 'matplotlib' for graphing.
import numpy as np
import matplotlib.pyplot as plt
import simpy

# We set a "seed" so everyone gets the same random numbers.
# This ensures your results match the instructor's screen.
np.random.seed(2026)

print("Setup Complete. Libraries loaded.")

# %%
# ---------------------------------------------------------
# STEP 2: DEFINE THE INPUTS
# ---------------------------------------------------------
# We store our business assumptions in variables.
# This is like defining your "Input Cells" at the top of an Excel sheet.

unit_cost = 7.50       # Cost per calendar
unit_price = 10.00     # Selling price per calendar
unit_refund = 2.50     # Salvage value per unsold calendar

# Decision Variable
order_quantity = 200   # The number of calendars we buy

# Simulation Settings
n_trials = 1000        # We will simulate 1,000 different futures (days)

print(f"Simulation Parameters Set: Ordering {order_quantity} units for {n_trials} trials.")

# %%
# ---------------------------------------------------------
# STEP 3: GENERATE DEMAND (The Randomness)
# ---------------------------------------------------------
# In Excel, you might use: =NORMINV(RAND(), 200, 40)
# In Python, we generate all 1,000 scenarios at once using 'numpy'.

# Logic: Generate 1,000 random numbers from a Normal Distribution (Mean=200, SD=40)
demand_simulations = np.random.normal(loc=200, scale=40, size=n_trials)

# Round to nearest whole number (since we can't sell 0.5 calendars)
demand_simulations = np.round(demand_simulations)

# Print the first 10 scenarios to check our work
print("First 10 Demand Scenarios:", demand_simulations[:10])

# %%
# ---------------------------------------------------------
# STEP 4: CALCULATE FINANCIALS (The Business Logic)
# ---------------------------------------------------------
# This section translates business rules into code.
# In Python, we perform operations on the WHOLE array at once (Vectorization).

# --- A. Units Sold ---
# Logic: You cannot sell more calendars than you ordered.
#        You also cannot sell more than customers want (demand).
# Excel Equivalent: =MIN(Demand, Order_Qty)
units_sold = np.minimum(demand_simulations, order_quantity)

# --- B. Units Leftover ---
# Logic: Anything we ordered but didn't sell.
# Excel Equivalent: =MAX(0, Order_Qty - Demand)
units_leftover = order_quantity - units_sold

# --- C. Financials ---
# Now use standard math (+, -, *) to calculate the money.

# Revenue: Money IN from sales
revenue = units_sold * unit_price

# Cost: Money OUT for buying calendars
# Note: We pay for *everything* we ordered, not just what we sold.
cost = order_quantity * unit_cost

# Refund: Money BACK from unsold items
refund = units_leftover * unit_refund

# Profit: The Bottom Line
# Logic: Profit = Revenue - Cost + Refund
profit_array = revenue - cost + refund

# Check the first 5 outcomes
print("Calculation Complete.")
print(f"First 5 Profit Outcomes: {profit_array[:5]}")

# %%
# ---------------------------------------------------------
# STEP 5: ANALYZE RISK (The Output)
# ---------------------------------------------------------
# Now we analyze the 'profit_array' to answer the business question.

# 1. Calculate Average Profit
average_profit = np.mean(profit_array)
std_dev_profit = np.std(profit_array)

print("-" * 30)
print(f"DECISION SUMMARY (Order Qty: {order_quantity})")
print(f"Average Profit:     ${average_profit:.2f}")
print(f"Standard Deviation: ${std_dev_profit:.2f}")
print(f"Minimum Profit:     ${np.min(profit_array):.2f}")
print(f"Maximum Profit:     ${np.max(profit_array):.2f}")
print("-" * 30)

# 2. Visualizing Risk (Histogram)
# This shows the distribution of possible outcomes.
plt.figure(figsize=(10,6))
plt.hist(profit_array, bins=30, color='skyblue', edgecolor='black', alpha=0.7)

# Add a vertical line for the average
plt.axvline(average_profit, color='red', linestyle='dashed', linewidth=2, label=f"Avg: ${average_profit:.0f}")

plt.title(f"Risk Profile for Ordering {order_quantity} Calendars")
plt.xlabel("Profit ($)")
plt.ylabel("Frequency (out of 1000 trials)")
plt.legend()
plt.grid(axis='y', alpha=0.5)
plt.show()