# -*- coding: utf-8 -*-
"""Island Grid — Single Source of Truth for ALL Simulation Parameters.

All simulation code reads prices, CTA directives, and operational constants
from THIS file.  Enterprise models, runners, and optimizers import from here.
"""

import simpy


# ══════════════════════════════════════════════════════════════════════════
#  OPERATIONAL CONSTANTS (shared by all enterprises)
# ══════════════════════════════════════════════════════════════════════════

SHIFT_MIN      = 480          # 8-hour shift in minutes
LEASE_PER_UNIT = 400          # $/unit/day
ADMIN_OVERHEAD = 100          # $/day
FAILURE_RATE   = 0.05         # probability per batch
REPAIR_COST    = 300          # $ per failure
REPAIR_RANGE   = (120, 240)   # repair time range in minutes (2-4 hrs)
LIQUIDATION    = 0.50         # unsold goods recovery rate


# ══════════════════════════════════════════════════════════════════════════
#  PHASE PRICES (canonical source)
# ══════════════════════════════════════════════════════════════════════════

PHASE_1_PRICES = {
    # Energy & Waste
    "GAS": 0.05,
    "WASTE_TAX": 0.00,
    # Bakery Inputs
    "FLOUR": 1.10,
    "BASIL_IMPORT": 5.00,
    "ORDER_FEE_BASIL": 50.0,
    "STORAGE_FEE_BASIL": 1.00,
    # Fish Farm Inputs
    "FISH_FEED": 2.20,
    "OXYGEN": 10.00,
    "ORDER_FEE_OXYGEN": 200.0,
    "STORAGE_FEE_OXYGEN": 0.50,
    # Greenhouse Inputs
    "WATER": 0.50,
    "NUTRIENTS": 20.00,
    "ORDER_FEE_NUTRIENTS": 200.0,
    "STORAGE_FEE_NUTRIENTS": 0.50,
    # Revenue
    "BREAD_PRICE": 8.00,
    "FISH_PRICE": 24.00,
    "BASIL_PRICE": 8.00,
}

PHASE_2_PRICES = {
    # Energy & Waste (Shock)
    "GAS": 0.50,              # 10x surge
    "WASTE_TAX": 5.00,        # Circularity incentive
    # Bakery Inputs
    "FLOUR": 1.50,
    "BASIL_IMPORT": 10.00,
    "ORDER_FEE_BASIL": 150.0,
    "STORAGE_FEE_BASIL": 1.50,
    # Fish Farm Inputs
    "FISH_FEED": 3.00,
    "OXYGEN": 30.00,
    "ORDER_FEE_OXYGEN": 300.0,
    "STORAGE_FEE_OXYGEN": 0.75,
    # Greenhouse Inputs
    "WATER": 1.00,
    "NUTRIENTS": 30.00,
    "ORDER_FEE_NUTRIENTS": 300.0,
    "STORAGE_FEE_NUTRIENTS": 0.75,
    # Revenue (Locked)
    "BREAD_PRICE": 8.00,
    "FISH_PRICE": 24.00,
    "BASIL_PRICE": 8.00,
}


# ══════════════════════════════════════════════════════════════════════════
#  CTA DIRECTIVES — Council of Tri-Island Alliance Policy Parameters
# ══════════════════════════════════════════════════════════════════════════

CTA_DIRECTIVES = {
    # ── Bakery: Thermal Connection Mandate ──
    # Bakery pipes oven heat to Fish Farm → exempt from heat waste tax
    # + receives $5.50/unit Resilience Credit
    "HEAT_CREDIT_PER_UNIT": 5.50,
    "HEAT_TAX_EXEMPT_IF_CAPTURED": True,

    # ── Bakery: Carbon Neutrality Mandate ──
    # Bakery diverts CO2 to Greenhouse → Greenhouse gives 15% discount on basil
    "CO2_TAX_CREDIT_PER_KG": 5.00,
    "BASIL_PARTNER_DISCOUNT": 0.15,     # 15% off local basil price for Bakery

    # ── Bakery: 3:1 Oven-to-Baker Ratio ──
    "MAX_OVENS_PER_BAKER": 3,

    # ── Aquaculture → Agriculture: Effluent & Heat Rebate ──
    # Effluent supplied from Aqua to Agriculture: waste tax rebate
    "EFFLUENT_REBATE_PER_KG": 5.00,
    # Warm effluent also carries heat → heat rebate for Agriculture
    "HEAT_REBATE_PER_KG": 5.00,

    # ── CO2 Capture Credit (Aqua + Bakery → Agriculture) ──
    # CO2 reused in greenhouse maturation earns carbon tax credit
    "CO2_CAPTURE_CREDIT_PER_KG": 5.00,

    # ── Tiered Waste Tax (for non-captured/overflow waste) ──
    # Replaces flat $5/kg for firms not fully participating in circular system
    "WASTE_TAX_TIERS": [
        (100, 2.00),              # first 100 kg at $2.00/kg
        (100, 3.00),              # next 100 kg at $3.00/kg
        (float("inf"), 5.00),     # remainder at $5.00/kg (standard rate)
    ],

    # ── Brand Damage: Service Level Floor ──
    "BRAND_DAMAGE_SL_FLOOR": 50,   # SL can scale down to 50%
}


# ══════════════════════════════════════════════════════════════════════════
#  HELPER: Tiered waste tax calculation
# ══════════════════════════════════════════════════════════════════════════

def compute_tiered_waste_tax(kg_vented, directives=None):
    """Compute waste tax using CTA tiered schedule.

    Parameters
    ----------
    kg_vented : float
        Total kg of waste vented (not captured).
    directives : dict, optional
        CTA directives dict. Defaults to module-level CTA_DIRECTIVES.

    Returns
    -------
    float
        Total waste tax owed.
    """
    if directives is None:
        directives = CTA_DIRECTIVES
    tiers = directives.get("WASTE_TAX_TIERS")
    if tiers is None:
        # fallback to flat rate
        return kg_vented * PHASE_2_PRICES["WASTE_TAX"]

    remaining = kg_vented
    total_tax = 0.0
    for band_kg, rate in tiers:
        taxable = min(remaining, band_kg)
        total_tax += taxable * rate
        remaining -= taxable
        if remaining <= 0:
            break
    return total_tax


# ══════════════════════════════════════════════════════════════════════════
#  ISLAND GRID — Shared SimPy Infrastructure
# ══════════════════════════════════════════════════════════════════════════

class IslandGrid:
    """
    The Shared Infrastructure for the 'Archipelago Survival' Capstone.
    Updated for Case (B): The Great Re-Alignment.
    """
    def __init__(self, env):
        self.env = env

        # --- THE RESOURCE CONTAINERS (Circular Logic) ---
        self.energy = simpy.Container(env, capacity=10000, init=0)  # Heat
        self.water = simpy.Container(env, capacity=5000, init=0)    # Nitrated Water
        self.produce = simpy.Container(env, capacity=500, init=0)   # Fresh Basil

    def get_prices(self, phase):
        """Returns the Market Price Dictionary for Phase A or Phase B."""
        if phase == "PHASE_1":
            return PHASE_1_PRICES
        elif phase == "PHASE_2":
            return PHASE_2_PRICES
        else:
            raise ValueError(f"Invalid Phase '{phase}'. Use PHASE_1 or PHASE_2.")


if __name__ == "__main__":
    print("IslandGrid (Full Logistics B) loaded.")
    print(f"Phase 2 Gas: ${PHASE_2_PRICES['GAS']}")
    print(f"CTA Heat Credit: ${CTA_DIRECTIVES['HEAT_CREDIT_PER_UNIT']}/unit")
    print(f"CTA Basil Discount: {CTA_DIRECTIVES['BASIL_PARTNER_DISCOUNT']*100:.0f}%")
    print(f"Tiered waste tax on 250 kg: ${compute_tiered_waste_tax(250):.2f}")
