# Southern Reefs — Phase B Simulation Results

**Island-Wide Integrated Simulation with CTA Directives**
**Date:** 2026-02-13 | **Config:** Default (3 ovens, 6 tanks, 3 GH units)

---

## 1. CTA Directives & Phase 2 Prices

### CTA Policy Parameters (from `island_grid_B.py`)

| Directive | Value |
|---|---|
| Bakery Heat → Fish Farm credit | $5.50/unit |
| Captured heat exempt from waste tax | Yes |
| CO2 Capture Credit (Bakery/Fish → GH) | $5.00/kg |
| Basil Partner Discount (GH → Bakery) | 15% |
| Max Ovens per Baker | 3 |
| Effluent Rebate (Fish → GH) | $5.00/kg |
| Heat Rebate (warm effluent → GH) | $5.00/kg |
| Tiered Waste Tax | $2/kg (first 100) → $3/kg (next 100) → $5/kg (remainder) |
| Brand Damage SL Floor | 50% |

### Phase 2 Prices (Shock)

| Item | Price |
|---|---|
| Gas | $0.50/unit (10x Phase 1) |
| Waste Tax (flat, pre-CTA) | $5.00/kg |
| Flour | $1.50/kg |
| Basil Import (mainland) | $10.00/kg |
| Fish Feed | $3.00/kg |
| Oxygen | $30.00/unit |
| Bread Price | $8.00/loaf |
| Fish Price | $24.00/kg |
| Basil Price | $8.00/kg |

---

## 2. Single-Day Results (seed=42)

### Circular + CTA Mode

| Metric | Bakery | Fish Farm | Greenhouse | Island |
|---|---:|---:|---:|---:|
| **Profit** | $276.80 | $5,006.00 | -$1,713.80 | **$3,569.00** |
| Service Level | 91.6% | 100.0% | 5.2% | — |
| Revenue | $2,880.00 | $7,272.00 | $400.00 | — |
| Production | 360 loaves | 400 kg | 50 kg | — |
| Batches (start/done/fail) | 12/9/3 | 8/8/0 | 5/2/3 | — |
| Gas Used | 12.0 | 24.0 | 10.0 | 46.0 |
| Repair Cost | $900.00 | $0.00 | $900.00 | $1,800.00 |
| CTA Credits | $126.00 | $240.00 | $120.00 | **$486.00** |
| Waste Tax | $0.00 | — | — | **$0.00** |

**CTA Credits Breakdown:**
- Bakery: Heat credit $66.00 + CO2 credit $60.00 = $126.00
- Fish Farm: Effluent rebate $120.00 + CO2 credit $120.00 = $240.00
- Greenhouse: Heat rebate (warm effluent) $120.00

**Circular Resource Flows:**
- Heat: 12.0 units supplied → 12.0 to Fish Farm / 0.0 surplus
- CO2: 12.0 kg (bakery) + 24.0 kg (fish) = 36.0 kg captured by Greenhouse
- Effluent: 24.0 kg captured by Greenhouse
- Basil: 24.0 kg to Bakery (15% discount) / 26.0 kg to market / 0.0 imported

### Standalone Mode (No Circular, No CTA)

| Metric | Bakery | Fish Farm | Greenhouse | Island |
|---|---:|---:|---:|---:|
| **Profit** | -$77.92 | $4,664.00 | -$1,805.00 | **$2,781.08** |
| Waste Tax | — | — | — | **$144.00** |
| Basil Imported | 24.0 kg | — | — | — |

**Single-day CTA advantage: +$787.92 (+28.3%)**

### Real-Time Mode (Shared SimPy Environment, seed=42)

| Metric | Bakery | Fish Farm | Greenhouse | Island |
|---|---:|---:|---:|---:|
| **Profit** | -$30.50 | $3,215.00 | $491.00 | **$3,675.50** |
| Service Level | 81.4% | 100.0% | 47.5% | — |

Grid container levels at end-of-shift:
- Energy (heat): 11.0 units remaining
- Water (effluent): 21.0 units remaining
- Produce (basil): 225.0 kg remaining

---

## 3. Monte Carlo Results (500 trials)

### Circular + CTA

| Metric | Mean | Std Dev | 5th %ile | 95th %ile |
|---|---:|---:|---:|---:|
| **Island GDP** | **$6,526.37** | $972.43 | $4,618.28 | $7,810.50 |
| Bakery Profit | $1,712.04 | — | — | — |
| Fish Farm Profit | $4,505.54 | — | — | — |
| Greenhouse Profit | $308.79 | — | — | — |
| Waste Tax | $0.00 | — | — | — |
| CTA Credits | $503.75 | — | — | — |
| Basil Imported | 0.0 kg | — | — | — |
| Heat Surplus | 0.0 units | — | — | — |

**Service Levels:**

| Enterprise | Mean SL | P(SL >= 90%) |
|---|---:|---:|
| Bakery | 99.0% | 96.0% |
| Fish Farm | 100.0% | 100.0% |
| Greenhouse | 35.7% | 0.0% |

### Standalone (No CTA)

| Metric | Mean | Std Dev | 5th %ile | 95th %ile |
|---|---:|---:|---:|---:|
| **Island GDP** | **$5,694.13** | $949.67 | $3,876.82 | $6,953.31 |
| Bakery Profit | $1,302.01 | — | — | — |
| Fish Farm Profit | $4,167.40 | — | — | — |
| Greenhouse Profit | $224.72 | — | — | — |
| Waste Tax | $151.37 | — | — | — |
| CTA Credits | $0.00 | — | — | — |
| Basil Imported | 28.5 kg | — | — | — |
| Heat Surplus | 14.2 units | — | — | — |

---

## 4. Circular + CTA vs Standalone Comparison

| Metric | Circular + CTA | Standalone | Delta |
|---|---:|---:|---:|
| **Island GDP** | **$6,526.37** | **$5,694.13** | **+$832.23/day (+14.6%)** |
| Bakery Profit | $1,712.04 | $1,302.01 | +$410.02 |
| Fish Farm Profit | $4,505.54 | $4,167.40 | +$338.14 |
| Greenhouse Profit | $308.79 | $224.72 | +$84.07 |
| Waste Tax | $0.00 | $151.37 | -$151.37 (saved) |
| CTA Credits | $503.75 | $0.00 | +$503.75 |

### How CTA Creates Value

1. **Waste tax elimination ($151/day saved):** All heat, CO2, and effluent are captured through the circular system, so zero waste is vented. Standalone mode pays $151/day in tiered waste taxes.

2. **CTA incentive credits ($504/day earned):**
   - Bakery heat credit: ~$78/day (heat piped to Fish Farm)
   - Bakery CO2 credit: ~$71/day (CO2 diverted to Greenhouse)
   - Fish effluent rebate: ~$143/day (effluent sent to Greenhouse)
   - Fish CO2 credit: ~$143/day (CO2 captured by Greenhouse)
   - Greenhouse heat rebate: ~$69/day (warm effluent from Fish Farm)

3. **Basil import elimination:** Circular mode sources all basil locally from Greenhouse at a 15% CTA discount, avoiding $10/kg mainland import + ordering fees.

4. **Fish Farm gas savings:** Heat from Bakery ovens offsets Fish Farm gas consumption, reducing gas cost at Phase 2's $0.50/unit rate.

---

## 5. Key Observations

- **Fish Farm is the profit engine:** Contributes ~69% of island GDP in both modes. High yield (50 kg/batch), strong demand coverage (100% SL), and minimal failures.

- **Greenhouse has low service level (35.7%):** With only 3 units and high demand (500 mean), production capacity is insufficient. This is a bottleneck — increasing units would improve both SL and GDP.

- **Bakery benefits most from CTA (+31.5%):** Heat credits + CO2 credits + cheap local basil transform Bakery from a marginal enterprise ($1,302 standalone) to a strong contributor ($1,712 circular).

- **Zero waste achieved:** The circular economy fully captures all byproducts — no heat, CO2, or effluent is vented. This is the CTA's primary design goal.

- **CTA oven cap (3) is binding:** The default config already uses 3 ovens, which is the CTA maximum. The grid search optimizer respects this constraint.

---

## 6. Configuration

**Default config used for all results above:**

```python
{
    "num_ovens": 3,        # Bakery (CTA cap: 3)
    "bakery_stagger": 15,  # minutes between oven starts
    "bakery_demand": 400,  # Poisson mean
    "num_tanks": 6,        # Fish Farm
    "fish_mode": "optimal",
    "fish_stagger": 35,
    "fish_demand": 200,
    "num_units": 3,        # Greenhouse
    "gh_mode": "high",     # 25 kg/batch, 1 gas/hr
    "gh_stagger": 14,
    "gh_demand": 500,
    "use_grid_basil": True,
    "use_grid_heat": True,
    "use_grid_nutrients": True,
}
```

**Simulation parameters:**
- Shift length: 480 minutes (8 hours)
- Failure rate: 5% per batch
- Repair cost: $300, repair time: 2-4 hours
- Liquidation rate: 50% (unsold goods)
- Lease: $400/unit/day
- Admin overhead: $100/day
