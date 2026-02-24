# Southern Reefs — Phase B Island Runner

## Architecture

```
island_grid_B.py          (Single Source of Truth)
  Prices, CTA Directives, Operational Constants, IslandGrid

enterprise_base.py        (Anemic ABC — Pure Contract)
  Abstract properties + abstract methods only

bakery_model.py           (Team A — Flourish & Crumb)
fish_farm_model.py        (Team B — Blue Horizon)
greenhouse_model.py       (Team C — Verdant Canopy)
  Each: full operational logic (SimPy process, market, costs, simulate/run/end_of_day)

island_runner.py          (Orchestrator)
  simulate_island_day()        — Option B: standalone envs + post-hoc circular accounting
  simulate_island_day_realtime() — Option A: shared SimPy env + IslandGrid containers
  island_monte_carlo()         — Monte Carlo wrapper
  island_grid_search()         — Grid search over topology

island_optimizer.py       (Optimizer)
  Grid search + circular vs standalone comparison + reporting
```

## Data Flow

```
island_grid_B.py
  ├── PHASE_2_PRICES ──────────→ enterprise constructors (prices=)
  ├── CTA_DIRECTIVES ──────────→ island_runner.py (circular accounting)
  ├── SHIFT_MIN, FAILURE_RATE, → enterprise models (operational constants)
  │   LEASE_PER_UNIT, ...
  └── IslandGrid ──────────────→ Option A shared containers
```

## Circular Economy (CTA Directives)

```
Bakery ──heat──→ Fish Farm        ($5.50/unit credit, tax exempt)
Bakery ──CO2───→ Greenhouse       ($5.00/kg capture credit)
Fish   ──CO2───→ Greenhouse       ($5.00/kg capture credit)
Fish   ──effluent→ Greenhouse     ($5.00/kg rebate + $5.00/kg heat rebate)
Greenhouse ──basil→ Bakery        (15% partner discount)
```

Surplus waste: tiered tax $2/kg (first 100) → $3/kg (next 100) → $5/kg (remainder).

## Design Principles

- **`enterprise_base.py` is anemic**: defines the contract (what), not the behavior (how)
- **Each enterprise model is self-contained**: owns its full SimPy process, market logic, cost accounting
- **`island_grid_B.py` is the single source of truth**: all prices, CTA parameters, and operational constants live there
- **`island_runner.py` orchestrates**: instantiates enterprises, runs simulations, applies CTA post-hoc

## Usage

```bash
# Single day (circular + CTA)
python island_runner.py --ovens 3 --tanks 6 --units 3

# Single day (standalone, no CTA)
python island_runner.py --standalone

# Real-time shared environment (Option A)
python island_runner.py --realtime

# Grid search optimization
python island_runner.py --search --trials 200

# Full optimizer with comparison report
python island_optimizer.py
```

## Known Issues

See `issue_we_have.md`:
- CO2 from Bakery/Fish goes to waste tax in standalone mode because Greenhouse's CO2 input/output is 1:1 (self-consumed). CTA assumes CO2 can be recycled through Greenhouse but case data doesn't provide room for extra CO2 absorption. The circular mode handles this by granting capture credits regardless.
