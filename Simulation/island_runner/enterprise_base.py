"""
Enterprise Contract — Pure Abstract Base Class (Anemic Model).

Defines ONLY the interface that island_runner.py depends on.
Contains NO operational logic, NO constants, NO concrete methods.

All data (prices, constants) comes from island_grid_B.py (single source of truth).
All operational logic lives in the concrete subclasses (bakery/fish/greenhouse).
"""

from abc import ABC, abstractmethod


class EnterpriseDay(ABC):
    """
    Abstract contract for all Tri-Island Alliance enterprises.

    island_runner.py calls:
      - __init__(config, env, grid, prices)
      - .simulate(seed)    → standalone mode (Option B)
      - .run()             → shared-env SimPy generator (Option A)
      - .end_of_day()      → post-shift P&L (Option A)
      - .get_state()       → access production state

    Each subclass (Team A/B/C) implements the full operational logic:
      SimPy machine process, market computation, cost accounting,
      and enterprise-specific production/waste behavior.
    """

    # ── Abstract Properties (Enterprise Parameters) ────────────────────

    @property
    @abstractmethod
    def enterprise_name(self) -> str:
        """Identifier: 'bakery', 'fish_farm', 'greenhouse'."""
        ...

    @property
    @abstractmethod
    def num_machines(self) -> int:
        """Number of production units (ovens / tanks / hydroponic units)."""
        ...

    @property
    @abstractmethod
    def stagger_minutes(self) -> float:
        """Minutes between successive machine starts."""
        ...

    @property
    @abstractmethod
    def demand_mean(self) -> float:
        """Mean of Poisson demand distribution."""
        ...

    @property
    @abstractmethod
    def setup_time(self) -> float:
        """Worker time for loading/seeding (minutes)."""
        ...

    @property
    @abstractmethod
    def production_time(self) -> float:
        """Autonomous cycle time (bake/maturation/growth) in minutes."""
        ...

    @property
    @abstractmethod
    def teardown_time(self) -> float:
        """Worker time for unloading/extraction/harvest (minutes)."""
        ...

    @property
    @abstractmethod
    def yield_per_batch(self) -> float:
        """Output per successful batch (loaves or kg)."""
        ...

    @property
    @abstractmethod
    def sale_price(self) -> float:
        """Revenue per unit sold."""
        ...

    @property
    @abstractmethod
    def gas_per_batch(self) -> float:
        """Gas consumed during one production cycle."""
        ...

    @property
    @abstractmethod
    def output_key(self) -> str:
        """State dict key for production output: 'loaves', 'fish_kg', 'basil_kg'."""
        ...

    # ── Abstract Methods (Operations) ──────────────────────────────────

    @abstractmethod
    def simulate(self, seed=None) -> dict:
        """Run one simulated day in standalone mode (own SimPy env).
        Returns a results dict with at least: profit, revenue, service_level."""
        ...

    @abstractmethod
    def run(self):
        """SimPy generator for shared-environment mode (Option A).
        Register with: env.process(enterprise.run())"""
        ...

    @abstractmethod
    def end_of_day(self) -> dict:
        """Post-shift market + P&L. Call after env.run() in Option A.
        Returns same dict structure as simulate()."""
        ...

    # ── Constructor (stores references only) ───────────────────────────

    def __init__(self, config: dict, env=None, grid=None, prices=None):
        if prices is None:
            raise ValueError(
                "prices dict is required. Import PHASE_2_PRICES from "
                "island_grid_B and pass it to the enterprise constructor."
            )
        self.config = config
        self.env = env
        self.grid = grid
        self.prices = prices
        self._state = None

    def get_state(self) -> dict:
        """Access production state after simulation completes."""
        return self._state
