"""Emergency Trading Kill Switch and Risk Circuit Breaker for WealthPilot AI.

Provides automated and manual circuit-breaker kill-switch mechanisms halting autonomous
execution during extreme market volatility (e.g. VIX > 40 or error spike > 1%), severe market shocks,
or manual emergency intervention.
"""

from __future__ import annotations

import datetime
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class CircuitBreakerState(str, Enum):
    NORMAL = "NORMAL"          # Full autonomous trading enabled
    THROTTLED = "THROTTLED"    # Heightened risk; reduced limits and mandatory advisory confirmations
    HALTED = "HALTED"          # Platform or portfolio-level autonomous execution strictly frozen


class KillSwitch:
    """Instant execution pause and circuit breaker across single portfolios or the entire platform."""

    def __init__(
        self,
        vix_halt_threshold: float = 40.0,
        vix_throttle_threshold: float = 30.0,
        error_rate_halt_threshold: float = 0.01,  # 1%
        daily_market_drop_halt_threshold: float = -0.05,  # -5% in 1 day
    ) -> None:
        self.vix_halt = vix_halt_threshold
        self.vix_throttle = vix_throttle_threshold
        self.error_rate_halt = error_rate_halt_threshold
        self.daily_market_drop_halt = daily_market_drop_halt_threshold

        self._state: CircuitBreakerState = CircuitBreakerState.NORMAL
        self._halt_reason: Optional[str] = None
        self._tripped_by: Optional[str] = None
        self._tripped_at_utc: Optional[str] = None

        self._halted_portfolios: Set[str] = set()
        self._audit_log: List[Dict[str, Any]] = []

    @property
    def state(self) -> CircuitBreakerState:
        return self._state

    @property
    def is_globally_halted(self) -> bool:
        return self._state == CircuitBreakerState.HALTED

    def evaluate_automated_triggers(
        self,
        vix_level: Optional[float] = None,
        failed_executions: int = 0,
        total_executions: int = 0,
        daily_market_return: Optional[float] = None,
    ) -> Tuple[CircuitBreakerState, Optional[str]]:
        """Evaluates live market and operational telemetry to trigger automated circuit breakers."""
        # 1. Market Volatility (VIX)
        if vix_level is not None:
            if vix_level >= self.vix_halt:
                reason = f"AUTOMATED TRIP: Extreme India VIX spike ({vix_level:.1f} >= {self.vix_halt:.1f})"
                self.activate_global_halt(reason=reason, source="AUTOMATED_VIX_MONITOR")
                return CircuitBreakerState.HALTED, reason
            elif vix_level >= self.vix_throttle and self._state == CircuitBreakerState.NORMAL:
                self._state = CircuitBreakerState.THROTTLED
                self._log_event("CIRCUIT_THROTTLED", f"India VIX elevated at {vix_level:.1f}", "SYSTEM")
                return CircuitBreakerState.THROTTLED, f"India VIX elevated at {vix_level:.1f}"

        # 2. Execution / Optimizer Error Spike (> 1%)
        if total_executions > 100:
            error_rate = failed_executions / total_executions
            if error_rate >= self.error_rate_halt:
                reason = f"AUTOMATED TRIP: Operational error rate ({error_rate:.2%}) exceeded safety threshold ({self.error_rate_halt:.2%})"
                self.activate_global_halt(reason=reason, source="AUTOMATED_ERROR_MONITOR")
                return CircuitBreakerState.HALTED, reason

        # 3. Sudden Market Crash / Flash Drop
        if daily_market_return is not None and daily_market_return <= self.daily_market_drop_halt:
            reason = f"AUTOMATED TRIP: Severe market flash drop ({daily_market_return:.2%} <= {self.daily_market_drop_halt:.2%})"
            self.activate_global_halt(reason=reason, source="AUTOMATED_MARKET_MONITOR")
            return CircuitBreakerState.HALTED, reason

        return self._state, None

    def activate_global_halt(self, reason: str, source: str = "MANUAL_OPERATOR") -> None:
        """Instantly halts all autonomous rebalancing execution across the platform."""
        self._state = CircuitBreakerState.HALTED
        self._halt_reason = reason
        self._tripped_by = source
        self._tripped_at_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        logger.critical(f"GLOBAL KILL SWITCH ACTIVATED by {source}: {reason}")
        self._log_event("GLOBAL_HALT_ACTIVATED", reason, source)

    def deactivate_global_halt(self, reset_by: str = "ADMIN", justification: str = "Conditions normalized") -> None:
        """Resets the global circuit breaker back to NORMAL state."""
        prev_reason = self._halt_reason
        self._state = CircuitBreakerState.NORMAL
        self._halt_reason = None
        self._tripped_by = None
        self._tripped_at_utc = None
        logger.info(f"Global kill switch deactivated by {reset_by}: {justification} (Previous: {prev_reason})")
        self._log_event("GLOBAL_HALT_DEACTIVATED", f"{justification} (Prior: {prev_reason})", reset_by)

    def halt_portfolio(self, portfolio_id: str, reason: str, source: str = "RISK_AGENT") -> None:
        """Freezes autonomous execution for a specific portfolio."""
        self._halted_portfolios.add(portfolio_id)
        self._log_event("PORTFOLIO_HALTED", f"Portfolio {portfolio_id}: {reason}", source)

    def resume_portfolio(self, portfolio_id: str, source: str = "ADVISOR") -> None:
        """Unfreezes execution for a specific portfolio."""
        self._halted_portfolios.discard(portfolio_id)
        self._log_event("PORTFOLIO_RESUMED", f"Portfolio {portfolio_id} execution resumed", source)

    def is_execution_allowed(self, portfolio_id: Optional[str] = None) -> Tuple[bool, str]:
        """Validates if autonomous execution is currently permitted."""
        if self._state == CircuitBreakerState.HALTED:
            return False, f"Global trading halt active: {self._halt_reason}"
        if portfolio_id and portfolio_id in self._halted_portfolios:
            return False, f"Portfolio {portfolio_id} is individually frozen."
        if self._state == CircuitBreakerState.THROTTLED:
            return True, "Trading throttled (Advisory confirmation advised)."
        return True, "Execution permitted."

    def _log_event(self, action: str, details: str, operator: str) -> None:
        self._audit_log.append({
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "action": action,
            "details": details,
            "operator": operator,
        })

    def get_audit_log(self) -> List[Dict[str, Any]]:
        return list(self._audit_log)
