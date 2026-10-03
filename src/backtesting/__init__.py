"""Backtesting Engine and Scenario Simulation Package for WealthPilot AI.

Components:
- PerformanceAnalyser: Calculates institutional Sharpe, Sortino, Calmar, MaxDD, Tracking Error, and Tax Alpha
- BacktestEngine: Simulates 252 trading days across AI Agent, Calendar, Threshold, and Buy-and-Hold
- StrategyComparator: Parallel multi-strategy tournament ranking and alpha attribution
- ScenarioRunner: Severe historical crisis stress tests (COVID-19 2020 crash, VIX spikes, Kill Switch triggers)
"""

from src.backtesting.performance_analyser import PerformanceAnalyser
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.strategy_comparator import StrategyComparator
from src.backtesting.scenario_runner import ScenarioRunner

__all__ = [
    "PerformanceAnalyser",
    "BacktestEngine",
    "StrategyComparator",
    "ScenarioRunner",
]
