"""
Utility functions for trading strategies in Backtrader.
Centralizes indicators, stops, risk management, and debug printing.
"""

import logging
from typing import Literal

import backtrader as bt

logger = logging.getLogger(__name__)


# ================= INDICATORS =================


def get_atr(data: bt.feeds.DataBase, period: int = 14) -> bt.indicators.ATR:
    """Return backtrader ATR indicator."""
    return bt.ind.ATR(data, period=period)


def get_ema(data: bt.feeds.DataBase, period: int = 20) -> bt.indicators.EMA:
    """Return backtrader EMA indicator on close prices."""
    return bt.ind.EMA(data.close, period=period)


def get_rsi(data: bt.feeds.DataBase, period: int = 14) -> bt.indicators.RSI:
    """Return backtrader RSI indicator on close prices."""
    return bt.ind.RSI(data.close, period=period)


# ================= STOPS & RISK ================


def calculate_stop_loss(
    entry_price: float,
    atr: float,
    atr_mult: float = 1.0,
    direction: Literal["long", "short"] = "long",
) -> float:
    """Calculate stop-loss level using ATR distance from entry."""
    if entry_price <= 0 or atr < 0:
        logger.warning("Invalid entry_price (%.4f) or atr (%.4f)", entry_price, atr)
        return entry_price
    if direction == "long":
        return entry_price - atr * atr_mult
    return entry_price + atr * atr_mult


def calculate_take_profit(
    entry_price: float,
    stop_loss: float,
    rr_ratio: float = 2.0,
    direction: Literal["long", "short"] = "long",
) -> float:
    """Calculate take-profit level using risk-reward ratio."""
    risk = abs(entry_price - stop_loss)
    if direction == "long":
        return entry_price + risk * rr_ratio
    return entry_price - risk * rr_ratio


def calculate_position_size(
    account_value: float,
    risk_per_trade: float,
    stop_distance: float,
) -> float:
    """Calculate position size in asset units based on risk budget.

    Returns 0 if inputs are invalid (non-positive account, zero stop distance).
    """
    if account_value <= 0 or risk_per_trade <= 0 or stop_distance <= 0:
        return 0.0
    size = (account_value * risk_per_trade) / stop_distance
    return round(size, 4)


def print_debug(msg: str) -> None:
    """Log a debug message (prefer logger over print in production)."""
    logger.debug(msg)
