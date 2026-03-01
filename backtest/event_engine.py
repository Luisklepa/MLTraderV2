"""
Event-driven backtesting engine for ML trading strategies.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Class to track position information."""

    symbol: str
    type: str  # 'long' or 'short'
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float | None
    pnl: float = 0.0
    unrealized_pnl: float = 0.0


class BacktestEngine:
    """Event-driven backtesting engine for ML strategies."""

    def __init__(
        self, strategy: Any, data: pd.DataFrame, reference_data: dict[str, pd.DataFrame], config: dict[str, Any]
    ):
        """Initialize the backtest engine."""
        self.strategy = strategy
        self.data = data
        self.reference_data = reference_data
        self.config = config

        # State variables
        self.positions: dict[str, Position] = {}
        self.cash = config["risk_config"]["initial_capital"]
        self.equity = [self.cash]
        self.trades = []
        self.current_time = None

        # Performance tracking
        self.returns = []
        self.drawdowns = []
        self.exposure = []
        self.trade_stats = defaultdict(list)

        # Commission and slippage
        self.commission_rate = config["risk_config"].get("commission_rate", 0.001)
        self.slippage_rate = config["risk_config"].get("slippage_rate", 0.0005)
        # Queue for next-bar execution to avoid lookahead
        self.pending_orders: list[dict[str, Any]] = []

    def calculate_slippage(self, price: float, size: float, is_buy: bool) -> float:
        """Calculate slippage cost."""
        direction = 1 if is_buy else -1
        return price * (1 + direction * self.slippage_rate)

    def calculate_commission(self, price: float, size: float) -> float:
        """Calculate commission cost."""
        return price * size * self.commission_rate

    def execute_order(self, order: dict[str, Any], current_price: float) -> Position | None:
        """Execute a trade order."""
        symbol = order["symbol"]
        order_type = order["type"]
        size = order["size"]

        if size <= 0:
            logger.warning(f"Invalid order size ({size}), skipping: {order}")
            return None

        # Apply slippage
        execution_price = self.calculate_slippage(current_price, size, is_buy=(order_type == "buy"))

        # Calculate commission
        commission = self.calculate_commission(execution_price, size)

        # Check if we have enough cash (for buys: full cost; for shorts: margin requirement)
        if order_type == "buy":
            cost = execution_price * size + commission
            if cost > self.cash:
                logger.warning(f"Insufficient cash for buy order: {order}")
                return None
        else:
            margin_required = execution_price * size * 0.5 + commission
            if margin_required > self.cash:
                logger.warning(f"Insufficient margin for short order: {order}")
                return None

        # Create position
        position = Position(
            symbol=symbol,
            type="long" if order_type == "buy" else "short",
            size=size,
            entry_price=execution_price,
            entry_time=self.current_time,
            stop_loss=order.get("stop_loss"),
            take_profit=order.get("take_profit"),
        )

        # Update cash
        if order_type == "buy":
            self.cash -= execution_price * size + commission
        else:
            self.cash += execution_price * size - commission

        # Record trade
        self.trades.append(
            {
                "time": self.current_time,
                "symbol": symbol,
                "type": order_type,
                "size": size,
                "price": execution_price,
                "commission": commission,
                "reason": order.get("reason", "signal"),
            }
        )

        return position

    def close_position(self, position: Position, current_price: float, reason: str) -> None:
        """Close a position and record the trade."""
        # Calculate PnL
        # Apply slippage on exit
        if position.type == "long":
            exit_price = self.calculate_slippage(current_price, position.size, is_buy=False)
            pnl = (exit_price - position.entry_price) * position.size
        else:  # short
            exit_price = self.calculate_slippage(current_price, position.size, is_buy=True)
            pnl = (position.entry_price - exit_price) * position.size

        # Apply commission on exit price
        commission = self.calculate_commission(exit_price, position.size)
        pnl -= commission

        # Return full position value (not just PnL) to cash
        if position.type == "long":
            self.cash += exit_price * position.size - commission
        else:
            self.cash -= exit_price * position.size + commission

        # Record trade
        self.trades.append(
            {
                "time": self.current_time,
                "symbol": position.symbol,
                "type": "sell" if position.type == "long" else "buy",
                "size": position.size,
                "price": exit_price,
                "commission": commission,
                "pnl": pnl,
                "reason": reason,
                "duration": (self.current_time - position.entry_time).total_seconds() / 3600,  # hours
            }
        )

        # Update trade statistics
        self.trade_stats["pnl"].append(pnl)
        self.trade_stats["duration"].append((self.current_time - position.entry_time).total_seconds() / 3600)
        self.trade_stats["win"].append(pnl > 0)

    def update_positions(self, current_bar: pd.Series) -> None:
        """Update all positions with current market data."""
        closed_positions = []

        for symbol, position in self.positions.items():
            # Check stop loss
            if position.type == "long":
                if current_bar["low"] <= position.stop_loss:
                    self.close_position(position, position.stop_loss, "stop_loss")
                    closed_positions.append(symbol)
                    continue
            else:  # short position
                if current_bar["high"] >= position.stop_loss:
                    self.close_position(position, position.stop_loss, "stop_loss")
                    closed_positions.append(symbol)
                    continue

            # Check take profit
            if position.take_profit is not None:
                if position.type == "long" and current_bar["high"] >= position.take_profit:
                    self.close_position(position, position.take_profit, "take_profit")
                    closed_positions.append(symbol)
                    continue
                elif position.type == "short" and current_bar["low"] <= position.take_profit:
                    self.close_position(position, position.take_profit, "take_profit")
                    closed_positions.append(symbol)
                    continue

            # Update unrealized PnL
            if position.type == "long":
                position.unrealized_pnl = (current_bar["close"] - position.entry_price) * position.size
            else:
                position.unrealized_pnl = (position.entry_price - current_bar["close"]) * position.size

        # Remove closed positions
        for symbol in closed_positions:
            del self.positions[symbol]

    def update_metrics(self, current_close: float) -> None:
        """Update performance metrics using the current bar's close price."""
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        current_equity = self.cash + unrealized_pnl

        prev_equity = self.equity[-1]
        ret = (current_equity - prev_equity) / prev_equity if prev_equity != 0 else 0.0

        peak = max(self.equity)
        drawdown = (current_equity - peak) / peak if peak != 0 else 0.0

        total_exposure = sum(abs(pos.size * current_close) for pos in self.positions.values())
        exposure = total_exposure / current_equity if current_equity != 0 else 0.0

        self.equity.append(current_equity)
        self.returns.append(ret)
        self.drawdowns.append(drawdown)
        self.exposure.append(exposure)

    def run(self) -> pd.DataFrame:
        """Run the backtest."""
        logger.info("Starting backtest...")

        results = []
        index = self.data.index
        n_bars = len(index)

        for i in range(n_bars):
            idx = index[i]
            row = self.data.iloc[i]
            self.current_time = idx

            # 1) Execute pending orders at current bar's OPEN (next-bar execution)
            if self.pending_orders:
                exec_price = row["open"] if "open" in row else row["close"]
                for order in self.pending_orders:
                    position = self.execute_order(order, exec_price)
                    if position is not None:
                        self.positions[order["symbol"]] = position
                self.pending_orders = []

            # 2) Check SL/TP on current bar
            self.update_positions(row)

            # 3) Strategy sees data up to and including current bar's close
            try:
                orders = self.strategy.on_data(self.data.iloc[: i + 1].copy())
            except Exception as e:
                logger.error(f"Strategy error at {idx}: {str(e)}")
                orders = []

            # 4) Queue new orders for next bar execution
            for order in orders:
                if order["type"] in ["buy", "sell"]:
                    order = dict(order)
                    order["queued_at"] = idx
                    self.pending_orders.append(order)

            # 5) Update metrics using the current bar's close
            self.update_metrics(row["close"])

            results.append(
                {
                    "timestamp": idx,
                    "equity": self.equity[-1],
                    "cash": self.cash,
                    "returns": self.returns[-1],
                    "drawdown": self.drawdowns[-1],
                    "exposure": self.exposure[-1],
                    "positions": len(self.positions),
                    "unrealized_pnl": sum(pos.unrealized_pnl for pos in self.positions.values()),
                }
            )

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        results_df.set_index("timestamp", inplace=True)

        # Calculate summary statistics
        self.calculate_summary_statistics()

        logger.info("Backtest completed successfully")
        return results_df

    def calculate_summary_statistics(self) -> None:
        """Calculate and log summary statistics."""
        if not self.trades:
            logger.warning("No trades executed during backtest")
            return

        # Trading statistics
        total_trades = len(self.trades)
        profitable_trades = sum(1 for t in self.trades if t.get("pnl", 0) > 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0

        # PnL statistics
        total_pnl = sum(t.get("pnl", 0) for t in self.trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        # Risk statistics
        max_drawdown = min(self.drawdowns)
        avg_exposure = np.mean(self.exposure)

        # Returns statistics
        total_return = (self.equity[-1] - self.equity[0]) / self.equity[0]
        # Infer bar frequency to annualize correctly
        if len(self.equity) >= 2:
            idx = pd.to_datetime(self.data.index)
            diffs = np.diff(idx.values.astype("datetime64[ns]")).astype("timedelta64[s]").astype(float)
            sec_per_bar = np.median(diffs) if len(diffs) > 0 else 60 * 60 * 24
            bars_per_year = (365 * 24 * 3600) / sec_per_bar
        else:
            bars_per_year = 252.0
        n_returns = len(self.returns)
        annual_return = (1 + total_return) ** (bars_per_year / n_returns) - 1 if n_returns > 0 else 0
        vol = np.std(self.returns) if n_returns > 1 else 0.0
        volatility = vol * np.sqrt(bars_per_year)
        sharpe_ratio = (np.mean(self.returns) / vol * np.sqrt(bars_per_year)) if vol > 1e-12 else 0

        # Log statistics
        logger.info("\nBacktest Summary Statistics:")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Total PnL: ${total_pnl:,.2f}")
        logger.info(f"Average PnL per Trade: ${avg_pnl:,.2f}")
        logger.info(f"Maximum Drawdown: {max_drawdown:.2%}")
        logger.info(f"Average Exposure: {avg_exposure:.2%}")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Annualized Return: {annual_return:.2%}")
        logger.info(f"Annualized Volatility: {volatility:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
