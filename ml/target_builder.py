import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .signal_filter import MLSignalFilter

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SignalConfig:
    """Configuration for signal generation."""

    atr_multiplier: float
    profit_ratio: float
    max_overlap_ratio: float
    min_samples: int
    filters: dict


@dataclass
class TargetQualityMetrics:
    """Métricas de calidad de targets para facilitar reporting."""

    signal_count: int
    signal_ratio: float
    mean_return: float
    sharpe_ratio: float
    hit_ratio: float
    profit_loss_ratio: float
    overlap_ratio: float
    volume_correlation: float
    largest_cluster: int
    avg_cluster_duration: float
    max_drawdown: float
    clean_signal_ratio: float
    robustness_score: float
    warnings: list[str]
    recommendations: list[str]


class MLTargetBuilder:
    """
    Clase para construir targets para modelos de ML con soporte para targets dinámicos.
    """

    def __init__(
        self,
        horizon: int,
        atr_column: str,
        long_config: dict,
        short_config: dict,
        analyze_threshold_range: bool = True,
        experiment_tracking: bool = True,
        output_dir: str = "analysis",
    ):
        """
        Initialize target builder with dynamic threshold configuration.

        Args:
            horizon: Number of bars to look ahead
            atr_column: Name of ATR column for dynamic thresholds
            long_config: Configuration for long signals
            short_config: Configuration for short signals
            analyze_threshold_range: Whether to analyze different threshold values
            experiment_tracking: Whether to track experiments
            output_dir: Directory for analysis output
        """
        self.horizon = horizon
        self.atr_column = atr_column
        self.long_config = long_config
        self.short_config = short_config
        self.analyze_threshold_range = analyze_threshold_range
        self.experiment_tracking = experiment_tracking
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        filter_config = {
            "long": long_config,
            "short": short_config,
        }
        self.signal_filter = MLSignalFilter(filter_config)

        # Setup logging
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for target builder."""
        logger = logging.getLogger("MLTargetBuilder")
        logger.setLevel(logging.INFO)

        if self.experiment_tracking:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.output_dir / f"target_builder_{timestamp}.log"
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _apply_volatility_filter(self, df: pd.DataFrame, config: dict, side: str) -> pd.Series:
        """Apply volatility-based filters."""
        filters = config["filters"]["volatility"]
        if not filters["enabled"]:
            return pd.Series(True, index=df.index)

        atr = df[filters["atr_column"]]
        atr_rank = atr.rolling(100).rank(pct=True) * 100

        return (atr_rank >= filters["atr_percentile_min"] * 100) & (atr_rank <= filters["atr_percentile_max"] * 100)

    def _apply_volume_filter(self, df: pd.DataFrame, config: dict, side: str) -> pd.Series:
        """Apply volume-based filters."""
        filters = config["filters"]["volume"]
        if not filters["enabled"]:
            return pd.Series(True, index=df.index)

        volume_ratio = df[filters["volume_ratio_column"]]
        return volume_ratio >= filters["volume_ratio_min"]

    def _apply_trend_filter(self, df: pd.DataFrame, config: dict, side: str) -> pd.Series:
        """Apply trend-based filters."""
        filters = config["filters"]["trend"]
        if not filters["enabled"]:
            return pd.Series(True, index=df.index)

        ema_fast = df[f"ema_{filters['ema_fast']}"]
        ema_slow = df[f"ema_{filters['ema_slow']}"]
        trend_strength = (ema_fast - ema_slow) / ema_slow

        if side == "long":
            return trend_strength >= filters["min_trend_strength"]
        else:
            return trend_strength <= -filters["min_trend_strength"]

    def _generate_targets(self, df: pd.DataFrame, config: dict, side: str) -> pd.Series:
        """
        Generate trading targets based on dynamic thresholds.

        Args:
            df: Input DataFrame with price and indicator data
            config: Signal configuration
            side: 'long' or 'short'

        Returns:
            Series with target labels (0, 1 for long, -1 for short)
        """
        # Calculate dynamic threshold based on ATR
        atr = df[self.atr_column]
        threshold = atr * config["atr_multiplier"]

        # Calculate future returns
        future_returns = self._calculate_future_returns(df)
        future_returns / atr

        # Apply filters
        vol_filter = self._apply_volatility_filter(df, config, side)
        volume_filter = self._apply_volume_filter(df, config, side)
        trend_filter = self._apply_trend_filter(df, config, side)

        # Generate signals based on profit potential
        if side == "long":
            signals = (future_returns >= threshold * config["profit_ratio"]) & vol_filter & volume_filter & trend_filter
            return signals.astype(int)
        else:
            signals = (
                (future_returns <= -threshold * config["profit_ratio"]) & vol_filter & volume_filter & trend_filter
            )
            return -signals.astype(int)

    def build_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build trading targets for both long and short positions.

        Args:
            df: Input DataFrame with price and indicator data

        Returns:
            DataFrame with target labels
        """
        self.logger.info("Building trading targets...")

        # Generate long and short targets separately
        long_targets = self._generate_targets(df, self.long_config, "long")
        short_targets = self._generate_targets(df, self.short_config, "short")

        # Combine targets (prioritize stronger signals)
        combined_targets = pd.Series(0, index=df.index)
        combined_targets[long_targets == 1] = 1
        combined_targets[short_targets == -1] = -1

        # Remove overlapping signals
        clean_targets = self._remove_overlapping_signals(combined_targets)

        # Validate and analyze targets
        self._validate_targets(clean_targets, df)

        if self.analyze_threshold_range:
            self._analyze_threshold_sensitivity(df)

        return clean_targets

    def _remove_overlapping_signals(self, targets: pd.Series) -> pd.Series:
        """Remove overlapping signals within the horizon window."""
        clean_targets = targets.copy()
        signal_points = targets[targets != 0].index

        for i in range(len(signal_points) - 1):
            current_point = signal_points[i]
            next_point = signal_points[i + 1]

            # If signals are too close, remove the second one
            if (next_point - current_point).total_seconds() / 60 < self.horizon:
                clean_targets[next_point] = 0

        return clean_targets

    def _validate_targets(self, targets: pd.Series, df: pd.DataFrame):
        """Validate generated targets and log statistics."""
        signal_count = (targets != 0).sum()
        signal_ratio = signal_count / len(targets)

        self.logger.info(f"Generated {signal_count} signals ({signal_ratio:.2%} of bars)")
        self.logger.info(f"Long signals: {(targets == 1).sum()}")
        self.logger.info(f"Short signals: {(targets == -1).sum()}")

        # Calculate and log target quality metrics
        metrics = self._calculate_target_quality_metrics(targets, df)
        self._log_target_quality_metrics(metrics)

        if self.experiment_tracking:
            self._save_target_metrics(metrics)

    def _calculate_target_quality_metrics(
        self,
        targets: pd.Series,
        df: pd.DataFrame,
    ) -> TargetQualityMetrics:
        """Calculate comprehensive target quality metrics."""
        future_returns = self._calculate_future_returns(df)
        signal_mask = targets != 0

        signal_count = int(signal_mask.sum())
        signal_ratio = signal_count / max(len(targets), 1)

        signal_returns = future_returns[signal_mask]
        mean_return = float(signal_returns.mean()) if len(signal_returns) > 0 else 0.0
        ret_std = float(signal_returns.std()) if len(signal_returns) > 1 else 1.0
        sharpe = mean_return / ret_std if ret_std > 1e-12 else 0.0
        hit_ratio = float((signal_returns > 0).mean()) if len(signal_returns) > 0 else 0.0

        wins = signal_returns[signal_returns > 0]
        losses = signal_returns[signal_returns < 0]
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else -1.0
        pl_ratio = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-12 else 0.0

        # Overlap: fraction of signals within horizon bars of each other
        signal_idx = targets[signal_mask].index
        overlap_count = 0
        for i in range(len(signal_idx) - 1):
            diff = signal_idx[i + 1] - signal_idx[i]
            if hasattr(diff, "total_seconds"):
                gap = diff.total_seconds() / 60
            else:
                gap = float(diff)
            if gap < self.horizon:
                overlap_count += 1
        overlap_ratio = overlap_count / max(signal_count - 1, 1)

        # Volume correlation
        vol_corr = 0.0
        if "volume" in df.columns and signal_count > 5:
            vol_corr = float(df["volume"].corr(signal_mask.astype(float)))

        # Clustering: consecutive signal groups
        groups = (signal_mask != signal_mask.shift()).cumsum()
        cluster_sizes = signal_mask.groupby(groups).sum()
        cluster_sizes = cluster_sizes[cluster_sizes > 0]
        largest_cluster = int(cluster_sizes.max()) if len(cluster_sizes) > 0 else 0
        avg_cluster = float(cluster_sizes.mean()) if len(cluster_sizes) > 0 else 0.0

        # Max drawdown of signal returns
        cum_returns = (1 + signal_returns).cumprod() if len(signal_returns) > 0 else pd.Series([1.0])
        peak = cum_returns.cummax()
        dd = (cum_returns - peak) / peak
        max_dd = float(dd.min()) if len(dd) > 0 else 0.0

        clean_signal_ratio = 1.0 - overlap_ratio

        # Robustness heuristic
        robustness = min(
            1.0,
            max(
                0.0,
                0.3 * min(1.0, sharpe / 2.0)
                + 0.2 * hit_ratio
                + 0.2 * clean_signal_ratio
                + 0.15 * min(1.0, pl_ratio / 2.0)
                + 0.15 * (1.0 - abs(max_dd)),
            ),
        )

        warnings_list: list = []
        recommendations: list = []
        if signal_ratio < 0.01:
            warnings_list.append("Very few signals (<1% of bars)")
        if signal_ratio > 0.5:
            warnings_list.append("Too many signals (>50% of bars) — target may be too loose")
        if overlap_ratio > 0.3:
            warnings_list.append(f"High overlap ratio ({overlap_ratio:.2%}) — signals cluster too tightly")
            recommendations.append("Increase horizon or add minimum gap between signals")
        if hit_ratio < 0.4:
            recommendations.append("Consider tighter threshold to improve hit ratio")
        if pl_ratio < 1.0:
            recommendations.append("Profit/loss ratio < 1 — consider asymmetric thresholds")

        return TargetQualityMetrics(
            signal_count=signal_count,
            signal_ratio=signal_ratio,
            mean_return=mean_return,
            sharpe_ratio=sharpe,
            hit_ratio=hit_ratio,
            profit_loss_ratio=pl_ratio,
            overlap_ratio=overlap_ratio,
            volume_correlation=vol_corr,
            largest_cluster=largest_cluster,
            avg_cluster_duration=avg_cluster,
            max_drawdown=max_dd,
            clean_signal_ratio=clean_signal_ratio,
            robustness_score=robustness,
            warnings=warnings_list,
            recommendations=recommendations,
        )

    def _analyze_threshold_sensitivity(self, df: pd.DataFrame) -> None:
        """Analyze target sensitivity to ATR multiplier variations."""
        future_returns = self._calculate_future_returns(df)
        atr = df[self.atr_column]
        multipliers = np.linspace(0.5, 3.0, 11)

        self.logger.info("Threshold sensitivity analysis:")
        for mult in multipliers:
            thresh = atr * mult
            long_signals = (future_returns >= thresh).sum()
            short_signals = (future_returns <= -thresh).sum()
            total = len(df)
            self.logger.info(
                "  ATR*%.1f => long: %d (%.1f%%), short: %d (%.1f%%)",
                mult,
                long_signals,
                100 * long_signals / max(total, 1),
                short_signals,
                100 * short_signals / max(total, 1),
            )

    def _log_target_quality_metrics(self, metrics: TargetQualityMetrics) -> None:
        """Log target quality metrics to logger."""
        self.logger.info("=== Target Quality Metrics ===")
        self.logger.info("  Signals: %d (%.2f%% of bars)", metrics.signal_count, metrics.signal_ratio * 100)
        self.logger.info("  Mean return: %.4f", metrics.mean_return)
        self.logger.info("  Sharpe ratio: %.2f", metrics.sharpe_ratio)
        self.logger.info("  Hit ratio: %.2f%%", metrics.hit_ratio * 100)
        self.logger.info("  Profit/loss ratio: %.2f", metrics.profit_loss_ratio)
        self.logger.info("  Overlap ratio: %.2f%%", metrics.overlap_ratio * 100)
        self.logger.info("  Max drawdown: %.2f%%", metrics.max_drawdown * 100)
        self.logger.info("  Robustness score: %.2f", metrics.robustness_score)
        for w in metrics.warnings:
            self.logger.warning("  WARNING: %s", w)
        for r in metrics.recommendations:
            self.logger.info("  RECOMMENDATION: %s", r)

    def _save_target_metrics(self, metrics: TargetQualityMetrics) -> None:
        """Save target metrics to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"target_metrics_{timestamp}.json"
        data = {
            "signal_count": metrics.signal_count,
            "signal_ratio": metrics.signal_ratio,
            "mean_return": metrics.mean_return,
            "sharpe_ratio": metrics.sharpe_ratio,
            "hit_ratio": metrics.hit_ratio,
            "profit_loss_ratio": metrics.profit_loss_ratio,
            "overlap_ratio": metrics.overlap_ratio,
            "volume_correlation": metrics.volume_correlation,
            "largest_cluster": metrics.largest_cluster,
            "avg_cluster_duration": metrics.avg_cluster_duration,
            "max_drawdown": metrics.max_drawdown,
            "clean_signal_ratio": metrics.clean_signal_ratio,
            "robustness_score": metrics.robustness_score,
            "warnings": metrics.warnings,
            "recommendations": metrics.recommendations,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.logger.info("Target metrics saved to %s", path)

    def _calculate_future_returns(self, df: pd.DataFrame) -> pd.Series:
        """Calculate future returns over the horizon period."""
        close = df["close"]
        future_close = close.shift(-self.horizon)
        return (future_close - close) / close

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """
        Carga la configuración desde un archivo YAML.

        Args:
            config_path: Ruta al archivo de configuración

        Returns:
            Diccionario con configuración
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config["target_config"]

    def analyze_threshold_range(self, df: pd.DataFrame) -> dict[str, dict[float, dict]]:
        """
        Analiza diferentes multiplicadores de ATR para optimización.

        Args:
            df: DataFrame con datos de precios

        Returns:
            Diccionario con análisis por multiplicador de ATR
        """
        if not self.target_config["long"]["analyze_threshold_range"]:
            return {}

        results = {"long": {}, "short": {}}
        atr_column = self.target_config["atr_column"]

        # Definir rango de multiplicadores a analizar
        multipliers = np.linspace(0.5, 2.0, 10)
        horizons = [3, 5, 10]

        for _horizon in horizons:
            future_returns = self._calculate_future_returns(df)

            for multiplier in multipliers:
                # Convertir multiplier a float Python nativo
                multiplier = float(multiplier)

                # Análisis long
                threshold_long = df[atr_column] * multiplier
                target_long = (future_returns > threshold_long).astype(int)

                # Aplicar filtros long
                filter_mask_long = self.signal_filter.apply_filters(df, "long")
                target_long = target_long & filter_mask_long

                # Eliminar señales solapadas
                target_long = self._remove_overlapping_signals(target_long)
                signal_returns_long = future_returns[target_long == 1]

                results["long"][multiplier] = {
                    "signal_metrics": {"long_ratio": float(target_long.mean()), "long_count": int(target_long.sum())},
                    "return_metrics": {
                        "mean_return": float(signal_returns_long.mean()) if len(signal_returns_long) > 0 else 0.0,
                        "sharpe": float(signal_returns_long.mean() / signal_returns_long.std())
                        if len(signal_returns_long) > 0 and signal_returns_long.std() != 0
                        else 0.0,
                    },
                }

                # Análisis short
                threshold_short = -df[atr_column] * multiplier
                target_short = (future_returns < threshold_short).astype(int)

                # Aplicar filtros short
                filter_mask_short = self.signal_filter.apply_filters(df, "short")
                target_short = target_short & filter_mask_short

                # Eliminar señales solapadas
                target_short = self._remove_overlapping_signals(target_short)
                signal_returns_short = future_returns[target_short == 1]

                results["short"][multiplier] = {
                    "signal_metrics": {
                        "short_ratio": float(target_short.mean()),
                        "short_count": int(target_short.sum()),
                    },
                    "return_metrics": {
                        "mean_return": float(signal_returns_short.mean()) if len(signal_returns_short) > 0 else 0.0,
                        "sharpe": float(signal_returns_short.mean() / signal_returns_short.std())
                        if len(signal_returns_short) > 0 and signal_returns_short.std() != 0
                        else 0.0,
                    },
                }

        return results
