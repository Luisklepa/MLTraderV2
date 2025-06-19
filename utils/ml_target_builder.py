import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List, Union, Protocol, Callable, Any
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import yaml
from dataclasses import dataclass
from tabulate import tabulate
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from abc import ABC, abstractmethod
import os
from .ml_signal_filter import MLSignalFilter

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
    filters: Dict

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
    warnings: List[str]
    recommendations: List[str]

class MLTargetBuilder:
    """
    Clase para construir targets para modelos de ML con soporte para targets dinámicos.
    """
    
    def __init__(
        self,
        horizon: int,
        atr_column: str,
        long_config: Dict,
        short_config: Dict,
        analyze_threshold_range: bool = True,
        experiment_tracking: bool = True,
        output_dir: str = 'analysis'
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
        
        # Initialize signal filter
        self.signal_filter = MLSignalFilter()
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for target builder."""
        logger = logging.getLogger('MLTargetBuilder')
        logger.setLevel(logging.INFO)
        
        if self.experiment_tracking:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = self.output_dir / f'target_builder_{timestamp}.log'
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _apply_volatility_filter(
        self,
        df: pd.DataFrame,
        config: Dict,
        side: str
    ) -> pd.Series:
        """Apply volatility-based filters."""
        filters = config['filters']['volatility']
        if not filters['enabled']:
            return pd.Series(True, index=df.index)
            
        atr = df[filters['atr_column']]
        atr_rank = atr.rolling(100).apply(lambda x: stats.percentileofscore(x, x[-1]))
        
        return (atr_rank >= filters['atr_percentile_min'] * 100) & \
               (atr_rank <= filters['atr_percentile_max'] * 100)

    def _apply_volume_filter(
        self,
        df: pd.DataFrame,
        config: Dict,
        side: str
    ) -> pd.Series:
        """Apply volume-based filters."""
        filters = config['filters']['volume']
        if not filters['enabled']:
            return pd.Series(True, index=df.index)
            
        volume_ratio = df[filters['volume_ratio_column']]
        return volume_ratio >= filters['volume_ratio_min']

    def _apply_trend_filter(
        self,
        df: pd.DataFrame,
        config: Dict,
        side: str
    ) -> pd.Series:
        """Apply trend-based filters."""
        filters = config['filters']['trend']
        if not filters['enabled']:
            return pd.Series(True, index=df.index)
            
        ema_fast = df[f'ema_{filters["ema_fast"]}']
        ema_slow = df[f'ema_{filters["ema_slow"]}']
        trend_strength = (ema_fast - ema_slow) / ema_slow
        
        if side == 'long':
            return trend_strength >= filters['min_trend_strength']
        else:
            return trend_strength <= -filters['min_trend_strength']

    def _generate_targets(
        self,
        df: pd.DataFrame,
        config: Dict,
        side: str
    ) -> pd.Series:
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
        threshold = atr * config['atr_multiplier']
        
        # Calculate future returns
        future_returns = self._calculate_future_returns(df)
        volatility_adjusted_returns = future_returns / atr
        
        # Apply filters
        vol_filter = self._apply_volatility_filter(df, config, side)
        volume_filter = self._apply_volume_filter(df, config, side)
        trend_filter = self._apply_trend_filter(df, config, side)
        
        # Generate signals based on profit potential
        if side == 'long':
            signals = (future_returns >= threshold * config['profit_ratio']) & \
                     vol_filter & volume_filter & trend_filter
            return signals.astype(int)
        else:
            signals = (future_returns <= -threshold * config['profit_ratio']) & \
                     vol_filter & volume_filter & trend_filter
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
        long_targets = self._generate_targets(df, self.long_config, 'long')
        short_targets = self._generate_targets(df, self.short_config, 'short')
        
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
        df: pd.DataFrame
    ) -> TargetQualityMetrics:
        """Calculate comprehensive target quality metrics."""
        # Implementation of quality metrics calculation
        pass

    def _analyze_threshold_sensitivity(self, df: pd.DataFrame):
        """Analyze sensitivity to different threshold values."""
        # Implementation of threshold sensitivity analysis
        pass

    def _log_target_quality_metrics(self, metrics: TargetQualityMetrics):
        """Log target quality metrics."""
        # Implementation of metrics logging
        pass

    def _save_target_metrics(self, metrics: TargetQualityMetrics):
        """Save target metrics to file."""
        # Implementation of metrics saving
        pass

    def _calculate_future_returns(self, df: pd.DataFrame) -> pd.Series:
        """Calculate future returns over the horizon period."""
        close = df['close']
        future_close = close.shift(-self.horizon)
        return (future_close - close) / close
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Carga la configuración desde un archivo YAML.
        
        Args:
            config_path: Ruta al archivo de configuración
            
        Returns:
            Diccionario con configuración
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['target_config']

    def analyze_threshold_range(self, df: pd.DataFrame) -> Dict[str, Dict[float, Dict]]:
        """
        Analiza diferentes multiplicadores de ATR para optimización.
        
        Args:
            df: DataFrame con datos de precios
            
        Returns:
            Diccionario con análisis por multiplicador de ATR
        """
        if not self.target_config['long']['analyze_threshold_range']:
            return {}
            
        results = {'long': {}, 'short': {}}
        atr_column = self.target_config['atr_column']
        
        # Definir rango de multiplicadores a analizar
        multipliers = np.linspace(0.5, 2.0, 10)
        horizons = [3, 5, 10]
        
        for horizon in horizons:
            future_returns = self._calculate_future_returns(df)
            
            for multiplier in multipliers:
                # Convertir multiplier a float Python nativo
                multiplier = float(multiplier)
                
                # Análisis long
                threshold_long = df[atr_column] * multiplier
                target_long = (future_returns > threshold_long).astype(int)
                
                # Aplicar filtros long
                filter_mask_long = self.signal_filter.apply_filters(df, 'long')
                target_long = target_long & filter_mask_long
                
                # Eliminar señales solapadas
                target_long = self._remove_overlapping_signals(target_long)
                signal_returns_long = future_returns[target_long == 1]
                
                results['long'][multiplier] = {
                    'signal_metrics': {
                        'long_ratio': float(target_long.mean()),
                        'long_count': int(target_long.sum())
                    },
                    'return_metrics': {
                        'mean_return': float(signal_returns_long.mean()) if len(signal_returns_long) > 0 else 0.0,
                        'sharpe': float(signal_returns_long.mean() / signal_returns_long.std()) if len(signal_returns_long) > 0 and signal_returns_long.std() != 0 else 0.0
                    }
                }
                
                # Análisis short
                threshold_short = -df[atr_column] * multiplier
                target_short = (future_returns < threshold_short).astype(int)
                
                # Aplicar filtros short
                filter_mask_short = self.signal_filter.apply_filters(df, 'short')
                target_short = target_short & filter_mask_short
                
                # Eliminar señales solapadas
                target_short = self._remove_overlapping_signals(target_short)
                signal_returns_short = future_returns[target_short == 1]
                
                results['short'][multiplier] = {
                    'signal_metrics': {
                        'short_ratio': float(target_short.mean()),
                        'short_count': int(target_short.sum())
                    },
                    'return_metrics': {
                        'mean_return': float(signal_returns_short.mean()) if len(signal_returns_short) > 0 else 0.0,
                        'sharpe': float(signal_returns_short.mean() / signal_returns_short.std()) if len(signal_returns_short) > 0 and signal_returns_short.std() != 0 else 0.0
                    }
                }
        
        return results 