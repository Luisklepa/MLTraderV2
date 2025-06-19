import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MLSignalFilter:
    """
    Clase para filtrar señales basadas en condiciones de mercado.
    """
    
    def __init__(self, config: dict):
        """
        Inicializa el filtro de señales.
        
        Args:
            config: Diccionario con configuración de filtros
        """
        self.config = config
        
    def apply_filters(self, df: pd.DataFrame, side: str) -> pd.Series:
        """
        Aplica filtros a las señales.
        
        Args:
            df: DataFrame con datos y señales
            side: 'long' o 'short'
            
        Returns:
            Serie booleana con señales filtradas
        """
        # Inicializar máscara
        mask = pd.Series(True, index=df.index)
        
        # Obtener configuración de filtros
        filters = self.config[side]['filters']
        
        # Aplicar filtros de volatilidad
        if filters['volatility']['enabled']:
            mask &= self._apply_volatility_filter(df, side)
            
        # Aplicar filtros de volumen
        if filters['volume']['enabled']:
            mask &= self._apply_volume_filter(df, side)
            
        # Aplicar filtros de tendencia
        if filters['trend']['enabled']:
            mask &= self._apply_trend_filter(df, side)
            
        return mask
        
    def _apply_volatility_filter(self, df: pd.DataFrame, side: str) -> pd.Series:
        """
        Aplica filtros basados en volatilidad.
        
        Args:
            df: DataFrame con datos
            side: 'long' o 'short'
            
        Returns:
            Serie booleana con señales que pasan el filtro
        """
        config = self.config[side]['filters']['volatility']
        
        # Filtro por ATR
        atr_min = config['atr_percentile_min']
        atr_max = config['atr_percentile_max']
        atr_percentile = df[config['atr_column']].rolling(20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        
        return (atr_percentile >= atr_min) & (atr_percentile <= atr_max)
        
    def _apply_volume_filter(self, df: pd.DataFrame, side: str) -> pd.Series:
        """
        Aplica filtros basados en volumen.
        
        Args:
            df: DataFrame con datos
            side: 'long' o 'short'
            
        Returns:
            Serie booleana con señales que pasan el filtro
        """
        config = self.config[side]['filters']['volume']
        
        # Filtro por ratio de volumen
        vol_min = config['volume_ratio_min']
        volume_ratio = df[config['volume_ratio_column']]
        
        return volume_ratio >= vol_min
        
    def _apply_trend_filter(self, df: pd.DataFrame, side: str) -> pd.Series:
        """
        Aplica filtros basados en tendencia.
        
        Args:
            df: DataFrame con datos
            side: 'long' o 'short'
            
        Returns:
            Serie booleana con señales que pasan el filtro
        """
        config = self.config[side]['filters']['trend']
        
        # Filtro por dirección de EMAs
        if side == 'long':
            trend_condition = (
                (df['price_ema_20_ratio'] > 1.0) &  # Precio sobre EMA20
                (df['ema_20'] > df['ema_50'])  # EMA20 sobre EMA50
            )
        else:
            trend_condition = (
                (df['price_ema_20_ratio'] < 1.0) &  # Precio bajo EMA20
                (df['ema_20'] < df['ema_50'])  # EMA20 bajo EMA50
            )
            
        return trend_condition 