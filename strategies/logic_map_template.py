"""
PLANTILLA DE MAPEO Y ANÁLISIS DE LÓGICAS DE TRADING

Utiliza esta plantilla para documentar, analizar y comparar cada lógica/estrategia candidata antes de integrarla en el portafolio multi-estrategia.
"""

from typing import List, Dict, Any

class LogicMap:
    def __init__(self,
                 nombre: str,
                 tipo: str,
                 descripcion: str,
                 hipotesis: str,
                 reglas: str,
                 parametros: Dict[str, Any],
                 performance: Dict[str, float],
                 overlap: Dict[str, float],
                 robustez: str,
                 notas: str = ""):
        self.nombre = nombre  # Ej: "Trend Following EMA"
        self.tipo = tipo      # Ej: "Trend Following", "Mean Reversion", etc.
        self.descripcion = descripcion
        self.hipotesis = hipotesis
        self.reglas = reglas
        self.parametros = parametros  # Ej: {'ema_fast': 12, 'ema_slow': 26}
        self.performance = performance  # Ej: {'sharpe': 1.2, 'max_dd': 0.18, 'winrate': 0.45}
        self.overlap = overlap  # Ej: {'con_trend2': 0.12, 'con_meanrev': 0.05}
        self.robustez = robustez  # Resumen de tests de robustez
        self.notas = notas

    def resumen(self):
        print(f"\n{'='*40}")
        print(f"Nombre: {self.nombre}")
        print(f"Tipo: {self.tipo}")
        print(f"Descripción: {self.descripcion}")
        print(f"Hipótesis: {self.hipotesis}")
        print(f"Reglas: {self.reglas}")
        print(f"Parámetros: {self.parametros}")
        print(f"Performance: {self.performance}")
        print(f"Overlap: {self.overlap}")
        print(f"Robustez: {self.robustez}")
        print(f"Notas: {self.notas}")
        print(f"{'='*40}\n")

# Ejemplo de uso:
if __name__ == "__main__":
    trend_ema = LogicMap(
        nombre="Trend Following EMA",
        tipo="Trend Following",
        descripcion="Compra cuando EMA rápida cruza sobre EMA lenta y volatilidad es suficiente.",
        hipotesis="Las tendencias persisten y los cruces de medias capturan el inicio de movimientos fuertes.",
        reglas="EMA rápida cruza sobre EMA lenta, ATR > umbral, filtro de distancia al precio.",
        parametros={'ema_fast': 12, 'ema_slow': 26, 'atr_period': 14, 'atr_threshold': 0.5},
        performance={'sharpe': 1.1, 'max_dd': 0.22, 'winrate': 0.48},
        overlap={'con_trend2': 0.15, 'con_meanrev': 0.03},
        robustez="Funciona en BTC, ETH y SPY. Sensible a parámetros de ATR. Out-of-sample estable.",
        notas="Mejorar gestión de trailing stop."
    )
    trend_ema.resumen() 