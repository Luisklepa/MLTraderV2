"""
Script to download historical data from Binance
"""
import sys
from pathlib import Path

# Agregar el directorio raíz al path
root_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(root_dir)

from utils.data_fetcher import BinanceDataFetcher

def main():
    print("=== DESCARGANDO DATOS HISTÓRICOS ===")
    fetcher = BinanceDataFetcher()
    
    # Descargar datos
    df = fetcher.get_klines()
    
    if df is not None:
        print(f"Datos descargados: {len(df)} velas")
        # Guardar datos
        df.to_csv('btcusdt_prices.csv', index=False)
        print("Datos guardados en btcusdt_prices.csv")
    else:
        print("Error al descargar datos")

if __name__ == '__main__':
    main() 