"""
Script to download historical data from Binance
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import logging
import argparse

def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    return logger

def get_binance_klines(
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int,
    limit: int = 1000
) -> list:
    """Get klines data from Binance API."""
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': limit
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def align_timestamp_to_timeframe(ts_ms, interval):
    """Alinea un timestamp en ms al múltiplo inferior del timeframe."""
    minutos_por_vela = {'1m':1, '3m':3, '5m':5, '15m':15, '30m':30, '1h':60, '2h':120, '4h':240, '6h':360, '8h':480, '12h':720, '1d':1440}
    min_per_candle = minutos_por_vela.get(interval, 15)
    dt = datetime.utcfromtimestamp(ts_ms/1000)
    # Redondear hacia abajo al múltiplo de minutos
    minutos_redondeados = (dt.minute // min_per_candle) * min_per_candle
    dt_alineado = dt.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minutos_redondeados)
    return int(dt_alineado.timestamp() * 1000)

def download_historical_data(
    symbol: str,
    interval: str,
    days: int,
    output_file: str
) -> None:
    """Download historical data from Binance."""
    logger = logging.getLogger()
    logger.info(f"Downloading {days} days of {interval} data for {symbol}")
    
    # Calculate time range
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    start_time = align_timestamp_to_timeframe(start_time, interval)
    logger.info(f"[INFO] Rango solicitado (alineado): {datetime.utcfromtimestamp(start_time/1000)} a {datetime.utcfromtimestamp(end_time/1000)}")
    logger.info(f"[INFO] Parámetros enviados: symbol={symbol}, interval={interval}, days={days}")
    
    # Download data in chunks
    all_klines = []
    current_start = start_time
    chunk_count = 0
    while current_start < end_time:
        try:
            klines = get_binance_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_time,
                limit=1000
            )
            chunk_count += 1
            if not klines:
                logger.info(f"[INFO] No se recibieron más velas en el chunk {chunk_count}. Fin de la descarga.")
                break
            all_klines.extend(klines)
            current_start = klines[-1][0] + 1
            logger.info(f"[INFO] Chunk {chunk_count}: descargadas {len(klines)} velas. Total acumulado: {len(all_klines)}")
            if len(klines) < 1000:
                logger.info(f"[ADVERTENCIA] Chunk {chunk_count} devolvió menos de 1000 velas. Puede que no haya más datos disponibles.")
            time.sleep(0.1)  # Rate limit compliance
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            break
    
    if not all_klines:
        logger.error("No se descargaron datos. Verifica el símbolo, el intervalo y la conexión.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(
        all_klines,
        columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ]
    )
    
    # Convert timestamps
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    
    # Convert numeric columns
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_columns] = df[numeric_columns].astype(float)
    
    # Clean up columns
    df = df[['timestamp'] + numeric_columns]
    
    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"[INFO] Descargadas {len(df)} velas en total.")
    logger.info(f"[INFO] Data saved to {output_file}")
    logger.info(f"[INFO] Date range descargado: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Calcular velas esperadas
    minutos_por_vela = {'1m':1, '3m':3, '5m':5, '15m':15, '30m':30, '1h':60, '2h':120, '4h':240, '6h':360, '8h':480, '12h':720, '1d':1440}
    min_per_candle = minutos_por_vela.get(interval, 15)
    expected_candles = int((days * 24 * 60) / min_per_candle)
    logger.info(f"[INFO] Velas esperadas (aprox): {expected_candles}")
    if len(df) < expected_candles * 0.8:
        logger.warning(f"[ADVERTENCIA] Se descargaron muchas menos velas ({len(df)}) de las esperadas ({expected_candles}). Verifica el símbolo, el intervalo o el histórico disponible en Binance.")

def main():
    parser = argparse.ArgumentParser(description='Download historical data from Binance')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--interval', default='15m', help='Candle interval')
    parser.add_argument('--days', type=int, default=30, help='Number of days to download')
    parser.add_argument('--output-file', default='data/raw/btcusdt_15m.csv', help='Output file path')
    args = parser.parse_args()
    
    logger = setup_logging()
    
    try:
        download_historical_data(
            symbol=args.symbol,
            interval=args.interval,
            days=args.days,
            output_file=args.output_file
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 