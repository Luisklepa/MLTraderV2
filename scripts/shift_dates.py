import pandas as pd

# Load the original data
input_path = 'data/processed/ml_signals.csv'
output_path = 'data/processed/ml_signals_shifted.csv'

# Read CSV with datetime index
print(f'Loading {input_path}...')
df = pd.read_csv(input_path, index_col='datetime', parse_dates=True)

# Shift all dates back by 5 years
print('Shifting all dates back by 5 years...')
df.index = df.index - pd.DateOffset(years=5)

# Drop rows with invalid or missing datetimes
df = df[~df.index.isnull()]

# Print min and max date for verification
print(f'Min date after shift: {df.index.min()}')
print(f'Max date after shift: {df.index.max()}')

# Save the shifted data
print(f'Saving shifted data to {output_path}...')
df.to_csv(output_path)

print('Done!') 