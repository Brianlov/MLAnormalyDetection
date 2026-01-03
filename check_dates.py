import pandas as pd

df = pd.read_csv('arrivals (1).csv')
df['date'] = pd.to_datetime(df['date'])
df_total = df[df['country'] == 'ALL']

print(f"Min Date: {df_total['date'].min()}")
print(f"Max Date: {df_total['date'].max()}")
print(f"Total rows: {len(df_total)}")
print(f"Rows before 2020-03-01: {len(df_total[df_total['date'] < '2020-03-01'])}")
