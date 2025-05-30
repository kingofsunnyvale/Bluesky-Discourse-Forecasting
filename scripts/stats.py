import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

df = pd.read_parquet("../data/2025-01-20.parquet")

# Make histogram of the number of posts per hour of the day, only considering posts from 2025-01-20
df['timestamp'] = pd.to_datetime(df['createdAt'], format='mixed', utc=True)
df['hour'] = df['timestamp'].dt.hour
df['date'] = df['timestamp'].dt.date
df_filtered = df[df['timestamp'].dt.date == pd.to_datetime('2025-01-20').date()]
hour_counts = df_filtered.groupby('hour').size().reindex(range(24), fill_value=0)

plt.figure(figsize=(10, 6))
plt.bar(hour_counts.index, hour_counts.values,
        width=0.8, edgecolor='black', color='cornflowerblue')

def comma_formatter(x, p):
    return format(int(x), ',')
plt.gca().yaxis.set_major_formatter(FuncFormatter(comma_formatter))

plt.xlabel('Hour of the day (UTC)')
plt.ylabel('Number of posts')
plt.title('Number of posts per hour of the day')
plt.xticks(range(24))
plt.grid(axis='y', alpha=0.5)
plt.show()