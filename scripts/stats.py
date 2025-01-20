import pandas as pd
import matplotlib.pyplot as plt

parquet_filename = "../data/2025-01-20.parquet"

# load the parquet into pandas
df = pd.read_parquet(parquet_filename)

# create a histogram based on the timestamp 
df['timestamp'] = pd.to_datetime(df['createdAt'], format='mixed', utc=True)
df['hour'] = df['timestamp'].dt.hour
df['hour'].hist(bins=24)
plt.xlabel('Hour of the day')
plt.ylabel('Number of posts')
plt.title('Number of posts per hour of the day')
plt.show()

