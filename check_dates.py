import pandas as pd
import os

DATA_DIR = 'data'
# Check if file exists first to avoid crash if it doesn't
if os.path.exists(os.path.join(DATA_DIR, 'logins.csv')):
    logins = pd.read_csv(os.path.join(DATA_DIR, 'logins.csv'), parse_dates=['login'])
    dates = sorted(logins['login'].dt.date.unique())
    print("Available dates in the data:")
    for date in dates[:10]:  # Show first 10 dates
        print(f"  {date}")
    if len(dates) > 10:
        print(f"  ... and {len(dates) - 10} more dates")
    print(f"\nLatest date: {dates[-1]}")
    print(f"Total unique dates: {len(dates)}")
else:
    print("Error: data/logins.csv not found.")
