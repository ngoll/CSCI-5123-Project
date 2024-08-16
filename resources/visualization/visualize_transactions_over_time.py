import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os
os.chdir(os.path.dirname(os.path.dirname(os.getcwd())))

from resources.constants import *

# Group time period for transactions by week
GROUPING_TIME_PERIOD = "W"

# Load data
user_triplets_df = pd.read_csv(USER_ACTIVITY_TRIPLETS_CSV_PATH, sep=CSV_SEPARATOR)
original_orders_df = pd.read_csv(ORIGINAL_ORDERS_CSV_PATH, sep=CSV_SEPARATOR)
spot_rentals_df = pd.read_csv(SPOT_RENTALS_CSV_PATH, sep=CSV_SEPARATOR)

# Group rental start dates by the specified time period
def process_dataframe(df):
    df['rentalPeriod.start'] = pd.to_datetime(df['rentalPeriod.start'])
    weekly_data = df.groupby(df['rentalPeriod.start'].dt.to_period(GROUPING_TIME_PERIOD)).size().reset_index(name='count')
    weekly_data['rentalPeriod.start'] = weekly_data['rentalPeriod.start'].dt.to_timestamp()
    return weekly_data.sort_values('rentalPeriod.start')

weekly_transactions = process_dataframe(user_triplets_df)
weekly_spot_rentals = process_dataframe(spot_rentals_df)
weekly_original_orders = process_dataframe(original_orders_df)

# Plot the number of transactions per week
plt.figure(figsize=(16, 8))
plt.grid(True, linestyle='--', alpha=0.7)

plt.plot(weekly_transactions['rentalPeriod.start'], weekly_transactions['count'], label='Post 2020 Subscription Rentals', linewidth=2)
plt.plot(weekly_spot_rentals['rentalPeriod.start'], weekly_spot_rentals['count'], label='Post 2020 Spot Rentals', linewidth=2)
plt.plot(weekly_original_orders['rentalPeriod.start'], weekly_original_orders['count'], label='Pre 2020 Rentals', linewidth=2)

plt.title('Number of Transactions per Week', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Week', fontsize=14, labelpad=10)
plt.ylabel('Number of Transactions', fontsize=14, labelpad=10)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gcf().autofmt_xdate()  # Rotate and align the tick labels

plt.legend(fontsize=12, loc='upper left')

plt.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.savefig('reports/figures/weekly_transactions_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')