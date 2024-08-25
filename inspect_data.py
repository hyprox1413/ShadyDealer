import pandas as pd
import torch
import torch.nn
import math
import random

if __name__ == '__main__':
    
    df = pd.read_csv('sp500_stocks.csv')
    print(df.head())
    print(df.columns)
    df = df[['Date', 'Symbol', 'Close']]
    print(df.head())
    num_tracked_tickers = df["Symbol"].nunique()
    print(f'number of tracked tickers: {num_tracked_tickers}')
    num_tracked_days = df.shape[0] / num_tracked_tickers
    print(f'number of tracked days: {num_tracked_days}')
    nan_tickers = 0
    """
    for x, y in (df.groupby('Symbol')):
        print(x)
        has_nan = False
        for index, row in y.iterrows():
            if math.isnan(row['Close']):
                has_nan = True
        if has_nan:
            nan_tickers += 1
        if y.size != tracked_days:
            print('irregular data')
            quit()
    print(f'number of tickers with NaN: {nan_tickers}')
    """
    print(f'earliest tracked day: {df["Date"].min()}')
    print(f'latest tracked day: {df["Date"].max()}')