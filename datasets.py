import torch
import numpy as np
import pandas as pd
import math
import random
from tqdm import tqdm
from pathlib import Path
import pickle

class DayCloseDataset(torch.utils.data.Dataset):
    def __init__(self, start_date, end_date, n_dim, day_stride_length, in_length, out_length, ticker_redundancy=1):
        self.day_stride_length = day_stride_length
        self.in_length = in_length
        self.out_length = out_length
        
        df = pd.read_csv('sp500_stocks.csv')
        df = df[['Date', 'Symbol', 'Close']]
        self.data_by_ticker = []

        data_path = Path('pickled_close_data.pkl')
        if data_path.is_file():
            self.data_by_ticker = pickle.load(open(data_path, 'rb'))
        else:
            print("Unpacking dataset...")
            for x, y in tqdm(df.groupby('Symbol')):
                has_nan = False
                for index, row in y.iterrows():
                    if math.isnan(row['Close']):
                        has_nan = True
                if not has_nan:
                    self.data_by_ticker.append(y.sort_values(by='Date').set_index('Date').loc[start_date:end_date])
            pickle.dump(self.data_by_ticker, open(data_path, 'wb'))

        self.X = torch.stack([torch.from_numpy(x['Close'].to_numpy()) for x in self.data_by_ticker])
        data_shape = self.X.shape
        print(f'dataset shape: {data_shape}')

        for i in range(data_shape[0]):
            self.X[i] = self.X[i] / torch.mean(self.X[i])

        pickable_ticker_indices = np.arange(data_shape[0]).tolist()
        self.picked_ticker_indices = []
        for _ in range(math.ceil(data_shape[0] * ticker_redundancy / n_dim)):
            picked = []
            for _ in range(n_dim):
                picked.append(pickable_ticker_indices.pop(random.randrange(len(pickable_ticker_indices))))
                if len(pickable_ticker_indices) == 0:
                    pickable_ticker_indices = np.arange(data_shape[0]).tolist()
            self.picked_ticker_indices.append(picked)
        
        num_picked_days = math.ceil((data_shape[1] - in_length - out_length + 1) / day_stride_length)
        self.len = len(self.picked_ticker_indices) * num_picked_days
        
    def __len__(self):
        return self.len
        
    def __getitem__(self, idx):
        x = idx % len(self.picked_ticker_indices)
        y = idx // len(self.picked_ticker_indices)
        return self.X[self.picked_ticker_indices[x], y * self.day_stride_length:y * self.day_stride_length + self.in_length].swapaxes(0, 1).float(), self.X[self.picked_ticker_indices[x], y * self.day_stride_length + self.in_length:y * self.day_stride_length + self.in_length + self.out_length].swapaxes(0, 1).float()
    
if __name__ == '__main__':
    dataset = DayCloseDataset('2010-01-04', '2024-08-21', 256, 5, 30, 10)
    dataset.normalize()
    print(next(iter(dataset)))