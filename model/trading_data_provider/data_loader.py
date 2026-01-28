import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Any
import pdb
import json
import sys
from collections import defaultdict
import torch

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(utils_path)

class Dataset_General(Dataset):
    def __init__(self, root_dir_path, stock_ids, size, flag,
                target, split_dates):
        
        # size [seq_len, label_len, pred_len]
        self.seq_len, self.label_len, self.pred_len = size

        self.target = target
        
        self.means = {stock_id:{} for stock_id in stock_ids}
        self.stds  = {stock_id:{} for stock_id in stock_ids}
        self.prevs = {stock_id:{} for stock_id in stock_ids}

        self.flag = flag
        
        # split train/valid/test by date
        self.split_dates = [pd.to_datetime(split_date) for split_date in split_dates]
        # splits:   0         1       2       3
        #           | train   | valid | test  |
        
        self.root_dir_path = root_dir_path
        
        self.df_flat = [] 
        for stock_id in stock_ids:
            data_dict = self.__read_data__(stock_id)
            if (data_dict == -1):
                continue
            for date, (seq_x, seq_y) in data_dict.items():
                self.df_flat.append({
                    'stock_id': stock_id,
                    'date': date,
                    'seq_x': seq_x,
                    'seq_y': seq_y
                })
        self.df_flat = pd.DataFrame(self.df_flat)

        self.df_flat.index = self.df_flat['date']

        self.df_flat.sort_index(inplace=True)
        
        self.set_goal()
        self.norm_x()
    
    def set_goal(self, goal = 'mean_30'):
        if goal == 'mean_30':
            self.df_flat['y'] = self.df_flat['seq_y'].apply(lambda seq_y: np.mean(seq_y)if len(seq_y) >= self.pred_len else -999)
            self.df_flat['y_max'] = self.df_flat['seq_y'].apply(lambda seq_y: np.max(seq_y)if len(seq_y) >= self.pred_len else -999)
            self.df_flat['y_min'] = self.df_flat['seq_y'].apply(lambda seq_y: np.min(seq_y)if len(seq_y) >= self.pred_len else -999)
            self.df_flat['y_last'] = self.df_flat['seq_y'].apply(lambda seq_y: seq_y[-1] if len(seq_y) >= self.pred_len else -999)
        
        elif goal == 'other':
            pass
        
        else:
            raise NameError(f"{goal} goal not provided")    
        
    def norm_x(self):
        pass
    
    def norm_y(self):
        pass
        

    # read all data based on stock id 
    def __read_data__(self, stock_id):
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if (stock_id == -1):
            csv_path = os.path.join(self.root_dir_path, 'general_daily_data', f"data.csv")
        else:
            csv_path = os.path.join(self.root_dir_path, f"{stock_id}.csv")
        try:
            df_raw = pd.read_csv(csv_path)
        except:
            return -1
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        dates = pd.to_datetime(df_raw['date'])
        
        self.n_variates = len(df_raw.columns) - 1
        
        df_raw.index = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.drop(['date'], axis=1)
        
        # Choose partition based on mode
        if self.flag == "train":
            test_start_date = self.split_dates[0] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[1] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
            try:
                idx = dates[dates >= self.split_dates[0]].index.min()
                test_start_date = dates[idx-self.seq_len+1]
            except:
                test_start_date = self.split_dates[0] - pd.Timedelta(days=self.seq_len * 7 // 5)
            try:
                idx = dates[pd.to_datetime(dates) < self.split_dates[1]].index.max()
                test_end_date = dates[idx+self.pred_len]
            except:
                test_end_date = self.split_dates[1] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
        elif self.flag == "val":
            test_start_date = self.split_dates[1] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[2] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
            try:
                idx = dates[dates >= self.split_dates[1]].index.min()
                test_start_date = dates[idx-self.seq_len+1]
            except:
                test_start_date = self.split_dates[1] - pd.Timedelta(days=self.seq_len * 7 // 5)
            try:
                idx = dates[pd.to_datetime(dates) < self.split_dates[2]].index.max()
                test_end_date = dates[idx]
            except:
                test_end_date = self.split_dates[2] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
        elif self.flag == "test":
            test_start_date = self.split_dates[2] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[3] + pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
            try:
                idx = dates[dates >= self.split_dates[2]].index.min()
                test_start_date = dates[idx-self.seq_len+1]
            except:
                test_start_date = self.split_dates[2] - pd.Timedelta(days=self.seq_len * 7 // 5)
            try:
                idx = dates[pd.to_datetime(dates) >= self.split_dates[3]].index.min()
                test_end_date = dates[idx+self.pred_len]
            except:
                test_end_date = self.split_dates[3] + pd.Timedelta(days=999)
        elif self.flag == 'train_val':
            test_start_date = self.split_dates[0] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[2] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
            try:
                idx = dates[dates >= self.split_dates[0]].index.min()
                test_start_date = dates[idx-self.seq_len+1]
            except:
                test_start_date = self.split_dates[0] - pd.Timedelta(days=self.seq_len * 7 // 5)
            try:
                idx = dates[pd.to_datetime(dates) < self.split_dates[2]].index.max()
                test_end_date = dates[idx]
            except:
                test_end_date = self.split_dates[2] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
        else:
            raise ValueError(f"Unknown flag: {self.flag}")
        df = df_raw.loc[test_start_date:test_end_date]

        data_len = len(df) - self.seq_len - self.pred_len + 1
        if test_end_date > pd.Timestamp.today():
            data_len += self.pred_len
        #    return {}
        #    raise ValueError("Data length is insufficient for the given sequence and prediction lengths.")

        date_index = df.index
        
        data = {}
        #df = df[df.columns[:8]]
        
        
        for index in range(data_len):
            end_date = date_index[index + self.seq_len - 1]
            s_begin, s_end = index, index + self.seq_len
            r_begin, r_end = index + self.seq_len, index + self.seq_len + self.pred_len
            seq_x = df.iloc[s_begin: s_end]
            seq_y = df.iloc[r_begin: r_end][self.target].values
        
            # X 跟 y 一起做標準化
            means = np.mean(seq_x, axis=0)
            stdev = np.sqrt(np.var(seq_x, axis=0, ddof=0) + 1e-5)
            self.means[stock_id][end_date] = means[0]
            self.stds[stock_id][end_date] = stdev[0]
            self.prevs[stock_id][end_date] = seq_x[self.target].iloc[-1]
            seq_x = (seq_x - means) / stdev
            seq_y = (seq_y - means[0]) / stdev[0]
            
            data[end_date] = [seq_x, seq_y]
        return data
    
    def __getitem__(self, index):
        #  date stock_id seq_x seq_y mean 
        row = self.df_flat.iloc[index]
        date = row['date']
        date = [date.year, date.month, date.day]
        return row['seq_x'].values, row['y'], date, row['stock_id'], row['y_min'], row['y_max'], row['y_last']

    def __len__(self):
        return len(self.df_flat) 
    
    def to_pct(self, y, date: str, stock_id: str):
        #return y
        date = pd.to_datetime(date)
        mean, std, prev = self.means[stock_id][date], self.stds[stock_id][date], self.prevs[stock_id][date]
        
        return ((y * std) + mean) / prev - 1

# without get_item and len
class Dataset_Basic(Dataset):
    def __init__(self, root_dir_path, stock_ids, size, flag,
                target, split_dates, batch_per_day=False):
        
        # size [seq_len, label_len, pred_len]
        self.seq_len, self.label_len, self.pred_len = size

        self.target = target

        self.root_dir_path = root_dir_path
        self.flag = flag
        
        # split train/valid/test by date
        self.split_dates = [pd.to_datetime(split_date) for split_date in split_dates]
        # splits:   0         1       2       3
        #           | train   | valid | test  |
    
        self.stock_ids = stock_ids
        self.batch_per_day = batch_per_day
        
        # data_dict_x[stock_id][date] = x
        self.data_dict_x = defaultdict(dict)
        self.data_dict_y = defaultdict(dict)
        
        for stock_id in stock_ids:
            seq_x_per_stock, y_per_stock = self.__read_data__(stock_id)
            self.data_dict_x[stock_id] = seq_x_per_stock
            self.data_dict_y[stock_id] = y_per_stock
        
        self.df_x = pd.DataFrame.from_dict(self.data_dict_x)
        self.df_y = pd.DataFrame.from_dict(self.data_dict_y)
        
        self.df_x.sort_index(inplace=True)
        self.df_y.sort_index(inplace=True)

        na_y = self.df_y.isna().sum()
        na_stock_ids = [stock_id for stock_id, na_count in na_y.items() if na_count > len(self.df_y) // 5]
        
        # index: date, columns: stock_id
        if self.batch_per_day:
            # fill nan with ones_df and zeros_arr
            ones_df = pd.DataFrame(1.0, index=np.arange(self.seq_len), columns=self.df_x_cols)
            zeros_arr = np.ones(self.pred_len)
            self.df_x = self.df_x.map(lambda _df : _df if isinstance(_df, pd.DataFrame) else ones_df)
            self.df_y = self.df_y.map(lambda _arr : _arr if isinstance(_arr, np.ndarray) else zeros_arr)
        else:
            # drop na stock_ids
            self.df_x = self.df_x.drop(columns=na_stock_ids)
            self.df_y = self.df_y.drop(columns=na_stock_ids)
        
        flat_df_x = self.flat(self.df_x, 'seq_x')
        flat_df_y = self.flat(self.df_y, 'close_y')
        
        flat_df_y = flat_df_y.dropna(axis=0)
        flat_df_x = flat_df_x.dropna(axis=0)
        
        self.df_flat = pd.merge(
            flat_df_x,
            flat_df_y,
            how='inner',
            on = ['date', 'stock_id']
        )
    
    def flat(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        flat_df = df.stack(future_stack=True).reset_index()
        flat_df.columns = ['date', 'stock_id', col_name]
        return flat_df

    # read all data based on stock id 
    def __read_data__(self, stock_id):
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        csv_path = os.path.join(self.root_dir_path, f"{stock_id}.csv")
        if not os.path.exists(csv_path):
            return np.nan, np.nan
        df_raw = pd.read_csv(csv_path)
        
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + [self.target] + cols]
        
        self.n_variates = len(df_raw.columns) - 1
        self.df_x_cols = [self.target] + cols # record the columns (for filling nan)
        
        df_raw.index = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.drop(['date'], axis=1)
        
        # Choose partition based on mode     
        if self.flag == "train":
            test_start_date = self.split_dates[0] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[1] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
        elif self.flag == "val":
            test_start_date = self.split_dates[1] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[2] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
        elif self.flag == "test":
            test_start_date = self.split_dates[2] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[3] + pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
        elif self.flag == 'train_val':
            test_start_date = self.split_dates[0] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[2] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
        else:
            raise ValueError(f"Unknown flag: {self.flag}")
        df = df_raw.loc[test_start_date:test_end_date]

        data_len = len(df) - self.seq_len - self.pred_len + 1
        if data_len <= 0:
            print(f"{stock_id} insufficient")
            # raise ValueError("Data length is insufficient for the given sequence and prediction lengths.")

        date_index = df.index
        
        seq_x_one_stock = {}
        y_one_stock = {}
        
        for index in range(data_len):
            end_date = date_index[index + self.seq_len - 1]
            s_begin, s_end = index, index + self.seq_len
            r_begin, r_end = index + self.seq_len, index + self.seq_len + self.pred_len
            seq_x = df.iloc[s_begin: s_end]
            y = df.iloc[r_begin: r_end][self.target].values
            seq_x_one_stock[end_date] = seq_x
            y_one_stock[end_date] = y
            
        return seq_x_one_stock, y_one_stock
    
    def __getitem__(self, index):
        row = self.df_flat.iloc[index]
        return row

    def __len__(self):
        return len(self.df_flat)
    
class Dataset_NormInd_Pct(Dataset_Basic):
    def __init__(self, goal,
                 take_profit, stop_loss, trail_stop,
                 seq_x_pct_target=True, *args, **kwargs):
        """
        Initializes the Dataset_NormInd_Pct instance.

        Inherits from Dataset_Basic and applies normalization.
        """

        self.seq_x_pct_target = seq_x_pct_target

        # Initialize the parent class with all necessary parameters
        super().__init__(*args, **kwargs)

        self.set_goal(goal, take_profit, stop_loss, trail_stop)
        self.norm_x_individual()

    def set_goal(self, goal='max_roi',
                 take_profit=0, stop_loss=0, trail_stop=0):
        assert take_profit >= 0, "take_profit should be positive"
        assert stop_loss >= 0, "stop_loss should be positive"
        assert trail_stop >= 0, "trail_stop should be positive"

        self.df_flat['last_close'] = self.df_flat['seq_x'].apply(lambda seq: seq[self.target].iloc[-1])
        raw_pct = self.df_flat['close_y'] / self.df_flat['last_close']
        raw_pct = raw_pct.apply(lambda pcts: np.nan_to_num(pcts, nan=1.0, posinf=1.0, neginf=1.0))

        if goal == 'max_roi_30' or goal == 'max_roi':
            self.df_flat['y'] = raw_pct.apply(lambda pcts: max(pcts) - 1)

        elif goal in ['stop_limit', 'exp_stop_limit', 'lin_stop_limit']:
            # take_profit, stop_loss, and trail_stop
            def get_first_true_index(arr:np.ndarray):
                idx = np.argmax(arr)
                return idx if arr[idx] else len(arr) - 1

            def get_stop_index(raw_pct:np.ndarray):
                stop_idx = len(raw_pct) - 1
                if take_profit is not None:
                    stop_idx = min(stop_idx, get_first_true_index(raw_pct >= (1 + take_profit)))
                if stop_loss is not None:
                    stop_idx = min(stop_idx, get_first_true_index(raw_pct <= (1 - stop_loss)))
                if trail_stop is not None:
                    stop_pct = np.maximum.accumulate(raw_pct) * (1 - trail_stop)
                    stop_idx = min(stop_idx, get_first_true_index(raw_pct <= stop_pct))

                return stop_idx

            def get_stop_roi(raw_pct:np.ndarray):
                stop_idx = get_stop_index(raw_pct)
                pct = raw_pct[stop_idx]
                if pct <= 0:
                    return -1.
                elif goal == 'exp_stop_limit':
                    return (pct ** (self.pred_len / (stop_idx + 1))) - 1.
                elif goal == 'lin_stop_limit':
                    return (pct - 1.) * (self.pred_len / (stop_idx + 1))

                return pct - 1.

            rois = raw_pct.apply(lambda pcts: get_stop_roi(pcts))
            self.df_flat['y'] = rois

        elif goal == 'Other':
            '''
            TODO:
            Implement Strategy
            Set it as self.df_flat['y']
            '''
            pass
    
    def norm_x_individual(self):
        
        def normalization(seq_x):
            data_x = seq_x.values
            mean = np.mean(data_x, axis = 0).reshape(1, -1)
            std = np.std(data_x, axis = 0).reshape(1, -1)
            std_safe = np.where(std == 0, 1, std)
            data_x_normalized = (data_x - mean) / std_safe

            if self.seq_x_pct_target:
                # transform the target to percentage
                target_last = seq_x[self.target].iloc[-1]
                target_last = 1 if target_last == 0 else target_last 
                data_x_normalized[:, 0] = seq_x[self.target] / target_last - 1

            normalized_df = pd.DataFrame(data_x_normalized, columns=seq_x.columns, index=seq_x.index)
            return normalized_df
        
        self.df_flat['seq_x'] = self.df_flat['seq_x'].apply(normalization)

    def to_pct(self, y, date: str, stock_id: str):
        return y
    
    def __getitem__(self, index):
        #  date stock_id seq_x seq_y mean 
        row = self.df_flat.iloc[index]
        date = row['date']
        date = [date.year, date.month, date.day]
        return row['seq_x'].values, row['y'], date, row['stock_id']


class Dataset_PctPrev_NormInd_Pct_NormPrev(Dataset_Basic):
    def __init__(self, goal, *args, **kargs):
        """
        Initializes the Dataset_PCT_To_Prev instance.

        Inherits from Dataset_Basic and applies percentage change and normalization.
        """
        # Initialize the parent class with all necessary parameters
        super().__init__(*args, **kargs)
        self.goal = goal
        
        # caculate the y_pct
        self.set_goal(goal)
        self.pct_change_x()
        self.norm_x_individual()
        self.norm_y_on_prev()

    def set_goal(self, goal = 'max_roi_30'):
        self.df_flat['last_close'] = self.df_flat['seq_x'].apply(lambda seq: seq[self.target].iloc[-1])
        
        if goal == 'max_roi_30':
            self.df_flat['max_close'] = self.df_flat['close_y'].apply(lambda close_y: max(close_y))
            self.df_flat['y'] = self.df_flat['max_close'] / self.df_flat['last_close'] - 1
        
        elif goal == 'roi_30':
            self.df_flat['30_close'] = self.df_flat['close_y'].apply(lambda close_y: close_y[-1])
            self.df_flat['y'] = self.df_flat['30_close'] / self.df_flat['last_close'] - 1
            
        elif goal == 'stop_10_take_20_max_roi_30':
            stop_loss, take_profit = -0.1, 0.2
            
            def find_goal(row):
                last_close = row['last_close']
                close_y = row['close_y']

                for i in range(min(30, len(close_y))):  # Ensure we don't go out of bounds
                    roi = close_y[i] / last_close - 1
                    if roi < stop_loss:
                        return stop_loss
                    elif roi > take_profit:
                        return take_profit
                # If neither stop_loss nor take_profit conditions are met, return max ROI
                return max(close_y) / last_close - 1

            self.df_flat['y'] = self.df_flat.apply(find_goal, axis=1)
        
        elif goal == 'stop_5_take_10_max_roi_30':
            stop_loss, take_profit = -0.05, 0.1
            
            def find_goal(row):
                last_close = row['last_close']
                close_y = row['close_y']

                for i in range(min(30, len(close_y))):  # Ensure we don't go out of bounds
                    roi = close_y[i] / last_close - 1
                    if roi < stop_loss:
                        return stop_loss
                    elif roi > take_profit:
                        return take_profit
                # If neither stop_loss nor take_profit conditions are met, return max ROI
                return max(close_y) / last_close - 1

            self.df_flat['y'] = self.df_flat.apply(find_goal, axis=1)
            
        elif goal == 'Other':
            '''
            TODO:
            Implement Strategy
            Set it as self.df_flat['y']
            '''
            pass
    
        self.df_flat['y'] = self.df_flat['y'].replace([np.inf, -np.inf, np.nan], 0)
        
    def pct_change_x(self):
        # do pct_change on seq_x
        def compute_pct(seq_x):
            data_x = seq_x.values.astype(float)
            devisor = np.vstack((data_x[0, :], data_x[:-1, :]))
            devisor[devisor == 0] = 1
            pct_x = data_x / devisor
            pct_x = pct_x - 1
            pct_x_df = pd.DataFrame(pct_x, columns=seq_x.columns, index=seq_x.index)
            return pct_x_df
        
        self.df_flat['seq_x'] = self.df_flat['seq_x'].apply(compute_pct)
        
    def norm_x_individual(self):
        
        def normalization(seq_x):
            data_x = seq_x.values
            mean = np.mean(data_x, axis = 0).reshape(1, -1)
            std = np.std(data_x, axis = 0).reshape(1, -1)
            std_safe = np.where(std == 0, 1, std)
            data_x_normalized = (data_x - mean) / std_safe
            normalized_df = pd.DataFrame(data_x_normalized, columns=seq_x.columns, index=seq_x.index)
            return normalized_df
        
        self.df_flat['seq_x'] = self.df_flat['seq_x'].apply(normalization)
    
    def norm_y_on_prev(self): 
        # Sort the DataFrame by 'date' and optionally by 'stock_id' for consistency
        self.df_flat = self.df_flat.sort_values(by=['date', 'stock_id']).reset_index(drop=True)
        
        # Compute mean and std of 'y' per date
        daily_stats = self.df_flat.groupby('date')['y'].agg(['mean', 'std']).reset_index()

        # Shift the statistics by one day
        daily_stats_shifted = daily_stats.copy()
        daily_stats_shifted['mean_prev'] = daily_stats_shifted['mean'].shift(1)
        daily_stats_shifted['std_prev'] = daily_stats_shifted['std'].shift(1)

        # Drop the original 'mean' and 'std' columns as they are now shifted
        # The first day of this shifted stat will ba Nan
        daily_stats_shifted = daily_stats_shifted.drop(['mean', 'std'], axis=1)
        daily_stats_shifted = daily_stats_shifted.dropna(axis=0)
        
        self.df_flat = self.df_flat.merge(
            daily_stats_shifted[['date', 'mean_prev', 'std_prev']],
            on='date',
            how='inner'
        )

        self.mean_std_map = self.df_flat.set_index(['date', 'stock_id'])[['mean_prev', 'std_prev']].to_dict('index') 
        self.df_flat['y'] = (self.df_flat['y'] - self.df_flat['mean_prev']) / self.df_flat['std_prev']
        #self.df_flat = self.df_flat[self.df_flat['std_prev'] != 0]
    
    def to_pct(self, y, date: str, stock_id: str):
        stock_id = str(stock_id)
        date = pd.to_datetime(date)
        val = self.mean_std_map[(date, stock_id)]
        mean, std = val['mean_prev'], val['std_prev']
        
        return (y * std) + mean

    def __len__(self):
        return len(self.df_flat)
    
    def __getitem__(self, index):
        #  date stock_id seq_x seq_y mean 
        row = self.df_flat.iloc[index]
        date = row['date']
        date = [date.year, date.month, date.day]
        return row['seq_x'].values, row['y'], date, row['stock_id']
    
    
    
class Dataset_Allen(Dataset):
    def __init__(self, root_dir_path, stock_ids, size, flag,
                target, split_dates):
        
        # size [seq_len, label_len, pred_len]
        self.seq_len, self.label_len, self.pred_len = size

        self.target = target
        
        self.means = {stock_id:{} for stock_id in stock_ids}
        self.stds  = {stock_id:{} for stock_id in stock_ids}
        self.prevs = {stock_id:{} for stock_id in stock_ids}
        self.adjust= {stock_id:{} for stock_id in stock_ids}
        
        self.raw_by_date = {}
        self.means_by_date = {}
        self.stds_by_date  = {}

        self.flag = flag
        
        # split train/valid/test by date
        self.split_dates = [pd.to_datetime(split_date) for split_date in split_dates]
        # splits:   0         1       2       3
        #           | train   | valid | test  |
        
        self.root_dir_path = root_dir_path
        
        self.df_flat = [] 
        for stock_id in stock_ids:
            data_dict = self.__read_data__(stock_id)
            if (data_dict == -1):
                continue
            for date, (seq_x, seq_y, raw) in data_dict.items():
                self.df_flat.append({
                    'stock_id': stock_id,
                    'date': date,
                    'seq_x': seq_x,
                    'seq_y': seq_y,
                    'raw': raw
                })
                if date not in self.raw_by_date:
                    self.raw_by_date[date] = [raw]
                else:
                    self.raw_by_date[date].append(raw)
                    
        for date in self.raw_by_date:
            self.raw_by_date[date] = pd.concat(self.raw_by_date[date], axis=1)
            self.means_by_date[date] = self.raw_by_date[date].mean(axis=1)
            self.stds_by_date[date]  = self.raw_by_date[date].std(axis=1) + 0.00001
        
        for i in range (len(self.df_flat)):
            date = self.df_flat[i]['date']
            adjustment = (self.df_flat[i]["raw"] - self.means_by_date[date]) / self.stds_by_date[date]
            self.adjust[self.df_flat[i]["stock_id"]][date] = adjustment[0]
            self.df_flat[i]['seq_x'] = self.df_flat[i]['seq_x'] + adjustment
            self.df_flat[i]['seq_y'] = self.df_flat[i]['seq_y'] + adjustment[0]
            
        self.df_flat = pd.DataFrame(self.df_flat)

        self.df_flat.index = self.df_flat['date']

        self.df_flat.sort_index(inplace=True)

        self.set_goal()
        self.norm_x()
    
    def set_goal(self, goal = 'mean_30'):
        if goal == 'mean_30':
            self.df_flat['y'] = self.df_flat['seq_y'].apply(lambda seq_y: np.mean(seq_y)if len(seq_y) >= self.pred_len else -999)
            self.df_flat['y_max'] = self.df_flat['seq_y'].apply(lambda seq_y: np.max(seq_y)if len(seq_y) >= self.pred_len else -999)
            self.df_flat['y_min'] = self.df_flat['seq_y'].apply(lambda seq_y: np.min(seq_y)if len(seq_y) >= self.pred_len else -999)
            self.df_flat['y_last'] = self.df_flat['seq_y'].apply(lambda seq_y: seq_y[-1] if len(seq_y) >= self.pred_len else -999)
        
        elif goal == 'other':
            pass
        
        else:
            raise NameError(f"{goal} goal not provided")    
        
    def norm_x(self):
        pass
    
    def norm_y(self):
        pass
        

    # read all data based on stock id 
    def __read_data__(self, stock_id):
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if (stock_id == -1):
            csv_path = os.path.join(self.root_dir_path, 'general_daily_data', f"data.csv")
        else:
            csv_path = os.path.join(self.root_dir_path, f"{stock_id}.csv")
        try:
            df_raw = pd.read_csv(csv_path)
        except:
            return -1
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        dates = pd.to_datetime(df_raw['date'])
        df_raw = df_raw[['date'] + [self.target] + cols]
        
        self.n_variates = len(df_raw.columns) - 1
        
        df_raw.index = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.drop(['date'], axis=1)
        
        # Choose partition based on mode        
        if self.flag == "train":
            test_start_date = self.split_dates[0] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[1] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
            try:
                idx = dates[dates >= self.split_dates[0]].index.min()
                test_start_date = dates[idx-self.seq_len+1]
            except:
                test_start_date = self.split_dates[0] - pd.Timedelta(days=self.seq_len * 7 // 5)
            try:
                idx = dates[pd.to_datetime(dates) < self.split_dates[1]].index.max()
                test_end_date = dates[idx+self.pred_len]
            except:
                test_end_date = self.split_dates[1] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
        elif self.flag == "val":
            test_start_date = self.split_dates[1] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[2] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
            try:
                idx = dates[dates >= self.split_dates[1]].index.min()
                test_start_date = dates[idx-self.seq_len+1]
            except:
                test_start_date = self.split_dates[1] - pd.Timedelta(days=self.seq_len * 7 // 5)
            try:
                idx = dates[pd.to_datetime(dates) < self.split_dates[2]].index.max()
                test_end_date = dates[idx]
            except:
                test_end_date = self.split_dates[2] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
        elif self.flag == "test":
            test_start_date = self.split_dates[2] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[3] + pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
            try:
                idx = dates[dates >= self.split_dates[2]].index.min()
                test_start_date = dates[idx-self.seq_len+1]
            except:
                test_start_date = self.split_dates[2] - pd.Timedelta(days=self.seq_len * 7 // 5)
            try:
                idx = dates[pd.to_datetime(dates) >= self.split_dates[3]].index.min()
                test_end_date = dates[idx+self.pred_len]
            except:
                test_end_date = self.split_dates[3] + pd.Timedelta(days=999)
        elif self.flag == 'train_val':
            test_start_date = self.split_dates[0] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[2] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
            try:
                idx = dates[dates >= self.split_dates[0]].index.min()
                test_start_date = dates[idx-self.seq_len+1]
            except:
                test_start_date = self.split_dates[0] - pd.Timedelta(days=self.seq_len * 7 // 5)
            try:
                idx = dates[pd.to_datetime(dates) < self.split_dates[2]].index.max()
                test_end_date = dates[idx]
            except:
                test_end_date = self.split_dates[2] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
        else:
            raise ValueError(f"Unknown flag: {self.flag}")
 
        df = df_raw.loc[test_start_date:test_end_date]        
        df = pd.DataFrame(np.where(df >= 1, np.log(df), 0), columns=df.columns, index=df.index)

        data_len = len(df) - self.seq_len - self.pred_len + 1
        if test_end_date > pd.Timestamp.today():
            data_len += self.pred_len
        #    return {}
        #    raise ValueError("Data length is insufficient for the given sequence and prediction lengths.")

        date_index = df.index
        
        data = {}
        #df = df[df.columns[:8]]
        
        
        for index in range(data_len):
            end_date = date_index[index + self.seq_len - 1]
            s_begin, s_end = index, index + self.seq_len
            r_begin, r_end = index + self.seq_len, index + self.seq_len + self.pred_len
            seq_x = df.iloc[s_begin: s_end]
            seq_y = df.iloc[r_begin: r_end][self.target].values
        
            # X 跟 y 一起做標準化
            means = np.mean(seq_x, axis=0)
            stdev = np.sqrt(np.var(seq_x, axis=0, ddof=0) + 1e-5)
            self.means[stock_id][end_date] = means[0]
            self.stds[stock_id][end_date] = stdev[0]
            self.prevs[stock_id][end_date] = seq_x[self.target].iloc[-1]
            seq_x = (seq_x - means) / stdev
            seq_y = (seq_y - means[0]) / stdev[0]
            
            data[end_date] = [seq_x, seq_y, df.iloc[index + self.seq_len - 1]]
        return data
    
    def __getitem__(self, index):
        #  date stock_id seq_x seq_y mean 
        row = self.df_flat.iloc[index]
        date = row['date']
        date = [date.year, date.month, date.day]
        return row['seq_x'].values, row['y'], date, row['stock_id'], row['y_min'], row['y_max'], row['y_last']

    def __len__(self):
        return len(self.df_flat) 
    
    def to_pct(self, y, date: str, stock_id: str):
        #return y
        date = pd.to_datetime(date)
        mean, std, prev, adj = self.means[stock_id][date], self.stds[stock_id][date], self.prevs[stock_id][date], self.adjust[stock_id][date]
        if isinstance(y, torch.Tensor):
            mean = torch.from_numpy( np.asarray(mean) ).to(device=y.device, dtype=y.dtype)
            std  = torch.from_numpy( np.asarray(std ) ).to(device=y.device, dtype=y.dtype)
            prev = torch.from_numpy( np.asarray(prev) ).to(device=y.device, dtype=y.dtype)
            adj  = torch.from_numpy( np.asarray(adj ) ).to(device=y.device, dtype=y.dtype)
            return torch.exp((((y - adj) * std) + mean)) / torch.exp(prev) - 1 
        else:
            return np.exp((((y - adj) * std) + mean)) / np.exp(prev) - 1 
    

class Dataset_Allen2(Dataset):
    def __init__(self, root_dir_path, stock_ids, size, flag,
                target, split_dates):
        
        # size [seq_len, label_len, pred_len]
        self.seq_len, self.label_len, self.pred_len = size

        self.target = target
        
        self.means = {stock_id:{} for stock_id in stock_ids}
        self.stds  = {stock_id:{} for stock_id in stock_ids}
        self.prevs = {stock_id:{} for stock_id in stock_ids}
        self.adjust= {stock_id:{} for stock_id in stock_ids}
        
        self.raw_by_date = {}
        self.means_by_date = {}
        self.stds_by_date  = {}

        self.flag = flag
        
        # split train/valid/test by date
        self.split_dates = [pd.to_datetime(split_date) for split_date in split_dates]
        # splits:   0         1       2       3
        #           | train   | valid | test  |
        
        self.root_dir_path = root_dir_path
        
        self.df_flat = [] 
        for stock_id in stock_ids:
            data_dict = self.__read_data__(stock_id)
            if (data_dict == -1):
                continue
            for date, (seq_x, seq_y, raw) in data_dict.items():
                self.df_flat.append({
                    'stock_id': stock_id,
                    'date': date,
                    'seq_x': seq_x,
                    'seq_y': seq_y,
                    'raw': raw
                })
                if date not in self.raw_by_date:
                    self.raw_by_date[date] = [raw]
                else:
                    self.raw_by_date[date].append(raw)
                    
        for date in self.raw_by_date:
            self.raw_by_date[date] = pd.concat(self.raw_by_date[date], axis=1)
            self.means_by_date[date] = self.raw_by_date[date].mean(axis=1)
            self.stds_by_date[date]  = self.raw_by_date[date].std(axis=1) + 0.00001
        
        for i in range (len(self.df_flat)):
            date = self.df_flat[i]['date']
            self.df_flat[i]['seq_x'] = (self.df_flat[i]['seq_x'] - self.means_by_date[date]) / self.stds_by_date[date]
            self.df_flat[i]['seq_y'] = (self.df_flat[i]['seq_y'] - self.means_by_date[date][0]) / self.stds_by_date[date][0]
            #self.df_flat[i]['seq_x'] = np.clip(self.df_flat[i]['seq_x'], -2, 2)
            
        self.df_flat = pd.DataFrame(self.df_flat)

        self.df_flat.index = self.df_flat['date']

        self.df_flat.sort_index(inplace=True)

        self.set_goal()
        self.norm_x()
    
    def set_goal(self, goal = 'mean_30'):
        if goal == 'mean_30':
            self.df_flat['y'] = self.df_flat['seq_y'].apply(lambda seq_y: np.mean(seq_y)if len(seq_y) >= self.pred_len else -999)
            self.df_flat['y_max'] = self.df_flat['seq_y'].apply(lambda seq_y: np.max(seq_y)if len(seq_y) >= self.pred_len else -999)
            self.df_flat['y_min'] = self.df_flat['seq_y'].apply(lambda seq_y: np.min(seq_y)if len(seq_y) >= self.pred_len else -999)
            self.df_flat['y_last'] = self.df_flat['seq_y'].apply(lambda seq_y: seq_y[-1] if len(seq_y) >= self.pred_len else -999)
        
        elif goal == 'other':
            pass
        
        else:
            raise NameError(f"{goal} goal not provided")    
        
    def norm_x(self):
        pass
    
    def norm_y(self):
        pass
        

    # read all data based on stock id 
    def __read_data__(self, stock_id):
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if (stock_id == -1):
            csv_path = os.path.join(self.root_dir_path, 'general_daily_data', f"data.csv")
        else:
            csv_path = os.path.join(self.root_dir_path, f"{stock_id}.csv")
        try:
            df_raw = pd.read_csv(csv_path)
        except:
            return -1
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        dates = pd.to_datetime(df_raw['date'])
        df_raw = df_raw[['date'] + [self.target] + cols]
        
        self.n_variates = len(df_raw.columns) - 1
        
        df_raw.index = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.drop(['date'], axis=1)
        
        # Choose partition based on mode        
        if self.flag == "train":
            test_start_date = self.split_dates[0] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[1] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
            try:
                idx = dates[dates >= self.split_dates[0]].index.min()
                test_start_date = dates[idx-self.seq_len+1]
            except:
                test_start_date = self.split_dates[0] - pd.Timedelta(days=self.seq_len * 7 // 5)
            try:
                idx = dates[pd.to_datetime(dates) < self.split_dates[1]].index.max()
                test_end_date = dates[idx+self.pred_len]
            except:
                test_end_date = self.split_dates[1] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
        elif self.flag == "val":
            test_start_date = self.split_dates[1] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[2] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
            try:
                idx = dates[dates >= self.split_dates[1]].index.min()
                test_start_date = dates[idx-self.seq_len+1]
            except:
                test_start_date = self.split_dates[1] - pd.Timedelta(days=self.seq_len * 7 // 5)
            try:
                idx = dates[pd.to_datetime(dates) < self.split_dates[2]].index.max()
                test_end_date = dates[idx]
            except:
                test_end_date = self.split_dates[2] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
        elif self.flag == "test":
            test_start_date = self.split_dates[2] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[3] + pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
            try:
                idx = dates[dates >= self.split_dates[2]].index.min()
                test_start_date = dates[idx-self.seq_len+1]
            except:
                test_start_date = self.split_dates[2] - pd.Timedelta(days=self.seq_len * 7 // 5)
            try:
                idx = dates[pd.to_datetime(dates) >= self.split_dates[3]].index.min()
                test_end_date = dates[idx+self.pred_len]
            except:
                test_end_date = self.split_dates[3] + pd.Timedelta(days=999)
        elif self.flag == 'train_val':
            test_start_date = self.split_dates[0] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[2] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
            try:
                idx = dates[dates >= self.split_dates[0]].index.min()
                test_start_date = dates[idx-self.seq_len+1]
            except:
                test_start_date = self.split_dates[0] - pd.Timedelta(days=self.seq_len * 7 // 5)
            try:
                idx = dates[pd.to_datetime(dates) < self.split_dates[2]].index.max()
                test_end_date = dates[idx]
            except:
                test_end_date = self.split_dates[2] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
        else:
            raise ValueError(f"Unknown flag: {self.flag}")
 
        df = df_raw.loc[test_start_date:test_end_date]        
        df = pd.DataFrame(np.where(df >= 1, np.log(df), 0), columns=df.columns, index=df.index)

        data_len = len(df) - self.seq_len - self.pred_len + 1
        if test_end_date > pd.Timestamp.today():
            data_len += self.pred_len
        #    return {}
        #    raise ValueError("Data length is insufficient for the given sequence and prediction lengths.")

        date_index = df.index
        
        data = {}
        #df = df[df.columns[:8]]
        
        
        for index in range(data_len):
            end_date = date_index[index + self.seq_len - 1]
            s_begin, s_end = index, index + self.seq_len
            r_begin, r_end = index + self.seq_len, index + self.seq_len + self.pred_len
            seq_x = df.iloc[s_begin: s_end]
            seq_y = df.iloc[r_begin: r_end][self.target].values
        
            # X 跟 y 一起做標準化
            means = np.mean(seq_x, axis=0)
            stdev = np.sqrt(np.var(seq_x, axis=0, ddof=0) + 1e-5)
            self.means[stock_id][end_date] = means[0]
            self.stds[stock_id][end_date] = stdev[0]
            self.prevs[stock_id][end_date] = seq_x[self.target].iloc[-1]
            #seq_x = (seq_x - means) / stdev
            #seq_y = (seq_y - means[0]) / stdev[0]
             
            data[end_date] = [seq_x, seq_y, df.iloc[index + self.seq_len - 1]]
        return data
    
    def __getitem__(self, index):
        #  date stock_id seq_x seq_y mean 
        row = self.df_flat.iloc[index]
        date = row['date']
        date = [date.year, date.month, date.day]
        return row['seq_x'].values, row['y'], date, row['stock_id'], row['y_min'], row['y_max'], row['y_last']

    def __len__(self):
        return len(self.df_flat) 
    
    def to_pct(self, y, date: str, stock_id: str):
        date = pd.to_datetime(date)
        mean, std, prev = self.means_by_date[date][0], self.stds_by_date[date][0], self.prevs[stock_id][date]
        if isinstance(y, torch.Tensor):
            mean = torch.from_numpy( np.asarray(mean) ).to(device=y.device, dtype=y.dtype)
            std  = torch.from_numpy( np.asarray(std ) ).to(device=y.device, dtype=y.dtype)
            prev = torch.from_numpy( np.asarray(prev) ).to(device=y.device, dtype=y.dtype)
            return torch.exp(((y * std) + mean)) / torch.exp(prev) - 1 
        else:
            return np.exp(((y * std) + mean)) / np.exp(prev) - 1 
       
       
       
   
class Dataset_Allen3(Dataset):
    def __init__(self, root_dir_path, stock_ids, size, flag,
                target, split_dates):
        
        # size [seq_len, label_len, pred_len]
        self.seq_len, self.label_len, self.pred_len = size

        self.target = target 
        
        self.means = {stock_id:{} for stock_id in stock_ids}
        self.stds  = {stock_id:{} for stock_id in stock_ids}
        self.prevs = {stock_id:{} for stock_id in stock_ids}
        self.adjust= {stock_id:{} for stock_id in stock_ids}
        
        self.raw_by_date = {}
        self.means_by_date = {}
        self.stds_by_date  = {}

        self.flag = flag
        
        # split train/valid/test by date
        self.split_dates = [pd.to_datetime(split_date) for split_date in split_dates]
        # splits:   0         1       2       3
        #           | train   | valid | test  |
        
        self.root_dir_path = root_dir_path
        
        self.df_flat = [] 
        for stock_id in stock_ids:
            data_dict = self.__read_data__(stock_id)
            if (data_dict == -1):
                continue
            for date, (seq_x, seq_y, raw) in data_dict.items():
                self.df_flat.append({
                    'stock_id': stock_id,
                    'date': date,
                    'seq_x': seq_x,
                    'seq_y': seq_y,
                    'raw': raw
                })
                if date not in self.raw_by_date:
                    self.raw_by_date[date] = [raw.iloc[-1]]
                else:
                    self.raw_by_date[date].append(raw.iloc[-1])
                    
        for date in self.raw_by_date:
            self.raw_by_date[date] = pd.concat(self.raw_by_date[date], axis=1)
            self.means_by_date[date] = self.raw_by_date[date].mean(axis=1)
            self.stds_by_date[date]  = self.raw_by_date[date].std(axis=1) + 0.00001
        
        for i in range (len(self.df_flat)):
            date = self.df_flat[i]['date']
            adjustment = (self.df_flat[i]['raw'] - self.means_by_date[date]) / self.stds_by_date[date]
            self.df_flat[i]['seq_x'] = pd.concat([self.df_flat[i]['seq_x'], adjustment], axis=1)
            
        self.df_flat = pd.DataFrame(self.df_flat)

        self.df_flat.index = self.df_flat['date']

        self.df_flat.sort_index(inplace=True)

        self.set_goal()
        self.norm_x()
    
    def set_goal(self, goal = 'mean_30'):
        if goal == 'mean_30':
            self.df_flat['y'] = self.df_flat['seq_y'].apply(lambda seq_y: np.mean(seq_y)if len(seq_y) >= self.pred_len else -999)
            self.df_flat['y_max'] = self.df_flat['seq_y'].apply(lambda seq_y: np.max(seq_y)if len(seq_y) >= self.pred_len else -999)
            self.df_flat['y_min'] = self.df_flat['seq_y'].apply(lambda seq_y: np.min(seq_y)if len(seq_y) >= self.pred_len else -999)
            self.df_flat['y_last'] = self.df_flat['seq_y'].apply(lambda seq_y: seq_y[-1] if len(seq_y) >= self.pred_len else -999)
        
        elif goal == 'other':
            pass
        
        else:
            raise NameError(f"{goal} goal not provided")    
        
    def norm_x(self):
        pass
    
    def norm_y(self):
        pass
        

    # read all data based on stock id 
    def __read_data__(self, stock_id):
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if (stock_id == -1):
            csv_path = os.path.join(self.root_dir_path, 'general_daily_data', f"data.csv")
        else:
            csv_path = os.path.join(self.root_dir_path, f"{stock_id}.csv")
        try:
            df_raw = pd.read_csv(csv_path)
        except:
            return -1
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        dates = pd.to_datetime(df_raw['date'])
        df_raw = df_raw[['date'] + [self.target] + cols]
        
        self.n_variates = len(df_raw.columns) - 1
        
        df_raw.index = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.drop(['date'], axis=1)
        
        # Choose partition based on mode        
        if self.flag == "train":
            test_start_date = self.split_dates[0] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[1] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
            try:
                idx = dates[dates >= self.split_dates[0]].index.min()
                test_start_date = dates[idx-self.seq_len+1]
            except:
                test_start_date = self.split_dates[0] - pd.Timedelta(days=self.seq_len * 7 // 5)
            try:
                idx = dates[pd.to_datetime(dates) < self.split_dates[1]].index.max()
                test_end_date = dates[idx+self.pred_len]
            except:
                test_end_date = self.split_dates[1] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
        elif self.flag == "val":
            test_start_date = self.split_dates[1] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[2] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
            try:
                idx = dates[dates >= self.split_dates[1]].index.min()
                test_start_date = dates[idx-self.seq_len+1]
            except:
                test_start_date = self.split_dates[1] - pd.Timedelta(days=self.seq_len * 7 // 5)
            try:
                idx = dates[pd.to_datetime(dates) < self.split_dates[2]].index.max()
                test_end_date = dates[idx]
            except:
                test_end_date = self.split_dates[2] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
        elif self.flag == "test":
            test_start_date = self.split_dates[2] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[3] + pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
            try:
                idx = dates[dates >= self.split_dates[2]].index.min()
                test_start_date = dates[idx-self.seq_len+1]
            except:
                test_start_date = self.split_dates[2] - pd.Timedelta(days=self.seq_len * 7 // 5)
            try:
                idx = dates[pd.to_datetime(dates) >= self.split_dates[3]].index.min()
                test_end_date = dates[idx+self.pred_len]
            except:
                test_end_date = self.split_dates[3] + pd.Timedelta(days=999)
        elif self.flag == 'train_val':
            test_start_date = self.split_dates[0] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[2] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
            try:
                idx = dates[dates >= self.split_dates[0]].index.min()
                test_start_date = dates[idx-self.seq_len+1]
            except:
                test_start_date = self.split_dates[0] - pd.Timedelta(days=self.seq_len * 7 // 5)
            try:
                idx = dates[pd.to_datetime(dates) < self.split_dates[2]].index.max()
                test_end_date = dates[idx]
            except:
                test_end_date = self.split_dates[2] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
        else:
            raise ValueError(f"Unknown flag: {self.flag}")
 
        df = df_raw.loc[test_start_date:test_end_date]        
        df = pd.DataFrame(np.where(df >= 1, np.log(df), 0), columns=df.columns, index=df.index)

        data_len = len(df) - self.seq_len - self.pred_len + 1
        if test_end_date > pd.Timestamp.today():
            data_len += self.pred_len
        #    return {}
        #    raise ValueError("Data length is insufficient for the given sequence and prediction lengths.")

        date_index = df.index
        
        data = {}
        #df = df[df.columns[:8]]
        
        
        for index in range(data_len):
            end_date = date_index[index + self.seq_len - 1]
            s_begin, s_end = index, index + self.seq_len
            r_begin, r_end = index + self.seq_len, index + self.seq_len + self.pred_len
            seq_x = df.iloc[s_begin: s_end]
            seq_y = df.iloc[r_begin: r_end][self.target].values
        
            # X 跟 y 一起做標準化
            means = np.mean(seq_x, axis=0)
            stdev = np.sqrt(np.var(seq_x, axis=0, ddof=0) + 1e-5)
            self.means[stock_id][end_date] = means[0]
            self.stds[stock_id][end_date] = stdev[0]
            self.prevs[stock_id][end_date] = seq_x[self.target].iloc[-1]
            seq_x = (seq_x - means) / stdev
            seq_y = (seq_y - means[0]) / stdev[0]
            
            data[end_date] = [seq_x, seq_y, df.iloc[s_begin: s_end]]
        return data
    
    def __getitem__(self, index):
        #  date stock_id seq_x seq_y mean 
        row = self.df_flat.iloc[index]
        date = row['date']
        date = [date.year, date.month, date.day]
        return row['seq_x'].values, row['y'], date, row['stock_id'], row['y_min'], row['y_max'], row['y_last']

    def __len__(self):
        return len(self.df_flat) 
    
    def to_pct(self, y, date: str, stock_id: str):
        #return y
        date = pd.to_datetime(date)
        mean, std, prev = self.means[stock_id][date], self.stds[stock_id][date], self.prevs[stock_id][date]
        if isinstance(y, torch.Tensor):
            mean = torch.from_numpy( np.asarray(mean) ).to(device=y.device, dtype=y.dtype)
            std  = torch.from_numpy( np.asarray(std ) ).to(device=y.device, dtype=y.dtype)
            prev = torch.from_numpy( np.asarray(prev) ).to(device=y.device, dtype=y.dtype)
            return torch.exp(((y * std) + mean)) / torch.exp(prev) - 1 
        else:
            return np.exp(((y * std) + mean)) / np.exp(prev) - 1 
     

if __name__ == "__main__":
    print("data loader testing")
    root_path = "../trading_data/all_data"
    data_filenames = os.listdir(root_path)
    size = (60, 48, 30)

    with open("../trading_data/basic_info.json", "r") as f:
        basic_info = json.load(f)
    
    with open("../trading_data/train_setting.json") as f:
        train_setting = json.load(f)
    
    stock_ids = basic_info["Semiconductor"]
    
    dataset_basic = Dataset_Basic(
        root_dir_path=root_path, 
        stock_ids=stock_ids, 
        size=size,
        flag='test', 
        target="etl:adj_close", 
        split_dates=["2015-01-01", "2017-01-01", "2018-01-01", "2019-01-01"]
    )

    dataset_general = Dataset_General(
        root_dir_path=root_path, 
        size=size,
        flag='test', 
        target="market_transaction_info:收盤指數_TAIEX", 
        split_dates=["2015-01-01", "2017-01-01", "2018-01-01", "2019-01-01"]
    )
    
    dataset_pct = Dataset_PctPrev_NormInd_Pct_NormPrev(
        root_dir_path=root_path, 
        stock_ids=stock_ids, 
        size=size,
        flag='test', 
        target="etl:adj_close", 
        split_dates=["2015-01-01", "2017-01-01", "2018-01-01", "2019-01-01"],
        goal='stop_10_take_20_max_roi_30'
    )

    dataset_pct_batch = Dataset_NormInd_Pct(
        root_dir_path=root_path, 
        stock_ids=stock_ids, 
        size=size,
        flag='train', 
        target="etl:adj_close", 
        split_dates=["2015-01-01", "2017-01-01", "2018-01-01", "2019-01-01"],
        batch_per_day=True,
        goal='max_roi',
        take_profit=0,
        stop_loss=0,
        trail_stop=0
    )

    # na 2338, 3545,
    
    breakpoint()



class Dataset_Triple_Barrier(Dataset):
    def __init__(self, root_dir_path, stock_ids, size, barriers, flag,
                target, split_dates, num_labels):
        
        # size [seq_len, label_len, pred_len]
        self.seq_len, self.label_len, self.pred_len = size

        self.target = target

        self.flag = flag
        self.num_labels = num_labels
        
        # split train/valid/test by date
        self.split_dates = [pd.to_datetime(split_date) for split_date in split_dates]
        # splits:   0         1       2       3
        #           | train   | valid | test  |
        
        self.root_dir_path = root_dir_path
        #self.lower_barrier, self.upper_barrier = self.__find_barrier__(stock_ids)
        self.lower_barrier, self.upper_barrier = barriers
        
        self.df_flat = [] 
        for stock_id in stock_ids:
            data_dict = self.__read_data__(stock_id)
            if (data_dict == -1):
                continue
            for date, (seq_x, seq_y) in data_dict.items():
                self.df_flat.append({
                    'stock_id': stock_id,
                    'date': date,
                    'seq_x': seq_x,
                    'y': seq_y,
                })
        self.df_flat = pd.DataFrame(self.df_flat)

        self.df_flat.index = self.df_flat['date']

        self.df_flat.sort_index(inplace=True)
        
    def __find_barrier__(self, stock_ids):
        mins, maxs = [], []
        for stock_id in stock_ids:
            csv_path = os.path.join(self.root_dir_path, f"{stock_id}.csv")
            try:
                df_raw = pd.read_csv(csv_path)
            except:
                continue
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + [self.target] + cols]
            df_raw.index = pd.to_datetime(df_raw['date'])
            df_raw = df_raw.drop(['date'], axis=1)
            
            test_start_date = self.split_dates[0] - pd.Timedelta(days=365)
            test_end_date   = self.split_dates[0] - pd.Timedelta(days=1)
            df = df_raw.loc[test_start_date:test_end_date]

            data_len = len(df) - self.pred_len
            date_index = df.index
        
            for index in range(data_len):
                today = df.iloc[index][self.target]
                seq_y = df.iloc[index+1: index+self.pred_len+1][self.target].values
                maxs.append(np.max(seq_y)/today)
                mins.append(np.min(seq_y)/today)
        maxs = sorted(maxs)
        mins = sorted(mins)
        return mins[len(mins)//3], maxs[len(maxs)//3*2]

    # read all data based on stock id 
    def __read_data__(self, stock_id):
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if (stock_id == -1):
            csv_path = os.path.join(self.root_dir_path, 'general_daily_data', f"data.csv")
        else:
            csv_path = os.path.join(self.root_dir_path, f"{stock_id}.csv")
        try:
            df_raw = pd.read_csv(csv_path)
        except:
            return -1
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + [self.target] + cols]
        
        self.n_variates = len(df_raw.columns) - 1
        
        df_raw.index = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.drop(['date'], axis=1)
        
        # Choose partition based on mode        
        if self.flag == "train":
            test_start_date = self.split_dates[0] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[1] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
        elif self.flag == "val":
            test_start_date = self.split_dates[1] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[2] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
        elif self.flag == "test":
            test_start_date = self.split_dates[2] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[3] + pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
        elif self.flag == 'train_val':
            test_start_date = self.split_dates[0] - pd.Timedelta(days=self.seq_len * 7 // 5)
            test_end_date   = self.split_dates[2] - pd.Timedelta(days=self.pred_len* 7 // 5 + 7)
        else:
            raise ValueError(f"Unknown flag: {self.flag}")
        df = df_raw.loc[test_start_date:test_end_date]

        data_len = len(df) - self.seq_len - self.pred_len + 1
        if test_end_date > pd.Timestamp.today():
            data_len += self.pred_len
        #    return {}
        #    raise ValueError("Data length is insufficient for the given sequence and prediction lengths.")

        date_index = df.index
        
        data = {}
        #df = df[df.columns[:8]]
        
        
        for index in range(data_len):
            end_date = date_index[index + self.seq_len - 1]
            s_begin, s_end = index, index + self.seq_len
            r_begin, r_end = index + self.seq_len, index + self.seq_len + self.pred_len
            seq_x = df.iloc[s_begin: s_end]
            seq_y = df.iloc[r_begin: r_end][self.target].values
            
            above_mask = seq_y > seq_x[self.target].iloc[-1] * self.upper_barrier
            below_mask = seq_y < seq_x[self.target].iloc[-1] * self.lower_barrier
            above_idx = np.argmax(above_mask) if np.any(above_mask) else seq_y.shape[0]
            below_idx = np.argmax(below_mask) if np.any(below_mask) else seq_y.shape[0]
            if (self.num_labels == 3):
                if (above_idx < below_idx):
                    seq_y = 2
                elif (above_idx > below_idx):
                    seq_y = 0
                else:
                    seq_y = 1
            else:
                if (above_idx < below_idx):
                    seq_y = 1
                else:
                    seq_y = 0
            data[end_date] = [seq_x, seq_y]
        return data
    
    def __getitem__(self, index):
        #  date stock_id seq_x seq_y mean 
        row = self.df_flat.iloc[index]
        date = row['date']
        date = [date.year, date.month, date.day]
        return row['seq_x'].values, row['y'], date, row['stock_id'], row['y_min'], row['y_max'], row['y_last']

    def __len__(self):
        return len(self.df_flat) 
    
    def to_pct(self, y, date: str, stock_id: str):
        return y
        date = pd.to_datetime(date)
        mean, std, prev = self.means[stock_id][date], self.stds[stock_id][date], self.prevs[stock_id][date]
        
        return ((y * std) + mean) / prev - 1
