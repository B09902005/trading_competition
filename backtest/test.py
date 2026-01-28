import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

import sys
import warnings
warnings.filterwarnings("ignore")
import os
import warnings
import pandas as pd
import requests
from pandas import MultiIndex
import quantstats as qs
from Allenbacktest import *
from finlab import data
import finlab
 
def get_valid_stocks(Target):
    bad = []
    for target in Target:
        try:
            df_stock = pd.read_csv(f"data/{target}.csv")
        except:
            bad.append(target)
            continue
        df_stock["high_grow"] = df_stock['etl:adj_high'] / df_stock['etl:adj_close'].shift(1)
        df_stock[ "low_grow"] =  df_stock['etl:adj_low'] / df_stock['etl:adj_close'].shift(1)
    for b in bad:
        Target.remove(b)
    return Target

def calculate_adx(df, period=14):
    # 計算 TR (True Range)
    df['high_low'] = df['etl:adj_high'] - df['etl:adj_low']
    df['high_close'] = abs(df['etl:adj_high'] - df['etl:adj_close'].shift(1))
    df['low_close'] = abs(df['etl:adj_low'] - df['etl:adj_close'].shift(1))
    df['TR'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    
    # 計算 +DM 和 -DM
    df['+DM'] = np.where((df['etl:adj_high'].diff() > 0) & (df['etl:adj_high'].diff() > -df['etl:adj_low'].diff()), df['etl:adj_high'].diff(), 0)
    df['-DM'] = np.where((-df['etl:adj_low'].diff() > 0) & (-df['etl:adj_low'].diff() > df['etl:adj_high'].diff()), -df['etl:adj_low'].diff(), 0)
    
    # 取 14 期加總值
    df['TR_smooth'] = df['TR'].rolling(window=period).sum()
    df['+DM_smooth'] = df['+DM'].rolling(window=period).sum()
    df['-DM_smooth'] = df['-DM'].rolling(window=period).sum()
    
    # 計算 +DI 和 -DI
    df['+DI'] = (df['+DM_smooth'] / df['TR_smooth']) * 100
    df['-DI'] = (df['-DM_smooth'] / df['TR_smooth']) * 100
    
    # 計算 DX
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    
    # 計算 ADX
    df['ADX'] = df['DX'].rolling(window=period).mean()
    
    # 選擇需要的欄位
    return df[['ADX']]


# 把價格記錄下來，存進 filtered_df
def get_filtered_df(Target):
    data.set_storage(data.FileStorage(path="finlab", use_cache=False))
    adj_open = data.get('etl:adj_open')[Target]
    adj_close = data.get('etl:adj_close')[Target]
    open = data.get('price:開盤價')[Target]
    close = data.get('price:收盤價')[Target]
    
    adj_close = close
    adj_open = open
    
    adj_close = pd.DataFrame(adj_close)
    adj_open = pd.DataFrame(adj_open)
    #adj_close = adj_close.combine_first(close_new)
    #adj_open = adj_open.combine_first(open_new)
    
    adj_open.columns = pd.MultiIndex.from_product([adj_open.columns, ['adj_open']], names=['Ticker', 'Price Type'])
    adj_close.columns = pd.MultiIndex.from_product([adj_close.columns, ['adj_close']], names=['Ticker', 'Price Type'])
    filtered_df = pd.concat([adj_close, adj_open], axis=1)
    filtered_df = filtered_df.sort_index(axis=1, level=1) 
    filtered_df = filtered_df.sort_index(axis=1, level=0, sort_remaining=True) 
    
    filtered_df.columns = MultiIndex.from_tuples(
        [(col[0], 'Close' if col[1] == 'adj_close' else 'Open') for col in filtered_df.columns]
    )
    filtered_df.fillna(method="bfill", inplace=True)
    return filtered_df

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Backtest script")

    parser.add_argument("--backtest_start_year", type=int, required=True)
    parser.add_argument("--backtest_end_year", type=int, required=True)
    parser.add_argument("--train_period", type=int, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--loss", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # get our target
    f = open("basic_info.json", "r")
    t = json.load(f)
    Target = t[args.category]
    Target = list(get_valid_stocks(Target))
    for i in range (len(Target)):
        Target[i] = str(Target[i])

    # calculate adx
    df_canbuy = pd.DataFrame()
    for target in Target:
        df_stock = pd.read_csv(f"data/{target}.csv")
        df_stock.index = df_stock["date"]
        df_stock = calculate_adx(df_stock).fillna(0)
        df_stock = df_stock > 40
        df_canbuy = pd.concat([df_canbuy, df_stock], axis=1)
    df_canbuy = df_canbuy.fillna(False)
    df_canbuy.columns = Target
    df_canbuy = df_canbuy.sort_index()
    df_canbuy.to_csv("outputs/adx.csv")

    # finlab
    finlab.login('ntSS3778pZi2FfkeYxXP0p+S0iI4AggkcphAUxh/lTVrWqT2FreKQsDkTA92CM7d#vip_m')

    filtered_df = get_filtered_df(Target)
    filtered_df = filtered_df[(filtered_df.index >= f'{args.backtest_start_year}-01-01') & (filtered_df.index <= f'{args.backtest_end_year}-12-31')]

    filtered_df.fillna(method="bfill", inplace=True)
    
    stock_ids = [column[0] for column in filtered_df.columns]
    stock_ids_series = pd.Series(stock_ids)
    stock_ids = stock_ids_series.drop_duplicates().tolist()
    date_index = filtered_df.index 

    # 模型預測的 .csv 相關。視檔名決定
    m = args.model
    goal = 'max_roi'
    train_years = [(i-args.train_period, i-1) for i in range(args.backtest_start_year, args.backtest_end_year+1)]
    result_dir = 'results'
    loss = args.loss
    data_category = args.category

    positions = pd.DataFrame(
        0.0, 
        index=date_index,
        columns=stock_ids
    )
    
    # 從 .csv 來生成買賣的訊號，存在 positions
    df_all = pd.DataFrame()
    for y1, y2 in train_years:
        result_file_dir = f"{args.category}_{args.model}_{y1}_{y2}_{args.loss}_{args.data}" 
        
        try:
            df = pd.read_csv(os.path.join(result_dir, result_file_dir, 'test', 'whole_output.csv'))
        except:
            print("no file ", os.path.join(result_dir, result_file_dir, 'test', 'whole_output.csv'))
            continue
        df_all = pd.concat([df_all, df], axis=0)
        
        df_val = pd.read_csv(os.path.join(result_dir, result_file_dir, 'test', 'whole_output.csv'))              
        df_val = df_val[df_val['date'].str.startswith(str(y2))]
        df_val = df_val.sort_values(by='pred_pct', ascending = False)
        date_len = len(df_val['date'].unique())
        
        df = df[df['stock_id'].astype(str).isin(Target)]
            
        df_train = pd.read_csv(os.path.join(result_dir, result_file_dir, 'train_vali', 'whole_output.csv')) 
        df_train_sorted = df_train.sort_values(['pred_pct'], ascending=[False])
        date_len_train = len(df_train['date'].unique())
        
        buy_signals, sell_signals = [], []
        for stock_id in Target:
            
            df_train_sorted_onestock = df_train_sorted  #[df_train_sorted['stock_id'] == int(stock_id)]
            super_high_threshold = df_train_sorted_onestock.iloc[df_train_sorted_onestock.shape[0]//4]["pred_pct"] if df_train_sorted_onestock.shape[0] != 0 else 0
            high_threshold = df_train_sorted_onestock.iloc[df_train_sorted_onestock.shape[0]//2]["pred_pct"] if df_train_sorted_onestock.shape[0] != 0 else 0
            low_threshold = df_train_sorted_onestock.iloc[-df_train_sorted_onestock.shape[0]//4]["pred_pct"] if df_train_sorted_onestock.shape[0] != 0 else 0
            
            # Method 1: Top 1 stock
            # df_sorted = df.sort_values(['date', 'pred_pct'], ascending=[True, False])
            # top_stocks = df_sorted.groupby('date').head(1)
            
            # Method 2: More than threshold stock
            df_sorted = df[df['stock_id'] == int(stock_id)].sort_values(['pred_pct'], ascending=[False])
            df_sorted.index = df_sorted["date"]
            
            top_stocks = df_sorted[(df_sorted["pred_pct"] > super_high_threshold) & (df_canbuy[stock_id])]
            buy_signals = buy_signals + list(zip(top_stocks['date'], top_stocks['stock_id']))
            
            bottom_stocks = df_sorted[df_sorted["pred_pct"] < low_threshold]
            sell_signals = sell_signals + list(zip(bottom_stocks['date'], bottom_stocks['stock_id']))
            
        from datetime import timedelta

        # init the position:
        position = pd.DataFrame(
            0.0, 
            index=date_index,
            columns=stock_ids
        )

        # print(len(buy_signals))
        # Iterate through each buy signal and update the position DataFrame
        for date, stock_id in buy_signals:
            stock_id = str(stock_id)
            if stock_id == '881':
                stock_id = '00881'
            if stock_id == '757':
                stock_id = '00757'
            date = pd.to_datetime(date)
            if stock_id not in position.columns:
                raise ValueError(f"{stock_id} not in position.columns")

            if date in position.index:
                position.loc[date, stock_id] += 1
            
        for date, stock_id in sell_signals:
            stock_id = str(stock_id)
            if stock_id == '881':
                stock_id = '00881'
            if stock_id == '757':
                stock_id = '00757'
            date = pd.to_datetime(date)
            if stock_id not in position.columns:
                raise ValueError(f"{stock_id} not in position.columns")

            if date in position.index:
                position.loc[date, stock_id] += -1
            
        start_date = df['date'].unique()[0]
        end_date = df['date'].unique()[-1]
        
        positions.loc[:, :] += position.values

    backtest = Backtest(Strategy, filtered_df, commission=.001425, cash=1e9)
    result = backtest.run(
        positions > 0,
        positions < 0,
        targets = Target, 
        max_positions = len(Target) * 1,
        is_benchmark = False
    )
    
    positions.to_csv("outputs/positions.csv")
    trades = pd.DataFrame(result.trades)
    trades.to_csv("outputs/trades.csv")
    open_positions = pd.DataFrame(result.open_positions)
    open_positions.to_csv("outputs/current_stocks.csv")
    returns = pd.DataFrame(result.returns)
    returns.to_csv("outputs/returns.csv")
    print("回測完成，輸出已經儲存至 outputs 資料夾")

