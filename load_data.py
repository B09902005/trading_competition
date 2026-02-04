from datetime import datetime, time, timedelta
import pandas as pd
import json
import csv

import numpy as np
from finlab import data, login

def feat_id_inverse(data_by_feat, all_stock_ids):
    datas_by_stock_id = {}
    feats = list(data_by_feat.keys())

    for stock_id in all_stock_ids:
        valid = True
        for feat in feats:
            if stock_id not in data_by_feat[feat].columns:
                print(stock_id, feat)
                valid = False
        if valid:
            datas_by_stock_id[stock_id] = pd.DataFrame()
            for feat in feats:
                datas_by_stock_id[stock_id][feat] = data_by_feat[feat][stock_id]
                datas_by_stock_id[stock_id].dropna(inplace=True) 
    return datas_by_stock_id

        
if __name__ == "__main__":
    now = datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

    f = open("basic_info.json", "r")
    t = json.load(f)
    target = []
    for category in t:
        target = target + t[category]
    target = list(set(target))
    # 下載最新訓練資料 可見 data/
    login('ntSS3778pZi2FfkeYxXP0p+S0iI4AggkcphAUxh/lTVrWqT2FreKQsDkTA92CM7d#vip_m')
    #data.set_storage(data.FileStorage(path="finlab", use_cache=False))
    individual_daily_features_dictionary = {
        'etl': ['adj_close', 'adj_open', 'adj_high', 'adj_low'],
        'price': ['成交股數', '成交筆數', '成交金額'],
        'institutional_investors_trading_summary': ['外陸資買賣超股數(不含外資自營商)', '外資自營商買賣超股數',
                                                    '投信買賣超股數', '自營商買賣超股數(自行買賣)', '自營商買賣超股數(避險)'],
        'foreign_investors_shareholding':['全體外資及陸資持股比率']
    }
    individual_stock_features = []
    for key, columns in individual_daily_features_dictionary.items():
        for feat in columns:
            individual_stock_features.append(f"{key}:{feat}")
            
    data_by_feat = {}

    for feat in individual_stock_features:
        data_by_feat[feat] = data.get(feat)
        if 'etl' not in feat and 'price' not in feat:
            # Replace NaN values with 0
            data_by_feat[feat] = data_by_feat[feat].fillna(0)
        
    datas_by_stock_id = feat_id_inverse(data_by_feat, target)

    data_dir = "data/"

    for stock_id, df in datas_by_stock_id.items():
        df.to_csv(f"{data_dir}/{stock_id}.csv")
    print("資料獲取完成，股票資料已經儲存至 data 資料夾")
