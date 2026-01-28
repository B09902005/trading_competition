backtest_start_year=2021
backtest_end_year=2026
train_period=3

data_category=Top100
model_name=iTransformer
loss=ConcordanceCorrelation
data=Dataset_Individual_Seq_Norm

python3 backtest/test.py \
    --backtest_start_year ${backtest_start_year} \
    --backtest_end_year ${backtest_end_year} \
    --train_period ${train_period} \
    --category ${data_category} \
    --model $model_name \
    --loss ${loss} \
    --data $data