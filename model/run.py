import argparse
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
from exp.exp_pct_prev import Exp_Pct_Prev
from exp.exp_regression import Exp_Regression
#from utils.print_args import print_args
import random
import numpy as np
import json
import time
import gc
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from datetime import datetime, timedelta

def parse_args():
    parser = argparse.ArgumentParser(description='model hyperparameter')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='forecast',
                        help='task name, options:[forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    
    # wandb and result saving
    parser.add_argument('--wandb_project_name', type=str, default='', help='wandb project name')
    parser.add_argument('--Notes', type=str, default='', help='what exp is it running')
    parser.add_argument('--result_file_name', type=str, default="default_result_file_name")

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--filename', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./model/checkpoints/', help='location of model checkpoints')
    parser.add_argument('--transform', type=str, default=None, help='options: [individual_seq_norm, pct_change_to_first, pct_change_to_prev]')
    '''
    another help: transform --> target
    individual_seq_norm --> normalized direct value
    pct_change_to_first --> pct to first
    pct_change_to_prev --> pct to last
    '''
    parser.add_argument('--target_transform', type=str, default=None, help='options: [individual_seq_norm, pct_change_to_first, pct_change_to_prev]')
    parser.add_argument('--n_features', type=int, required=True)

    # Data 
    parser.add_argument('--prediction_setting', type=str, default="short_term", help='see train_setting.json')
    parser.add_argument('--train_test_split', type=str, default="low_mismatch", help='see train_setting.json')
    parser.add_argument('--category', type=str, default="Test", help='see train_setting.json')
    
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--aggregation_method', type=str, default="default", help='max, min, max_min_diff, default')
    parser.add_argument('--goal', type=str, required=True, help='stop_10_take_20_max_roi_30, roi_30, max_roi_30')
    parser.add_argument('--pred_len', type=int, default=0)
    parser.add_argument('--seq_len', type=int, default=0)
    parser.add_argument('--stop_loss', type=float, default=0, help='stop loss')
    parser.add_argument('--take_profit', type=float, default=0, help='take profit')
    parser.add_argument('--trail_stop', type=float, default=0, help='trail stop')

    # region general model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--use_denorm', type=int, default=0, help='whether to use de-normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')
    parser.add_argument('--decomp_kernel_size', type=int, default = 0)
    parser.add_argument('--use_log', type=int, default=0)
    parser.add_argument('--make_dependent', type=int, default=0)
    # endregion
    
    # other features for individual models for MS
    parser.add_argument('--individual', action='store_true', help='Enable Individual of DLinear')
    parser.add_argument('--period_len', type=int, default=24, help='SparceTSF period length')
    parser.add_argument('--pooling', type=str, default="none",
                        help='different ways of pooling at the last layer.\
                        options: [average, max, attention]')
    
    # region covariates and static data setting
    parser.add_argument('--use_covariate', type=int, default=0, help="use covariate")
    parser.add_argument('--covariate_dim', type=int, default=0, 
                        help='dimension of covariate, for example, "[2007, 05, 23]" is 3. if not, if 0,\
                        it automaticly sets based on the freq')
    parser.add_argument('--use_static', type=int, default=0, help='use static data')
    parser.add_argument('--static_dim', type=int, default=0)
    parser.add_argument('--use_ts_as_covariate',type=int, default=0,)
    parser.add_argument('--use_general_daily', type=int, default=0)
    parser.add_argument('--batch_per_day', type=int, default=0)
    parser.add_argument('--maintain_order', type=int, default=0, help='maintain the batch order of data by time when using batch_per_day')
    parser.add_argument('--hard_no_shuffle', type=int, default=0)
    # endregion

    # region optimization
    parser.add_argument('--loss', type=str, default='MSE', help='option = [MSE, SpearmanRank, RankNet, ListNet, PairwiseRanking, \
                                                                        ContrastiveRanking, PearsonCorrelation, DistanceCorrelation, \
                                                                        ConcordanceCorrelation, CompositeLoss, NeuralNDCG, StockMixerLoss]')
    parser.add_argument('--output_path', type=str, default="results", help='output folder path')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    # endregion

    # region GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    # endregion

    # region de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    # endregion

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False, 
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    
    # region Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")
    
    parser.add_argument('--lower_barrier', type=float, default=0.9, help="lower barrier")
    parser.add_argument('--upper_barrier', type=float, default=1.1, help="upper barrier")
    parser.add_argument('--pred_lens', type=int, default=[30], nargs="+", help="list of pred len")
    #parser.add_argument('--test_day', type=str, nargs="+", help="test date")
    # endregion

    return parser.parse_args()

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args = parse_args()
    
    with open("model/train_setting.json", "r") as f:
        train_setting = json.load(f)
    
    args.target = train_setting["target"]
    if args.pred_len == 0:
        args.pred_len = train_setting["prediction_setting"][args.prediction_setting]["pred_len"]
    if args.seq_len == 0:
        args.seq_len = train_setting["prediction_setting"][args.prediction_setting]["seq_len"]
    args.label_len = train_setting["prediction_setting"][args.prediction_setting]["label_len"]
    args.train_start_date = train_setting["train_test_split"][args.train_test_split]["train_start_date"]
    args.valid_start_date = train_setting["train_test_split"][args.train_test_split]["valid_start_date"]
    args.test_start_date = train_setting["train_test_split"][args.train_test_split]["test_start_date"]
    args.test_end_date = train_setting["train_test_split"][args.train_test_split]["test_end_date"]
    '''
    dt = datetime.strptime(args.test_day[0], "%Y-%m-%d")   # 轉成 datetime
    prev_day = dt - timedelta(days=1)              # 減一天
    next_day = dt + timedelta(days=1)              # 加一天
    args.test_start_date = prev_day.strftime("%Y-%m-%d")   # 轉回字串
    args.test_end_date   = next_day.strftime("%Y-%m-%d")   # 轉回字串
    '''
    print("cuda available", torch.cuda.is_available())

    #print('Args in experiment:')
    #print_args(args)

    if args.task_name == 'regression':
        Exp = Exp_Regression
    elif args.task_name in ['pct_to_prev', 'from_scratch']:
        Exp = Exp_Pct_Prev
    else:
        raise NameError("Currently only support forecast")

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  
            setting = args.result_file_name

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            # breakpoint()
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = args.result_file_name
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
    print(f"模型訓練完成，預測結果已經存至 results/{setting} 資料夾")
    print(f"模型已經存至 model/checkpoints/{setting} 資料夾")