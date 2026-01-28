from .data_loader import (
    Dataset_PctPrev_NormInd_Pct_NormPrev,
    Dataset_NormInd_Pct, 
    Dataset_General,
    Dataset_Allen,
    Dataset_Allen2,
    Dataset_Allen3,
    Dataset_Triple_Barrier,
)
from torch.utils.data import DataLoader
import json
import argparse
from .utils import PreBatchRandomSampler
import os

def data_provider_trading(args, flag):
    timeenc = 0

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size  # bsz=1 for evaluation
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid

    with open("basic_info.json", "r") as f:
        basic_info = json.load(f)
    try:
        stock_ids = basic_info[args.category]
    except:
        stock_ids = basic_info[args.category+args.train_start_date[:4]+args.valid_start_date[:4]]

    if args.batch_per_day:
        if batch_size == 0:
            print(f"batch_size is 0, set batch_size to len(stock_ids)={len(stock_ids)}")
            batch_size = len(stock_ids)
        elif batch_size < len(stock_ids):
            raise ValueError(
                f"batch_size should not be less than the number of stock_ids, but got batch_size={args.batch_size} and len(stock_ids)={len(stock_ids)}"
            )
        
        if batch_size > len(stock_ids):
            print(f"fill up stock_ids to batch_size with nan: {len(stock_ids)} -> {batch_size}")
            stock_ids.extend([f"{i}" for i in range(len(stock_ids) - batch_size, 0)])
    if args.data == 'Dataset_PctPrev_NormInd_Pct_NormPrev':
        data_set = Dataset_PctPrev_NormInd_Pct_NormPrev(
            root_dir_path=args.root_path,
            stock_ids=stock_ids,
            size=[args.seq_len, args.label_len, args.pred_len],
            flag=flag, 
            target=args.target,
            split_dates=[args.train_start_date, args.valid_start_date, args.test_start_date, args.test_end_date],
            batch_per_day=args.batch_per_day,
            goal = args.goal,
            take_profit=args.take_profit,
            stop_loss=args.stop_loss,
            trail_stop=args.trail_stop
        )
    elif args.data == 'Dataset_NormInd_Pct':
        data_set = Dataset_NormInd_Pct(
            root_dir_path=args.root_path,
            stock_ids=stock_ids,
            size=[args.seq_len, args.label_len, args.pred_len],
            flag=flag, 
            target=args.target,
            split_dates=[args.train_start_date, args.valid_start_date, args.test_start_date, args.test_end_date],
            batch_per_day=args.batch_per_day,
            goal=args.goal,
            take_profit=args.take_profit,
            stop_loss=args.stop_loss,
            trail_stop=args.trail_stop
        )
    elif args.data == 'Dataset_General':
        data_set = Dataset_General(
            root_dir_path=args.root_path,
            stock_ids=[-1],
            size=[args.seq_len, args.label_len, args.pred_len],
            flag=flag, 
            target="taiex_total_index:收盤指數_TAIEX",
            split_dates=[args.train_start_date, args.valid_start_date, args.test_start_date, args.test_end_date]
        )
    elif args.data == 'Dataset_Individual_Seq_Norm':
        data_set = Dataset_General(
            root_dir_path=args.root_path,
            stock_ids=stock_ids,
            size=[args.seq_len, args.label_len, args.pred_len],
            flag=flag, 
            target=args.target,
            split_dates=[args.train_start_date, args.valid_start_date, args.test_start_date, args.test_end_date],
        )
    elif args.data == 'Dataset_Individual_Seq_Norm_with_GroupZ': # 13 個 feature，全員加上 Z 分數
        data_set = Dataset_Allen(
            root_dir_path=args.root_path,
            stock_ids=stock_ids,
            size=[args.seq_len, args.label_len, args.pred_len],
            flag=flag, 
            target=args.target,
            split_dates=[args.train_start_date, args.valid_start_date, args.test_start_date, args.test_end_date],
        )
    elif args.data == 'Dataset_GroupZ': # 13個 feature，全員純 Z 分數
        data_set = Dataset_Allen2(
            root_dir_path=args.root_path,
            stock_ids=stock_ids,
            size=[args.seq_len, args.label_len, args.pred_len],
            flag=flag, 
            target=args.target,
            split_dates=[args.train_start_date, args.valid_start_date, args.test_start_date, args.test_end_date],
        )
    elif args.data == 'Dataset_Individual_Seq_Norm_with_GroupZ2': # 26 個 feature，全員 + 純Z分數
        data_set = Dataset_Allen3(
            root_dir_path=args.root_path,
            stock_ids=stock_ids,
            size=[args.seq_len, args.label_len, args.pred_len],
            flag=flag, 
            target=args.target,
            split_dates=[args.train_start_date, args.valid_start_date, args.test_start_date, args.test_end_date],
        )
    elif args.data == 'Dataset_Triple_Barrier' and 'BCEWithLogits' in args.loss:
        data_set = Dataset_Triple_Barrier(
            root_dir_path=args.root_path,
            stock_ids=stock_ids,
            size=[args.seq_len, args.label_len, args.pred_len],
            barriers=[args.lower_barrier, args.upper_barrier],
            flag=flag, 
            target=args.target,
            split_dates=[args.train_start_date, args.valid_start_date, args.test_start_date, args.test_end_date],
            num_labels=2
        )
    elif args.data == 'Dataset_Triple_Barrier':
        data_set = Dataset_Triple_Barrier(
            root_dir_path=args.root_path,
            stock_ids=stock_ids,
            size=[args.seq_len, args.label_len, args.pred_len],
            barriers=[args.lower_barrier, args.upper_barrier],
            flag=flag, 
            target=args.target,
            split_dates=[args.train_start_date, args.valid_start_date, args.test_start_date, args.test_end_date],
            num_labels=3
        )
        
    else:
        raise Exception(f"{args.data} not provided")
    
    print(flag, len(data_set))

    sampler = None
    if args.hard_no_shuffle:
        shuffle_flag = False
        
    if args.batch_per_day and shuffle_flag:
        # shuffle without jeopardizing batch_per_day
        sampler = PreBatchRandomSampler(data_set, batch_size=batch_size, maintain_order=args.maintain_order)
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=None if sampler else shuffle_flag,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    
    for x in data_loader:
        pass
    
    return data_set, data_loader
        
if __name__ == "__main__":
    pass
    
