# from data_provider.data_factory import data_provider
from trading_data_provider.data_factory import data_provider_trading
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.losses import (
    mape_loss, 
    PairwiseRankingLoss, 
    SpearmanRankCorrelationLoss,
    RankNetLoss, 
    ListNetLoss, 
    ContrastiveRankingLoss,
    PearsonCorrelationLoss,
    DistanceCorrelationLoss,
    ConcordanceCorrelationCoefficientLoss,
    CompositeLoss,
    NeuralNDCGLoss
)
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import wandb
import shutil
from models.BasicModel import BasicModel
import gc
from torch.utils.data import ConcatDataset, DataLoader
import csv

warnings.filterwarnings('ignore')


class Exp_Regression(Exp_Basic):
    def __init__(self, args):
        super(Exp_Regression, self).__init__(args)
        if args.wandb_project_name != "":
            wandb.init(
                project=args.wandb_project_name,
                config=vars(args),
            )
        self.best_train_loss = 10e5
        self.best_valid_loss = 10e5
        self.best_test_loss = 10e5
        self.train_epochs = args.train_epochs

    def _build_model(self):
        model = BasicModel(self.args)
        return model
    
    def _release_model(self):
        del self.model
        torch.cuda.empty_cache()
        gc.collect()

    def _get_data(self, flag):
        data_set, data_loader = data_provider_trading(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = mape_loss()
        # criterion = nn.MSELoss()
        loss_map = {
            'MSE': nn.MSELoss, 
            'SpearmanRank': SpearmanRankCorrelationLoss, 
            'RankNet': RankNetLoss, 
            'ListNet': ListNetLoss, 
            'PairwiseRanking': PairwiseRankingLoss, 
            'ContrastiveRanking': ContrastiveRankingLoss,
            'PearsonCorrelation': PearsonCorrelationLoss, 
            'DistanceCorrelation': DistanceCorrelationLoss, 
            'ConcordanceCorrelation': ConcordanceCorrelationCoefficientLoss, 
            'CompositeLoss': CompositeLoss,
            'NeuralNDCG': NeuralNDCGLoss
        }
        criterion = loss_map[self.args.loss]()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_general, 
                    close_mean, close_stdev, org_close_start, org_close_last,
                    fluctuation_ratio, stock_id) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_general = batch_general.float().to(self.device)
                # decoder input
                dec_inp = None
                
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_general)
                # testing
                # outputs = outputs * fluctuation_ratio.float().to(self.device)
                batch_y = batch_y.to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_epochs = self.get_epoch()
        self._release_model()
        self.model = self._build_model().to(self.device)
        self.refit(setting, train_epochs)
        pass

    def get_epoch(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        train_epochs = self.args.train_epochs
        for epoch in range(self.args.train_epochs):
            
            train_epochs = epoch + 1
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_static, 
                    close_mean, close_stdev, org_close_start, org_close_last, 
                    fluctuation_ratio, stock_id) in enumerate(train_loader):               
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_static = batch_static.float().to(self.device)

                dec_inp = None

                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_static)
                batch_y = batch_y.to(self.device)
                
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            early_stopping(vali_loss, self.model, None)
            if early_stopping.early_stop:
                print("Early stopping")
                train_epochs = epoch - self.args.patience + 1
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        print(f"train_epochs = {train_epochs}")
        return train_epochs
    
    def refit(self, setting, train_epochs):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        train_vali_data = ConcatDataset([train_data, vali_data])
        train_vali_loader = DataLoader(
            train_vali_data,
            batch_size=self.args.batch_size,
            shuffle=(not self.args.batch_per_day),
            num_workers=self.args.num_workers,
            drop_last=True
        )
        breakpoint()
        
        path = os.path.join(self.args.checkpoints, setting)
        if os.path.exists(path):
            shutil.rmtree(path)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(train_epochs):
            self.train_epochs = epoch
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_static, 
                    close_mean, close_stdev, org_close_start, org_close_last, 
                    fluctuation_ratio, stock_id) in enumerate(train_vali_loader):               
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_static = batch_static.float().to(self.device)

                dec_inp = None

                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_static)
                batch_y = batch_y.to(self.device)
                
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_vali_loss = self.vali(train_vali_data, train_vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Valid Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_vali_loss, test_loss))
            
            if self.args.wandb_project_name != "":
                info = {
                    "Train Vali Loss": train_vali_loss,
                    "Test Loss": test_loss,
                }
                print(f"wandb logging {info}")
                wandb.log(info, step = epoch+1)

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            
        torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')

        return self.model
    
    
    def test(self, setting, test=0):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        train_vali_data = ConcatDataset([train_data, vali_data])
        train_vali_loader = DataLoader(
            train_vali_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False
        )

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        
        folder_path = './results/' + setting
        
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        self.test_pct_change(folder_path, train_vali_loader, 'train_vali')
        self.test_pct_change(folder_path, test_loader, 'test')
            
    def test_pct_change(self, folder_path, data_loader, flag):

        folder_path = os.path.join(folder_path, flag)
        os.makedirs(folder_path)
        
        pct_preds = []
        pct_trues = []
        
        csv_list = [['stock_id', 'date', 'pred_pct', 'true_pct', 'flunctuation_ratio']]
        

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_static, 
                    close_mean, close_stdev, org_close_start, org_close_last, 
                    fluctuation_ratio, stock_id) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_static = batch_static.float().to(self.device)

                # decoder input
                dec_inp = None
                
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_static)
                batch_y = batch_y.to(self.device)
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                org_close_start = org_close_start.numpy()
                org_close_last = org_close_last.numpy()
                input = batch_x.detach().cpu().numpy()
                # testing:
                fluctuation_ratio = fluctuation_ratio.numpy()

                raw_pred = outputs
                raw_true = batch_y
                
                if self.args.target_transform == 'pct_change_to_prev':
                    pct_pred = raw_pred
                    pct_true = raw_true
                elif self.args.target_transform == 'pct_change_to_first':
                    ratio_true = raw_true + 1
                    ratio_pred = raw_pred + 1
                    ratio_last = org_close_last/org_close_start
                    
                    pct_true = ratio_true/ratio_last - 1
                    pct_pred = ratio_pred/ratio_last - 1
                elif self.args.target_transform == 'individual_seq_norm':
                    close_mean = close_mean.numpy()
                    close_stdev = close_stdev.numpy()
                    
                    inverse_pred = raw_pred * close_stdev + close_mean
                    inverse_true = raw_true * close_stdev + close_mean
                    
                    last_close = input[:, -1, 0] * close_stdev + close_mean
                    
                    pct_pred = inverse_pred / last_close - 1
                    pct_true = inverse_true / last_close - 1
                    
                else:
                    raise KeyError(f"target_transform: {self.args.target_transform} does not exist")

                stock_id = stock_id.detach().cpu().numpy().astype(int).tolist()
                batch_x_mark = batch_x_mark.detach().cpu().numpy().astype(int)
                last_date_mark = batch_x_mark[:, -1, :].tolist()
                last_date = [f"{year}-{month}-{day}" for year, month, day in last_date_mark]
                
                result  = [[sid, date, pred, true, fr] for sid, date, pred, true, fr in zip(stock_id, last_date, pct_pred, pct_true, fluctuation_ratio)]
                csv_list += result 
                
                pct_preds.append(pct_pred)
                pct_trues.append(pct_true)
                    
        pct_preds = np.concatenate(pct_preds, axis=0)
        pct_trues = np.concatenate(pct_trues, axis=0)
        
        print(f'{flag} shape:', pct_preds.shape, pct_trues.shape)
        
        dtw = -999

        mae, mse, rmse, mape, mspe = metric(pct_preds, pct_trues)
        raw_correlation = self.plot_correlation(pct_preds, pct_trues, folder_path, 'pct_scatter_plot')
        
        mean_truth = np.mean(pct_trues)
        
        stdev_truth = np.std(pct_trues)
        mad_truth = np.mean(np.abs(pct_trues - mean_truth))  # Mean Absolute Deviation
        
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        print(f'correlations: pct: {raw_correlation}\n')
        os.makedirs('result_log', exist_ok=True)
        f = open("result_log/result_regression.txt", 'a')
        f.write(str(vars(self.args)) + "  \n")
        f.write(folder_path + '\n')
        f.write('mse:{}, mae:{}, dtw:{}\n'.format(mse, mae, dtw))
        f.write(f'correlations: pct: {raw_correlation}\n')
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + '/metrics.npy', np.array([mae, mse, rmse, mad_truth, stdev_truth, mae/mad_truth, mae/stdev_truth]))
        np.save(folder_path + '/pred.npy', pct_preds)
        np.save(folder_path + '/true.npy', pct_trues)
        with open(f"{folder_path}/whole_output.csv", mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_list)
    
    def plot_correlation(self, preds, trues, folder_path, plotname='scatter_plot'):
        import matplotlib.pyplot as plt

        # Plotting the scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(preds, trues, color='blue', label=plotname)

        # Adding labels and title
        plt.xlabel('Predictions')
        plt.ylabel('True Values')
        plt.title('Scatter Plot of Predictions vs True Values')

        # Display the correlation on the plot
        correlation = np.corrcoef(preds, trues)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        # Show plot with grid
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{folder_path}/{plotname}.png')
        
        return correlation
    
    def wandb_finish(self):
        if self.args.wandb_project_name != "":
            wandb.finish()