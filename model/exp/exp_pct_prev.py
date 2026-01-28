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
    NeuralNDCGLoss,
    StockMixerLoss,
    StopProfitLoss,
    StopLossLoss
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

warnings.filterwarnings('ignore')


class Exp_Pct_Prev(Exp_Basic):
    def __init__(self, args):
        super(Exp_Pct_Prev, self).__init__(args)
        if args.wandb_project_name != "":
            wandb.init(
                project=args.wandb_project_name,
                config=vars(args),
            )
        self.best_train_loss = 10e5
        self.best_valid_loss = 10e5
        self.best_test_loss = 10e5

    def _build_model(self):
        model = BasicModel(self.args)
        return model
    
    def _release_model(self):
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        
    def _get_weights(self, data_set):
        class_counts = data_set.df_flat['y'].value_counts().sort_index()  # 確保順序是 0,1,2
        return torch.tensor([class_counts[0] / class_counts[1]], dtype=torch.float).to(self.device)

    def _get_data(self, flag):
        data_set, data_loader = data_provider_trading(self.args, flag)
        if ('Weighted' in self.args.loss) and (flag == "train_val"):
            self.weights = self._get_weights(data_set)
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
            'NeuralNDCG1': NeuralNDCGLoss,
            'StockMixerLoss': StockMixerLoss,
            'CrossEntropy': nn.CrossEntropyLoss,
            'BCEWithLogits': nn.BCEWithLogitsLoss,
            'WeightedBCEWithLogits': nn.BCEWithLogitsLoss,
            'StopProfit': StopProfitLoss,
            'StopLoss': StopLossLoss
        }
        if ('Weighted' in self.args.loss):
            criterion = loss_map[self.args.loss](pos_weight=self.weights)
        else:
            criterion = loss_map[self.args.loss]()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, dates, stock_ids, *others) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = None
                batch_y_mark = None
                batch_general = None
                # decoder input
                dec_inp = None
                
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_general)
                # testing
                # outputs = outputs * fluctuation_ratio.float().to(self.device)
                batch_y = batch_y.to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                
                if (self.args.loss == 'StockMixerLoss'):
                    last_date = [f"{year}-{month:02d}-{day:02d}" for year, month, day in zip(dates[0].tolist(), dates[1].tolist(),  dates[2].tolist())]
                    pct_y = [vali_data.to_pct(output, date, id) for(output, date, id) in zip(outputs, last_date, stock_ids)]
                    pct_true = [vali_data.to_pct(y, date, id) for(y, date, id) in zip(batch_y, last_date, stock_ids)]
                    pred = torch.stack(pct_y).cpu()
                    true = torch.stack(pct_true).cpu()
                
                if (self.args.loss == "CrossEntropy"):
                    true = true.long() 
                
                if ("Weighted" in self.args.loss):
                    c = nn.BCEWithLogitsLoss(pos_weight=self.weights.cpu())
                    loss = c(pred, true)
                else:
                    loss = criterion(pred, true)
                total_loss.append(loss)
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_epochs = 3 #self.get_epoch()
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
            for i, (batch_x, batch_y, dates, stock_ids, *others) in enumerate(train_loader):               
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = None
                batch_y_mark = None
                batch_static = None
                
                dec_inp = None

                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_static)
                batch_y = batch_y.to(self.device)
                
                if (self.args.loss == 'StockMixerLoss'):
                    last_date = [f"{year}-{month:02d}-{day:02d}" for year, month, day in zip(dates[0].tolist(), dates[1].tolist(),  dates[2].tolist())]
                    pct_y = [train_data.to_pct(output, date, id) for(output, date, id) in zip(outputs, last_date, stock_ids)]
                    pct_true = [train_data.to_pct(y, date, id) for(y, date, id) in zip(batch_y, last_date, stock_ids)]
                    outputs = torch.stack(pct_y)
                    batch_y = torch.stack(pct_true)
                if (self.args.loss == "CrossEntropy"):
                    batch_y = batch_y.long()
                    
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
        #train_data, train_loader = self._get_data(flag='train')
        #vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        train_vali_data, train_vali_loader = self._get_data(flag='train_val')
        
        print(f"train_val len = {train_vali_data.__len__()}")
        
        path = os.path.join(self.args.checkpoints, setting)
        if os.path.exists(path):
            shutil.rmtree(path)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        #train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(train_epochs):
            self.train_epochs = epoch
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, dates, stock_ids, *others) in enumerate(train_vali_loader):               
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = None
                batch_y_mark = None
                batch_static = None

                dec_inp = None

                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_static)
                batch_y = batch_y.to(self.device)
                
                if (self.args.loss == 'StockMixerLoss'):
                    last_date = [f"{year}-{month:02d}-{day:02d}" for year, month, day in zip(dates[0].tolist(), dates[1].tolist(),  dates[2].tolist())]
                    pct_y = [train_vali_data.to_pct(output, date, id) for(output, date, id) in zip(outputs, last_date, stock_ids)]
                    pct_true = [train_vali_data.to_pct(y, date, id) for(y, date, id) in zip(batch_y, last_date, stock_ids)]
                    outputs = torch.stack(pct_y)
                    batch_y = torch.stack(pct_true)
                if (self.args.loss == "CrossEntropy"):
                    batch_y = batch_y.long()
                
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    #left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, -1))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_vali_loss = self.vali(train_vali_data, train_vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Valid Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, "step", train_vali_loss, test_loss))
            
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
        #train_data, train_loader = self._get_data(flag='train')
        #vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        train_vali_data, train_vali_loader = self._get_data(flag='train_val')

        if test:
            print('loading model')
            try:
                self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))
                #path = os.path.join('model/checkpoints/', 'checkpoint.pth')
                #temp = torch.load(path, map_location=torch.device('cpu'))
                #self.model.load_state_dict(temp)
            except:
                #print((os.path.join('checkpoints/', 'checkpoint.pth')), "model not exists")
                print(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth'), "model not exists")
            
        folder_path = './results/' + setting

        os.makedirs(folder_path, exist_ok=True)
            
        self.test_pct_change(folder_path, train_vali_data, train_vali_loader, 'train_vali')
        self.test_pct_change(folder_path, test_data, test_loader, 'test')
            
    def test_pct_change(self, folder_path, dataset, data_loader, flag):

        folder_path = os.path.join(folder_path, flag)
        os.makedirs(folder_path, exist_ok=True)
        
        pct_preds = []
        pct_trues = []
        
        csv_list = [['stock_id', 'date', 'pred_pct', 'true_pct']]
        

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, dates, stock_id, *others) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = None
                batch_y_mark = None
                batch_static = None

                # decoder input
                dec_inp = None
                
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_static)
                batch_y = batch_y.to(self.device)
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                # testing:
                stock_id = np.array(stock_id).astype(str)
                last_date = [f"{year}-{month:02d}-{day:02d}" for year, month, day in zip(dates[0].tolist(), dates[1].tolist(),  dates[2].tolist())]
                
                pct_y = [dataset.to_pct(output, date, id) for(output, date, id) in zip(outputs, last_date, stock_id)]
                pct_true = [dataset.to_pct(y, date, id) for(y, date, id) in zip(batch_y, last_date, stock_id)]
                
                result  = [[sid, date, pred, true] for sid, date, pred, true in zip(stock_id, last_date, pct_y, pct_true)]
                csv_list += result 
                
                pct_preds.append(pct_y)
                pct_trues.append(pct_true)
                    
        pct_preds = np.concatenate(pct_preds, axis=0)
        pct_trues = np.concatenate(pct_trues, axis=0)

        mask = pct_trues >= -1
        pct_preds = pct_preds[mask]
        pct_trues = pct_trues[mask]
        
        print(f'{flag} shape:', pct_preds.shape, pct_trues.shape, len(csv_list))
        
        dtw = -999
        
        if (self.args.loss == "CrossEntropy"):
            y_true = pct_trues.astype(int)
            y_pred = np.argmax(pct_preds, axis=1)  
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')
            
            print('acc:{}, precision:{}, recall:{}, f1={}\n'.format(acc, precision, recall, f1))
            print(classification_report(y_true, y_pred))
            
            os.makedirs('model/result_log', exist_ok=True)
            f = open("model/result_log/result_classification.txt", 'a')
            f.write(str(vars(self.args)) + "  \n")
            f.write(folder_path + '\n')
            f.write('acc:{}, precision:{}, recall:{}, f1={}\n'.format(acc, precision, recall, f1))
            f.write('\n')
            f.write('\n')
            f.close()
            np.save(folder_path + '/metrics.npy', np.array([acc, precision, recall, f1]))
        else:
            mae, mse, rmse, mape, mspe = metric(pct_preds, pct_trues)
            raw_correlation = self.plot_correlation(pct_preds, pct_trues, folder_path, 'pct_scatter_plot')
            
            mean_truth = np.mean(pct_trues)
            
            stdev_truth = np.std(pct_trues)
            mad_truth = np.mean(np.abs(pct_trues - mean_truth))  # Mean Absolute Deviation
            
            print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
            print(f'correlations: pct: {raw_correlation}\n')
            os.makedirs('model/result_log', exist_ok=True)
            f = open("model/result_log/result_regression.txt", 'a')
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

        header = csv_list[0]
        rows = csv_list[1:]
        rows.sort(key=lambda x: (x[1], x[0]))
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