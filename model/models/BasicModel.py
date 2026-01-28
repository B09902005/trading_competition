import torch
import torch.nn as nn
from models import iTransformer, DLinear, LSTM, CATS, TimesNet


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class StaticModel(nn.Module):
    def __init__(self, configs):
        super(StaticModel, self).__init__()
        self.configs = configs
        
        self.linear = nn.Linear(configs.static_dim, configs.pred_len)
        
    def forward(self, batch_static):
        output = self.linear(batch_static)
        return output
        
class BasicModel(nn.Module):
    def __init__(self, configs):
        super(BasicModel, self).__init__()
        
    
        self.configs = configs
        self.task_name = configs.task_name
        model_dict = {
            'iTransformer': iTransformer.Model,
            'DLinear': DLinear.Model,
            'LSTM': LSTM.Model,
            'TimesNet': TimesNet.Model,
            'CATS': CATS.CATS
        }
        self.model_class = model_dict[configs.model]
        
        self.use_norm = configs.use_norm
        self.use_log = configs.use_log
        self.decomp_method = configs.decomp_method
        self.pred_len = configs.pred_len
        
        if self.decomp_method == 'moving_avg':
            kernel_size = configs.decomp_kernel_size
            self.decomposition = series_decomp(kernel_size)
            
            self.Model_Seasonal = self._build_model()
            self.Model_Trend = self._build_model()
            
        elif self.decomp_method == "":
            self.Model = self._build_model()
            
        else:
            raise Exception(f"decomp_method: {self.decomp_method} does not exist")
            
    def _build_model(self):
        model = self.model_class(self.configs).float()
        return model
    
    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, batch_static):
        if self.use_log:
            x_min, _ = torch.min(x, dim = 1, keepdim=True)
            x = x - x_min + 1
            x = torch.log(x + 1e-5)
            
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
        
            
        if self.decomp_method == 'moving_avg':
            seasonal_init, trend_init = self.decomposition(x)
            
            seasonal_output = self.Model_Seasonal(seasonal_init, x_mark_enc, x_dec, x_mark_dec, batch_static)
            trend_output = self.Model_Trend(trend_init, x_mark_enc, x_dec, x_mark_dec, batch_static)
            
            output = seasonal_output + trend_output
        else:
            output = self.Model(x, x_mark_enc, x_dec, x_mark_dec, batch_static)
            
        if self.use_log:
            # breakpoint()
            output = torch.exp(output) - 1e-5
            output = output + x_min - 1
            
        return output