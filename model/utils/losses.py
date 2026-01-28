# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb
import torch.nn.functional as F
import torch
from .utils import (
    deterministic_neural_sort,
    sinkhorn_scaling,
    stochastic_neural_sort,
    dcg,
    get_torch_device,
    sample_gumbel,
)


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, forecast: t.Tensor, target: t.Tensor, mask: t.Tensor = None) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        if mask == None:
            mask = t.ones_like(forecast)
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)
    
class PairwiseRankingLoss(nn.Module):
    def __init__(self):
        super(PairwiseRankingLoss, self).__init__()

    def forward(self, pred: t.Tensor, truth: t.Tensor) -> t.Tensor:
        """
        Compute the pairwise ranking loss across the batch.

        Parameters:
        pred (torch.Tensor): Predicted scores, shape (batch_size,)
        truth (torch.Tensor): True scores (relevance), shape (batch_size,)

        Returns:
        torch.Tensor: The pairwise ranking loss (scalar)
        """
        # Ensure inputs are of the correct shape
        assert pred.dim() == 1 and truth.dim() == 1, "Predictions and truths must be 1D tensors."

        # Compute pairwise differences
        pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)   # Shape: (batch_size, batch_size)
        truth_diff = truth.unsqueeze(0) - truth.unsqueeze(1) # Shape: (batch_size, batch_size)

        # Create a mask where the true difference is positive (i.e., where sample i should be ranked higher than sample j)
        positive_truth_mask = (truth_diff > 0).float()      # Shape: (batch_size, batch_size)

        # Apply hinge loss to the predicted differences where the true difference is positive
        # The loss encourages pred_diff > 0 where truth_diff > 0
        losses = F.relu(1 - pred_diff) * positive_truth_mask  # Shape: (batch_size, batch_size)

        # Compute the average loss
        num_positive_pairs = positive_truth_mask.sum()  # Scalar
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-8
        loss = losses.sum() / (num_positive_pairs + epsilon)  # Scalar

        return loss
    

class SpearmanRankCorrelationLoss(nn.Module):
    def __init__(self, regularization_strength=1e-3):
        super(SpearmanRankCorrelationLoss, self).__init__()
        self.regularization_strength = regularization_strength

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Spearman rank correlation coefficient loss.

        Parameters:
        pred (torch.Tensor): Predicted values, shape (batch_size,)
        target (torch.Tensor): True values, shape (batch_size,)

        Returns:
        torch.Tensor: Spearman rank correlation loss (scalar)
        """
        pred = pred.view(-1)
        target = target.view(-1)
        batch_size = pred.size(0)

        # Compute soft ranks for predictions
        pred_soft_rank = self.soft_rank(pred)

        # Compute ranks for targets (since targets are fixed, we can use hard ranks)
        target_rank = torch.argsort(torch.argsort(target))

        # Compute covariance between ranks
        pred_rank_mean = pred_soft_rank.mean()
        target_rank_mean = target_rank.float().mean()

        cov = ((pred_soft_rank - pred_rank_mean) * (target_rank.float() - target_rank_mean)).sum()
        pred_var = ((pred_soft_rank - pred_rank_mean) ** 2).sum()
        target_var = ((target_rank.float() - target_rank_mean) ** 2).sum()

        spearman_corr = cov / (torch.sqrt(pred_var * target_var) + 1e-8)
        loss = 1 - spearman_corr  # We want to maximize the correlation, so minimize 1 - correlation

        return loss

    def soft_rank(self, x: torch.Tensor, regularization_strength=None) -> torch.Tensor:
        """
        Compute a differentiable approximation to the rank of elements in x.

        Parameters:
        x (torch.Tensor): Input tensor, shape (batch_size,)
        regularization_strength (float): Regularization parameter for smoothing.

        Returns:
        torch.Tensor: Soft ranks, shape (batch_size,)
        """
        if regularization_strength is None:
            regularization_strength = self.regularization_strength

        x_expand = x.unsqueeze(0)  # Shape: (1, batch_size)
        x_expand2 = x.unsqueeze(1)  # Shape: (batch_size, 1)
        diff = x_expand - x_expand2  # Shape: (batch_size, batch_size)
        soft_rank = torch.sigmoid(-diff / regularization_strength).sum(dim=1)

        # Normalize the ranks
        soft_rank = soft_rank + 0.5  # Add 0.5 to account for self-comparison
        return soft_rank
    
class RankNetLoss(nn.Module):
    def __init__(self):
        super(RankNetLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the RankNet loss.

        Parameters:
        pred (torch.Tensor): Predicted scores, shape (batch_size,)
        target (torch.Tensor): True scores (relevance), shape (batch_size,)

        Returns:
        torch.Tensor: RankNet loss (scalar)
        """
        pred = pred.view(-1)
        target = target.view(-1)
        batch_size = pred.size(0)

        # Compute pairwise differences
        pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)   # Shape: (batch_size, batch_size)
        target_diff = target.unsqueeze(0) - target.unsqueeze(1) # Shape: (batch_size, batch_size)

        # Create target probabilities
        P_ij = (target_diff > 0).float()
        # Exclude pairs with zero difference in target
        mask = (target_diff != 0).float()

        # Apply sigmoid to predictions
        S_ij = torch.sigmoid(pred_diff)

        # Compute the loss
        loss = - (P_ij * torch.log(S_ij + 1e-8) + (1 - P_ij) * torch.log(1 - S_ij + 1e-8))
        loss = loss * mask  # Apply mask to ignore pairs with no difference

        # Average over the number of valid pairs
        num_pairs = mask.sum()
        loss = loss.sum() / (num_pairs + 1e-8)

        return loss
    

class ListNetLoss(nn.Module):
    def __init__(self):
        super(ListNetLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the ListNet loss.

        Parameters:
        pred (torch.Tensor): Predicted scores, shape (batch_size,)
        target (torch.Tensor): True scores (relevance), shape (batch_size,)

        Returns:
        torch.Tensor: ListNet loss (scalar)
        """
        pred = pred.view(-1)
        target = target.view(-1)

        # Apply softmax to get probabilities
        pred_prob = F.softmax(pred, dim=0)
        target_prob = F.softmax(target, dim=0)

        # Compute cross-entropy loss
        loss = - (target_prob * torch.log(pred_prob + 1e-8)).sum()

        return loss
    
class ContrastiveRankingLoss(nn.Module):
    def __init__(self, margin=1.0, threshold=0.5):
        super(ContrastiveRankingLoss, self).__init__()
        self.margin = margin
        self.threshold = threshold  # Threshold to determine if a pair is similar or dissimilar

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the contrastive loss for ranking regression values.

        Parameters:
        pred (torch.Tensor): Predicted values, shape (batch_size,)
        target (torch.Tensor): True values, shape (batch_size,)

        Returns:
        torch.Tensor: Contrastive loss (scalar)
        """
        pred = pred.view(-1)
        target = target.view(-1)
        batch_size = pred.size(0)

        # Compute pairwise differences and distances
        pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)     # Shape: (batch_size, batch_size)
        target_diff = target.unsqueeze(0) - target.unsqueeze(1) # Shape: (batch_size, batch_size)

        # Compute pairwise distances (Euclidean distance)
        pred_distance = torch.abs(pred_diff)  # Alternatively, use squared difference
        target_distance = torch.abs(target_diff)

        # Determine if pairs are similar or dissimilar based on the threshold
        similar_pairs = (target_distance <= self.threshold).float()
        dissimilar_pairs = (target_distance > self.threshold).float()

        # Compute loss for similar pairs
        loss_similar = 0.5 * similar_pairs * (pred_distance ** 2)

        # Compute loss for dissimilar pairs
        loss_dissimilar = 0.5 * dissimilar_pairs * F.relu(self.margin - pred_distance) ** 2

        # Combine losses
        loss = loss_similar + loss_dissimilar

        # Exclude self-comparisons (diagonal elements)
        mask = 1 - torch.eye(batch_size, device=pred.device)
        loss = loss * mask

        # Compute the average loss
        num_pairs = mask.sum()
        total_loss = loss.sum() / (num_pairs + 1e-8)

        return total_loss

class PearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super(PearsonCorrelationLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the negative Pearson correlation coefficient loss.

        Parameters:
        pred (torch.Tensor): Predicted values, shape (batch_size,)
        target (torch.Tensor): Actual target values, shape (batch_size,)

        Returns:
        torch.Tensor: Pearson correlation loss (scalar)
        """
        pred_mean = pred.mean()
        target_mean = target.mean()

        pred_diff = pred - pred_mean
        target_diff = target - target_mean

        numerator = (pred_diff * target_diff).sum()
        denominator = torch.sqrt((pred_diff ** 2).sum() * (target_diff ** 2).sum()) + 1e-8

        correlation = numerator / denominator

        # Since we want to minimize the loss, and correlation ranges from -1 to 1,
        # we can define the loss as 1 - correlation (to maximize correlation)
        loss = 1 - correlation

        return loss


class DistanceCorrelationLoss(nn.Module):
    def __init__(self):
        super(DistanceCorrelationLoss, self).__init__()

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute the distance correlation loss between X and Y.

        Parameters:
        X (torch.Tensor): Predicted values, shape (batch_size,)
        Y (torch.Tensor): Actual target values, shape (batch_size,)

        Returns:
        torch.Tensor: Distance correlation loss (scalar)
        """
        X = X.view(-1)
        Y = Y.view(-1)
        n = X.size(0)

        # Compute distance matrices
        a = torch.abs(X.unsqueeze(0) - X.unsqueeze(1))
        b = torch.abs(Y.unsqueeze(0) - Y.unsqueeze(1))

        # Double centering
        A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
        B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()

        # Compute distance covariance and variances
        dcov = (A * B).sum() / (n * n)
        dvar_X = (A * A).sum() / (n * n)
        dvar_Y = (B * B).sum() / (n * n)

        # Compute distance correlation
        dcor = dcov / (torch.sqrt(dvar_X * dvar_Y) + 1e-8)

        # Loss is 1 - distance correlation
        loss = 1 - dcor

        return loss


class ConcordanceCorrelationCoefficientLoss(nn.Module):
    def __init__(self):
        super(ConcordanceCorrelationCoefficientLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = pred.mean()
        target_mean = target.mean()

        pred_var = pred.var(unbiased=False)
        target_var = target.var(unbiased=False)

        pred_diff = pred - pred_mean
        target_diff = target - target_mean

        covariance = (pred_diff * target_diff).mean()

        numerator = 2 * covariance
        denominator = pred_var + target_var + (pred_mean - target_mean) ** 2

        ccc = numerator / (denominator + 1e-8)

        # Loss is 1 - concordance correlation coefficient
        loss = 1 - ccc

        return loss

class CompositeLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CompositeLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.pearson_loss = PearsonCorrelationLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = self.mse_loss(pred, target)
        corr_loss = self.pearson_loss(pred, target)
        return self.alpha * mse + (1 - self.alpha) * corr_loss

class NeuralNDCGLoss(nn.Module):
    # ref: https://github.com/allegro/allRank/blob/master/allrank/models/losses/approxNDCG.py
    def __init__(
        self,
        stochastic=False,
        powered_relevancies=True,
        k=None,
        n_samples=32,
        beta=0.1,
        temperature=1.0,
        log_scores=True,
    ):
        super(NeuralNDCGLoss, self).__init__()
        self.stochastic = stochastic
        self.powered_relevancies = powered_relevancies
        self.k = k
        self.n_samples = n_samples
        self.beta = beta
        self.temperature = temperature
        self.log_scores = log_scores

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the neural ncdg loss.

        Parameters:
        pred (torch.Tensor): Predicted values, shape (batch_size,)
        target (torch.Tensor): Actual target values, shape (batch_size,)

        Returns:
        torch.Tensor: Neural NCDG loss (scalar)
        """
        stochastic = self.stochastic
        powered_relevancies = self.powered_relevancies
        k = self.k
        n_samples = self.n_samples
        beta = self.beta
        temperature = self.temperature
        log_scores = self.log_scores

        # dev = get_torch_device()
        dev = pred.device

        pred = pred[None, :]  # too lazy to handle batch....
        target = target[None, :]
        mask = target == -1e32  # impossible(?) value


        if k is None:
            k = target.shape[-1]

        # Choose the deterministic/stochastic variant
        if stochastic:
            P_hat = stochastic_neural_sort(
                pred.unsqueeze(-1),
                n_samples=n_samples,
                tau=temperature,
                mask=mask,
                beta=beta,
                log_scores=log_scores,
            )
        else:
            P_hat = deterministic_neural_sort(
                pred.unsqueeze(-1), tau=temperature, mask=mask
            ).unsqueeze(0)
        if P_hat.isnan().any():
            print("P_hat has nan")

        # Perform sinkhorn scaling to obtain doubly stochastic permutation matrices
        P_hat = sinkhorn_scaling(
            P_hat.view(P_hat.shape[0] * P_hat.shape[1], P_hat.shape[2], P_hat.shape[3]),
            mask.repeat_interleave(P_hat.shape[0], dim=0),
            tol=1e-6,
            max_iter=50,
        )
        P_hat = P_hat.view(
            int(P_hat.shape[0] / pred.shape[0]),
            pred.shape[0],
            P_hat.shape[1],
            P_hat.shape[2],
        )

        # Mask P_hat and apply to true labels, ie approximately sort them
        P_hat = P_hat.masked_fill(mask[None, :, :, None] | mask[None, :, None, :], 0.0)
        target_masked = target.masked_fill(mask, 0.0).unsqueeze(-1).unsqueeze(0)
        if powered_relevancies:
            target_masked = torch.pow(2.0, target_masked) - 1.0

        ground_truth = torch.matmul(P_hat, target_masked).squeeze(-1)
        discounts = (
            torch.tensor(1.0)
            / torch.log2(torch.arange(target.shape[-1], dtype=torch.float) + 2.0)
        ).to(dev)
        discounted_gains = ground_truth * discounts

        if powered_relevancies:
            idcg = dcg(target, target, ats=[k]).permute(1, 0)
        else:
            idcg = dcg(target, target, ats=[k], gain_function=lambda x: x).permute(1, 0)

        discounted_gains = discounted_gains[:, :, :k]
        ndcg = discounted_gains.sum(dim=-1) / (idcg + 1e-8)
        idcg_mask = idcg == 0.0
        ndcg = ndcg.masked_fill(idcg_mask.repeat(ndcg.shape[0], 1), 0.0)

        assert (ndcg < 0.0).sum() >= 0, "every ndcg should be non-negative"
        if idcg_mask.all():
            return torch.tensor(0.0)

        mean_ndcg = ndcg.sum() / ((~idcg_mask).sum() * ndcg.shape[0])  # type: ignore

        return -1.0 * mean_ndcg  # -1 cause we want to maximize NDCG
    
class StockMixerLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(StockMixerLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = self.mse_loss(pred, target)
        
        N = target.shape[0]
        pred_diff = pred.view(N, 1) - pred.view(1, N)  # 生成 a[i] - a[j] 的 pairwise 矩陣
        target_diff = target.view(N, 1) - target.view(1, N)  # 生成 b[i] - b[j] 的 pairwise 矩陣
        loss_matrix = torch.clamp(-(pred_diff * target_diff), min=0)  # max(0, -(...))
        loss = loss_matrix.sum()  # sum over all pairs
        return mse + self.alpha * loss

class StopProfitLoss(nn.Module):
    def __init__(self, temperature=1000.0):
        super().__init__()
        # temperature (k) 控制 Sigmoid 的陡峭程度
        # 越大越接近真實的 if/else 邏輯，但也可能導致梯度消失
        self.k = temperature

    def forward(self, pred_tp: torch.Tensor, period_high: torch.Tensor, period_close: torch.Tensor):
        """
        pred_tp: 模型預測的停利價格 (Batch, 1)
        period_high: 設定天數內的最高價 (Batch, 1)
        period_close: 設定天數結束時的收盤價 (Batch, 1)
        """
        
        # 1. 計算 "成交機率" (Soft execution probability)
        # 如果 High > Pred，difference > 0，sigmoid -> 1 (成交)
        # 如果 High < Pred，difference < 0，sigmoid -> 0 (失敗，只能拿到 Close)
        diff = period_high - pred_tp
        prob_hit = torch.sigmoid(self.k * diff)
        
        # 2. 計算期望獲利 (Expected Return)
        # 這是整個 Loss 的精隨：它會自動權衡 "設高一點賺更多" vs "設太高會掉回 Close"
        expected_return = (prob_hit * pred_tp) + ((1 - prob_hit) * period_close)
        
        # 3. 因為我們要最大化獲利，所以 Loss 取負號
        loss = -torch.mean(expected_return)
        
        return loss
    
class StopLossLoss(nn.Module):
    def __init__(self, temperature=1000.0):
        super().__init__()
        # temperature (k) 控制 Sigmoid 的陡峭程度
        # 越大越接近真實的 if/else 邏輯，但也可能導致梯度消失
        self.k = temperature

    def forward(self, pred_tp: torch.Tensor, period_low: torch.Tensor, period_close: torch.Tensor):
        """
        pred_tp: 模型預測的停利價格 (Batch, 1)
        period_high: 設定天數內的最高價 (Batch, 1)
        period_close: 設定天數結束時的收盤價 (Batch, 1)
        """

        # 1. 計算 "成交機率" (Soft execution probability)
        # 如果 High > Pred，difference > 0，sigmoid -> 1 (成交)
        # 如果 High < Pred，difference < 0，sigmoid -> 0 (失敗，只能拿到 Close)
        diff = pred_tp - period_low
        prob_hit = torch.sigmoid(self.k * diff)
        
        # 2. 計算期望獲利 (Expected Return)
        # 這是整個 Loss 的精隨：它會自動權衡 "設高一點賺更多" vs "設太高會掉回 Close"
        expected_return = (prob_hit * pred_tp) + ((1 - prob_hit) * period_close)
        
        # 3. 因為我們要最大化獲利，所以 Loss 取負號
        loss = -torch.mean(expected_return)
        
        return loss