"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

    
    
    
class LabeledContrastiveLoss(nn.Module):
    def __init__(self, temp=0.5, eps=1e-3,
                 pos_in_denom=False, # True
                 log_first=True,  # False
                 a_lc=1.0, a_spread=0.0,
                 noise_rate=0.0, K=-1, lc_norm = False, old_lnew = False,
                 detect_noise=False, correct_noise=False, num_classes=10):
        super().__init__()
        self.temp = temp
        self.eps  = eps
        self.log_first = log_first
        self.a_lc=a_lc
        self.a_spread=a_spread
        self.pos_in_denom = pos_in_denom
        self.noise_rate = noise_rate
        p = 1 - noise_rate + noise_rate * (1. / num_classes)
        self.p = p
        self.num_classes = num_classes
        self.lc_norm = lc_norm
        self.K = K
        self.old_lnew = old_lnew
        self.detect_noise = detect_noise
        self.correct_noise = correct_noise

    def forward(self, x, labels):
        # x has shape batch * num views * dimension
        # labels has shape batch * num views
        labels = labels.unsqueeze(1).repeat(1, 2)
        
        b, nViews, d = x.size()
        vs = torch.split(x,1, dim=1) # images indexed by view
        ts = torch.split(labels, 1, dim=1) # labels indexed by view
        l = 0.
        pairs = nViews*(nViews-1)//2
        
        for ii in range(nViews):
            vi = vs[ii].squeeze()
            ti = ts[ii].squeeze()
            
            ti_np = np.array([int(label) for label in ti])
            for jj in range(ii):
                vj = vs[jj].squeeze()
                tj = ts[jj].squeeze()
                
                if self.log_first:
                    if self.old_lnew:
                        # old implementation of L_new
                        # don't include these in positives
                        _mask = ti.unsqueeze(0) != tj.unsqueeze(1)

                        # num[i,j] is f(xi) * f(xj) / tau, for i,j
                        if self.lc_norm:
                            num = torch.einsum('b d, c d -> b c', vi, vj).div(self.temp).div(
                                torch.norm(vi, dim=1) * torch.norm(vj, dim=1)
                            )
                        else:
                            num = torch.einsum('b d, c d -> b c', vi, vj).div(self.temp)
                            
                        # _mask_denom is True when yi != yj
                        _mask_denom = (ti.unsqueeze(0) != tj.unsqueeze(1)).float()
                        if self.noise_rate > 0.:
                            _mask_denom[ti.unsqueeze(0) == tj.unsqueeze(1)] = self.noise_rate
                            _mask_denom[torch.eye(ti.shape[0], dtype=bool)] = 0.

                        # for numerical stability, see log-sum-exp trick
                        num_max, _ = torch.max(num, dim=1, keepdim=True)
                        
                        # log_denom[i,j] is log[exp(f(xi) * f(xj) / tau) +
                        #   + sum_{j in _mask_denom[i]} exp(f(xi) * f(xj) / tau)
                        log_denom = (
                            # sum_{j in _mask_denom[i]} exp(f(xi) * f(xj) / tau)
                            (torch.exp(num - num_max) * _mask_denom).sum(-1, keepdim=True) + 
                            # exp(f(xi) * f(xj) / tau)
                            torch.exp(num - num_max)
                        ).log() + num_max

                        log_prob = num - log_denom
                        
                        if self.noise_rate > 0.:
                            _mask_mult = (ti.unsqueeze(0) == tj.unsqueeze(1)).float() * (1. - self.noise_rate)
                            _mask_mult[torch.eye(ti.shape[0], dtype=bool)] = 1.
                            log_prob = log_prob * _mask_mult

                        _mask_nans_infs = torch.isnan(log_prob) + torch.isinf(log_prob)

                        a = -log_prob.masked_fill(
                            _mask,
                            math.log(self.eps)
                        ).masked_fill(
                            _mask_nans_infs,
                            math.log(self.eps)
                        ).sum(-1).div(_mask.sum(-1))
                        l += a.mean()
                    else:
                        # new implementation of L_new/L_out
                        # don't include these in positives
                        _mask = ti.unsqueeze(0) != tj.unsqueeze(1)

                        # num[i,j] is f(xi) * f(xj) / tau, for i,j
                        if self.lc_norm:
                            num = torch.einsum('b d, c d -> b c', vi, vj).div(self.temp).div(
                                torch.norm(vi, dim=1) * torch.norm(vj, dim=1)
                            )
                        else:
                            num = torch.einsum('b d, c d -> b c', vi, vj).div(self.temp)
                        
                        if self.detect_noise:
                            # new implementation of L_new
                            _mask_same_class = ti.unsqueeze(0) == tj.unsqueeze(1)
                            _mask_same_class[torch.eye(ti.shape[0], dtype=bool)] = False
                            _mask_diff_class = ti.unsqueeze(0) != tj.unsqueeze(1)
                            
                            num_norm = torch.einsum('b d, c d -> b c', vi, vj).div(self.temp).div(
                                torch.norm(vi, dim=1) * torch.norm(vj, dim=1)
                            )
                            num_norm_numpy = num_norm.detach().cpu().numpy()

                            if (min(torch.sum(_mask_same_class, axis=0)) == 0 or
                                min(torch.sum(_mask_diff_class, axis=0)) == 0):
                                # crazy edge case where every element in the batch has the same class??
                                preds_class_correct = np.ones(ti_np.shape[0], dtype=bool)
                            else:
                                same_class_sim = np.average(num_norm_numpy, axis=0,
                                                            weights=_mask_same_class.detach().cpu().numpy())
                                diff_class_sim = np.average(num_norm_numpy, axis=0,
                                                            weights=_mask_diff_class.detach().cpu().numpy())

                                difference = same_class_sim - diff_class_sim

                                preds_class_correct = np.ones(ti_np.shape[0], dtype=bool)
                                preds_class_correct[np.argsort(difference)[:int((1-self.p) * ti.shape[0])]] = False
                                
                            if self.correct_noise:
                                class_sims = []
                                for cls in range(self.num_classes):
                                    class_sim = np.mean(
                                        num_norm_numpy, axis=1,
                                        where=np.tile((ti_np == cls), (len(ti_np), 1))
                                    )
                                    class_sims.append(class_sim)

                                class_preds = np.argmax(class_sims, axis=0)

                                # naive, correct *all* the labels
                                preds_incorrect = np.invert(preds_class_correct)
                                ti_np[preds_incorrect] = class_preds[preds_incorrect]
                                preds_class_correct = np.ones(ti_np.shape[0], dtype=bool)
                        
                        pos_ones = [] # store the first positive (augmentation of the same view)
                        neg_ones = [] # store the first negative
                        M_indices = []
                        div_factor = []
                           
                        for i, cls in enumerate(ti_np):
                            # fall back to SimCLR
                            pos_indices = torch.tensor([i]).cuda()
                            if cls != -1:
                                if not self.detect_noise:
                                    pos_indices = torch.where(ti == cls)[0]
                                elif preds_class_correct[i]:
                                    # only keep positive indices if the classes are correct
                                    pos_indices = torch.where(
                                        torch.tensor((ti_np == int(cls)) * preds_class_correct).cuda())[0]
                            
                            # fall back to SimCLR
                            neg_indices = torch.tensor([idx for idx in range(ti.shape[0]) if idx != i]).cuda()
                            if cls != -1:
                                if not self.pos_in_denom:
                                    # L_new
                                    if not self.detect_noise:
                                        neg_indices = torch.where(ti != cls)[0]
                                    elif preds_class_correct[i]:
                                        neg_indices = torch.where(
                                            torch.tensor((ti_np != int(cls)) | np.invert(preds_class_correct)).cuda())[0] 
                            
                            if self.K != -1:
                                m = self.K
                            else:
                                m = len(neg_indices)

                            pos_idx = int(torch.min(torch.where(pos_indices == i)[0]))
                            neg_pivots = torch.where(neg_indices > i)[0]
                            if len(neg_pivots) > 0:
                                neg_idx = int(torch.min(neg_pivots))
                            else:
                                neg_idx = 0

                            pos_indices = torch.roll(pos_indices, -1 * pos_idx)
                            neg_indices = torch.roll(neg_indices, -1 * neg_idx)
                            
                            if self.K == -1:
                                # set a different z_1^+ every time
                                all_indices = torch.stack([
                                    torch.cat((torch.roll(pos_indices, j)[:1], neg_indices[:m]))
                                    for j in range(len(pos_indices))
                                ])
                            else:
                                all_indices = torch.stack([
                                    torch.cat((torch.roll(pos_indices, j)[:1],
                                               torch.roll(neg_indices, j)[:m]))
                                    for j in range(len(pos_indices))
                                ])

                            # store all the positive indices
                            pos_ones.append(pos_indices)

                            # store all the negative indices that go up to m
                            neg_ones.append(neg_indices)
                            
                            M_indices.append(all_indices)
                            
                            div_factor.append(len(pos_indices))
                        
#                         import pdb; pdb.set_trace()
                        
                        # denominator for each point in the batch
                        denominator = torch.stack([
                            # reshape num with an extra dimension, then take the sum over everything
                            torch.logsumexp(num[i][M_indices[i]], 1).sum()
                            for i in range(len(ti))
                        ])

                        # numerator
                        numerator = torch.stack([
                            # sum over all the positives
                            torch.sum(-1 * num[i][pos_ones[i]])
        #                     -1 * num[i][pos_ones[i]]
                            for i in range(len(ti))
                        ])
                        
                        log_prob = numerator + denominator

                        if self.a_spread > 0.0:
                            assert(self.a_lc + self.a_spread != 0)
                            
                            numerator_spread = -1 * torch.diagonal(num, 0)
                            denominator_spread = torch.stack([
                                # reshape num with an extra dimension, then take the sum over everything
                                torch.logsumexp(num[i][pos_ones[i]], 0).sum()
                                for i in range(len(ti))
                            ])
                            log_prob_spread = numerator_spread + denominator_spread
                            
                            a = (self.a_lc * log_prob.div(torch.tensor(div_factor).cuda()) +
                                 self.a_spread * log_prob_spread) / (self.a_lc + self.a_spread)
                        else:
                            a = self.a_lc * log_prob.div(torch.tensor(div_factor).cuda())

                        l += a.mean()
                else: 
                    if self.pos_in_denom:
                        # L_in
                        # don't include these in positives
                        _mask = ti.unsqueeze(0) != tj.unsqueeze(1)
                        
                        if self.lc_norm:
                            num = torch.einsum('b d, c d -> b c', vi, vj).div(self.temp).div(
                                torch.norm(vi, dim=1) * torch.norm(vj, dim=1)
                            )
                        else:
                            num = torch.einsum('b d, c d -> b c', vi, vj).div(self.temp)
                        
                        a  = num.softmax(dim=-1)
                        # we want to sum over the rows that satisfy the mask, then log the mask.
                        a  = -a.masked_fill(_mask, self.eps).sum(-1).log()
                        l += a.mean()
                    else:
                        pass
                    
        return l/pairs
