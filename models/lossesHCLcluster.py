import torch
import numpy as np
import torch.nn.functional as F
import math
import dtaidistance
from dtaidistance import dtw_ndim
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.spatial import distance
def EuclideanDistance(x, y):
    (rowx, colx) = x.shape
    (rowy, coly) = y.shape
    if colx != coly:
        raise RuntimeError('colx must be equal with coly')
    xy = np.dot(x, y.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowy, axis=1)
    y2 = np.repeat(np.reshape(np.sum(np.multiply(y, y), axis=1), (rowy, 1)), repeats=rowx, axis=1).T
    dis =np.sqrt(x2 + y2 - 2 * xy)
    return dis
def HCL_loss(z, sublabel, aug_num=3, N_Local=20, alpha=0.33, beta=0.33):
    loss = torch.tensor(0., device=z.device)
    batch_p = int(z.size(0) / aug_num)
    z_pooling = z.to(device=z.device)
    z_pooling = F.max_pool1d(z_pooling, kernel_size=z_pooling.size(2))
    z_cash = z.transpose(0,1).to(device=z.device)
    sim_cash = torch.matmul(z_cash, z_cash.transpose(1, 2))  # T x 3B x 3B
    logits_A = torch.tril(sim_cash, diagonal=-1)[:, :, :-1]  # T x 3B x (3B-1)
    logits_A += torch.triu(sim_cash, diagonal=1)[:, :, 1:]
    similar_mask = torch.zeros(z.size(0), z.size(0))
    similar_avg = torch.zeros(z.size(0), z.size(0))
    z_pooling = z_pooling.cpu().detach().numpy()

    #DTW
    z_pooling = z_pooling.astype(np.double)
    for i in range(z.size(0)):
        print(i)
        for j in range(z.size(0)):
            if i <= j:
                if i==j:
                    similar_mask[i][j] = -2.0
                elif (abs(j - i) % batch_p) == 0:
                    similar_mask[i][j] = -2.0
                    # similar_avg[i][j] = distance.euclidean(z_pooling[i], z_pooling[j])
                    similar_avg[i][j] = dtaidistance.dtw_ndim.distance_fast(z_pooling[i], z_pooling[j], use_pruning=True)
                    similar_mask[j][i] = -2.0
                    similar_avg[j][i] = similar_avg[i][j]
                else:
                    # similar_avg[i][j] = distance.euclidean(z_pooling[i], z_pooling[j])
                    similar_mask[i][j] = dtaidistance.dtw_ndim.distance_fast(z_pooling[i], z_pooling[j], use_pruning=True)
                    similar_mask[j][i] = similar_mask[i][j]
    distribu, distribu_INDEX = torch.max(similar_avg, 1)
    for i in range(z.size(0)):
        for j in range(z.size(0)):
             if similar_mask[i][j]<=distribu[i]:
                similar_mask[i][j] = 2
             else:
                similar_mask[i][j] = 1
    logits_similar_mask = torch.tril(similar_mask, diagonal=-1)[:, :-1]
    logits_similar_mask += torch.triu(similar_mask, diagonal=1)[:, 1:]
    logits_similar_mask = torch.tensor(logits_similar_mask).to(device=z.device)
    logits_similar_mask[logits_similar_mask == 1] = 0
    logits_similar_mask[logits_similar_mask == 2] = 1
    loss_A = self_supervised_contrastive_loss_timestamp(z, aug_num, logits_A, logits_similar_mask)
    loss_B = self_supervised_contrastive_loss_local(z, N_Local, aug_num, logits_A, logits_similar_mask)
    loss_C = self_supervised_contrastive_loss_instance(z, aug_num, logits_A, logits_similar_mask)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(loss_A)
    print(loss_B)
    print(loss_C)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    loss = alpha*loss_A + beta*loss_B + (1-alpha-beta)*loss_C
    return loss
def self_supervised_contrastive_loss_timestamp(z, aug_num, logit_cash, mask_cash):
    B, T = int(z.size(0)/aug_num), z.size(1)
    if B == 1:
        return z.new_tensor(0.)

    logit_cash_timestamp = logit_cash.to(device=z.device)#T*3B*(3B-1)
    logit_cash_timestamp = -F.log_softmax(logit_cash_timestamp, dim=-1)
    logit_cash_timestamp = logit_cash_timestamp * mask_cash#mask_cash:3B*(3B-1)
    logits_ave = torch.sum(mask_cash, dim=1)
    loss = torch.div(torch.sum(logit_cash_timestamp, dim=-1), logits_ave).mean()
    return loss
def self_supervised_contrastive_loss_instance(z, aug_num, logit_cash, mask_cash):
    B, T = int(z.size(0)/aug_num), z.size(1)
    if B == 1:
        return z.new_tensor(0.)
    logit_cash_instance = logit_cash.to(device=z.device)
    logit_cash_instance = torch.sum(logit_cash_instance,0)#3B*(3B-1)
    logit_cash_instance = -F.log_softmax(logit_cash_instance, dim=-1)
    logit_cash_instance = logit_cash_instance * mask_cash
    logits_ave = torch.sum(mask_cash, dim=1)
    loss = torch.div(torch.sum(logit_cash_instance, dim=-1), logits_ave).mean()
    return loss
def self_supervised_contrastive_loss_local(z, N_Local, aug_num, logit_cash, mask_cash):
    #z1,z2:B*T*C
    B, T = int(z.size(0)/aug_num), z.size(1)
    num_local = N_Local
    if B == 1:
        return z.new_tensor(0.)
    logit_cash_local = logit_cash.to(device=z.device)
    loss_f = 0
    for j in range(1, num_local+1):
        local1 = np.random.randint(1, high=T)
        local2 = np.random.randint(1, high=T)
        if local1==local2:
            local1 = np.random.randint(int(T * 0.1), high=int(T * 0.5))
            local2 = np.random.randint(int(T * 0.5), high=T)
        elif local1>local2:
            local12 = local1
            local1 = local2
            local2 = local12
        #logit_cash_local = logit_cash.to(device=z.device)
        logit_cash_local_for = logit_cash_local[local1:local2,:,:].to(device=z.device)
        T_local =logit_cash_local_for.size(0)     #
        # softmax
        logit_cash_local_for = torch.sum(logit_cash_local_for, 0)#3B*(3B-1)
        logit_cash_local_for = -F.log_softmax(logit_cash_local_for, dim=-1)
        logit_cash_local_for = logit_cash_local_for * mask_cash
        logits_ave = torch.sum(mask_cash, dim=1)
        loss = torch.div(torch.sum(logit_cash_local_for, dim=-1), logits_ave).mean()
        loss_f = loss_f + loss
    return torch.div(loss_f, num_local)