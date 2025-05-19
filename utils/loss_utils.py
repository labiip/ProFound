import sys
import torch
import torch.nn
import numpy as np
from models.ProFound_utils import *

from extensions.Chamfer3D.fscore import fscore
from extensions.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from extensions.emd import emd_module as emd
chamfer_dist = chamfer_3DDist()

# CD-L1_CD-L2
def chamfer(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)

def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


def chamfer_single_side(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(d1)
    return d1

def chamfer_single_side_sqrt(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(torch.sqrt(d1))
    return d1


# Inpainting_loss
def inpainting_loss(out, gt, redius=0.4):
    gt = fps_subsample(gt)
    sqrdists = square_distance(gt, out)  # B, S, N
    idx = torch.argsort(sqrdists, dim=-1, descending=False)[:, :, 0:1]
    sqrdists = torch.gather(sqrdists, 2, idx)
    a = torch.zeros_like(sqrdists)
    sqrdists = torch.where(sqrdists > redius, sqrdists, a)

    return 1.1 * torch.mean(sqrdists)
    

# DCD
def calc_dcd(x, gt, alpha=10, n_lambda=1, return_raw=False, non_reg=False, traing=False):

    if traing:
        x = x.float()
        gt = gt.float()
    else:
        x = x.astype(np.float32)  
        x = torch.from_numpy(x.reshape(1, -1, 3))
        x = x.cuda()
        gt = gt.astype(np.float32)
        gt = torch.from_numpy(gt.reshape(1, -1, 3))
        gt = gt.cuda()

    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    dist1, dist2, idx1, idx2 = chamfer_dist(x, gt)

    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    count1 = torch.zeros_like(idx2)
    count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
    weight1 = count1.gather(1, idx1.long()).float().detach() ** n_lambda
    weight1 = (weight1 + 1e-6) ** (-1) * frac_21
    loss1 = (1 - exp_dist1 * weight1).mean(dim=1)

    count2 = torch.zeros_like(idx1)
    count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
    weight2 = count2.gather(1, idx2.long()).float().detach() ** n_lambda
    weight2 = (weight2 + 1e-6) ** (-1) * frac_12
    loss2 = (1 - exp_dist2 * weight2).mean(dim=1)

    loss = (loss1 + loss2) / 2

    res = loss.mean()
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])

    return res

def calc_dcd_1(x, gt, traing=False):

    if traing:
        x = x.float()
        gt = gt.float()
    else:
        x = x.astype(np.float32)  
        x = torch.from_numpy(x.reshape(1, -1, 3))
        x = x.cuda()
        gt = gt.astype(np.float32)
        gt = torch.from_numpy(gt.reshape(1, -1, 3))
        gt = gt.cuda()

    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    dist1, dist2, idx1, idx2 = chamfer_dist(x, gt)
    dist1 = torch.clamp(dist1*1e-3, min=1e-9)
    dist2 = torch.clamp(dist2*1e-3, min=1e-9)
    d1 = torch.tanh(dist1)
    d2 = torch.tanh(dist2)

    count1 = torch.zeros_like(idx2)
    count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
    m1 = torch.sum(count1 == 0, dim=1)/count1.size(1)
    count2 = torch.zeros_like(idx1)
    count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
    m2 = torch.sum(count2 == 0, dim=1)/count2.size(1)

    loss_d = ((d1 + d2) / 2).mean()
    loss_m = ((m1 + m2) / 2).mean()
    
    return loss_d + loss_m


# Hyper_CD
def calc_cd_like_hyperV2(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    # cham_loss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
    # dist1, dist2, idx1, idx2 = cham_loss(array1, array2)
    # dist1 = torch.clamp(dist1, min=1e-9)
    # dist2 = torch.clamp(dist2, min=1e-9)
    # d1 = torch.sqrt(dist1)
    # d2 = torch.sqrt(dist2)
    d1 = arcosh(1+ 1 * d1)
    d2 = arcosh(1+ 1 * d2)
    # print(d1.shape)
    # print(d2.shape)

    return torch.mean(d1) + torch.mean(d2)

def calc_cd_one_side_like_hyperV2(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    # cham_loss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
    # dist1, dist2, idx1, idx2 = cham_loss(array1, array2)
    # dist1 = torch.clamp(dist1, min=1e-9)
    # dist2 = torch.clamp(dist2, min=1e-9)
    # d1 = torch.sqrt(dist1)
    # d2 = torch.sqrt(dist2)
    d1 = arcosh(1+ 1 * d1)
    # d2 = arcosh(1+d2)
    # print(d1.shape)
    # print(d2.shape)

    return torch.mean(d1)

def arcosh(x, eps=1e-5):  # pragma: no cover
    # x = x.clamp(-1 + eps, 1 - eps)
    # x = x.clamp(1,)
    x = torch.clamp(x, min=1 + eps)
    return torch.log(x + torch.sqrt(1 + x) * torch.sqrt(x - 1))


# InfoCD
def calc_cd_like_InfoV2(p1, p2):

    dist1, dist2, idx1, idx2 = chamfer_dist(p1, p2)
    dist1 = torch.clamp(dist1, min=1e-9)
    dist2 = torch.clamp(dist2, min=1e-9)
    d1 = torch.sqrt(dist1)
    d2 = torch.sqrt(dist2)

    distances1 = - torch.log(torch.exp(-0.5 * d1)/(torch.sum(torch.exp(-0.5 * d1) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)
    distances2 = - torch.log(torch.exp(-0.5 * d2)/(torch.sum(torch.exp(-0.5 * d2) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)

    # cd_p = (torch.sqrt(dist1).mean() + torch.sqrt(dist2).mean()) / 2 # CD-L2
    
    return ((torch.sum(distances1) + torch.sum(distances2)) / 2)*1e-4

def calc_cd_one_side_like_InfoV2(p1, p2):

    dist1, dist2, idx1, idx2 = chamfer_dist(p1, p2)
    dist1 = torch.clamp(dist1, min=1e-9)
    d1 = torch.sqrt(dist1)
    distances1 = - torch.log(torch.exp(-0.5 * d1)/(torch.sum(torch.exp(-0.5 * d1) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)

    return (torch.sum(distances1))*1e-4


# Loss function
def get_loss(pcds_pred, complete_gt, epoch=1, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt: # CD-L2
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:    # CD-L1
        CD = chamfer
        PM = chamfer_single_side
    EMD = emd.emdModule()  
    DCD = calc_dcd_1
    Hyper_CD = calc_cd_like_hyperV2
    Hyper_PM = calc_cd_one_side_like_hyperV2
    InfoCD = calc_cd_like_InfoV2
    InfoPM = calc_cd_one_side_like_InfoV2

    Pc, P0, P1, P2, P3, partial_input = pcds_pred

    gt_2 = fps_subsample(complete_gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_0 = fps_subsample(gt_1, P0.shape[1])
    gt_c = fps_subsample(gt_0, Pc.shape[1])

    cdc = CD(Pc, gt_c)*DCD(Pc, gt_c,traing=True)
    cd0 = CD(P0, gt_0)*DCD(P0, gt_0,traing=True)
    emd1, _ = EMD(P1, gt_1, 0.005, 100)
    cd1 = torch.mean(torch.sqrt(emd1))
    Inpainting_loss = inpainting_loss(P1, complete_gt)
    cd2 = InfoCD(P2, gt_2)
    cd3 = InfoCD(P3, complete_gt)

    partial_matching = InfoPM(partial_input, P3)

    loss_coarse = (cdc + cd0 + cd1 + cd2 + partial_matching + Inpainting_loss)
    loss_fine = cd3 
    
    loss_coarse = (cdc + cd0 + cd1 + partial_matching + Inpainting_loss)  
    loss_fine = cd2 + cd3
    
    return loss_coarse, loss_fine

# Loss function
def get_loss_CD(pcds_pred, complete_gt, epoch=1, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt: # CD-L2
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:    # CD-L1
        CD = chamfer
        PM = chamfer_single_side
    EMD = emd.emdModule()  
    DCD = calc_dcd_1
    Hyper_CD = calc_cd_like_hyperV2
    Hyper_PM = calc_cd_one_side_like_hyperV2
    InfoCD = calc_cd_like_InfoV2
    InfoPM = calc_cd_one_side_like_InfoV2

    Pc, P0, P1, P2, P3, partial_input = pcds_pred

    gt_2 = fps_subsample(complete_gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_0 = fps_subsample(gt_1, P0.shape[1])
    gt_c = fps_subsample(gt_0, Pc.shape[1])

    cdc = CD(Pc, gt_c)
    cd0 = CD(P0, gt_0)
    # emd1, _ = EMD(P1, gt_1, 0.005, 100)
    # cd1 = torch.mean(torch.sqrt(emd1))
    # Inpainting_loss = inpainting_loss(P1, complete_gt)
    cd1 = CD(P1, gt_1)
    cd2 = CD(P2, gt_2)
    cd3 = CD(P3, complete_gt)

    partial_matching = PM(partial_input, P3)

    loss_coarse = (cdc + cd0 + cd1 + cd2 + partial_matching)# + Inpainting_loss
    loss_fine = cd3 

    # losses = [cdc, cd0, cd1, cd2, cd3, partial_matching, Inpainting_loss]
    # print(losses)
    # sys.exit()
    return loss_coarse, loss_fine

def get_loss_InfoCD(pcds_pred, complete_gt, epoch=1, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt: # CD-L2
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:    # CD-L1
        CD = chamfer
        PM = chamfer_single_side
    EMD = emd.emdModule()  
    DCD = calc_dcd_1
    Hyper_CD = calc_cd_like_hyperV2
    Hyper_PM = calc_cd_one_side_like_hyperV2
    InfoCD = calc_cd_like_InfoV2
    InfoPM = calc_cd_one_side_like_InfoV2

    Pc, P0, P1, P2, P3, partial_input = pcds_pred

    gt_2 = fps_subsample(complete_gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_0 = fps_subsample(gt_1, P0.shape[1])
    gt_c = fps_subsample(gt_0, Pc.shape[1])

    cdc = InfoCD(Pc, gt_c)
    cd0 = InfoCD(P0, gt_0)
    # emd1, _ = EMD(P1, gt_1, 0.005, 100)
    # cd1 = torch.mean(torch.sqrt(emd1))
    # Inpainting_loss = inpainting_loss(P1, complete_gt)
    cd1 = InfoCD(P1, gt_1)
    cd2 = InfoCD(P2, gt_2)
    cd3 = InfoCD(P3, complete_gt)

    partial_matching = InfoPM(partial_input, P3)

    loss_coarse = (cdc + cd0 + cd1 + cd2 + partial_matching)# + Inpainting_loss
    loss_fine = cd3 

    # losses = [cdc, cd0, cd1, cd2, cd3, partial_matching, Inpainting_loss]
    # print(losses)
    # sys.exit()
    return loss_coarse, loss_fine
