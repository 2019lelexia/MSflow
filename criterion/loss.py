import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_FLOW = 40000
def sequence_loss(output, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(output['flow'])
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        loss_i = output['nf'][i]
        mask = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach())) & valid[:, None]
        if mask.sum() == 0:
            flow_loss += 0 * loss_i.sum()
        else:
            flow_loss += i_weight * ((mask * loss_i).sum()) / mask.sum()
        if 'sl' in output:
            flow_loss += output['sl'][i]
        if 'align' in output:
            loss_a = output['align'][i]
            if loss_a.dim() == 3:
                loss_a = loss_a.unsqueeze(1)
            if mask.sum() > 0:
                flow_loss += i_weight * ((mask * loss_a).sum()) / mask.sum()
        if 'mono' in output:
            loss_m = output['mono'][i]
            if loss_m.dim() == 3:
                loss_m = loss_m.unsqueeze(1)
            if mask.sum() > 0:
                flow_loss += i_weight * ((mask * loss_m).sum()) / mask.sum()

    return flow_loss

def sequence_loss_double_stage(output, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(output['flow'])
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        if i == 0:
            i_weight = 5
        loss_i = output['nf'][i]
        mask = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach())) & valid[:, None]
        if mask.sum() == 0:
            flow_loss += 0 * loss_i.sum()
        else:
            flow_loss += i_weight * ((mask * loss_i).sum()) / mask.sum()
        if 'mono' in output:
            loss_m = output['mono'][i]
            if loss_m.dim() == 3:
                loss_m = loss_m.unsqueeze(1)
            if mask.sum() > 0:
                flow_loss += i_weight * ((mask * loss_m).sum()) / mask.sum()
    return flow_loss

def sequence_loss_with_occl(output, flow_gt, valid, gamma=0.8, weight_occl=20, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(output['flow'])
    flow_loss = 0.0
    occl_loss = 0.0
    total_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    valid_number = (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        loss_i = output['nf'][i]
        loss_bce_i = output['bce'][i]
        mask = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach())) & valid[:, None]
        mask_bce = (~torch.isnan(loss_bce_i.detach())) & (~torch.isinf(loss_bce_i.detach())) & valid_number[:, None]
        if mask.sum() == 0:
            # flow_loss += 0 * loss_i.sum()
            flow_loss += 0.0
        else:
            flow_loss += i_weight * ((mask * loss_i).sum()) / mask.sum()
        if mask_bce.sum() == 0:
            occl_loss += 0.0
        else:
            occl_loss += i_weight * ((mask_bce * loss_bce_i).sum()) / mask_bce.sum()
    total_loss = flow_loss + weight_occl * occl_loss

    return total_loss, flow_loss, occl_loss
