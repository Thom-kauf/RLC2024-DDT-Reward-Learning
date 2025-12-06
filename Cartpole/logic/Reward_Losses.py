import torch
import random
import torch.nn as nn
import numpy as np
import os
import matplotlib.pylab as plt
from collections import defaultdict
from torch.utils.data import TensorDataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import yaml
from Utils import EarlyStopping

__AUTHORS__ = "Zack Freeman, Thomas Kauffman"

# def RSS_or_OT_Loss(r_theta, target, RSS_factor=1, OT_factor=1):
#     r_theta, target = r_theta.to("cpu"), target.to("cpu")
#     isGood_pos = torch.sigmoid(r_theta[0, target.item()])
#     isGood_neg = torch.sigmoid(r_theta[0, target.item() - 1])
    
#     B = isGood_pos + isGood_neg - (isGood_pos * isGood_neg)
#     A = torch.min(torch.tensor([1]), isGood_pos / isGood_neg)

#     return A + B - A * B

"""
This is the One True Constraint. It enforces a disjunction between the trajectory's rewards and is
implemented with the product t-norm disjunction
"""
def One_True_Loss(isGood_pos, isGood_neg):

    # print(f"is good pos is {isGood_pos} and is good neg is {isGood_neg}")
    return -1 * torch.log(isGood_pos + isGood_neg - (isGood_pos * isGood_neg))

def Reward_Penalty_Loss(isGood_pos, isGood_neg):
    # logic: isGood(positive) and not isBad(negative)
    # NLL  : -log( isGood_pos * isBad_neg ) = -log(isGood_pos * (1 - isGood_neg))
    isBad_neg = 1 - isGood_neg

    return -1 * torch.log(isGood_pos * isBad_neg)

def Richardson_Srikumar_Sabhahwal_Loss(isGood_pos, isGood_neg):

    return torch.max(torch.tensor([0]), torch.log(isGood_neg) - torch.log(isGood_pos))

def sigmoid_shift(x, shift_param = 0, temperature = 1):
    exp = torch.exp(-1 * (x - shift_param) / temperature)
    return 1 / (1 + exp)

def BT_OT_RSS_Loss(r_theta, target, RSS_factor=1, OT_factor=1, BT_factor=1, RP_factor=1):

    r_theta, target = r_theta.to("cpu"), target.to("cpu")

    shift_param = 2
    isGood_pos = sigmoid_shift(r_theta[0, target.item()], shift_param)
    isGood_neg = sigmoid_shift(r_theta[0, target.item() - 1], shift_param)

    BT_loss = nn.CrossEntropyLoss()
    BT = BT_loss(r_theta, target).to("cpu")
    
    RSS = Richardson_Srikumar_Sabhahwal_Loss(isGood_pos, isGood_neg)
    OT = One_True_Loss(isGood_pos, isGood_neg)
    RP = Reward_Penalty_Loss(isGood_pos, isGood_neg)

    return  (OT_factor * OT) + (RSS_factor * RSS) + (BT_factor * BT) + (RP_factor * RP)

