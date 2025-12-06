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


def Richardson_Srikumar_Sabhahwal_Loss(r_theta, target):

    r_theta, target = r_theta.to("cpu"), target.to("cpu")

    isGood_pos = torch.sigmoid(r_theta[0, target.item()])
    isGood_neg = torch.sigmoid(r_theta[0, target.item() - 1])

    loss = torch.max(torch.tensor([0]), torch.log(isGood_neg) - torch.log(isGood_pos))

    return loss

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
def One_True_Loss(r_theta, target):
    r_theta, target = r_theta.to("cpu"), target.to("cpu")

    isGood_pos = torch.sigmoid(r_theta[0, target.item()])
    isGood_neg = torch.sigmoid(r_theta[0, target.item() - 1])

    # print(f"is good pos is {isGood_pos} and is good neg is {isGood_neg}")
    loss = -1 * torch.log(isGood_pos + isGood_neg - (isGood_pos * isGood_neg))

    return loss

def sigmoid_shift(x, shift_param = 0, temperature = 1):
    exp = torch.exp(-1 * (x - shift_param) / temperature)
    return 1 / (1 + exp)

def Reward_Penalty_Loss(r_theta, target):
    # logic: isGood(positive) and not isBad(negative)
    # NLL  : -log( isGood_pos * isBad_neg ) = -log(isGood_pos * (1 - isGood_neg))
    r_theta, target = r_theta.to("cpu"), target.to("cpu")

    shift_param = 2 # since the smallest reward is 0, we shift by 10 to allow the isGood function to output values close to 0
    isGood_pos = sigmoid_shift(r_theta[0, target.item()], shift_param)
    isBad_neg = 1 - sigmoid_shift(r_theta[0, target.item() - 1], shift_param)

    loss = -1 * torch.log(isGood_pos * isBad_neg)

    return loss

def BT_OT_RSS_Loss(r_theta, target, RSS_factor=1, OT_factor=1, BT_factor=1, RP_factor=1):

    BT_loss = nn.CrossEntropyLoss()
    BT = BT_loss(r_theta, target).to("cpu")
    
    RSS = Richardson_Srikumar_Sabhahwal_Loss(r_theta, target)

    OT = One_True_Loss(r_theta, target)
    RP = Reward_Penalty_Loss(r_theta, target)

    return  (OT_factor * OT) + (RSS_factor * RSS) + (BT_factor * BT) + (RP_factor * RP)

