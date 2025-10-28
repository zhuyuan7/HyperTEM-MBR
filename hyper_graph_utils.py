import numpy as np
import scipy.sparse as sp
from scipy.sparse import *
import torch as t
from torch import nn


import datetime
from Params import args
import pickle


device = t.device(f"cuda:{args.cuda_num}" if t.cuda.is_available() else "cpu")

def get_hyper_use(behaviors_data):

    hyper_behavior_mats = {}

    behaviors_data = (behaviors_data != 0) * 1  #shape:(31882, 31232)
    
    hyper_behavior = sp.coo_matrix(behaviors_data) 
    hyper_behavior_mats['A'] = makeTorchAdjs(hyper_behavior)  # USER
    hyper_behavior_mats['AT'] = makeTorchAdjs(hyper_behavior.T)  # ITEM
    hyper_behavior_mats['A_ori'] = None

    return hyper_behavior_mats

def normalizeAdj(mat):
    degree = np.array(mat.sum(axis=-1))  
    dInvSqrt = np.reshape(np.power(degree, -0.5), [-1]) 
    dInvSqrt[np.isinf(dInvSqrt)] = 0.0
    dInvSqrtMat = sp.diags(dInvSqrt)  
    HO = mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo() 
    return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo() 

def makeTorchAdjs(hyper_behavior):
    # make ui adj
    userNum = hyper_behavior.shape[0] 
    itemNum = hyper_behavior.shape[1]  
    a = sp.csr_matrix((userNum, userNum)) 
    b = sp.csr_matrix((itemNum, itemNum))  
    mat = sp.vstack([sp.hstack([a, hyper_behavior]), sp.hstack([hyper_behavior.transpose(), b])])  
    mat = (mat != 0) * 1.0
    mat = normalizeAdj(mat)

    # make cuda tensor
    idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))  
    vals = t.from_numpy(mat.data.astype(np.float32))
    shape = t.Size(mat.shape)
    return t.sparse.FloatTensor(idxs, vals, shape).to(device)

