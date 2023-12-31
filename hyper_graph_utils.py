import numpy as np
import scipy.sparse as sp
from scipy.sparse import *
import torch as t
from torch import nn


import datetime
from Params import args
import pickle


def get_hyper_use(behaviors_data):

    hyper_behavior_mats = {}

    behaviors_data = (behaviors_data != 0) * 1  #shape:(31882, 31232)
    
    hyper_behavior = sp.coo_matrix(behaviors_data) 
    hyper_behavior_mats['A'] = makeTorchAdjs(hyper_behavior)  # USER
    hyper_behavior_mats['AT'] = makeTorchAdjs(hyper_behavior.T)  # ITEM
    hyper_behavior_mats['A_ori'] = None

    return hyper_behavior_mats

def normalizeAdj(mat):
    degree = np.array(mat.sum(axis=-1))  #np.array(adj.sum(1)) 
    dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
    dInvSqrt[np.isinf(dInvSqrt)] = 0.0
    dInvSqrtMat = sp.diags(dInvSqrt)
    return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

def makeTorchAdjs(hyper_behavior):
    # make ui adj
    userNum = hyper_behavior.shape[0]  # (2174,30113)
    itemNum = hyper_behavior.shape[1]  #30113)
    a = sp.csr_matrix((userNum, userNum))
    b = sp.csr_matrix((itemNum, itemNum))
    mat = sp.vstack([sp.hstack([a, hyper_behavior]), sp.hstack([hyper_behavior.transpose(), b])])  #<32287x32287 sparse matrix of type '<class 'numpy.float64'>'
    mat = (mat != 0) * 1.0
    mat = normalizeAdj(mat)

    # make cuda tensor
    idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
    vals = t.from_numpy(mat.data.astype(np.float32))
    shape = t.Size(mat.shape)
    return t.sparse.FloatTensor(idxs, vals, shape).cuda()

def mixer(meta_user_embed, meta_item_embed, meta_user_embeds, meta_item_embeds,
          hyper_user_embed, hyper_item_embed, hyper_user_embeds, hyper_item_embeds):
    
    mlp = nn.Sequential(nn.Linear(16 * 2, 16), nn.ReLU()).cuda()
                                    #torch.Size([2174, 16]) torch.Size([2174, 16])
    concatenated_tensor1 = t.cat((meta_user_embed, hyper_user_embed), dim=1).cuda()  # torch.Size([4, 31882, 32])
    concatenated_tensor2 = t.cat((meta_item_embed, hyper_item_embed), dim=1).cuda() 
    concatenated_tensor3 = t.cat((meta_user_embeds, hyper_user_embeds), dim=2).cuda() 
    concatenated_tensor4 = t.cat((meta_item_embeds, hyper_item_embeds), dim=2).cuda() 

    user_embed = mlp(concatenated_tensor1).cuda()
    item_embed = mlp(concatenated_tensor2 ).cuda()
    user_embeds = mlp(concatenated_tensor3).cuda()
    item_embeds = mlp(concatenated_tensor4).cuda()
    
    return user_embed,item_embed, user_embeds, item_embeds
