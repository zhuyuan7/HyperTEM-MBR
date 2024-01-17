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
    """
    adjacency matrix를 Symmetrically normalize
    노드의 연결성을 차원(degree)라고 부르는데, 인접 행렬 A의 차원(D)에 대한 역행렬을 곱해주면 정규화를 할 수 있다.
    
    """
    degree = np.array(mat.sum(axis=-1))  # # D np.array(adj.sum(1)) USER NODE DEGREE
    dInvSqrt = np.reshape(np.power(degree, -0.5), [-1]) # # D^-0.5// D(USER NODE)^-1/2
    dInvSqrt[np.isinf(dInvSqrt)] = 0.0
    dInvSqrtMat = sp.diags(dInvSqrt)  # # D^-0.5
    return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()  # # D^-0.5 A D^0.5 를 만들어 준다.

def makeTorchAdjs(hyper_behavior):
    # make ui adj
    userNum = hyper_behavior.shape[0]  # (31882, 31232)
    itemNum = hyper_behavior.shape[1]  # 31232)
    a = sp.csr_matrix((userNum, userNum))  # (31882, 31882)
    b = sp.csr_matrix((itemNum, itemNum))  # (31232, 31232)
    mat = sp.vstack([sp.hstack([a, hyper_behavior]), sp.hstack([hyper_behavior.transpose(), b])])  #<32287x32287 sparse matrix of type '<class 'numpy.float64'>'
    mat = (mat != 0) * 1.0
    # mat = (mat + sp.eye(mat.shape[0])) * 1.0
    mat = normalizeAdj(mat)

    # make cuda tensor
    idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
    vals = t.from_numpy(mat.data.astype(np.float32))
    shape = t.Size(mat.shape)
    return t.sparse.FloatTensor(idxs, vals, shape).to(device)

# useless
def mixer(meta_user_embed, meta_item_embed, meta_user_embeds, meta_item_embeds,
          hyper_user_embed, hyper_item_embed, hyper_user_embeds, hyper_item_embeds):
    
    mlp = nn.Sequential(nn.Linear(16 * 2, 16), nn.ReLU()).to(device)
                                    #torch.Size([2174, 16]) torch.Size([2174, 16])
    concatenated_tensor1 = t.cat((meta_user_embed, hyper_user_embed), dim=1).to(device)  # torch.Size([4, 31882, 32])
            # concatenated_tensor = torch.cat((attn_user_embeddings, user_embeds), dim=2).to(device)  # torch.Size([4, 31882, 32])
    concatenated_tensor2 = t.cat((meta_item_embed, hyper_item_embed), dim=1).to(device)
    concatenated_tensor3 = t.cat((meta_user_embeds, hyper_user_embeds), dim=2).to(device) 
    concatenated_tensor4 = t.cat((meta_item_embeds, hyper_item_embeds), dim=2).to(device) 

    user_embed = mlp(concatenated_tensor1).to(device)
    item_embed = mlp(concatenated_tensor2 ).to(device)
    user_embeds = mlp(concatenated_tensor3).to(device)
    item_embeds = mlp(concatenated_tensor4).to(device)
    
    return user_embed,item_embed, user_embeds, item_embeds

