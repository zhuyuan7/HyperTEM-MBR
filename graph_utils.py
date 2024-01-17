import numpy as np
import scipy.sparse as sp
from scipy.sparse import *
import torch
import datetime
from Params import args
device = torch.device(f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu")


def get_user(behavior_mats):
    
    user = behavior_mats['A'] # shape:torch.Size([31882, 31232])
    item = behavior_mats['AT']

    return user, item


def get_use(behaviors_data):

    behavior_mats = {}
        
    behaviors_data = (behaviors_data != 0) * 1  #shape:(31882, 31232)

    behavior_mats['A'] = matrix_to_tensor(normalize_adj(behaviors_data))  # USER
    behavior_mats['AT'] = matrix_to_tensor(normalize_adj(behaviors_data.T))  # ITEM
    behavior_mats['A_ori'] = None

    return behavior_mats


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj) # shape:(31882, 31232) #<2174x30113 sparse matrix of type '<class 'numpy.int32'>'
    rowsum = np.array(adj.sum(1)) #shape:(31232, 1)  #dtype('int32')
    rowsum_diag = sp.diags(np.power(rowsum+1e-8, -0.5).flatten()) #shape:(31232, 31232)

    colsum = np.array(adj.sum(0))
    colsum_diag = sp.diags(np.power(colsum+1e-8, -0.5).flatten())

    return rowsum_diag*adj*colsum_diag
    # return adj


def matrix_to_tensor(cur_matrix):
    if type(cur_matrix) != sp.coo_matrix:
        cur_matrix = cur_matrix.tocoo()  
    indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  
    values = torch.from_numpy(cur_matrix.data)  
    shape = torch.Size(cur_matrix.shape) #torch.Size([31882, 31232])

    return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).to(device)  

