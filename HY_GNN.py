import os
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import scipy.sparse as sp

import tqdm
from Params import args

os.environ['TORCH'] = torch.__version__
os.environ['DGLBACKEND'] = "pytorch"
import dgl



torch.backends.cudnn.enabled = False   # RuntimeError: cuDNN error: CUDNN_STATUS_MAPPING_ERROR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()

    return Variable(x, requires_grad=requires_grad)


class myModel(nn.Module):
    def __init__(self, userNum, itemNum, behavior, hyper_behavior_Adj): 
        super(myModel, self).__init__()  
        
        self.userNum = userNum  # 31882
        self.itemNum = itemNum  # 31232
        self.behavior = behavior  #['pv', 'fav', 'cart', 'buy']
        self.hyper_behavior_Adj = hyper_behavior_Adj  # {0:{'A':tensor(indices=tensor..parse_coo),'AT':tensor(indices=tensor..parse_coo),'A_ori':None}, {1:{}},{2:{}},{3:{}}}
        
        self.hidden_dim = args.hidden_dim 
        self.output_dim = args.hidden_dim 
        self.embedding_dim = args.hidden_dim 
        self.embedding_dict = self.init_embedding() #{'user_embedding':None, 'item_embedding':None,'user_embeddings':None, 'item_embeddings':None,  }
        self.weight_dict = self.init_weight()
        self.hgnn = HGNN(self.userNum, self.itemNum, self.behavior, self.hyper_behavior_Adj)


    def init_embedding(self):
        
        embedding_dict = {  
            'user_embedding': None,
            'item_embedding': None,
            'user_embeddings': None,
            'item_embeddings': None,
        }
        return embedding_dict

    def init_weight(self):  
        initializer = nn.init.xavier_uniform_
        
        weight_dict = nn.ParameterDict({
            'w_self_attention_item': nn.Parameter(initializer(torch.empty([args.hidden_dim, args.hidden_dim]))),
            'w_self_attention_user': nn.Parameter(initializer(torch.empty([args.hidden_dim, args.hidden_dim]))),
            'w_self_attention_cat': nn.Parameter(initializer(torch.empty([args.head_num*args.hidden_dim, args.hidden_dim]))),
            'alpha': nn.Parameter(torch.ones(2)),
        })      
        return weight_dict  


    def forward(self):
        user_embed, item_embed, user_embeds, item_embeds = self.hgnn()
        return user_embed, item_embed, user_embeds, item_embeds 

                  
  

class HGNN(nn.Module):
    def __init__(self, userNum, itemNum, behavior, hyper_behavior_Adj):
        super(HGNN, self).__init__()  
        self.userNum = userNum  #22438
        self.itemNum = itemNum  #35573
        self.hidden_dim = args.hidden_dim # 16

        self.behavior = behavior
        self.hyper_behavior_Adj = hyper_behavior_Adj
        

        self.user_embedding, self.item_embedding = self.init_embedding()         
        
        self.alpha, self.i_concatenation_w, self.u_concatenation_w, self.i_input_w, self.u_input_w = self.init_weight()
        
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(args.drop_rate)

        self.hgnn_layer = eval(args.gnn_layer)
        self.layers = nn.ModuleList()
        for i in range(0, len(self.hgnn_layer)):   # [16, 16, 16]
            self.layers.append(HGCNLayer(args.hidden_dim, args.hidden_dim, self.userNum, self.itemNum, self.behavior, self.hyper_behavior_Adj))  
    


    # gcn 생성 위한 embedding
    def init_embedding(self):  
        user_embedding = torch.nn.Embedding(self.userNum, args.hidden_dim) #Embedding(31232, 16)
        item_embedding = torch.nn.Embedding(self.itemNum, args.hidden_dim) #Embedding(31882, 16)
        nn.init.xavier_uniform_(user_embedding.weight)
        nn.init.xavier_uniform_(item_embedding.weight)

        return user_embedding, item_embedding

    def init_weight(self):
        alpha = nn.Parameter(torch.ones(2))  # T:tensor([1., 1.], grad_fn=<PermuteBackward0>)
        i_concatenation_w = nn.Parameter(torch.Tensor(len(eval(args.gnn_layer))*args.hidden_dim, args.hidden_dim)) #torch.Size([48, 16])
        u_concatenation_w = nn.Parameter(torch.Tensor(len(eval(args.gnn_layer))*args.hidden_dim, args.hidden_dim)) #torch.Size([48, 16])
        i_input_w = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim)) # torch.Size([16, 16])
        u_input_w = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim)) # torch.Size([16, 16])
        init.xavier_uniform_(i_concatenation_w) # torch.Size([48, 16])
        init.xavier_uniform_(u_concatenation_w) # torch.Size([48, 16])
        init.xavier_uniform_(i_input_w)  #torch.Size([16, 16])
        init.xavier_uniform_(u_input_w)  #torch.Size([16, 16])

        return alpha, i_concatenation_w, u_concatenation_w, i_input_w, u_input_w

    def forward(self, user_embedding_input=None, item_embedding_input=None):
        all_user_embeddings = []
        all_item_embeddings = []
        all_user_embeddingss = []
        all_item_embeddingss = []

        user_embedding = self.user_embedding.weight #torch.Size([31882, 16])  #dtype:torch.float32
        item_embedding = self.item_embedding.weight #torch.Size([31232, 16])

        for i, layer in enumerate(self.layers):
            # print("layer:",i)
            user_embedding, item_embedding, user_embeddings, item_embeddings = layer(user_embedding, item_embedding)

            norm_user_embeddings = F.normalize(user_embedding, p=2, dim=1) #shape: torch.Size([31882, 16])
            norm_item_embeddings = F.normalize(item_embedding, p=2, dim=1) #shape: torch.Size([31232, 16])

            all_user_embeddings.append(user_embedding)   
            all_item_embeddings.append(item_embedding)
            all_user_embeddingss.append(user_embeddings)
            all_item_embeddingss.append(item_embeddings)

        user_embedding = torch.cat(all_user_embeddings, dim=1) #torch.Size([31882, 16])   #### shape:torch.Size([31882, 48])
        item_embedding = torch.cat(all_item_embeddings, dim=1) #torch.Size([31232, 16])   #### shape:torch.Size([31232, 48])
        
        user_embeddings = torch.cat(all_user_embeddingss, dim=2)
        item_embeddings = torch.cat(all_item_embeddingss, dim=2)

        user_embedding = torch.matmul(user_embedding , self.u_concatenation_w) #torch.Size([31882, 16])
        item_embedding = torch.matmul(item_embedding , self.i_concatenation_w) # torch.Size([31232, 16])
        user_embeddings = torch.matmul(user_embeddings , self.u_concatenation_w) #torch.Size([4, 31882, 16])
        item_embeddings = torch.matmul(item_embeddings , self.i_concatenation_w) #torch.Size([4, 31232, 16])

        return user_embedding, item_embedding, user_embeddings, item_embeddings  #[31882, 16], [31232, 16], [4, 31882, 16], [4, 31232, 16]


class HGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, userNum, itemNum, behavior, hyper_behavior_Adj): # (16,16,22438, 35573,)
        super(HGCNLayer, self).__init__()

        self.behavior = behavior
        self.hyper_behavior_Adj = hyper_behavior_Adj
        
        self.userNum = userNum
        self.itemNum = itemNum
        
        self.act = torch.nn.ReLU()
        self.i_w = nn.Parameter(torch.Tensor(in_dim, out_dim)) #shape:torch.Size([16, 16]) device: device(type='cpu')
        self.u_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.ii_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        init.xavier_uniform_(self.i_w)
        init.xavier_uniform_(self.u_w)

        self.uEmbeds = nn.Parameter(torch.empty(userNum, args.hidden_dim))
        self.iEmbeds = nn.Parameter(torch.empty(itemNum, args.hidden_dim))
        self.uHyper = nn.Parameter(torch.empty(args.hyperNum, args.hidden_dim))
        self.iHyper = nn.Parameter(torch.empty(args.hyperNum, args.hidden_dim))
        init.xavier_uniform_(self.uEmbeds)
        init.xavier_uniform_(self.iEmbeds)       
        init.xavier_uniform_(self.uHyper)
        init.xavier_uniform_(self.iHyper)
        self.dropout = torch.nn.Dropout(args.drop_rate1)


    def gcnLayer(self, adj, embeds):
        return torch.spmm(adj, embeds)   # adj torch.Size([32287, 32287]), embeds torch.Size([32287, 16])
    
    def hgnnLayer(self, embeds, hyper):  # hyper torch.Size([128, 16])
        return embeds @ (hyper.T @ hyper) @ (embeds.T @ embeds)
    
 

 
    def forward(self, user_embedding, item_embedding): # 여기의 user_embedding을 gru거친 user_embedding으로 바꿔

        user_embedding_list = [None]*len(self.behavior)
        item_embedding_list = [None]*len(self.behavior)



       # 031-639-1333
        for i in range(len(self.behavior)):
            hyper_behavior_mat = self.hyper_behavior_Adj[i]['A'] #.clone().detach()   #torch.Size([31882, 31232])
            embeds = torch.concat([self.uEmbeds, self.iEmbeds], dim=0)  #U: torch.Size([2174, 16]) I: torch.Size([2174, 16]) CONCAT: torch.Size([32287, 16])
            embeds_list = [embeds]
            for j in range(args.gcn_hops):
                temlat = self.gcnLayer(hyper_behavior_mat, embeds_list[-1])
                embeds_list.append(temlat)
            tem_embeds = sum(embeds_list)
            
            # tem_embeds = self.gcnLayer(hyper_behavior_mat , embeds_list[-1])   # torch.Size([32287, 16])
            user_embedding_list[i]  = self.hgnnLayer(tem_embeds[:self.userNum].detach(), self.uHyper)  #embeds[j][:self.userNum] ([16])    torch.Size([2174, 16]) torch.Size([16])
            item_embedding_list[i]= self.hgnnLayer(tem_embeds[self.userNum:].detach(), self.iHyper) 
 
        user_embeddings = torch.stack(user_embedding_list, dim=0)  #shape:torch.Size([4, 31882, 16])   torch.Size([3, 2174, 16])
        item_embeddings = torch.stack(item_embedding_list, dim=0)  #shape:torch.Size([4, 31232, 16])   torch.Size([3, 30113, 16])  

        user_embedding = self.act(torch.matmul(torch.mean(user_embeddings, dim=0), self.u_w)) # shape:torch.Size([31882, 16])  torch.Size([2174, 16])
        item_embedding = self.act(torch.matmul(torch.mean(item_embeddings, dim=0), self.i_w)) # shape:torch.Size([31232, 16])  torch.Size([30113, 16])

        user_embeddings = self.act(torch.matmul(user_embeddings, self.u_w)) #shape:torch.Size([4, 31882, 16])  torch.Size([3, 2174, 16])
        item_embeddings = self.act(torch.matmul(item_embeddings, self.i_w)) #shape:torch.Size([4, 31232, 16])  torch.Size([3, 30113, 16])
      

        return user_embedding, item_embedding, user_embeddings, item_embeddings  
