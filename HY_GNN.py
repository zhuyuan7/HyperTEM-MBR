import os
# import dgl.sparse as dglsp
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


torch.backends.cudnn.enabled = False   
device = torch.device(f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu")

    
def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.to(device)

    return Variable(x, requires_grad=requires_grad)


class myModel(nn.Module):
    def __init__(self, userNum, itemNum, behavior, hyper_behavior_Adj): 
        super(myModel, self).__init__()  
        
        self.userNum = userNum 
        self.itemNum = itemNum 
        self.behavior = behavior 
        self.hyper_behavior_Adj = hyper_behavior_Adj  
        self.hidden_dim = args.hidden_dim 
        self.output_dim = args.hidden_dim 
        self.embedding_dim = args.hidden_dim 
        self.embedding_dict = self.init_embedding() 
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
                 
class SpAdjDropEdge(nn.Module):
	def __init__(self):
		super(SpAdjDropEdge, self).__init__()


	def forward(self, adj, keepRate):
		if keepRate == 1.0:
			return adj
		vals = adj._values() 
		idxs = adj._indices()   
		edgeNum = vals.size() 
		mask = ((torch.rand(edgeNum) + keepRate).floor()).type(torch.bool)  
		newVals = vals[mask] / keepRate  
		newIdxs = idxs[:, mask]   
		return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)   
     

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
        
        self.sigmoid = torch.nn.Sigmoid()
        # self.act = torch.nn.PReLU()
        # self.act = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(args.drop_rate)

        self.hgnn_layer = eval(args.gnn_layer)
        self.layers = nn.ModuleList()
        for i in range(0, len(self.hgnn_layer)):   
            self.layers.append(HGCNLayer(args.hidden_dim, args.hidden_dim, self.userNum, self.itemNum, self.behavior, self.hyper_behavior_Adj))  
    

    def init_embedding(self):  
        user_embedding = torch.nn.Embedding(self.userNum, args.hidden_dim) 
        item_embedding = torch.nn.Embedding(self.itemNum, args.hidden_dim) 
        nn.init.xavier_uniform_(user_embedding.weight)
        nn.init.xavier_uniform_(item_embedding.weight)

        return user_embedding, item_embedding

    def init_weight(self):
        alpha = nn.Parameter(torch.ones(2))  
        i_concatenation_w = nn.Parameter(torch.Tensor(len(eval(args.gnn_layer))*args.hidden_dim, args.hidden_dim)) 
        u_concatenation_w = nn.Parameter(torch.Tensor(len(eval(args.gnn_layer))*args.hidden_dim, args.hidden_dim))
        i_input_w = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim)) 
        u_input_w = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim)) 
        init.xavier_uniform_(i_concatenation_w) 
        init.xavier_uniform_(u_concatenation_w) 
        init.xavier_uniform_(i_input_w)  
        init.xavier_uniform_(u_input_w) 

        return alpha, i_concatenation_w, u_concatenation_w, i_input_w, u_input_w

    def forward(self, user_embedding_input=None, item_embedding_input=None):
        all_user_embeddings = []
        all_item_embeddings = []
        all_user_embeddingss = []
        all_item_embeddingss = []

        user_embedding = self.user_embedding.weight 
        item_embedding = self.item_embedding.weight 

        for i, layer in enumerate(self.layers):
            # print("layer:",i)
            user_embedding, item_embedding, user_embeddings, item_embeddings = layer(user_embedding, item_embedding)

            norm_user_embeddings = F.normalize(user_embedding, p=2, dim=1)
            norm_item_embeddings = F.normalize(item_embedding, p=2, dim=1)
            all_user_embeddings.append(user_embedding)   
            all_item_embeddings.append(item_embedding)
            all_user_embeddingss.append(user_embeddings)
            all_item_embeddingss.append(item_embeddings)
        
        user_embedding = torch.cat(all_user_embeddings, dim=1) 
        item_embedding = torch.cat(all_item_embeddings, dim=1) 
        user_embeddings = torch.cat(all_user_embeddingss, dim=2)
        item_embeddings = torch.cat(all_item_embeddingss, dim=2)
        
        user_embedding = torch.matmul(user_embedding , self.u_concatenation_w) 
        item_embedding = torch.matmul(item_embedding , self.i_concatenation_w) 
        user_embeddings = torch.matmul(user_embeddings , self.u_concatenation_w) 
        item_embeddings = torch.matmul(item_embeddings , self.i_concatenation_w)           

        return user_embedding, item_embedding, user_embeddings, item_embeddings  


class HGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, userNum, itemNum, behavior, hyper_behavior_Adj): 
        super(HGCNLayer, self).__init__()

        self.behavior = behavior
        self.hyper_behavior_Adj = hyper_behavior_Adj
        
        self.userNum = userNum
        self.itemNum = itemNum

        self.keepRate = args.keepRate 
        
        self.act = torch.nn.ReLU()
        # self.act = torch.nn.LeakyReLU()
        self.i_w = nn.Parameter(torch.Tensor(in_dim, out_dim)) 
        self.u_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.ii_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        init.xavier_uniform_(self.i_w)
        init.xavier_uniform_(self.u_w)

        #### HYPER_GRAPH_EMBEDDING  ####

        self.uEmbeds = nn.Parameter(torch.empty(userNum, args.hidden_dim)) 
        self.iEmbeds = nn.Parameter(torch.empty(itemNum, args.hidden_dim))  
        self.uHyper = nn.Parameter(torch.empty( args.hidden_dim, args.hyperNum)) 
        self.iHyper = nn.Parameter(torch.empty(args.hidden_dim, args.hyperNum))  
        init.xavier_uniform_(self.uEmbeds)
        init.xavier_uniform_(self.iEmbeds)       
        init.xavier_uniform_(self.uHyper)
        init.xavier_uniform_(self.iHyper)
        self.dropout = torch.nn.Dropout(args.drop_rate1)
        self.edgeDropper = SpAdjDropEdge()
        # self.gcnLayer = GCNLayer()
        # self.hgnnLayer = HGNNLayer()

    def gcnLayer(self, adj, embeds):
        return torch.spmm(adj, embeds)   
    
    def hgnnLayer(self, adj, embeds):
        lat = adj.T @ embeds   
        ret = adj @ lat   
        return ret


    def forward(self, user_embedding, item_embedding): 

        user_embedding_list = [None]*len(self.behavior)
        item_embedding_list = [None]*len(self.behavior)


        for i in range(len(self.behavior)):
            hyper_behavior_mat = self.hyper_behavior_Adj[i]['A'].to(device) 
            keepRate =args.keepRate
            embeds = torch.concat([self.uEmbeds, self.iEmbeds], dim=0)  
            embeds_list = [embeds]
            gcn_embeds_list = []
            hyper_embeds_list = []
            uu_hyper = self.uEmbeds @ self.uHyper * args.mult 
            ii_hyper = self.iEmbeds @ self.iHyper * args.mult 

           
            temEmbeds = self.gcnLayer(self.edgeDropper(hyper_behavior_mat, keepRate), embeds_list[-1]) 
            hyperULat = self.hgnnLayer(F.dropout(uu_hyper, p=1-keepRate), embeds_list[-1][:self.userNum]) 
            hyperILat = self.hgnnLayer(F.dropout(ii_hyper, p=1-keepRate), embeds_list[-1][self.userNum:])  


            user_embedding_list[i] = hyperULat.detach() + temEmbeds[:self.userNum].detach()
            item_embedding_list[i] = hyperILat.detach() + temEmbeds[self.userNum:].detach()

                
        user_embeddings = torch.stack(user_embedding_list, dim=0) 
        item_embeddings = torch.stack(item_embedding_list, dim=0)    

        ##  hgnn embedding
        user_embedding = self.act(torch.mean(user_embeddings, dim=0)) 
        item_embedding = self.act(torch.mean(item_embeddings, dim=0)) 
        user_embeddings = self.act(user_embeddings) 
        item_embeddings = self.act(item_embeddings) 

        return user_embedding, item_embedding, user_embeddings, item_embeddings  

