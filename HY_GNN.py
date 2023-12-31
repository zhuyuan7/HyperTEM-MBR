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
# # device = torch.device("cuda:1")
# print('Device:', device)

    
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

        # self.attn = AttentionModel(self.behavior, self.userNum, self.itemNum, self.embedding_dim, self.hidden_dim)  #userNum, itemNum, embedding_dim, hidden_dim
        # self.gru = GRUModel(self.behavior, self.userNum, self.itemNum, self.userNum, self.hidden_dim, self.hidden_dim)
        # self.gcn = GCN(self.userNum, self.itemNum, self.behavior, self.behavior_mats)
        


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
        # user_embed, item_embed, user_embeds, item_embeds = self.gcn()  #torch.Size([31882, 16])  torch.Size([31232, 16]) torch.Size([4, 31882, 16])  torch.Size([4, 31232, 16])
        user_embed, item_embed, user_embeds, item_embeds = self.hgnn()
        return user_embed, item_embed, user_embeds, item_embeds 


    def para_dict_to_tenser(self, para_dict):     
        tensors = []
        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        tensors = torch.stack(tensors, dim=0)

        return tensors.float()

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_parameters(), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_parameters()(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  
                    self.set_param(self, name, param)
                    
  

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
        
        # self.sigmoid = torch.nn.Sigmoid()
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
        # init.xavier_uniform_(alpha)

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
            # print("layer_user_embedding:",user_embedding.shape) #layer_user_embedding: torch.Size([31882, 16])
            # print("layer_item_embedding:",item_embedding.shape) #layer_item_embedding: torch.Size([31232, 16])
            all_user_embeddings.append(user_embedding)   
            all_item_embeddings.append(item_embedding)
            all_user_embeddingss.append(user_embeddings)
            all_item_embeddingss.append(item_embeddings)
        
        # print("all_user_embeddings:", len(all_user_embeddings))  # 3 
        # print("all_item_embeddings:", len(all_item_embeddings))  # 3
        user_embedding = torch.cat(all_user_embeddings, dim=1) #torch.Size([31882, 16])   #### shape:torch.Size([31882, 48])
        item_embedding = torch.cat(all_item_embeddings, dim=1) #torch.Size([31232, 16])   #### shape:torch.Size([31232, 48])
        
        # print("쉐입확인_user_embedding:", user_embedding.shape)  #torch.Size([31882, 48])
        # print("쉐입확인_item_embedding:", item_embedding.shape)  #torch.Size([31232, 48])
        
        
        
        # [DONE!] TODO 차원 정리하고 에러 해결해야함.
        # IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
        user_embeddings = torch.cat(all_user_embeddingss, dim=2)
        # print("user_embedding:", user_embeddings.shape)   # torch.Size([4, 31882, 48])
        item_embeddings = torch.cat(all_item_embeddingss, dim=2)
        # print("item_embedding:", item_embeddings.shape)   # torch.Size([4, 31232, 48])
        
        
        
        # self.u_concatenation_w = torch.Size([48,16])
        # self.i_concatenation_w = torch.Size([48,16])
        user_embedding = torch.matmul(user_embedding , self.u_concatenation_w) #torch.Size([31882, 16])
        item_embedding = torch.matmul(item_embedding , self.i_concatenation_w) # torch.Size([31232, 16])
        user_embeddings = torch.matmul(user_embeddings , self.u_concatenation_w) #torch.Size([4, 31882, 16])
        item_embeddings = torch.matmul(item_embeddings , self.i_concatenation_w) #torch.Size([4, 31232, 16])
        # print("user_embedding_dtype:", user_embedding.dtype)
        # print("item_embedding_dtype:", item_embedding.dtype)
        # print("user_embeddings_dtype:", user_embeddings.dtype)
        # print("item_embeddings_dtype:", item_embeddings.dtype)
            

        return user_embedding, item_embedding, user_embeddings, item_embeddings  #[31882, 16], [31232, 16], [4, 31882, 16], [4, 31232, 16]


class HGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, userNum, itemNum, behavior, hyper_behavior_Adj): # (16,16,22438, 35573,)
        super(HGCNLayer, self).__init__()

        self.behavior = behavior
        self.hyper_behavior_Adj = hyper_behavior_Adj
        
        self.userNum = userNum
        self.itemNum = itemNum
        # self.hidden_dim = args.hidden_dim 
        
        self.act = torch.nn.ReLU()
        self.i_w = nn.Parameter(torch.Tensor(in_dim, out_dim)) #shape:torch.Size([16, 16]) device: device(type='cpu')
        self.u_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.ii_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        init.xavier_uniform_(self.i_w)
        init.xavier_uniform_(self.u_w)

        #### HYPER_GRAPH_EMBEDDING  ####
        # self.user_embedding = torch.nn.Embedding(self.userNum, args.hidden_dim) #Embedding(31232, 16)
        # self.item_embedding = torch.nn.Embedding(self.itemNum, args.hidden_dim) #Embedding(31882, 16)
        # nn.init.xavier_uniform_(self.user_embedding.weight)
        # nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # self.uHyper = nn.Parameter(torch.Tensor(args.hyperNum, args.hidden_dim))
        # self.iHyper = nn.Parameter(torch.Tensor(args.hyperNum, args.hidden_dim))
        # init.xavier_uniform_(self.uHyper)
        # init.xavier_uniform_(self.iHyper)

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
        # HGNN can also be seen as learning a transformation in hidden space, with args.hyperNum hidden units (hyperedges)
        return embeds @ (hyper.T @ hyper)# @ (embeds.T @ embeds)
    
 

 
    def forward(self, user_embedding, item_embedding): # 여기의 user_embedding을 gru거친 user_embedding으로 바꿔

        user_embedding_list = [None]*len(self.behavior)
        item_embedding_list = [None]*len(self.behavior)



       # 031-639-1333
        for i in range(len(self.behavior)):
            hyper_behavior_mat = self.hyper_behavior_Adj[i]['A'] #.clone().detach()   #torch.Size([31882, 31232])
            # user_behavior_mat = self.behavior_mats[i]['A'].clone().detach()   #torch.Size([31882, 31232])
            # item_behavior_mat = self.behavior_mats[i]['AT'].clone().detach()    #shape: torch.Size([31882, 31232])
            embeds = torch.concat([self.uEmbeds, self.iEmbeds], dim=0)  #U: torch.Size([2174, 16]) I: torch.Size([2174, 16]) CONCAT: torch.Size([32287, 16])
            embeds_list = [embeds]
            gcn_embeds_list = []
            hyper_embeds_list = []
            # uu_hyper = self.uEmbeds @ self.uHyper * args.mult
            # ii_hyper = self.iEmbeds @ self.iHyper * args.mult

            # for j in range(len(self.behavior)):
            tem_embeds = self.gcnLayer(hyper_behavior_mat , embeds_list[-1])   # torch.Size([32287, 16])
                # tem_embeds = self.gcn_layer(hyper_behavior_mat, embeds_list[-1])
            user_embedding_list[i]  = self.hgnnLayer(tem_embeds[:self.userNum].detach(), self.uHyper)  #embeds[j][:self.userNum] ([16])    torch.Size([2174, 16]) torch.Size([16])
            item_embedding_list[i]= self.hgnnLayer(tem_embeds[self.userNum:].detach(), self.iHyper) 
                
                # hyper_user_embeds = self.hgnn_layer(F.dropout(uu_hyper, p=1-keep_rate), embeds_list[-1][:self.user_num])
                # hyper_item_embeds = self.hgnn_layer(F.dropout(ii_hyper, p=1-keep_rate), embeds_list[-1][self.user_num:])
            # gcn_embeds_list.append(tem_embeds)
                # hyper_embeds_list.append(torch.concat([hyper_user_embeds, hyper_item_embeds], dim=0))

                # embeds_list.append(tem_embeds + hyper_embeds_list[-1])
            # user_embedding_list[i] = self.hgnnLayer(gcn_embeds_list[i][:self.userNum], self.uHyper)   #embeds[j][:self.userNum] ([16])    torch.Size([2174, 16]) torch.Size([16])
            # item_embedding_list[i] = self.hgnnLayer(gcn_embeds_list[i][self.userNum:], self.iHyper) 
                # user_embedding_list[i] = hyper_user_embeds
                # item_embedding_list[i] = hyper_item_embeds
            # embeds = sum(embeds_list)
            # return embeds, gcn_embeds_list, hyper_embeds_list
            
            
            
            
           ### 
            # embeds = torch.concat([self.uEmbeds, self.iEmbeds], dim=0)
            # # embeds = torch.concat([self.user_embedding, self.item_embedding], dim=0)
            # lats = [embeds]
            # for j in range(len(self.layer_num)):
            #     temlat = self.gcnLayer(hyper_behavior_mat , lats[-1])
            #     lats.append(temlat)
            # # embeds = sum(lats)
            # embeds = lats
            
            # # this detach helps eliminate the mutual influence between the local GCN and the global HGNN
            # user_embedding_list[i] = self.hgnnLayer(embeds[j][:self.userNum].detach(), self.uHyper)   #torch.Size([2174, 16])
            # item_embedding_list[i] = self.hgnnLayer(embeds[j][self.userNum:].detach(), self.iHyper)   # torch.Size([30113, 16])
            # hyperUEmbeds = self.hgnnLayer(embeds[:self.userNum].detach(), self.uHyper)   #torch.Size([2174, 16])
            # hyperIEmbeds = self.hgnnLayer(embeds[self.userNum:].detach(), self.iHyper)   # torch.Size([30113, 16])
            # user_embedding_list[i] = user_embedding_list.append(hyperUEmbeds)
            # item_embedding_list[i] = item_embedding_list.append(hyperIEmbeds)
            # return embeds, hyperUEmbeds, hyperIEmbeds
            
            # dense_user_behavior_mat = user_behavior_mat.to_dense()
            # dense_item_behavior_mat = item_behavior_mat.to_dense()
            


#             d_V = torch.sum(dense_user_behavior_mat, dim=1)
#             d_E = torch.sum(dense_user_behavior_mat, dim=0)

#             # Compute the inverse of the square root of the diagonal D_v.
#             D_v_invsqrt = torch.diag(torch.pow(d_V, -0.5))

#             # Compute the inverse of the diagonal D_e.
#             D_e_inv = torch.diag(1.0 / d_E)

#             # In our example, B is an identity matrix.
#             n_edges = d_E.shape[0]  # 31232
#             identity = torch.eye(n_edges)

#             Laplacian = torch.matmul(D_v_invsqrt, torch.matmul(dense_user_behavior_mat, torch.matmul(identity, torch.matmul(D_e_inv, torch.matmul(dense_item_behavior_mat, D_v_invsqrt))))).to(device) 
#             user_node = torch.matmul(Laplacian, self.W1(self.dropout(user_node)))
            
#             user_node = self.act(user_node)
#             user_node = torch.matmul(Laplacian, self.W2(self.dropout(user_node)))
#             user_embedding_list[i] = user_node

# ############ ITEM_NODE, USER_EDGE ##################
#             item_d_V = torch.sum(dense_item_behavior_mat, dim=1)
#             # Compute Item edge degree.
#             item_d_E = torch.sum(dense_item_behavior_mat, dim=0)
#             # Compute the inverse of the square root of the diagonal D_v.
#             item_D_v_invsqrt = torch.diag(torch.pow(item_d_V, -0.5))
#             # Compute the inverse of the diagonal D_e.
#             item_D_e_inv = torch.diag(1.0 / item_d_E)
#             # In our example, B is an identity matrix.
#             item_n_edges = item_d_E.shape[0]
#             item_identity = torch.eye(item_n_edges)
#             item_identity = dglsp.identity((item_n_edges, item_n_edges))
#             # Compute Laplacian from the equation above.
#             Laplacian = torch.matmul(item_D_v_invsqrt, torch.matmul(dense_item_behavior_mat, torch.matmul(item_identity, torch.matmul(item_D_e_inv, torch.matmul(dense_user_behavior_mat, item_D_v_invsqrt))))).to(device) 
#             item_node = torch.matmul(Laplacian, self.W1(self.dropout(item_node)))
#             item_node = self.act(item_node)
#             item_node = torch.matmul(Laplacian, self.W2(self.dropout(item_node)))
#             item_embedding_list[i] = item_node

        user_embeddings = torch.stack(user_embedding_list, dim=0)  #shape:torch.Size([4, 31882, 16])   torch.Size([3, 2174, 16])
        item_embeddings = torch.stack(item_embedding_list, dim=0)  #shape:torch.Size([4, 31232, 16])   torch.Size([3, 30113, 16])  
        # print(" item_embeddings:",  item_embeddings.shape)  # torch.Size([4, 31882, 16])
       
       
       
        # # print("gru_stack 완성")
        user_embedding = self.act(torch.matmul(torch.mean(user_embeddings, dim=0), self.u_w)) # shape:torch.Size([31882, 16])  torch.Size([2174, 16])
        item_embedding = self.act(torch.matmul(torch.mean(item_embeddings, dim=0), self.i_w)) # shape:torch.Size([31232, 16])  torch.Size([30113, 16])

        user_embeddings = self.act(torch.matmul(user_embeddings, self.u_w)) #shape:torch.Size([4, 31882, 16])  torch.Size([3, 2174, 16])
        item_embeddings = self.act(torch.matmul(item_embeddings, self.i_w)) #shape:torch.Size([4, 31232, 16])  torch.Size([3, 30113, 16])
      

        return user_embedding, item_embedding, user_embeddings, item_embeddings  


            # user_behavior_mat = ex_user_behavior_mat + dglsp.identity(ex_user_behavior_mat.shape)

            # user_behavior_mat = ex_user_behavior_mat + dglsp.identity(ex_user_behavior_mat.shape)
            # item_behavior_mat = self.behavior_mats[i]['AT'].clone().detach()   #shape:torch.Size([31232, 31882])
            ###########################################################
            # (HIGHLIGHT) Compute the Laplacian with Sparse Matrix API
            ###########################################################
            # Compute  USER node degree.
            # d_V = user_behavior_mat.sum(1)
            # Compute Item edge degree.
            # d_E = user_behavior_mat.sum(0)
            # # Compute the inverse of the square root of the diagonal D_v.
            # D_v_invsqrt = dglsp.diag(d_V**-0.5)
            # # Compute the inverse of the diagonal D_e.
            # D_e_inv = dglsp.diag(d_E**-1)
            # # In our example, B is an identity matrix.
            # n_edges = d_E.shape[0]  # 31232
            # identity = dglsp.identity((n_edges, n_edges))
            # Compute Laplacian from the equation above.
            # Laplacian = D_v_invsqrt @ dense_user_behavior_mat @ identity @ D_e_inv @ dense_item_behavior_mat @ D_v_invsqrt
            # Laplacian = torch.matmul(D_v_invsqrt, torch.matmul(dense_user_behavior_mat, torch.matmul(identity, torch.matmul(D_e_inv, torch.matmul(dense_item_behavior_mat, D_v_invsqrt)))))
            # user_node = Laplacian @ self.W1(self.dropout(user_node))
            # user_node = self.act(user_node)
            # user_node = Laplacian @ self.W2(self.dropout(user_node))
            # user_embedding_list[i] = user_node

############ ITEM_NODE, USER_EDGE ##################
            # ex_item_behavior_mat = self.behavior_mats[i]['AT'].clone().detach() 
            # item_behavior_mat = ex_item_behavior_mat + dglsp.identity(ex_item_behavior_mat.shape)
            # Compute  USER node degree.
            # item_d_V = item_behavior_mat.sum(1)
            # # Compute Item edge degree.
            # item_d_E = item_behavior_mat.sum(0)
            # # Compute the inverse of the square root of the diagonal D_v.
            # item_D_v_invsqrt = dglsp.diag(item_d_V**-0.5)
            # # Compute the inverse of the diagonal D_e.
            # item_D_e_inv = dglsp.diag(item_d_E**-1)
            # # In our example, B is an identity matrix.
            # n_edges = d_E.shape[0]
            # item_identity = dglsp.identity((n_edges, n_edges))
            # # Compute Laplacian from the equation above.
            # Laplacian = item_D_v_invsqrt @ item_behavior_mat @ item_identity @ item_D_e_inv @ item_behavior_mat.T @ item_D_v_invsqrt
            
            # item_node = Laplacian @ self.W1(self.dropout(item_node))
            # item_node = self.act(item_node)
            # item_node = Laplacian @ self.W2(self.dropout(item_node))
            # print(item_node)
            # item_embedding_list[i] = item_node
            # # print(i)
            # print("시작")
# #######################################################################################################################################################################################            
#             # GRU_ USER PERSONALIZED item interation pattern FEATURE
#             # user_behavior_mat = self.behavior_mats[i]['A'].clone().detach()    #shape: torch.Size([31882, 31232])
#             # item_behavior_mat = self.behavior_mats[i]['AT'].clone().detach()   #shape:torch.Size([31232, 31882])


#             # GCN item, user embedding
#             user_embedding_list[i] = torch.spmm(self.behavior_mats[i]['A'], item_embedding) # torch.Size([31882, 16])
#             # print(user_embedding_list[i].shape)
#             item_embedding_list[i] = torch.spmm(self.behavior_mats[i]['AT'], user_embedding) # torch.Size([31232, 16])
#             # print(item_embedding_list[i].shape)
        

        
        # user_embeddings = torch.stack(user_embedding_list, dim=0)  #shape:torch.Size([4, 31882, 16])  
        # item_embeddings = torch.stack(item_embedding_list, dim=0)  #shape:torch.Size([4, 31232, 16])      
        # # print(" item_embeddings:",  item_embeddings.shape)  # torch.Size([4, 31882, 16])
       
       
       
        # # # print("gru_stack 완성")
        # user_embedding = self.act(torch.matmul(torch.mean(user_embeddings, dim=0), self.u_w)) # shape:torch.Size([31882, 16])
        # item_embedding = self.act(torch.matmul(torch.mean(item_embeddings, dim=0), self.i_w)) # shape:torch.Size([31232, 16])

        # user_embeddings = self.act(torch.matmul(user_embeddings, self.u_w)) #shape:torch.Size([4, 31882, 16])
        # item_embeddings = self.act(torch.matmul(item_embeddings, self.i_w)) #shape:torch.Size([4, 31232, 16])
      

        # return user_embedding, item_embedding, user_embeddings, item_embeddings             

#------------------------------------------------------------------------------------------------------------------------------------------------


# class HGNN(nn.Module):
#     def __init__(self, userNum, itemNum, behavior, behavior_mats, H, in_size, out_size, hidden_dim):
#         super(HGNN, self).__init__()
#         self.userNum = userNum
#         self.itemNum = itemNum
#         self.behavior = behavior
#         self.behavior_mats = behavior_mats
#         self.hidden_dim = args.hidden_dim
#         self.output_dim = args.hidden_dim
        
#         self.embedding_dict = self.init_embedding() 
#         self.weight_dict = self.init_weight()
#         # self.gcn = META_GCN(self.userNum, self.itemNum, self.behavior, self.behavior_mats)

#         self.W1 = nn.Linear(in_size, hidden_dim)
#         self.W2 = nn.Linear(hidden_dim, out_size)
#         self.dropout = nn.Dropout(0.5)

#         ###########################################################
#         # (HIGHLIGHT) Compute the Laplacian with Sparse Matrix API
#         ###########################################################
#         # Compute node degree.
#         d_V = H.sum(1)
#         # Compute edge degree.
#         d_E = H.sum(0)
#         # Compute the inverse of the square root of the diagonal D_v.
#         D_v_invsqrt = dglsp.diag(d_V**-0.5)
#         # Compute the inverse of the diagonal D_e.
#         D_e_inv = dglsp.diag(d_E**-1)
#         # In our example, B is an identity matrix.
#         n_edges = d_E.shape[0]
#         B = dglsp.identity((n_edges, n_edges))
#         # Compute Laplacian from the equation above.
#         self.L = D_v_invsqrt @ H @ B @ D_e_inv @ H.T @ D_v_invsqrt

#     def forward(self, X):
#         X = self.L @ self.W1(self.dropout(X))
#         X = F.relu(X)
#         X = self.L @ self.W2(self.dropout(X))
#         print(X)
#         return X



# import torch
# import torch.nn as nn
# import torch.optim as optim

# class HypergraphConvLayer(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(HypergraphConvLayer, self).__init__()
#         self.linear = nn.Linear(input_dim, output_dim)

#     def forward(self, hypergraph_adjacency, node_features):
#         hypergraph_adjacency = hypergraph_adjacency.unsqueeze(2)
#         hypergraph_adjacency = hypergraph_adjacency.expand(-1, -1, node_features.size(2))

#         hypergraph_features = torch.sum(hypergraph_adjacency * node_features, dim=1)
#         output = self.linear(hypergraph_features)
#         return output

# class HypergraphModel(nn.Module):
#     def __init__(self, num_nodes, input_dim, hidden_dim, output_dim):
#         super(HypergraphModel, self).__init__()
#         self.embedding_layer = nn.Embedding(num_nodes, input_dim)
#         self.conv_layer = HypergraphConvLayer(input_dim, hidden_dim)
#         self.output_layer = nn.Linear(hidden_dim, output_dim)

#     def forward(self, hypergraph_adjacency, node_indices):
#         node_features = self.embedding_layer(node_indices)
#         conv_output = self.conv_layer(hypergraph_adjacency, node_features)
#         final_output = self.output_layer(conv_output)
#         return final_output

# # Hypergraph adjacency matrix (example)
# hypergraph_adjacency = torch.tensor([[1, 1, 0],
#                                      [1, 0, 1],
#                                      [0, 1, 1]])

# # Node indices (example)
# node_indices = torch.tensor([0, 1, 2])

# # Instantiate and run the model
# num_nodes = len(node_indices)
# input_dim = 64
# hidden_dim = 128
# output_dim = 10

# model = HypergraphModel(num_nodes, input_dim, hidden_dim, output_dim)
# output = model(hypergraph_adjacency, node_indices)
# print(output)
