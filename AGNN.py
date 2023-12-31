import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from Params import args
import graph_utils
import pickle
import numpy as np
from tqdm import tqdm


# import Gru
torch.backends.cudnn.enabled = False   # RuntimeError: cuDNN error: CUDNN_STATUS_MAPPING_ERROR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()

    return Variable(x, requires_grad=requires_grad)

# TODO Attn, GRU CODE

class AttentionModel(nn.Module):
    def __init__(self, behavior, userNum, itemNum, embedding_dim, hidden_dim):
        super(AttentionModel, self).__init__()
        
        self.behavior = behavior
        self.userNum = userNum
        self.itemNum = itemNum
        self.embedding_dim = args.hidden_dim
        self.hidden_dim = args.hidden_dim

        self.user_embedding = nn.Embedding(itemNum, embedding_dim)
        self.attn = nn.Linear(embedding_dim, hidden_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self):
        attn_u_embed_list = [None]*len(self.behavior)
        for i in range(len(self.behavior)):    
            # user_attn_data = pickle.load(open('/home/joo/JOOCML/data/retail_rocket/{}_gru_padding.pkl'.format(i), 'rb')).to(device)#.cuda()  # torch.Size([31882, 293])
            user_attn_data = pickle.load(open('/home/joo/JOOCML/data/retail_rocket/{}_gru_padding.pkl'.format(i), 'rb')).to(device)#.cuda()  # torch.Size([31882, 293])
            user_embedded = self.user_embedding(user_attn_data)  # torch.Size([31882, 293, 16])
            attn_weights = torch.softmax(self.attn(user_embedded), dim=1) # torch.Size([31882, 293, 16])
            user_embedding = torch.sum(attn_weights * user_embedded, dim=1)  # torch.Size([31882, 16])
            attn_u_embed_list[i] = self.fc(user_embedding)  #torch.Size([31882, 16])
        attn_user_embeddings = torch.stack(attn_u_embed_list, dim=0).to(device) #torch.Size([4, 31882, 16])
        
        return attn_user_embeddings  


# # Gru 모델 정의
class GRUModel(nn.Module):
    def __init__(self, behavior, userNum, itemNum, input_size, hidden_dim, output_dim):
        super(GRUModel, self).__init__()
        self.behavior = behavior 
        self.userNum = userNum  #22438
        self.itemNum = itemNum
        self.hidden_dim = args.hidden_dim 
        self.output_dim = args.hidden_dim 
        self.user_embedding_table =  torch.nn.Embedding(self.itemNum, self.hidden_dim).to(device)  # Embedding(31882, 16)
        
        self.gru = nn.GRU(input_size, hidden_dim, batch_first=True)  # (batch, seq, feature)
        self.fc = nn.Linear(hidden_dim, userNum)
        
    def forward(self):
        gru_u_embed_list = [None]*len(self.behavior)
        gru_i_embed_list = [None]*len(self.behavior)
        for i in range(len(self.behavior)):    
            # user_gru_data = pickle.load(open('/home/joo/JOOCML/data/retail_rocket/{}_gru_padding.pkl'.format(i), 'rb')).to(device)#.cuda()  # torch.Size([31882, 293])
            # user_gru_data = pickle.load(open('/home/joo/JOOCML/data/IJCAI_15/{}_gru_padding.pkl'.format(i), 'rb')).to(device)#.cuda()
            user_gru_data = pickle.load(open('C:\\Users\\choi\\Desktop\\115\\JOOCML\\data\\retail_rocket\\{}_retailrocket_gru_padding.pkl'.format(i), 'rb')).to(device)
            # user_gru_data = pickle.load(open('/home/joo/JOOCML/data/Tmall/{}_gru_padding copy.pkl'.format(i), 'rb')).to(device)#.cuda()
            gru_user_embeddings= self.user_embedding_table(user_gru_data).to(device)  #torch.Size([31882, 293, 16])    
                                  
            gru_u_embed = gru_user_embeddings.transpose(0, 2)  #torch.Size([16, 293, 31882])
            gru_u_output, _ = self.gru(gru_u_embed)   # torch.Size([16, 293, 16])
            gru_u_embed_squeeze = gru_u_output.squeeze(0) # torch.Size([16, 293, 16])
            fc_output = self.fc(gru_u_embed_squeeze)  # torch.Size([16, 293, 31882])
            trans_output = fc_output.transpose(0, 2)  # torch.Size([31882, 293, 16])
            gru_u_embed_list[i] = trans_output[:, -1, :]    # torch.Size([31882, 16])

        gru_user_embeddings = torch.stack(gru_u_embed_list, dim=0).to(device) #torch.Size([4, 31882, 16])

        return gru_user_embeddings
    
    

    
class myModel(nn.Module):
    def __init__(self, userNum, itemNum, behavior, behavior_mats): 
        super(myModel, self).__init__()  
        
        self.userNum = userNum  # 31882
        self.itemNum = itemNum  # 31232
        self.behavior = behavior  #['pv', 'fav', 'cart', 'buy']
        self.behavior_mats = behavior_mats  # {0:{'A':tensor(indices=tensor..parse_coo),'AT':tensor(indices=tensor..parse_coo),'A_ori':None}, {1:{}},{2:{}},{3:{}}}
        self.hidden_dim = args.hidden_dim 
        self.output_dim = args.hidden_dim 
        self.embedding_dim = args.hidden_dim 
        self.embedding_dict = self.init_embedding() #{'user_embedding':None, 'item_embedding':None,'user_embeddings':None, 'item_embeddings':None,  }
        self.weight_dict = self.init_weight()
        # self.attn = AttentionModel(self.behavior, self.userNum, self.itemNum, self.embedding_dim, self.hidden_dim)  #userNum, itemNum, embedding_dim, hidden_dim
        self.gru = GRUModel(self.behavior, self.userNum, self.itemNum, self.userNum, self.hidden_dim, self.hidden_dim)
        self.gcn = GCN(self.userNum, self.itemNum, self.behavior, self.behavior_mats)
#  userNum, itemNum, input_size, hidden_dim, output_dim

# TODO: [ok] gru의 인풋 찾아서 넣고, gcn의 들어갈 수있게 output size고려해

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
        # for i in range(len(self.behavior)):
        #     user_gru_data = pickle.load(open('/home/joo/JOOCML/data/Tmall/{}_minus_gru_padding.pkl'.format(i), 'rb')).to(device) 
        # attn_user_embeddings = self.attn()

        # 회복 
        gru_user_embeddings = self.gru() # torch.Size([31882, 16])

        user_embed, item_embed, user_embeds, item_embeds = self.gcn()  #torch.Size([31882, 16])  torch.Size([31232, 16]) torch.Size([4, 31882, 16])  torch.Size([4, 31232, 16])
        
        # TODO GRU, GCN FUSION CODE
        # 1) CONCAT GRU, GCN USER EMBEDS MANNER
        # gru_user_embeddings =gru_user_embeddings.unsqueeze(3)  #torch.Size([4, 31882, 16, 1])
        # user_embeds=user_embeds.unsqueeze(3)   # torch.Size([4, 31882, 16, 1])
        # concat_user_embeds = torch.cat((gru_user_embeddings,user_embeds), dim=3) #torch.Size([4, 31882, 16,2])
        # user_embeds =torch.mean(concat_user_embeds, dim=3)   #torch.Size([4, 31882,16])
        
        # 2) MLP GRU, GCN USER EMBEDS MANNER
        mlp = nn.Sequential(
                nn.Linear(16 * 2, 16),  
                nn.ReLU()  
             ).to(device)
        

        concatenated_tensor = torch.cat((gru_user_embeddings, user_embeds), dim=2).to(device)  # torch.Size([4, 31882, 32])
        # concatenated_tensor = torch.cat((attn_user_embeddings, user_embeds), dim=2).to(device)  # torch.Size([4, 31882, 32])
        user_embeds = mlp(concatenated_tensor).to(device)  #torch.Size([4, 31882, 16])

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
         


class GCN(nn.Module):
    def __init__(self, userNum, itemNum, behavior, behavior_mats):
        super(GCN, self).__init__()  
        self.userNum = userNum  #22438
        self.itemNum = itemNum  #35573
        self.hidden_dim = args.hidden_dim # 16

        self.behavior = behavior
        self.behavior_mats = behavior_mats
        

        self.user_embedding, self.item_embedding = self.init_embedding()         
        
        self.alpha, self.i_concatenation_w, self.u_concatenation_w, self.i_input_w, self.u_input_w = self.init_weight()
        
        self.sigmoid = torch.nn.Sigmoid()
        self.act = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(args.drop_rate)

        self.gnn_layer = eval(args.gnn_layer)
        self.layers = nn.ModuleList()
        for i in range(0, len(self.gnn_layer)):   # [16, 16, 16]
            self.layers.append(GCNLayer(args.hidden_dim, args.hidden_dim, self.userNum, self.itemNum, self.behavior, self.behavior_mats))  
    


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


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, userNum, itemNum, behavior, behavior_mats): # (16,16,22438, 35573,)
        super(GCNLayer, self).__init__()

        self.behavior = behavior
        self.behavior_mats = behavior_mats
        

        self.userNum = userNum
        self.itemNum = itemNum
        self.hidden_dim = args.hidden_dim 
        # self.input_size = max_len

        self.act = torch.nn.Sigmoid()
        self.i_w = nn.Parameter(torch.Tensor(in_dim, out_dim)) #shape:torch.Size([16, 16]) device: device(type='cpu')
        self.u_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.ii_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        init.xavier_uniform_(self.i_w)
        init.xavier_uniform_(self.u_w)


    def forward(self, user_embedding, item_embedding): # 여기의 user_embedding을 gru거친 user_embedding으로 바꿔

        user_embedding_list = [None]*len(self.behavior)
        item_embedding_list = [None]*len(self.behavior)
        gru_u_embed_list = [None]*len(self.behavior)
        gru_i_embed_list = [None]*len(self.behavior)
        
        
        for i in range(len(self.behavior)):

            user_behavior_mat = self.behavior_mats[i]['A'].clone().detach()    #shape: torch.Size([31882, 31232])
            # item_behavior_mat = self.behavior_mats[i]['AT'].clone().detach()   #shape:torch.Size([31232, 31882])

            # GCN item, user embedding
            user_embedding_list[i] = torch.spmm(self.behavior_mats[i]['A'], item_embedding) # torch.Size([31882, 16])
            item_embedding_list[i] = torch.spmm(self.behavior_mats[i]['AT'], user_embedding) # torch.Size([31232, 16])

        
        user_embeddings = torch.stack(user_embedding_list, dim=0)  #shape:torch.Size([4, 31882, 16])  
        item_embeddings = torch.stack(item_embedding_list, dim=0)  #shape:torch.Size([4, 31232, 16])      

        user_embedding = self.act(torch.matmul(torch.mean(user_embeddings, dim=0), self.u_w)) # shape:torch.Size([31882, 16])
        item_embedding = self.act(torch.matmul(torch.mean(item_embeddings, dim=0), self.i_w)) # shape:torch.Size([31232, 16])
        user_embeddings = self.act(torch.matmul(user_embeddings, self.u_w)) #shape:torch.Size([4, 31882, 16])
        item_embeddings = self.act(torch.matmul(item_embeddings, self.i_w)) #shape:torch.Size([4, 31232, 16])

        return user_embedding, item_embedding, user_embeddings, item_embeddings             

#------------------------------------------------------------------------------------------------------------------------------------------------


