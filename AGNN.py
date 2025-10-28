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
torch.backends.cudnn.enabled = False

device = torch.device(f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu")

    
def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.to(device)

    return Variable(x, requires_grad=requires_grad)



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
            user_attn_data = pickle.load(open('/home/joo/JOOCML/data/Tmall/{}_cpu_gru_padding.pkl'.format(i), 'rb')).to(device)
            user_embedded = self.user_embedding(user_attn_data)  
            attn_weights = torch.softmax(self.attn(user_embedded), dim=1) 
            user_embedding = torch.sum(attn_weights * user_embedded, dim=1)  
            attn_u_embed_list[i] = self.fc(user_embedding)  
        attn_user_embeddings = torch.stack(attn_u_embed_list, dim=0).to(device) 
        
        return attn_user_embeddings  


# # Gru 모델 정의
class GRUModel(nn.Module):
    def __init__(self, behavior, userNum, itemNum, input_size, hidden_dim, output_dim):
        super(GRUModel, self).__init__()
        self.behavior = behavior 
        self.userNum = userNum  
        self.itemNum = itemNum
        self.hidden_dim = args.hidden_dim 
        self.output_dim = args.hidden_dim 
        self.user_embedding_table =  torch.nn.Embedding(self.itemNum, self.hidden_dim).to(device)  
        
        self.gru = nn.GRU(input_size, hidden_dim, batch_first=True)  
        self.fc = nn.Linear(hidden_dim, userNum)
        
    def forward(self):
        gru_u_embed_list = [None]*len(self.behavior)
        gru_i_embed_list = [None]*len(self.behavior)
        for i in range(len(self.behavior)):    
            user_gru_data = pickle.load(open('/home/joo/JOOCML/data/Tmall/{}_cpu_gru_padding.pkl'.format(i), 'rb')).to(device)
            gru_user_embeddings= self.user_embedding_table(user_gru_data).to(device)                                   
            gru_u_embed = gru_user_embeddings.transpose(0, 2)  
            gru_u_output, _ = self.gru(gru_u_embed)  
            gru_u_embed_squeeze = gru_u_output.squeeze(0) 
            fc_output = self.fc(gru_u_embed_squeeze) 
            trans_output = fc_output.transpose(0, 2)  

            ## gru  connection 
            connect_block = gru_user_embeddings +trans_output
            gru_u_embed_list[i] = connect_block[:, -1, :]  

        gru_user_embeddings = torch.stack(gru_u_embed_list, dim=0).to(device) 

        return gru_user_embeddings
    
    

    
class myModel(nn.Module):
    def __init__(self, userNum, itemNum, behavior, behavior_mats): 
        super(myModel, self).__init__()  
        
        self.userNum = userNum  
        self.itemNum = itemNum  
        self.behavior = behavior  
        self.behavior_mats = behavior_mats  
        self.hidden_dim = args.hidden_dim 
        self.output_dim = args.hidden_dim 
        self.embedding_dim = args.hidden_dim 
        self.embedding_dict = self.init_embedding() 
        self.weight_dict = self.init_weight()
        self.gru = GRUModel(self.behavior, self.userNum, self.itemNum, self.userNum, self.hidden_dim, self.hidden_dim)
        self.gcn = GCN(self.userNum, self.itemNum, self.behavior, self.behavior_mats)


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
        gru_user_embeddings = self.gru()
        user_embed, item_embed, user_embeds, item_embeds = self.gcn()  

        # GRU, GCN FUSION CODE
        # 1) CONCAT GRU, GCN USER EMBEDS MANNER
        # gru_user_embeddings =gru_user_embeddings.unsqueeze(3) 
        # user_embeds=user_embeds.unsqueeze(3) 
        # concat_user_embeds = torch.cat((gru_user_embeddings,user_embeds), dim=3) 
        # user_embeds =torch.mean(concat_user_embeds, dim=3) 
        
        # 2) MLP GRU, GCN USER EMBEDS MANNER
        mlp = nn.Sequential(
                nn.Linear(16 * 2, 16),  
                nn.ReLU()  
             ).to(device)
        

        concatenated_tensor = torch.cat((gru_user_embeddings, user_embeds), dim=2).to(device)  
        user_embeds = mlp(concatenated_tensor).to(device) 

        return user_embed, item_embed, user_embeds, item_embeds 



class GCN(nn.Module):
    def __init__(self, userNum, itemNum, behavior, behavior_mats):
        super(GCN, self).__init__()  
        self.userNum = userNum 
        self.itemNum = itemNum  
        self.hidden_dim = args.hidden_dim 

        self.behavior = behavior
        self.behavior_mats = behavior_mats
        

        self.user_embedding, self.item_embedding = self.init_embedding()         
        
        self.alpha, self.i_concatenation_w, self.u_concatenation_w, self.i_input_w, self.u_input_w = self.init_weight()
        
        self.sigmoid = torch.nn.Sigmoid()
        self.act = torch.nn.PReLU()
        # self.act = torch.nn.ReLU()
        # self.act = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(args.drop_rate)

        self.gnn_layer = eval(args.gnn_layer)
        self.layers = nn.ModuleList()
        for i in range(0, len(self.gnn_layer)): 
            self.layers.append(GCNLayer(args.hidden_dim, args.hidden_dim, self.userNum, self.itemNum, self.behavior, self.behavior_mats))  
    

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


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, userNum, itemNum, behavior, behavior_mats):
        super(GCNLayer, self).__init__()

        self.behavior = behavior
        self.behavior_mats = behavior_mats
        

        self.userNum = userNum
        self.itemNum = itemNum
        self.hidden_dim = args.hidden_dim 


        self.act = torch.nn.Sigmoid()
        self.i_w = nn.Parameter(torch.Tensor(in_dim, out_dim)) 
        self.u_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.ii_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        init.xavier_uniform_(self.i_w)
        init.xavier_uniform_(self.u_w)
   

        
 
    def forward(self, user_embedding, item_embedding): 

        user_embedding_list = [None]*len(self.behavior)
        item_embedding_list = [None]*len(self.behavior)
        gru_u_embed_list = [None]*len(self.behavior)
        gru_i_embed_list = [None]*len(self.behavior)
        
        # GCN item, user embedding        
        for i in range(len(self.behavior)):
            user_embedding_list[i] = torch.spmm(self.behavior_mats[i]['A'].to(device) , item_embedding)
            # print(user_embedding_list[i].shape)
            item_embedding_list[i] = torch.spmm(self.behavior_mats[i]['AT'].to(device), user_embedding) 
            # print(item_embedding_list[i].shaped)
  
        user_embeddings = torch.stack(user_embedding_list, dim=0)   
        item_embeddings = torch.stack(item_embedding_list, dim=0)  
        user_embedding = self.act(torch.matmul(torch.mean(user_embeddings, dim=0), self.u_w)) 
        item_embedding = self.act(torch.matmul(torch.mean(item_embeddings, dim=0), self.i_w)) 

        user_embeddings = self.act(torch.matmul(user_embeddings, self.u_w))
        item_embeddings = self.act(torch.matmul(item_embeddings, self.i_w)) 
        return user_embedding, item_embedding, user_embeddings, item_embeddings             



