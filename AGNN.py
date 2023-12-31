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
# # device = torch.device("cuda:1")
# print('Device:', device)

    
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
            # trans = user_embedded.transpose(0, 2) # torch.Size([16, 293, 31882])
            attn_weights = torch.softmax(self.attn(user_embedded), dim=1) # torch.Size([31882, 293, 16])
            # user_embedding = attn_weights.unsqueeze(3)
            user_embedding = torch.sum(attn_weights * user_embedded, dim=1)  # torch.Size([31882, 16])
            attn_u_embed_list[i] = self.fc(user_embedding)  #torch.Size([31882, 16])
        attn_user_embeddings = torch.stack(attn_u_embed_list, dim=0).to(device) #torch.Size([4, 31882, 16])
        
        return attn_user_embeddings  #, attn_weights


# # Gru 모델 정의
class GRUModel(nn.Module):
    def __init__(self, behavior, userNum, itemNum, input_size, hidden_dim, output_dim):
    # def __init__(self.userNum, self.embedding_size, self.userNum) 
        super(GRUModel, self).__init__()
        # self.behavior_mats = behavior_mats self.userNum, self.itemNum,
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
                    


# # # Gru 모델 정의
# class GRUModel(nn.Module):
#     def __init__(self, userNum, itemNum,input_size, hidden_dim, output_dim):
#         super(GRUModel, self).__init__()
#         # self.behavior_mats = behavior_mats self.userNum, self.itemNum,
#         self.userNum = userNum  #22438
#         self.itemNum = itemNum
#         self.user_embedding_table =  torch.nn.Embedding(self.itemNum, self.embedding_size).to(device)  # Embedding(31882, 16)
#         # self.user_gru_model = GRUModel(self.userNum, self.embedding_size, self.userNum) 
#         self.gru = nn.GRU(input_size, hidden_dim, batch_first=True)  # (batch, seq, feature)
#         # self.user_gru = nn.GRU(userNum, hidden_dim, batch_first=True)
#         # self.item_gru = nn.GRU(itemNum, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)
        
#     def forward(self, x):
#         gru_u_embed_list = [None]*len(self.behavior)
#         gru_i_embed_list = [None]*len(self.behavior)
#         for i in range(len(self.behavior)):
#             user_gru_data = pickle.load(open('/home/joo/CML/data/Tmall/{}_minus_gru_padding.pkl'.format(i), 'rb')).to(device)  
#                 # print("gru_BEH")          
#                 #  gru_user input data embedding
#             # else:
#             #     result = pickle.load(open('/home/joo/JOOCML/data/Tmall/{}_META_gru_padding.pkl'.format(i), 'rb')).to(device)
#             #     # print("gru_METAPATH")
#             gru_user_embeddings= self.user_embedding_table(user_gru_data).to(device) #shape:torch.Size([4, 31882, 16])   shape:torch.Size([31882, 63, 16])     
#             # print('Device:', device)   
#                 # print("gru_user_embeddings", gru_user_embeddings.shape)   
#             # gru_user_embeddings= self.user_embedding_table(result).cuda()   
#             # print("gru 시작")
                       
#             gru_u_embed_list[i] = self.user_gru_model(gru_user_embeddings) 

#             # output, _ = self.gru(x.unsqueeze(0))  
#             # print(self.gru)  # GRU(31882, 16, batch_first=True)  --> input tensor of shape (16, 293, 31882) 
#             # print(x.shape)   # torch.Size([31882, 293, 16])   
#             x = x.transpose(0, 2)   #ro__torch.Size([2174, 3808, 16])
#             # print("x:", x.shape)    # x: torch.Size([16, 293, 31882])
#                                                 # CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)` -->input_size ;31882로 해놓고 31882 이상인 777777로 패딩해서 에러
#             output, _ = self.gru(x)             
#             # print(output.shape)   # torch.Size([16, 293, 16])
#             output = output.squeeze(0)
#             fc_output = self.fc(output)  ## gru_u_embed_list[i] : torch.Size([16, 293, 31882])  
#             trans_output = fc_output.transpose(0, 2)   # shape:torch.Size([31882, 293, 16])
            
#             # [DONE!]TODO 여기서 torch.Size를 ([31882, 16]) 으로 바꿔야함.
#             # TODO 정보손실 없이 torch.Size([31882, 293, 16])를 torch.Size([31882, 16])로 차원을 줄이는 방법
#             # output = trans_output.squeeze(dim=1) 
#             output = trans_output[:, -1, :] 
#             # 
#             # print("output.shape:", output.shape)   #  output.shape: torch.Size([31882, 293, 16])
#             return output
    
    

        


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

        # self.user_embedding_table =  torch.nn.Embedding(self.userNum, self.embedding_size).to('cuda:1')
        # self.user_embedding_table =  torch.nn.Embedding(self.itemNum, self.embedding_size).to(device)  # Embedding(31882, 16)
        # print("self.user_embedding_table :" ,self.user_embedding_table )
        # self.item_embedding_table =  torch.nn.Embedding(self.itemNum, self.embedding_size).to(device) # Embedding(31232, 16)
        # self.gru_model = GRUModel(in_dim, self.embedding_size, out_dim) 
        # self.user_gru_model = GRUModel(self.userNum, self.embedding_size, self.userNum)  
                    #fc:Linear(in_features=16, out_features=16, bias=True)
                    
                    #gru: GRU(31882, 16, batch_first=True)
        # self.item_gru_model = GRUModel(self.itemNum, self.embedding_size, out_dim) 
        

        
 
    def forward(self, user_embedding, item_embedding): # 여기의 user_embedding을 gru거친 user_embedding으로 바꿔

        user_embedding_list = [None]*len(self.behavior)
        item_embedding_list = [None]*len(self.behavior)
        gru_u_embed_list = [None]*len(self.behavior)
        gru_i_embed_list = [None]*len(self.behavior)
        
        
        for i in range(len(self.behavior)):
            # print(i)
            # print("시작")
#######################################################################################################################################################################################            
            # GRU_ USER PERSONALIZED item interation pattern FEATURE
            user_behavior_mat = self.behavior_mats[i]['A'].clone().detach()    #shape: torch.Size([31882, 31232])
            # item_behavior_mat = self.behavior_mats[i]['AT'].clone().detach()   #shape:torch.Size([31232, 31882])

            # TODO item_indices의 max len구해서 패딩 후 embedding하고 gru 보내기 
            # TODO 각 USER의 대한 ITEM 패턴 구하기  
            # TODO 이후 GCN에 넣어 지금 sparse matrix를 dense로 만들었으니까 gru끝나고 다시 sparse matrix로 만들기

######################################################################################################################################################################################## 
# TODO gru padding 전처리 picklefile 만들기
       
            # user_item_indices = []
            # max_length = 0
            # item_indices_max = 0 
            # for user_id in range(self.userNum):
            #     # print("user_id:",user_id)
            #     user_indices = (user_behavior_mat._indices()[0] == user_id).nonzero().squeeze()
            #     item_indices = user_behavior_mat._indices()[1][user_indices]                      
            #     max_length = max(max_length, item_indices.numel())    #  item_indices를 Python 스칼라로 변환하기 전에 요소의 수를 구해야함.
            # #                                                         # int() 함수는 하나의 스칼라 값만을 변환할 수 있기 때문에, 길이가 1보다 큰 텐서를 int() 함수로 변환할 수 없습니다.
            # #                                                         # item_indices가 텐서인 경우, len() 함수는 텐서의 요소 수를 반환하지 않고, 텐서의 모양(shape) 정보를 반환하게 됩니다.
            # #                                                         # item_indices가 텐서일 경우 len(item_indices) 대신 item_indices.numel()을 사용하여 텐서의 요소 수를 구할 수 있음.
            
            #     # print("item_indices:", item_indices.shape)
            # #     # item 인덱스를 dense 텐서로 변환하고 reshape
            #     item_indices_tensor = torch.tensor(item_indices)
            #     # print("notunsqeeze_item_indices_tensor:", item_indices_tensor.shape) 
                
            #     #TODO 패딩 후 임베딩
            #     item_indices_tensor = item_indices_tensor.unsqueeze(0)
            #     # print("item_indices_tensor:", item_indices_tensor.shape) #torch.Size([1, 35])

                
            #     # gru embedding input_size보다 작은 수로 padding하기 위해 item id의 최대값 찾아서 [최대값+1]로 패딩
            #     if item_indices.numel() == 0  :
            #         pass
            #     else :
            #         item_indices_max = torch.max(item_indices_tensor).item()
          
            # #     # 텐서를 리스트에 추가
            #     user_item_indices.append(item_indices_tensor)
   
            # # print("max_length:", max_length)   # max_length: 293
            # # print("item_indices_max:", item_indices_max)   # item_indices_max: 29481
            
            #  # 각 user id의 item개수만큼 max_length padding     
            #  # gru embedding input_size보다 작은 수로 padding.          
            # padding_value = item_indices_max + 1  # padding
            # padding_value = torch.tensor([padding_value]).to(item_indices.device)    
            
            
            # padded_item_indices = []
            
            # for item_indices in user_item_indices:
            #     # while len(item_indices) < max_length:
            #         # padded_item = item_indices.append(padding_value)
            #     if len(item_indices.shape) == 1 :
            #         item_indices = item_indices.unsqueeze(0)
                    
                    
 
            #     padding_tensor = torch.tensor([padding_value] * (max_length - item_indices.numel())).unsqueeze(0).to(item_indices.device) 
            #     padded_item = torch.cat((item_indices, padding_tensor), 1)
            #     # padded_item = torch.cat((item_indices, padding_value.unsqueeze(0)), 1)
            #     padded_item_tensor = torch.tensor(padded_item)                                          # clone() 메서드와 detach() 메서드를 사용하여 텐서를 복사할 수 있음
            #                                                                                             # clone() 메서드는 원본 텐서와 동일한 크기와 데이터를 가진 새로운 텐서를 생성
            #                                                                                             # detach() 메서드는 그래디언트 계산을 중단하고 새로운 텐서를 생성
            #                                                                                             # requires_grad_() 메서드를 사용하여 새로운 텐서의 requires_grad 속성을 True로 설정할 수 있음
            #     # print(padded_item_tensor.shape)   # torch.Size([1, 293])
            #     padded_item_indices.append(padded_item_tensor)

            
            # # print("padded_item_tensor.shape:", padded_item_tensor.shape)  # torch.Size([1, 293])
            # # print("padded_item_indices:", len(padded_item_indices)) #padded_item_indices: 31882
            # result = torch.cat(padded_item_indices, dim=0).long()
            # # print("padded_item_indices_cat_result:",result.shape)  #torch.Size([31882, 293])
            # with open('/home/joo/CML/data/retail_rocket/{}_retailrocket_gru_padding.pkl'.format(i), 'wb') as file:    # james.p 파일을 바이너리 쓰기 모드(wb)로 열기
            #     pickle.dump(result, file)
            #     print(file)
##########################################################################################################################################################
            # if args.isJustbeh == True:  
            # result = pickle.load(open('/home/joo/CML/data/Tmall/{}_gru_padding.pkl'.format(i), 'rb')).to(device)  
            #     # print("gru_BEH")          
            #     #  gru_user input data embedding
            # # else:
            # #     result = pickle.load(open('/home/joo/JOOCML/data/Tmall/{}_META_gru_padding.pkl'.format(i), 'rb')).to(device)
            # #     # print("gru_METAPATH")
            # gru_user_embeddings= self.user_embedding_table(result).to(device) #shape:torch.Size([4, 31882, 16])   shape:torch.Size([31882, 63, 16])     
            # # print('Device:', device)   
            #     # print("gru_user_embeddings", gru_user_embeddings.shape)   
            # # gru_user_embeddings= self.user_embedding_table(result).cuda()   
            # # print("gru 시작")
                       
            # gru_u_embed_list[i] = self.user_gru_model(gru_user_embeddings) 
            # print("gru_u_embed_list[i] :", gru_u_embed_list[i].shape)  # gru_u_embed_list[i] :  torch.Size([31882, 16])
            # print(gru_u_embed_list[i])
            
##############################################################################################################################     


            # GCN item, user embedding
            user_embedding_list[i] = torch.spmm(self.behavior_mats[i]['A'], item_embedding) # torch.Size([31882, 16])
            # print(user_embedding_list[i].shape)
            item_embedding_list[i] = torch.spmm(self.behavior_mats[i]['AT'], user_embedding) # torch.Size([31232, 16])
            # print(item_embedding_list[i].shape)
        
        # ### 여기 중요 삭제 금지  #####
        # # # gru user embedding + gcn user embedding 과정
        # gcn_init_user_embeddings = torch.stack(user_embedding_list, dim=0)  #shape:torch.Size([4, 31882, 16])  dtype:torch.float32
        # gru_user_embeddings = torch.stack(gru_u_embed_list, dim=0)  #shape:torch.Size([4, 31882, 16])  dtype:torch.float32
        # user_embeddings = torch.mul(gcn_init_user_embeddings, gru_user_embeddings)  # dtype:torch.float32  shape:torch.Size([4, 31882, 16])
        # # print("torch.mul_user_embeddings.shape:", user_embeddings.shape)   #torch.mul_user_embeddings.shape: torch.Size([4, 31882, 16])

        ##################### try1 #####################
        # user_embeddings= torch.stack(gru_u_embed_list, dim=0) 



        #### 여기 중요 삭제 금지  #####

        
        user_embeddings = torch.stack(user_embedding_list, dim=0)  #shape:torch.Size([4, 31882, 16])  
        item_embeddings = torch.stack(item_embedding_list, dim=0)  #shape:torch.Size([4, 31232, 16])      
        # print(" item_embeddings:",  item_embeddings.shape)  # torch.Size([4, 31882, 16])
       
       
       
        # print("gru_stack 완성")
        user_embedding = self.act(torch.matmul(torch.mean(user_embeddings, dim=0), self.u_w)) # shape:torch.Size([31882, 16])
        item_embedding = self.act(torch.matmul(torch.mean(item_embeddings, dim=0), self.i_w)) # shape:torch.Size([31232, 16])
        # gru_user_embedding = self.act(torch.matmul(torch.mean(gru_user_embeddings, dim=0), self.u_w)) # shape:torch.Size([31882, 16])
        # gru_item_embedding = self.act(torch.matmul(torch.mean(gru_item_embeddings, dim=0), self.i_w)) # shape:torch.Size([31232, 16])     
        # print("gru_stack_mean:", gru_user_embedding.shape) #  torch.Size([31882, 16])
        
        # all_grucn_u_emb =torch.concat((user_embedding.unsqueeze(0), gru_user_embedding.unsqueeze(0)), dim=0)  #shape:torch.Size([2, 31882, 16])
        # mean_all_grucn_u_emb = torch.mean(all_grucn_u_emb, dim=0)  # shape:torch.Size([31882, 16])
        
        # all_grucn_i_emb =torch.concat(item_embedding, gru_item_embedding)
        # print("all_grc_i_emb;", all_grucn_i_emb.shape)
        # print("user_embedding.shape;", user_embedding.shape)   # torch.Size([31882, 16])
        # print("item_embedding.shape;", item_embedding.shape)  # torch.Size([31232, 16])
        
        
        #self.act(torch.matmul(mean_all_grucn_u_emb, self.u_w)): torch.Size([31882, 16])
        # user_embeddings = self.act(torch.matmul(mean_all_grucn_u_emb, self.u_w)) #shape:torch.Size([4, 31882, 16])
        # print("self.act(torch.matmul(mean_all_grucn_u_emb, self.u_w)):", user_embeddings.shape)
        user_embeddings = self.act(torch.matmul(user_embeddings, self.u_w)) #shape:torch.Size([4, 31882, 16])
        item_embeddings = self.act(torch.matmul(item_embeddings, self.i_w)) #shape:torch.Size([4, 31232, 16])
        # print("user_embeddings:", user_embeddings.shape)  # torch.Size([4, 31882, 16])
        # print("item_embeddings:", item_embeddings.shape)  # torch.Size([4, 31232, 16])
        # print("gru_user_embedding_ok!")

        return user_embedding, item_embedding, user_embeddings, item_embeddings             

#------------------------------------------------------------------------------------------------------------------------------------------------


