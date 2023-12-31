import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init

from torch.nn import init  
from Params import args
from tqdm import tqdm
import numpy as np



def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaWeightNet(nn.Module):
    def __init__(self, beh_num):
        super(MetaWeightNet, self).__init__()

        self.beh_num = beh_num

        self.sigmoid = torch.nn.Sigmoid()
        self.act = torch.nn.LeakyReLU(negative_slope=args.slope)  
        self.prelu = torch.nn.PReLU()
        self.relu = torch.nn.ReLU()
        self.tanhshrink = torch.nn.Tanhshrink()
        self.dropout7 = torch.nn.Dropout(args.drop_rate)
        self.dropout5 = torch.nn.Dropout(args.drop_rate1)
        self.batch_norm = torch.nn.BatchNorm1d(1)

        initializer = nn.init.xavier_uniform_


        self.SSL_layer1 = nn.Linear(args.hidden_dim*3, int((args.hidden_dim*3)/2))  # (48,24)
        self.SSL_layer2 = nn.Linear(int((args.hidden_dim*3)/2), 1)  #(24,1)
        self.SSL_layer3 = nn.Linear(args.hidden_dim*2, 1)  # (32,1)

        self.RS_layer1 = nn.Linear(args.hidden_dim*3, int((args.hidden_dim*3)/2)) # (48,24)
        self.RS_layer2 = nn.Linear(int((args.hidden_dim*3)/2), 1) #(24,1)
        self.RS_layer3 = nn.Linear(args.hidden_dim, 1) # (32,1)



        self.beh_embedding = nn.Parameter(initializer(torch.empty([beh_num, args.hidden_dim]))).cuda()
 

    def forward(self, infoNCELoss_list, behavior_loss_multi_list, user_step_index, user_index_list, user_embeds, user_embed):  
        
        infoNCELoss_list_weights = [None]*self.beh_num
        behavior_loss_multi_list_weights = [None]*self.beh_num
        # print("self.beh_num:",self.beh_num)
        for i in range(self.beh_num):
            
            

# #retailrocket--------------------------------------------------------------------------------------------------------------------------------------------------------------
#             ## encoded meta-knowledge
#             SSL_input = args.inner_product_mult*torch.cat((infoNCELoss_list[i].unsqueeze(1).repeat(1, args.hidden_dim)*args.inner_product_mult, user_embeds[i][user_step_index]), 1)  #[]  [1024, 16]
#             SSL_input = args.inner_product_mult*torch.cat((SSL_input, user_embed[user_step_index]), 1)
#             
#             infoNCELoss_list_weights[i] = self.dropout7(self.prelu(self.SSL_layer1(SSL_input)))
#             infoNCELoss_list_weights[i] = np.sqrt(SSL_input.shape[1])*self.dropout7(self.SSL_layer2(infoNCELoss_list_weights[i]).squeeze())
#             infoNCELoss_list_weights[i] = self.batch_norm(infoNCELoss_list_weights[i].unsqueeze(1)).squeeze()
#             infoNCELoss_list_weights[i] = args.inner_product_mult*self.sigmoid(infoNCELoss_list_weights[i])

#             ## weighting function 
#             SSL_input3 = args.inner_product_mult*((infoNCELoss_list[i].unsqueeze(1).repeat(1, args.hidden_dim*2))*torch.cat((user_embeds[i][user_step_index],user_embed[user_step_index]), 1))
#             SSL_weight3 = self.dropout7(self.prelu(self.SSL_layer3(SSL_input3)))
#             SSL_weight3 = self.batch_norm(SSL_weight3).squeeze()
#             SSL_weight3 = args.inner_product_mult*self.sigmoid(SSL_weight3)
#             infoNCELoss_list_weights[i] = (infoNCELoss_list_weights[i] + SSL_weight3)/2


#            ====================================================================================================================================================================
#             # BPR loss
#             RS_input = args.inner_product_mult*torch.cat((behavior_loss_multi_list[i].unsqueeze(1).repeat(1, args.hidden_dim)*args.inner_product_mult, user_embed[user_index_list[i]]), 1)
#             RS_input = args.inner_product_mult*torch.cat((RS_input, user_embeds[i][user_index_list[i]]), 1)
                        
#             behavior_loss_multi_list_weights[i] = self.dropout7(self.prelu(self.RS_layer1(RS_input))) 
#             behavior_loss_multi_list_weights[i] = np.sqrt(RS_input.shape[1])*self.dropout7(self.RS_layer2(behavior_loss_multi_list_weights[i]).squeeze())
#             behavior_loss_multi_list_weights[i] = self.batch_norm(behavior_loss_multi_list_weights[i].unsqueeze(1))
#             behavior_loss_multi_list_weights[i] = args.inner_product_mult*self.sigmoid(behavior_loss_multi_list_weights[i]).squeeze()
            

#             RS_input3 = args.inner_product_mult*((behavior_loss_multi_list[i].unsqueeze(1).repeat(1, args.hidden_dim))*user_embed[user_index_list[i]])
#             RS_weight3 = self.dropout7(self.prelu(self.RS_layer3(RS_input3)))
#             RS_weight3 = self.batch_norm(RS_weight3).squeeze()
#             RS_weight3 = args.inner_product_mult*self.sigmoid(RS_weight3).squeeze()
#             behavior_loss_multi_list_weights[i] = behavior_loss_multi_list_weights[i] + RS_weight3
#         return infoNCELoss_list_weights, behavior_loss_multi_list_weights

#retailrocket--------------------------------------------------------------------------------------------------------------------------------------------------------------

#IJCAI,Tmall--------------------------------------------------------------------------------------------------------------------------------------------------------------
            ### TODO INPUT dtype 확인
        
#           ## encoded meta-knowledge            
            SSL_input = args.inner_product_mult*torch.cat((infoNCELoss_list[i].unsqueeze(1).repeat(1, args.hidden_dim)*args.inner_product_mult, user_embeds[i][user_step_index]), 1)  #[]  [1024, 16] ### tensor([], device='cuda:0', size=(0, 32), dtype=torch.float64
            SSL_input = args.inner_product_mult*torch.cat((SSL_input, user_embed[user_step_index]), 1).to(torch.float32)  # tensor([], device='cuda:0', size=(0, 48), grad_fn=<MulBackward0>

            infoNCELoss_list_weights[i] = self.dropout7(self.prelu(self.SSL_layer1(SSL_input)))  
            infoNCELoss_list_weights[i] = np.sqrt(SSL_input.shape[1])*self.dropout7(self.SSL_layer2(infoNCELoss_list_weights[i]).squeeze())
            infoNCELoss_list_weights[i] = self.batch_norm(infoNCELoss_list_weights[i].unsqueeze(1)).squeeze()
            infoNCELoss_list_weights[i] = args.inner_product_mult*self.sigmoid(infoNCELoss_list_weights[i])

#           ## weighting function             
            SSL_input3 = args.inner_product_mult*((infoNCELoss_list[i].unsqueeze(1).repeat(1, args.hidden_dim*2))*torch.cat((user_embeds[i][user_step_index],user_embed[user_step_index]), 1)).to(torch.float32)
            SSL_weight3 = self.dropout7(self.prelu(self.SSL_layer3(SSL_input3)))
            SSL_weight3 = self.batch_norm(SSL_weight3).squeeze()   # tensor([], device='cuda:0', grad_fn=<SqueezeBackward0>)
            SSL_weight3 = args.inner_product_mult*self.sigmoid(SSL_weight3)
            infoNCELoss_list_weights[i] = (infoNCELoss_list_weights[i] + SSL_weight3)/2

#           #==============================================================================================================================================================================
#           ## BPR loss
            RS_input = args.inner_product_mult*torch.cat((behavior_loss_multi_list[i].unsqueeze(1).repeat(1, args.hidden_dim)*args.inner_product_mult, user_embed[user_index_list[i]]), 1).to(torch.float32)   # user_embed torch.Size([31882, 16])         
            RS_input = args.inner_product_mult*torch.cat((RS_input, user_embeds[i][user_index_list[i]]), 1)

            behavior_loss_multi_list_weights[i] = self.dropout7(self.prelu(self.RS_layer1(RS_input))) 
            behavior_loss_multi_list_weights[i] = np.sqrt(RS_input.shape[1])*self.dropout7(self.RS_layer2(behavior_loss_multi_list_weights[i]).squeeze(dim=-1))
            behavior_loss_multi_list_weights[i] = self.batch_norm(behavior_loss_multi_list_weights[i].unsqueeze(dim=1))
            behavior_loss_multi_list_weights[i] = args.inner_product_mult*self.sigmoid(behavior_loss_multi_list_weights[i]).squeeze()

            RS_input3 = args.inner_product_mult*((behavior_loss_multi_list[i].unsqueeze(1).repeat(1, args.hidden_dim))*user_embed[user_index_list[i]]).to(torch.float32)
            RS_weight3 = self.dropout7(self.prelu(self.RS_layer3(RS_input3)))
            RS_weight3 = self.batch_norm(RS_weight3).squeeze(dim = -1)
            RS_weight3 = args.inner_product_mult*self.sigmoid(RS_weight3).squeeze()
            behavior_loss_multi_list_weights[i] = behavior_loss_multi_list_weights[i] + RS_weight3
            
        return infoNCELoss_list_weights, behavior_loss_multi_list_weights
