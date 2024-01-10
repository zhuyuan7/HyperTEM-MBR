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
            if i == self.beh_num-1:
                args.inner_product_mult_last = args.inner_product_mult_last
            else:
                args.inner_product_mult_last = args.inner_product_mult

            
            

# #retailrocket--------------------------------------------------------------------------------------------------------------------------------------------------------------
#             ## encoded meta-knowledge
#             SSL_input = args.inner_product_mult*torch.cat((infoNCELoss_list[i].unsqueeze(1).repeat(1, args.hidden_dim)*args.inner_product_mult, user_embeds[i][user_step_index]), 1)  #[]  [1024, 16]
#             #[819,32]                                       [819,16]                                                                                [819,16] 
#             SSL_input = args.inner_product_mult*torch.cat((SSL_input, user_embed[user_step_index]), 1)
#             # torch.Size([819, 48])                                    user_embed[user_step_index] [819,16]   

#             # SSL_input3 = args.inner_product_mult*((infoNCELoss_list[i].unsqueeze(1).repeat(1, args.hidden_dim*2))*torch.cat((user_embeds[i][user_step_index],user_embed[user_step_index]), 1))
#             # # torch.Size([819, 32])
#             infoNCELoss_list_weights[i] = self.dropout7(self.prelu(self.SSL_layer1(SSL_input)))
#             #torch.Size([819, 24])                          [819, 24]              [819,24]
#             infoNCELoss_list_weights[i] = np.sqrt(SSL_input.shape[1])*self.dropout7(self.SSL_layer2(infoNCELoss_list_weights[i]).squeeze())
#             # torch.Size([819])
#             infoNCELoss_list_weights[i] = self.batch_norm(infoNCELoss_list_weights[i].unsqueeze(1)).squeeze()
#             # [819]                        [819,1]               [819,1]
#             infoNCELoss_list_weights[i] = args.inner_product_mult*self.sigmoid(infoNCELoss_list_weights[i])
#             # [819]  


#             ## weighting function 
#             SSL_input3 = args.inner_product_mult*((infoNCELoss_list[i].unsqueeze(1).repeat(1, args.hidden_dim*2))*torch.cat((user_embeds[i][user_step_index],user_embed[user_step_index]), 1))
#             # torch.Size([819, 32])
#             SSL_weight3 = self.dropout7(self.prelu(self.SSL_layer3(SSL_input3)))
#             # [819,1]
#             SSL_weight3 = self.batch_norm(SSL_weight3).squeeze()
#             # torch.Size([819])
#             SSL_weight3 = args.inner_product_mult*self.sigmoid(SSL_weight3)
#             # torch.Size([819])
#             infoNCELoss_list_weights[i] = (infoNCELoss_list_weights[i] + SSL_weight3)/2
#             # torch.Size([819])


# #==============================================================================================================================================================================
#             # BPR loss
#             RS_input = args.inner_product_mult*torch.cat((behavior_loss_multi_list[i].unsqueeze(1).repeat(1, args.hidden_dim)*args.inner_product_mult, user_embed[user_index_list[i]]), 1)
#             # torch.Size([8074, 32])                       # 8074,16                                                                                      8074, 16
#             RS_input = args.inner_product_mult*torch.cat((RS_input, user_embeds[i][user_index_list[i]]), 1)
#             # torch.Size([8074, 48])
                        
#             # RS_input3 = args.inner_product_mult*((behavior_loss_multi_list[i].unsqueeze(1).repeat(1, args.hidden_dim))*user_embed[user_index_list[i]])
#             # # torch.Size([8074, 16])
#             behavior_loss_multi_list_weights[i] = self.dropout7(self.prelu(self.RS_layer1(RS_input))) 
#             # 8074,24
#             behavior_loss_multi_list_weights[i] = np.sqrt(RS_input.shape[1])*self.dropout7(self.RS_layer2(behavior_loss_multi_list_weights[i]).squeeze())
#             # torch.Size([8074])
#             behavior_loss_multi_list_weights[i] = self.batch_norm(behavior_loss_multi_list_weights[i].unsqueeze(1))
#             # torch.Size([8074, 1])
#             behavior_loss_multi_list_weights[i] = args.inner_product_mult*self.sigmoid(behavior_loss_multi_list_weights[i]).squeeze()
#             #torch.Size([8074])
            

#             RS_input3 = args.inner_product_mult*((behavior_loss_multi_list[i].unsqueeze(1).repeat(1, args.hidden_dim))*user_embed[user_index_list[i]])
#             # torch.Size([8074, 16])
#             RS_weight3 = self.dropout7(self.prelu(self.RS_layer3(RS_input3)))
#             #torch.Size([8074,1])
#             RS_weight3 = self.batch_norm(RS_weight3).squeeze()
#             #torch.Size([8074])
#             RS_weight3 = args.inner_product_mult*self.sigmoid(RS_weight3).squeeze()
#             # torch.Size([8074])
#             behavior_loss_multi_list_weights[i] = behavior_loss_multi_list_weights[i] + RS_weight3
#             #torch.Size([8083])
#         return infoNCELoss_list_weights, behavior_loss_multi_list_weights

            # print("I=======================================================:", i)
            # SSL_input = args.inner_product_mult*torch.cat((infoNCELoss_list[i].unsqueeze(1).repeat(1, args.hidden_dim)*args.inner_product_mult, user_embeds[i][user_step_index]), 1)  #[]  [1024, 16]
            # print("1SSL:", SSL_input.shape) #torch.Size([0, 32])
            # SSL_input = args.inner_product_mult*torch.cat((SSL_input, user_embed[user_step_index]), 1)
            # print("2SSL:", SSL_input.shape) #torch.Size([0, 32])
            # SSL_input3 = args.inner_product_mult*((infoNCELoss_list[i].unsqueeze(1).repeat(1, args.hidden_dim*2))*torch.cat((user_embeds[i][user_step_index],user_embed[user_step_index]), 1))
            # print("3SSL:", SSL_input.shape) #torch.Size([0, 32])

            # infoNCELoss_list_weights[i] = self.dropout7(self.prelu(self.SSL_layer1(SSL_input)))
            # print("drop_info:",infoNCELoss_list_weights[i].shape )
            # infoNCELoss_list_weights[i] = np.sqrt(SSL_input.shape[1])*self.dropout7(self.SSL_layer2(infoNCELoss_list_weights[i]).squeeze())
            # print("np.sqrt_info:",infoNCELoss_list_weights[i].shape )           
            # infoNCELoss_list_weights[i] = self.batch_norm(infoNCELoss_list_weights[i].unsqueeze(1)).squeeze()
            # print("batch_norm_info:",infoNCELoss_list_weights[i].shape )
            # infoNCELoss_list_weights[i] = args.inner_product_mult*self.sigmoid(infoNCELoss_list_weights[i])
            # print("inner_product_mult_info:",infoNCELoss_list_weights[i].shape )
            
            # SSL_weight3 = self.dropout7(self.prelu(self.SSL_layer3(SSL_input3)))
            # print("dropout7 SSL_weight3:", SSL_weight3.shape) #torch.Size([0, 32])
            # SSL_weight3 = self.batch_norm(SSL_weight3).squeeze()
            # print("batch_norm SSL_weight3:", SSL_weight3.shape) 
            # SSL_weight3 = args.inner_product_mult*self.sigmoid(SSL_weight3)
            # print("inner_product_mult SSL_weight3:", SSL_weight3.shape) 
            # infoNCELoss_list_weights[i] = (infoNCELoss_list_weights[i] + SSL_weight3)/2
            # print("infoNCELoss_list_weights info:",infoNCELoss_list_weights[i].shape )

            # RS_input = args.inner_product_mult*torch.cat((behavior_loss_multi_list[i].unsqueeze(1).repeat(1, args.hidden_dim)*args.inner_product_mult, user_embed[user_index_list[i]]), 1)
            # print("1RS_input:", RS_input.shape)
            # RS_input = args.inner_product_mult*torch.cat((RS_input, user_embeds[i][user_index_list[i]]), 1)
            # print("2RS_input:", RS_input.shape)
            # RS_input3 = args.inner_product_mult*((behavior_loss_multi_list[i].unsqueeze(1).repeat(1, args.hidden_dim))*user_embed[user_index_list[i]])
            # print("3RS_input:", RS_input.shape)

            # behavior_loss_multi_list_weights[i] = self.dropout7(self.prelu(self.RS_layer1(RS_input))) 
            # print("dropout7_behavior_loss_multi_list_weights[i] :",behavior_loss_multi_list_weights[i].shape)
            # behavior_loss_multi_list_weights[i] = np.sqrt(RS_input.shape[1])*self.dropout7(self.RS_layer2(behavior_loss_multi_list_weights[i]).squeeze())
            # print("np.sqrt_behavior_loss_multi_list_weights[i] :",behavior_loss_multi_list_weights[i].shape)
            # behavior_loss_multi_list_weights[i] = self.batch_norm(behavior_loss_multi_list_weights[i].unsqueeze(1))
            # print("batch_norm_behavior_loss_multi_list_weights[i] :",behavior_loss_multi_list_weights[i].shape)
            # behavior_loss_multi_list_weights[i] = args.inner_product_mult*self.sigmoid(behavior_loss_multi_list_weights[i]).squeeze()
            # print("inner_product_mult_behavior_loss_multi_list_weights[i] :",behavior_loss_multi_list_weights[i].shape)
            
            # RS_weight3 = self.dropout7(self.prelu(self.RS_layer3(RS_input3)))
            # print("dropout7_RS_weight3:",RS_weight3.shape)
            # RS_weight3 = self.batch_norm(RS_weight3).squeeze()
            # print("batch_norm_RS_weight3:",RS_weight3.shape)
            # RS_weight3 = args.inner_product_mult*self.sigmoid(RS_weight3).squeeze()
            # print("inner_product_mult_RS_weight3:",RS_weight3.shape)

            # behavior_loss_multi_list_weights[i] = behavior_loss_multi_list_weights[i] + RS_weight3
            # print("final_RS_weight3:",behavior_loss_multi_list_weights[i].shape) 
#retailrocket--------------------------------------------------------------------------------------------------------------------------------------------------------------

#IJCAI,Tmall--------------------------------------------------------------------------------------------------------------------------------------------------------------
            ### TODO INPUT dtype 확인
        
            
            SSL_input = args.inner_product_mult*torch.cat((infoNCELoss_list[i].unsqueeze(1).repeat(1, args.hidden_dim)*args.inner_product_mult, user_embeds[i][user_step_index]), 1)  #[]  [1024, 16] ### tensor([], device='cuda:0', size=(0, 32), dtype=torch.float64
            # print("1SSL:", SSL_input.shape) #torch.Size([0, 32])
                #tensor([], device='cuda:0', size=(0, 32), grad_fn=<MulBackward0>)
            SSL_input = args.inner_product_mult*torch.cat((SSL_input, user_embed[user_step_index]), 1).to(torch.float32)  # tensor([], device='cuda:0', size=(0, 48), grad_fn=<MulBackward0>
            # print("2SSL:", SSL_input.shape) #torch.Size([0, 48])
                #tensor([], device='cuda:0', size=(0, 48))
            SSL_input3 = args.inner_product_mult*((infoNCELoss_list[i].unsqueeze(1).repeat(1, args.hidden_dim*2))*torch.cat((user_embeds[i][user_step_index],user_embed[user_step_index]), 1)).to(torch.float32)
            # print("3SSL:", SSL_input3.shape) #torch.Size([0, 32])
                #tensor([], device='cuda:0', size=(0, 32), grad_fn=<MulBackward0>)
                
            # print("SSL_input:", SSL_input.dtype) #torch.float32
            # print("SSL_input3:", SSL_input3.dtype) #torch.float32
            
            infoNCELoss_list_weights[i] = self.dropout7(self.prelu(self.SSL_layer1(SSL_input)))  
            # print("drop_info:",infoNCELoss_list_weights[i].shape ) #torch.Size([0, 24])
            # SSL_input tensor([], device='cuda:0', size=(0, 48), dtype=torch.float64,
            # RuntimeError: mat1 and mat2 must have the same dtype
            
            infoNCELoss_list_weights[i] = np.sqrt(SSL_input.shape[1])*self.dropout7(self.SSL_layer2(infoNCELoss_list_weights[i]).squeeze())
            # print("1_info:",infoNCELoss_list_weights[i].shape ) #torch.Size([0])
            infoNCELoss_list_weights[i] = self.batch_norm(infoNCELoss_list_weights[i].unsqueeze(1)).squeeze()
            # print("2_info:",infoNCELoss_list_weights[i].shape ) #torch.Size([0])
            # infoNCELoss_list_weights[i] = args.inner_product_mult*self.sigmoid(infoNCELoss_list_weights[i])
            infoNCELoss_list_weights[i] = args.inner_product_mult_last*self.sigmoid(infoNCELoss_list_weights[i])
            # print("3_info:",infoNCELoss_list_weights[i].shape ) #torch.Size([0])
            
            SSL_weight3 = self.dropout7(self.prelu(self.SSL_layer3(SSL_input3)))
            # print("1_SSL:",SSL_weight3.shape) #torch.Size([0, 1])
            SSL_weight3 = self.batch_norm(SSL_weight3).squeeze()   # tensor([], device='cuda:0', grad_fn=<SqueezeBackward0>)
            # print("2_SSL:",SSL_weight3.shape) #torch.Size([0])
            # SSL_weight3 = args.inner_product_mult*self.sigmoid(SSL_weight3)
            SSL_weight3 = args.inner_product_mult_last*self.sigmoid(SSL_weight3)
            # print("3_SSL:",SSL_weight3.shape) #torch.Size([0])
            infoNCELoss_list_weights[i] = (infoNCELoss_list_weights[i] + SSL_weight3)/2
            # print("info:",infoNCELoss_list_weights[i].shape ) #torch.Size([0])
            # print("I:",i)


            RS_input = args.inner_product_mult*torch.cat((behavior_loss_multi_list[i].unsqueeze(1).repeat(1, args.hidden_dim)*args.inner_product_mult, user_embed[user_index_list[i]]), 1).to(torch.float32)   # user_embed torch.Size([31882, 16])
            # print("1RS_input:", RS_input.shape) #torch.float32
            # data:tensor([], device='cuda:0', size=(0, 48))
            # shape: torch.Size([1, 48])           
            RS_input = args.inner_product_mult*torch.cat((RS_input, user_embeds[i][user_index_list[i]]), 1)
            # print("2RS_input:", RS_input.shape)  #torch.float32
            RS_input3 = args.inner_product_mult*((behavior_loss_multi_list[i].unsqueeze(1).repeat(1, args.hidden_dim))*user_embed[user_index_list[i]]).to(torch.float32)
            
            # print("RS_input3:", RS_input3.shape)  #torch.float32
            
            # print("RS_INPUT:", RS_input.shape)
            # print("RS_INPUT:", RS_input)
            # if i == 2 and RS_input.shape[0] == 1:  # [1, 48]
            #     RS_input = torch.empty([0] + list(RS_input.shape[1:])).cuda()
            #     # RS_input = torch.Size([0] + list(RS_input[1:]))
            #     # RS_input = torch.zeros(RS_input.shape[0]).cuda()
            # else:
            #     pass
                # print("new_RS_INPUT:", RS_input.shape)
            behavior_loss_multi_list_weights[i] = self.dropout7(self.prelu(self.RS_layer1(RS_input))) 
            # print("dropout7_behavior_loss_multi_list_weights[i] :",behavior_loss_multi_list_weights[i].shape)  #torch.Size([2, 24]) ,  torch.Size([1, 24])
            
            
            
            behavior_loss_multi_list_weights[i] = np.sqrt(RS_input.shape[1])*self.dropout7(self.RS_layer2(behavior_loss_multi_list_weights[i]).squeeze(dim=-1))
            # Dimension out of range (expected to be in range of [-1, 0], but got 1)
            # print("sqat_behavior_loss_multi_list_weights[i] :",behavior_loss_multi_list_weights[i].shape) #batchnorm: torch.Size([2])  , torch.Size([])
            # print("self.RS_layer2(behavior_loss_multi_list_weights[i]).squeeze():", self.RS_layer2(behavior_loss_multi_list_weights[i]).squeeze().shape)
                         
            
            ## [DONE!] ]TODO 425 IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
            #  에러의 원인은 batch_size를 2로 변경해서 임. 다시 default=8192으로 원상복귀하니 잘 돌아감.
            behavior_loss_multi_list_weights[i] = self.batch_norm(behavior_loss_multi_list_weights[i].unsqueeze(dim=1))
            # print("bc_norm_behavior_loss_multi_list_weights[i]:", behavior_loss_multi_list_weights[i].shape)  # torch.Size([2, 1])
            # behavior_loss_multi_list_weights[i] = args.inner_product_mult*self.sigmoid(behavior_loss_multi_list_weights[i]).squeeze()
            behavior_loss_multi_list_weights[i] = args.inner_product_mult_last*self.sigmoid(behavior_loss_multi_list_weights[i]).squeeze()
            # print("inner_norm_behavior_loss_multi_list_weights[i]:", behavior_loss_multi_list_weights[i].shape)  # torch.Size([2])
            
            # if i == 2 and RS_input3.shape[0] == 1:  # [1, 48]
            #     RS_input3 = torch.empty([0] + list(RS_input3.shape[1:])).cuda()
            #     # print("1_RS_weight3:",RS_input3.shape)  #torch.Size([2, 1])
            # else:
            #     pass
            
            RS_weight3 = self.dropout7(self.prelu(self.RS_layer3(RS_input3)))
            # print("new_1_RS_weight3:",RS_weight3.shape)  #torch.Size([2, 1])
            RS_weight3 = self.batch_norm(RS_weight3).squeeze(dim = -1)
            # print("2_RS_weight3:",RS_weight3.shape)  #torch.Size([2])
            # RS_weight3 = args.inner_product_mult*self.sigmoid(RS_weight3).squeeze()
            RS_weight3 = args.inner_product_mult_last*self.sigmoid(RS_weight3).squeeze()
            # print("3_RS_weight3:",RS_weight3.shape)  #torch.Size([2])
            behavior_loss_multi_list_weights[i] = behavior_loss_multi_list_weights[i] + RS_weight3
            # print("final_RS_weight3:",behavior_loss_multi_list_weights[i].shape)  # torch.Size([2])


        return infoNCELoss_list_weights, behavior_loss_multi_list_weights

## parser.add_argument('--batch', default=8192, type=int, help='batch size')   
# 1SSL: torch.Size([819, 32])
# 2SSL: torch.Size([819, 48])
# 3SSL: torch.Size([819, 32])
# SSL_input: torch.float32
# SSL_input3: torch.float32
# drop_info: torch.Size([819, 24])
# 1_info: torch.Size([819])
# 2_info: torch.Size([819])
# 3_info: torch.Size([819])
# 1_SSL: torch.Size([819, 1])
# 2_SSL: torch.Size([819])
# 3_SSL: torch.Size([819])
# info: torch.Size([819])
# 1RS_input: torch.float32
# 2RS_input: torch.float32
# RS_input3: torch.float32
# dropout7_behavior_loss_multi_list_weights[i] : torch.Size([8155, 24])
# sqat_behavior_loss_multi_list_weights[i] : torch.Size([8155])
# bc_norm_behavior_loss_multi_list_weights[i]: torch.Size([8155, 1])
# inner_norm_behavior_loss_multi_list_weights[i]: torch.Size([8155])
# 1_RS_weight3: torch.Size([8155, 1])
# 2_RS_weight3: torch.Size([8155])
# 3_RS_weight3: torch.Size([8155])
# final_RS_weight3: torch.Size([8155])
# 1SSL: torch.Size([819, 32])
# 2SSL: torch.Size([819, 48])
# 3SSL: torch.Size([819, 32])
# SSL_input: torch.float32
# SSL_input3: torch.float32
# drop_info: torch.Size([819, 24])
# 1_info: torch.Size([819])
# 2_info: torch.Size([819])
# 3_info: torch.Size([819])
# 1_SSL: torch.Size([819, 1])
# 2_SSL: torch.Size([819])
# 3_SSL: torch.Size([819])
# info: torch.Size([819])
# 1RS_input: torch.float32
# 2RS_input: torch.float32
# RS_input3: torch.float32
# dropout7_behavior_loss_multi_list_weights[i] : torch.Size([2989, 24])
# sqat_behavior_loss_multi_list_weights[i] : torch.Size([2989])
# bc_norm_behavior_loss_multi_list_weights[i]: torch.Size([2989, 1])
# inner_norm_behavior_loss_multi_list_weights[i]: torch.Size([2989])
# 1_RS_weight3: torch.Size([2989, 1])
# 2_RS_weight3: torch.Size([2989])
# 3_RS_weight3: torch.Size([2989])
# final_RS_weight3: torch.Size([2989])
# 1SSL: torch.Size([819, 32])
# 2SSL: torch.Size([819, 48])
# 3SSL: torch.Size([819, 32])
# SSL_input: torch.float32
# SSL_input3: torch.float32
# drop_info: torch.Size([819, 24])
# 1_info: torch.Size([819])
# 2_info: torch.Size([819])
# 3_info: torch.Size([819])
# 1_SSL: torch.Size([819, 1])
# 2_SSL: torch.Size([819])
# 3_SSL: torch.Size([819])
# info: torch.Size([819])
# 1RS_input: torch.float32
# 2RS_input: torch.float32
# RS_input3: torch.float32
# dropout7_behavior_loss_multi_list_weights[i] : torch.Size([6707, 24])
# sqat_behavior_loss_multi_list_weights[i] : torch.Size([6707])
# bc_norm_behavior_loss_multi_list_weights[i]: torch.Size([6707, 1])
# inner_norm_behavior_loss_multi_list_weights[i]: torch.Size([6707])
# 1_RS_weight3: torch.Size([6707, 1])
# 2_RS_weight3: torch.Size([6707])
# 3_RS_weight3: torch.Size([6707])
# final_RS_weight3: torch.Size([6707])
# 1SSL: torch.Size([819, 32])
# 2SSL: torch.Size([819, 48])
# 3SSL: torch.Size([819, 32])
# SSL_input: torch.float32
# SSL_input3: torch.float32
# drop_info: torch.Size([819, 24])
# 1_info: torch.Size([819])
# 2_info: torch.Size([819])
# 3_info: torch.Size([819])
# 1_SSL: torch.Size([819, 1])
# 2_SSL: torch.Size([819])
# 3_SSL: torch.Size([819])
# info: torch.Size([819])
# 1RS_input: torch.float32
# 2RS_input: torch.float32
# RS_input3: torch.float32
# dropout7_behavior_loss_multi_list_weights[i] : torch.Size([8192, 24])
# sqat_behavior_loss_multi_list_weights[i] : torch.Size([8192])
# bc_norm_behavior_loss_multi_list_weights[i]: torch.Size([8192, 1])
# inner_norm_behavior_loss_multi_list_weights[i]: torch.Size([8192])
# 1_RS_weight3: torch.Size([8192, 1])
# 2_RS_weight3: torch.Size([8192])
# 3_RS_weight3: torch.Size([8192])
# final_RS_weight3: torch.Size([8192])
