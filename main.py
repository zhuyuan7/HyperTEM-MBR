import numpy as np
from numpy import random
import pickle
from scipy.sparse import csr_matrix,coo_matrix
import math
import gc
import time
import random
import datetime
import os 

import torch as t
import torch.nn as nn
import torch.utils.data as dataloader
import torch.nn.functional as F
from torch.nn import init

import graph_utils
import DataHandler
#import Gru

import AGNN
import METAGNN
import MV_Net
import HY_GNN
import hyper_graph_utils
# import SE_NET

from Params import args
from Utils.TimeLogger import log
from tqdm import tqdm


import wandb
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"]='0'

t.backends.cudnn.benchmark=True

if t.cuda.is_available():
    use_cuda = True
    t.cuda.set_device(0)
else:
    use_cuda = False

MAX_FLAG = 0x7FFFFFFF

now_time = datetime.datetime.now()
modelTime = datetime.datetime.strftime(now_time,'%Y_%m_%d__%H_%M_%S')

t.autograd.set_detect_anomaly(True)
os.getcwd()
class Model():
    def __init__(self):
   
        self.train_file = args.predir + 'train_mat_'  
        self.val_file = args.predir + 'test_mat.pkl'
        self.test_file = args.predir + 'test_mat.pkl'

        self.trn_file = args.path + args.dataset + '/trn_'
        self.tst_file = args.path + args.dataset + '/tst_int'     
        # self.tst_file = args.path + args.dataset + '/BST_tst_int_59' 
        #Tmall: 3,4,5,6,8,59
        #IJCAI_15: 5,6,8,10,13,53

        # self.meta_multi_file = args.path + args.dataset + '/meta_multi_beh_user_index'
        # self.meta_single_file = args.path + args.dataset + '/meta_single_beh_user_index'
        self.meta_multi_single_file = args.path + args.dataset + '/meta_multi_single_beh_user_index_shuffle' #[2]  # len : 11690
        #                                                         /meta_multi_single_beh_user_index_shuffle
        #                                                         /new_multi_single

        # self.meta_multi = pickle.load(open(self.meta_multi_file, 'rb'))
        # self.meta_single = pickle.load(open(self.meta_single_file, 'rb'))
        self.meta_multi_single = pickle.load(open(self.meta_multi_single_file, 'rb')) #[3]

        self.t_max = -1 
        self.t_min = 0x7FFFFFFF
        self.time_number = -1
 
        self.user_num = -1
        self.item_num = -1
        self.behavior_mats = {} 
        self.behaviors = []
        self.behaviors_data = {}
        self.hyper_data = {}
        self.beh_meta_path_data = {}
        self.beh_meta_path_mats = {}
        self.hyper_behavior_mats = {}
        self.hyper_behavior_Adj = {}
     
        #history
        self.train_loss = []
        self.his_hr = []
        self.his_ndcg = []
        gc.collect()  #

        self.relu = t.nn.ReLU()
        self.sigmoid = t.nn.Sigmoid()
        self.curEpoch = 0

# TODO 아예 모델인풋을 BEH와 META를 합쳐서 들어가자. 오늘 테스트 
        if args.dataset == 'Tmall':
            self.predir = '/home/joo/JOOCML/data/Tmall/'
            self.behaviors_SSL = ['pv','fav', 'cart', 'buy']
            self.behaviors = ['pv','fav', 'cart', 'buy']        

        elif args.dataset == 'IJCAI_15':
            self.predir = '/home/joo/JOOCML/data/IJCAI_15/'
            self.behaviors = ['click','fav', 'cart', 'buy']
            self.behaviors_SSL = ['click','fav', 'cart', 'buy']

        elif args.dataset == 'retail_rocket':
            self.predir = '/home/joo/JOOCML/data/retail_rocket'
            self.behaviors = ['view','cart', 'buy']
            self.behaviors_SSL = ['view','cart', 'buy']




################ LOAD_ BEH_DATA ####################################################################################
        for i in range(0, len(self.behaviors)): #[5]
            with open(self.trn_file + self.behaviors[i], 'rb') as fs:
                data = pickle.load(fs)   # '/home/joo/CML/data/Tmall/trn_pv'  trn_fav' trn_fav' trn_buy'>
                self.behaviors_data[i] = data 

                if data.get_shape()[0] > self.user_num:  #data.get_shape()[0] :31882
                    self.user_num = data.get_shape()[0]  
                if data.get_shape()[1] > self.item_num:  #data.get_shape()[1] :31232
                    self.item_num = data.get_shape()[1]  

             
                if data.data.max() > self.t_max:  # k-타임스텝 값 설정 유닉스타임
                    self.t_max = data.data.max() # 1512316799  시간대의 날짜: 2017. 12. 4. 오전 12:59:59
                if data.data.min() < self.t_min:
                    self.t_min = data.data.min() # 1511539200  시간대의 날짜: 2017. 11. 25. 오전 1:00:00

        
                if self.behaviors[i]==args.target: #buy
                    self.trainMat = data   #(2174, 30113)
                    self.trainLabel = 1*(self.trainMat != 0)   #shape:(31882, 31232)
                    self.labelP = np.squeeze(np.array(np.sum(self.trainLabel, axis=0)))   #shape:(31232,)
################ LOAD_ BEH_DATA ####################################################################################
        
        self.test_mat = pickle.load(open(self.test_file, 'rb')) #shape:(31882, 31232)
        self.userNum = self.behaviors_data[0].shape[0]  #self.behaviors_data[0] shape:(31882, 31232)  (2174,30113)
        self.itemNum = self.behaviors_data[0].shape[1]  #self.behaviors_data[0] shape:(31882, 31232)

        
    # ==>  BEHAVIOR_BUILDING
        time = datetime.datetime.now()
        
        print("Start BEHAVIOR_building:  ", time)
        for i in range(0, len(self.behaviors_data)):
            self.behaviors_data[i] = 1*(self.behaviors_data[i]!=0)  #<2174x30113 sparse matrix of type '<class 'numpy.int32'>'
            self.behavior_mats[i] = graph_utils.get_use(self.behaviors_data[i]) #  
            self.hyper_behavior_Adj[i] = hyper_graph_utils.get_hyper_use(self.behaviors_data[i])            
        time = datetime.datetime.now()
        print("End BEHAVIOR_building:", time)

        print("user_num: ", self.user_num) # 31881
        print("item_num: ", self.item_num) # 31232 
        print("\n")

        train_u, train_v = self.trainMat.nonzero()  #--> 실질적으로 trn_buy 파일 [6] train_u: shape:(167862,), train_v: shape:(167862,) 
        train_data = np.hstack((train_u.reshape(-1,1), train_v.reshape(-1,1))).tolist() # [train_u shape:(167862,1), train_v shape:(167862,1)] # [[0, 0], [0, 1], [0, 3], [0, 8], [0, 27], [1, 47], [1, 48], [1, 54], [1, 78], [1, 79], [
        


        train_dataset = DataHandler.RecDataset_beh(self.behaviors, train_data, self.item_num, self.behaviors_data, True)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

        # test_data  
        with open(self.tst_file, 'rb') as fs:   
            data = pickle.load(fs)  # [22, None, 133,None,None,None,None,None,527,None,None,None,804,None,.., ]
                                    # len():22438         #len():2174
        test_user = np.array([idx for idx, i in enumerate(data) if i is not None])  #array([    6,     9,    11, ..., 31872, 31874, 31876])
        test_item = np.array([i for idx, i in enumerate(data) if i is not None])    #array([  253,   392,   462, ..., 21210, 17045,  2486])
        test_data = np.hstack((test_user.reshape(-1,1), test_item.reshape(-1,1))).tolist() #[[6, 253], [9, 392], [11, 462], [17, 726], [18, 738], [24, 1008], [28, 1142], [30, 1210],..., ]
        test_dataset = DataHandler.RecDataset(test_data, self.item_num, self.trainMat, 0, False)  
        self.test_loader = dataloader.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)  
        # -------------------------------------------------------------------------------------------------->>>>>

    def prepareModel(self):
        self.modelName = self.getModelName()  
        self.hidden_dim = args.hidden_dim  #16
        

        if args.isload == True:
            self.loadModel(args.loadModelPath)
        else:
            self.model = AGNN.myModel(self.user_num, self.item_num, self.behaviors, self.behavior_mats).cuda()
            self.meta_weight_net = MV_Net.MetaWeightNet(len(self.behaviors)).cuda()
            


        # #IJCAI_15
        # self.opt = t.optim.AdamW(self.model.parameters(), lr = args.lr, weight_decay = args.opt_weight_decay)
        # self.meta_opt =  t.optim.AdamW(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay)
        # # self.meta_opt =  t.optim.RMSprop(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay, momentum=0.95, centered=True)
        # self.scheduler = t.optim.lr_scheduler.CyclicLR(self.opt, args.opt_base_lr, args.opt_max_lr, step_size_up=5, step_size_down=10, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
        # self.meta_scheduler = t.optim.lr_scheduler.CyclicLR(self.meta_opt, args.meta_opt_base_lr, args.meta_opt_max_lr, step_size_up=2, step_size_down=3, mode='triangular', gamma=0.98, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.9, max_momentum=0.99, last_epoch=-1)
        #       


        # #Tmall
        # self.opt = t.optim.AdamW(self.model.parameters(), lr = args.lr, weight_decay = args.opt_weight_decay)
        # self.meta_opt =  t.optim.AdamW(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay)
        # # self.meta_opt =  t.optim.RMSprop(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay, momentum=0.95, centered=True)
        # self.scheduler = t.optim.lr_scheduler.CyclicLR(self.opt, args.opt_base_lr, args.opt_max_lr, step_size_up=5, step_size_down=10, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
        # self.meta_scheduler = t.optim.lr_scheduler.CyclicLR(self.meta_opt, args.meta_opt_base_lr, args.meta_opt_max_lr, step_size_up=3, step_size_down=7, mode='triangular', gamma=0.98, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.9, max_momentum=0.99, last_epoch=-1)
        # # #                                                                                                                                                                           0.993                                             

        # # retailrocket
        self.opt = t.optim.AdamW(self.model.parameters(), lr = args.lr, weight_decay = args.opt_weight_decay)
        # self.meta_opt =  t.optim.AdamW(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay)
        self.meta_opt =  t.optim.SGD(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay, momentum=0.95, nesterov=True)
        self.scheduler = t.optim.lr_scheduler.CyclicLR(self.opt, args.opt_base_lr, args.opt_max_lr, step_size_up=1, step_size_down=3, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
        self.meta_scheduler = t.optim.lr_scheduler.CyclicLR(self.meta_opt, args.meta_opt_base_lr, args.meta_opt_max_lr, step_size_up=1, step_size_down=3, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.9, max_momentum=0.99, last_epoch=-1)
                                                                                                                                            #  exp_range  step_size_up=1, step_size_down=2,


        if use_cuda:
            self.model = self.model.cuda()

    def innerProduct(self, u, i, j):  
        pred_i = t.sum(t.mul(u,i), dim=1)*args.inner_product_mult  
        pred_j = t.sum(t.mul(u,j), dim=1)*args.inner_product_mult
        return pred_i, pred_j

    def SSL(self, user_embeddings, item_embeddings, target_user_embeddings, target_item_embeddings, user_step_index):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[t.randperm(embedding.size()[0])]  
            return corrupted_embedding
        
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[t.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:,t.randperm(corrupted_embedding.size()[1])]  
            return corrupted_embedding
        
        def score(x1, x2):
            return t.sum(t.mul(x1, x2), 1)

        def neg_sample_pair(x1, x2, τ = 0.05):  
            for i in range(x1.shape[0]):
                index_set = set(np.arange(x1.shape[0]))
                index_set.remove(i)
                index_set_neg = t.as_tensor(np.array(list(index_set))).long().cuda()  

                x_pos = x1[i].repeat(x1.shape[0]-1, 1)
                x_neg = x2[index_set]  
                
                if i==0:
                    x_pos_all = x_pos
                    x_neg_all = x_neg
                else:
                    x_pos_all = t.cat((x_pos_all, x_pos), 0)
                    x_neg_all = t.cat((x_neg_all, x_neg), 0)
            x_pos_all = t.as_tensor(x_pos_all)  #[9900, 100]
            x_neg_all = t.as_tensor(x_neg_all)  #[9900, 100]  

            return x_pos_all, x_neg_all

        def one_neg_sample_pair_index(i, step_index, embedding1, embedding2):

            index_set = set(np.array(step_index))
            index_set.remove(i.item())
            neg2_index = t.as_tensor(np.array(list(index_set))).long().cuda()

            neg1_index = t.ones((2,), dtype=t.long)
            neg1_index = neg1_index.new_full((len(index_set),), i)

            neg_score_pre = t.sum(compute(embedding1, embedding2, neg1_index, neg2_index).squeeze())
            return neg_score_pre

        def multi_neg_sample_pair_index(batch_index, step_index, embedding1, embedding2):  #small, big, target, beh: [30], [819], [31882, 16], [31882, 16]

            index_set = set(np.array(step_index.cpu()))   # len 804 {30729, 4106, 10255, 20504, 20505, 26, 22558, 20513, 14372, 4135, 20521, 16427, 22572,
            batch_index_set = set(np.array(batch_index.cpu()))  # len 30 {1415, 17288, 25612, 31251, 8340, 16918, 20505, 20378, 28444, 10908, 8874, 19636, 24886,
            neg2_index_set = index_set - batch_index_set                         #beh
            neg2_index = t.as_tensor(np.array(list(neg2_index_set))).long().cuda()  #[774]
            neg2_index = t.unsqueeze(neg2_index, 0)                              #torch.Size([1, 774])
            neg2_index = neg2_index.repeat(len(batch_index), 1)                  #torch.Size([30, 774])
            neg2_index = t.reshape(neg2_index, (1, -1))                          #torch.Size([1, 23220])
            neg2_index = t.squeeze(neg2_index)                                   #shape:torch.Size([23220])
                                                                                 #target
            neg1_index = batch_index.long().cuda()                               #torch.Size([1, 23220])
            neg1_index = t.unsqueeze(neg1_index, 1)                              #shape:torch.Size([30, 1])
            neg1_index = neg1_index.repeat(1, len(neg2_index_set))               #shape:torch.Size([30, 774])
            neg1_index = t.reshape(neg1_index, (1, -1))                          #shape:torch.Size([1, 23220])        
            neg1_index = t.squeeze(neg1_index)                                   #shape:torch.Size([23220])

            neg_score_pre = t.sum(compute(embedding1, embedding2, neg1_index, neg2_index).squeeze().view(len(batch_index), -1), -1)  #
            #                           torch.Size([31882, 16]) torch.Size([31882, 16])  torch.Size([23220]) torch.Size([23220])   torch.Size([30])
            return neg_score_pre  # torch.Size([30])

        def compute(x1, x2, neg1_index=None, neg2_index=None, τ = 0.05):  #[1024, 16], [1024, 16]

            if neg1_index!=None:
                x1 = x1[neg1_index]  # neg_score_pre torch.Size([23220, 16])
                x2 = x2[neg2_index]  # neg_score_pre torch.Size([23220, 16])

            N = x1.shape[0]   # 819   # neg_score_pre  23220
            D = x1.shape[1]   # 16

            x1 = x1  #torch.Size([819, 16])    # neg_score_pre  torch.Size([23220, 16])
            x2 = x2  #torch.Size([819, 16])    # neg_score_pre  torch.Size([23220, 16])

            scores = t.exp(t.div(t.bmm(x1.view(N, 1, D), x2.view(N, D, 1)).view(N, 1), np.power(D, 1)+1e-8))  #[1024, 1] torch.Size([819, 1])
            #                         torch.Size([23220, 1, 16]) torch.Size([23220, 16, 1])  16.00000001
            #torch.Size([23220, 1])
            return scores  # torch.Size([819, 1])
        
        def single_infoNCE_loss_simple(embedding1, embedding2):
            pos = score(embedding1, embedding2)  #[100]
            neg1 = score(embedding2, row_column_shuffle(embedding1))  
            one = t.cuda.FloatTensor(neg1.shape[0]).fill_(1)  #[100]
            # one = zeros = t.ones(neg1.shape[0])
            con_loss = t.sum(-t.log(1e-8 + t.sigmoid(pos))-t.log(1e-8 + (one - t.sigmoid(neg1))))  
            return con_loss

        #use_less    
        def single_infoNCE_loss(embedding1, embedding2):
            N = embedding1.shape[0]
            D = embedding1.shape[1]

            pos_score = compute(embedding1, embedding2).squeeze()  #[100, 1]

            neg_x1, neg_x2 = neg_sample_pair(embedding1, embedding2)  #[9900, 100], [9900, 100]
            neg_score = t.sum(compute(neg_x1, neg_x2).view(N, (N-1)), dim=1)  #[100]  
            con_loss = -t.log(1e-8 +t.div(pos_score, neg_score))   
            con_loss = t.mean(con_loss)  
            return max(0, con_loss)

        def single_infoNCE_loss_one_by_one(embedding1, embedding2, step_index):  #target, beh
            N = step_index.shape[0]  # 819
            D = embedding1.shape[1]  # 16

            pos_score = compute(embedding1[step_index], embedding2[step_index]).squeeze()  #torch.Size([819])
            neg_score = t.zeros((N,), dtype = t.float64).cuda()  #shape:torch.Size([819])

            #-------------------------------------------------multi version-----------------------------------------------------
            steps = int(np.ceil(N / args.SSL_batch))  #separate the batch to smaller one  819/ 30 =28
            for i in range(steps):
                st = i * args.SSL_batch
                ed = min((i+1) * args.SSL_batch, N)
                batch_index = step_index[st: ed] #torch.Size([30])

                neg_score_pre = multi_neg_sample_pair_index(batch_index, step_index, embedding1, embedding2) #torch.Size([30])
                if i ==0:
                    neg_score = neg_score_pre
                else:
                    neg_score = t.cat((neg_score, neg_score_pre), 0)
            #-------------------------------------------------multi version-----------------------------------------------------

            con_loss = -t.log(1e-8 +t.div(pos_score, neg_score+1e-8))  #[819]]/[819]==>819]  torch.Size([819])


            assert not t.any(t.isnan(con_loss))
            assert not t.any(t.isinf(con_loss))

            return t.where(t.isnan(con_loss), t.full_like(con_loss, 0+1e-8), con_loss)

        user_con_loss_list = []  # torch.Size([819])
        item_con_loss_list = []

        SSL_len = int(user_step_index.shape[0]/10)  # 8192/10 =819
        user_step_index = t.as_tensor(np.random.choice(user_step_index.cpu(), size=SSL_len, replace=False, p=None)).cuda() # torch.Size 819

        for i in range(len(self.behaviors_SSL)):  # ['pv', 'fav', 'cart', 'buy']

            user_con_loss_list.append(single_infoNCE_loss_one_by_one(user_embeddings[-1], user_embeddings[i], user_step_index))
                                                                #    torch.Size([4, 31882, 16]) torch.Size([4, 31882, 16])   torch.Size([819])
        user_con_losss = t.stack(user_con_loss_list, dim=0)  # torch.Size([4, 819])

        return user_con_loss_list, user_step_index  #4*shape:torch.Size([819])   torch.Size([819])

    def run(self):
        
        

        wandb.init(project="TempSSL")
        # run = wandb.init(settings=wandb.Settings(_service_wait=300))
        wandb.run.name = args.wandb
        wandb.config.update(args)
        
        
        
        self.prepareModel()
        # if args.isload == True:
        #     # print("----------------------pre test:")
        #     # HR, NDCG = self.testEpoch(self.test_loader)
        #     print(f"HR: {HR} , NDCG: {NDCG}")  
        # log('Model Prepared')


        cvWait = 0  
        self.best_HR = 0 
        self.best_NDCG = 0
        flag = 0

        self.user_embed = None 
        self.item_embed = None
        self.user_embeds = None
        self.item_embeds = None


        print("Test before train:")
        # HR, NDCG = self.testEpoch(self.test_loader)

        for e in tqdm(range(self.curEpoch, args.epoch+1)):  
            self.curEpoch = e

            self.meta_flag = 0
            if e%args.meta_slot == 0:
                self.meta_flag=1


            log("*****************Start epoch: %d ************************"%e)  
            #TODO 이부분에 metapath도 걸어야함.
            if args.isJustTest == False:
                epoch_loss, user_embed, item_embed, user_embeds, item_embeds = self.trainEpoch()
                
                self.train_loss.append(epoch_loss)  
                print(f"epoch {e/args.epoch},  epoch loss{epoch_loss}")
                self.train_loss.append(epoch_loss)
            else:
                break

            HR, NDCG = self.testEpoch(self.test_loader)
            self.his_hr.append(HR)
            self.his_ndcg.append(NDCG)

            self.scheduler.step()
            self.meta_scheduler.step()

            if HR > self.best_HR:
                self.best_HR = HR
                self.best_epoch = self.curEpoch 
                cvWait = 0
                print("--------------------------------------------------------------------------------------------------------------------------best_HR", self.best_HR)
                # print("--------------------------------------------------------------------------------------------------------------------------NDCG", self.best_NDCG)
                self.user_embed = user_embed 
                self.item_embed = item_embed
                self.user_embeds = user_embeds
                self.item_embeds = item_embeds

                self.saveHistory()
                self.saveModel()


            
            if NDCG > self.best_NDCG:
                self.best_NDCG = NDCG
                self.best_epoch = self.curEpoch 
                cvWait = 0
                # print("--------------------------------------------------------------------------------------------------------------------------HR", self.best_HR)
                print("--------------------------------------------------------------------------------------------------------------------------best_NDCG", self.best_NDCG)
                self.user_embed = user_embed 
                self.item_embed = item_embed
                self.user_embeds = user_embeds
                self.item_embeds = item_embeds

                self.saveHistory()
                self.saveModel()



            if (HR<self.best_HR) and (NDCG<self.best_NDCG): 
                cvWait += 1


            wandb.log({
                    "Train loss" : epoch_loss,
                    "Train HR" : HR,
                    "Train NDCG" : NDCG,
                })
            
            
            if cvWait == args.patience:
                print(f"Early stop at {self.best_epoch} :  best HR: {self.best_HR}, best_NDCG: {self.best_NDCG} \n")
                self.saveHistory()
                self.saveModel()
                break
               
        HR, NDCG = self.testEpoch(self.test_loader)
        self.his_hr.append(HR)
        self.his_ndcg.append(NDCG)
        
        
        
        wandb.log({
                    "Test HR" : HR,
                    "Test NDCG" : NDCG,
                })


    def negSamp(self, temLabel, sampSize, nodeNum):
        negset = [None] * sampSize    
        cur = 0
        while cur < sampSize:    # sampSize :5   [6010, 30706, 30107, 30111, 7159]
            rdmItm = np.random.choice(nodeNum)   #rdmItm:12042, nodeNum :31232
            if temLabel[rdmItm] == 0:     # temLabel: array([0, 0, 0, ..., 0, 0, 0])
                negset[cur] = rdmItm   # [12042, None, None, None, None]
                cur += 1
        return negset  # [12042, 22435, 9126, 19071, 7791]

    def sampleTrainBatch(self, batIds, labelMat):
        temLabel = labelMat[batIds.cpu()].toarray()  # shape:(32, 31232)  size:999424
        batch = len(batIds)
        user_id = []       # [27852, 27852, 27852, 27852, 27852]
        item_id_pos = []   # [27344, 7361, 21664, 27344, 21664]
        item_id_neg = []   # [6010, 30706, 30107, 30111, 7159]
 
        cur = 0
        for i in range(batch):  # batch = 32
            posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])  # array([  483,  5278, 10161, 11471, 11984])
            sampNum = min(args.sampNum, len(posset))   # args.sampNum:10, posset :5
            if sampNum == 0:
                poslocs = [np.random.choice(labelMat.shape[1])]  # array([27344,  7361, 21664, 27344, 21664])  
                neglocs = [poslocs[0]]                           # [6010, 30706, 30107, 30111, 7159]   
            else:
                poslocs = np.random.choice(posset, sampNum)   # array([ 5278, 10161,  5278,   483,  5278]) 
                neglocs = self.negSamp(temLabel[i], sampNum, labelMat.shape[1])  # [12042, 22435, 9126, 19071, 7791]

            for j in range(sampNum):
                user_id.append(batIds[i].item())  #user_id:[27852, 27852, 27852, 27852, 27852]  i:2 [27852, 27852, 27852, 27852, 27852, 1222, 1222, 1222, 1222, 1222]
                item_id_pos.append(poslocs[j].item())   # [13209, 13209, 13209, 13209, 13209, 29050, 29050, 29050, 29050, 29050, 21664, 21664, 21
                item_id_neg.append(neglocs[j])          # [12169, 12169, 12169, 12169, 12169, 26936, 26936, 26936, 26936, 26936, 9020, 9020, 9020
                cur += 1

        return t.as_tensor(np.array(user_id)).cuda(), t.as_tensor(np.array(item_id_pos)).cuda(), t.as_tensor(np.array(item_id_neg)).cuda() 


    def trainEpoch(self):   
        train_loader = self.train_loader
        train_meta_loader = self.train_loader
        time = datetime.datetime.now()

        print("start_beh_ng_samp:  ", time)
        train_loader.dataset.ng_sample()
        print("end__beh_ng_samp:  ", time)
        
        epoch_loss = 0
        
#-----------------------------------------------------------------------------------
        self.behavior_loss_list = [None]*len(self.behaviors)   # [None, None, None, None]     
        self.user_id_list = [None]*len(self.behaviors)         # [None, None, None, None]
        self.item_id_pos_list = [None]*len(self.behaviors)     # [None, None, None, None]
        self.item_id_neg_list = [None]*len(self.behaviors)     # [None, None, None, None]

        self.meta_start_index = 0   # args.meta_batch =32
        self.meta_end_index = self.meta_start_index + args.meta_batch  # 32 

#----------------------------------------------------------------------------------
        
        cnt = 0
        args.isJustbeh == True

        for user, item_i, item_j in train_loader:   # batch_size:8192 
        
            user = user.long().cuda()
            self.user_step_index = user

            self.meta_user = t.as_tensor(self.meta_multi_single[self.meta_start_index:self.meta_end_index]).cuda()  
            
            if self.meta_end_index == self.meta_multi_single.shape[0]:
                self.meta_start_index = 0  
            else:
                self.meta_start_index = (self.meta_start_index + args.meta_batch) % (self.meta_multi_single.shape[0] - 1) # 32 % 11689 =32  64 %11689= 64
            self.meta_end_index = min(self.meta_start_index + args.meta_batch, self.meta_multi_single.shape[0])


#---round one---------------------------------------------------------------------------------------------
            meta_behavior_loss_list = [None]*len(self.behaviors)   # [None, None, None, None]
            meta_user_index_list = [None]*len(self.behaviors)      # [None, None, None, None]


            meta_model = AGNN.myModel(self.user_num, self.item_num, self.behaviors, self.behavior_mats).cuda() 
            meta_opt = t.optim.AdamW(meta_model.parameters(), lr = args.lr, weight_decay = args.opt_weight_decay)
            meta_model.load_state_dict(self.model.state_dict())

            hyper_model = HY_GNN.myModel(self.user_num, self.item_num, self.behaviors,self.hyper_behavior_Adj).cuda()
            meta_opt = t.optim.AdamW(hyper_model.parameters(), lr=args.hyper_lr, weight_decay=0)
            
            
            meta_user_embed, meta_item_embed, meta_user_embeds, meta_item_embeds = meta_model()
        
            # TODO ! meta_model이랑 hyper graph 모델의 임베딩을 mix해서 학습시키자.
            hyper_user_embed, hyper_item_embed, hyper_user_embeds, hyper_item_embeds = hyper_model()
 

            meta_user_embed, meta_item_embed, meta_user_embeds, meta_item_embeds = hyper_graph_utils.mixer(meta_user_embed, meta_item_embed, meta_user_embeds, meta_item_embeds,
                                                      hyper_user_embed, hyper_item_embed, hyper_user_embeds, hyper_item_embeds)


            for index in range(len(self.behaviors)):

                not_zero_index = np.where(item_i[index].cpu().numpy()!=-1)[0]  #  [tensor([20632, 29259...6, 14611]), tensor([  -1,   -1, ...-1, 9490]), tensor([ 1248, ..., 
                                                                            # -1아닌 것 : torch.Size([8156]) ,(3046,),(6724,) ,(8192,)   -1인 것: torch.Size([32]),(5146,), (1468,)
                self.user_id_list[index] = user[not_zero_index].long().cuda()
                meta_user_index_list[index] = self.user_id_list[index]
                self.item_id_pos_list[index] = item_i[index][not_zero_index].long().cuda()
                self.item_id_neg_list[index] = item_j[index][not_zero_index].long().cuda()

                meta_userEmbed = meta_user_embed[self.user_id_list[index]]    # torch.Size([8156, 16])
                meta_posEmbed = meta_item_embed[self.item_id_pos_list[index]] # torch.Size([8156, 16])
                meta_negEmbed = meta_item_embed[self.item_id_neg_list[index]] # torch.Size([8156, 16])

                meta_pred_i, meta_pred_j = 0, 0
                meta_pred_i, meta_pred_j = self.innerProduct(meta_userEmbed, meta_posEmbed, meta_negEmbed)
                
                meta_behavior_loss_list[index] = - (meta_pred_i.view(-1) - meta_pred_j.view(-1)).sigmoid().log()

            # line 643 --> 405
            meta_infoNCELoss_list, SSL_user_step_index = self.SSL(meta_user_embeds, meta_item_embeds, meta_user_embed, meta_item_embed, self.user_step_index)

            meta_infoNCELoss_list_weights, meta_behavior_loss_list_weights = self.meta_weight_net(\
                                                                        meta_infoNCELoss_list, \
                                                                        meta_behavior_loss_list, \
                                                                        SSL_user_step_index, \
                                                                        meta_user_index_list, \
                                                                        meta_user_embeds, \
                                                                        meta_user_embed)



            for i in range(len(self.behaviors)):   # 각 4 behaviors에 대해  loss(torch.Size([819]))들을 합한 값들을 생성
                meta_infoNCELoss_list[i] = (meta_infoNCELoss_list[i]*meta_infoNCELoss_list_weights[i]).sum()  # 4 * torch.Size([])  tensor(2595.2078, tensor(2601.2222 tensor(2600.2822, tensor(2565.3550,  
                meta_behavior_loss_list[i] = (meta_behavior_loss_list[i]*meta_behavior_loss_list_weights[i]).sum()    # 0 ;tensor(6574.2588),  1 :tensor(2384.8611), 2: tensor(5448.2744,),3:tensor(6529.0820)


            meta_bprloss = sum(meta_behavior_loss_list) / len(meta_behavior_loss_list)   # tensor(5234.1191) = tensor(20936.4766)/4
            meta_infoNCELoss = sum(meta_infoNCELoss_list) / len(meta_infoNCELoss_list)  # tensor(13.5014= 0:tensor(2702.4526), 1:tensor(2705.1792), 2:tensor(2713.5522,3:tensor(2679.9612
            meta_regLoss = (t.norm(meta_userEmbed) ** 2 + t.norm(meta_posEmbed) ** 2 + t.norm(meta_negEmbed) ** 2)  # tensor(132528.5938,=  tensor(53051.6562,)  tensor(38932.2891,)  tensor(40544.6523, )

            meta_model_loss = (meta_bprloss + args.reg * meta_regLoss + args.beta*meta_infoNCELoss) / args.batch    # tensor(0.6568
                                            # 0.001                       0.005

            meta_opt.zero_grad(set_to_none=True)
            self.meta_opt.zero_grad(set_to_none=True)
            meta_model_loss.backward()
            nn.utils.clip_grad_norm_(self.meta_weight_net.parameters(), max_norm=20, norm_type=2)
            nn.utils.clip_grad_norm_(meta_model.parameters(), max_norm=20, norm_type=2)
            meta_opt.step()
            self.meta_opt.step()
            
#---round one---------------------------------------------------------------------------------------------



#---round two---------------------------------------------------------------------------------------------
            # print("round_2")
            behavior_loss_list = [None]*len(self.behaviors)
            user_index_list = [None]*len(self.behaviors)  #---

            user_embed, item_embed, user_embeds, item_embeds = meta_model()
            hyper_user_embed, hyper_item_embed, hyper_user_embeds, hyper_item_embeds = hyper_model()

            user_embed, item_embed, user_embeds, item_embeds = hyper_graph_utils.mixer(user_embed, item_embed, user_embeds, item_embeds,
                                                      hyper_user_embed, hyper_item_embed, hyper_user_embeds, hyper_item_embeds)


            for index in range(len(self.behaviors)):

                user_id, item_id_pos, item_id_neg = self.sampleTrainBatch(t.as_tensor(self.meta_user), self.behaviors_data[i])

                user_index_list[index] = user_id  # shape:torch.Size([176])


                userEmbed = user_embed[user_id]   # shape:torch.Size([176, 16])

                posEmbed = item_embed[item_id_pos]  # shape:torch.Size([225, 16])
                negEmbed = item_embed[item_id_neg]  # shape:torch.Size([225, 16])

                pred_i, pred_j = self.innerProduct(userEmbed, posEmbed, negEmbed)
                behavior_loss_list[index] = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log()  
            


            self.infoNCELoss_list, SSL_user_step_index = self.SSL(user_embeds, item_embeds, user_embed, item_embed, self.meta_user)

            infoNCELoss_list_weights, behavior_loss_list_weights = self.meta_weight_net(\
                                                                        self.infoNCELoss_list, \
                                                                        behavior_loss_list, \
                                                                        SSL_user_step_index, \
                                                                        user_index_list, \
                                                                        user_embeds, \
                                                                        user_embed)


            for i in range(len(self.behaviors)):
                self.infoNCELoss_list[i] = (self.infoNCELoss_list[i]*infoNCELoss_list_weights[i]).sum()
                behavior_loss_list[i] = (behavior_loss_list[i]*behavior_loss_list_weights[i]).sum()   

            bprloss = sum(behavior_loss_list) / len(self.behavior_loss_list)
            infoNCELoss = sum(self.infoNCELoss_list) / len(self.infoNCELoss_list)
            round_two_regLoss = (t.norm(userEmbed) ** 2 + t.norm(posEmbed) ** 2 + t.norm(negEmbed) ** 2)

            meta_loss = 0.5 * (bprloss + args.reg * round_two_regLoss  + args.beta*infoNCELoss) / args.batch

            self.meta_opt.zero_grad()
            meta_loss.backward()
            nn.utils.clip_grad_norm_(self.meta_weight_net.parameters(), max_norm=20, norm_type=2)
            self.meta_opt.step()

#---round two-----------------------------------------------------------------------------------------------


#---round three---------------------------------------------------------------------------------------------

            # print("round_3")
            user_embed, item_embed, user_embeds, item_embeds = self.model()
            hyper_user_embed, hyper_item_embed, hyper_user_embeds, hyper_item_embeds = hyper_model()

            user_embed, item_embed, user_embeds, item_embeds = hyper_graph_utils.mixer(user_embed, item_embed, user_embeds, item_embeds,
                                                      hyper_user_embed, hyper_item_embed, hyper_user_embeds, hyper_item_embeds)


            for index in range(len(self.behaviors)):


                userEmbed = user_embed[self.user_id_list[index]]  # torch.Size([8192, 16])
                posEmbed = item_embed[self.item_id_pos_list[index]]  # torch.Size([8192, 16])
                negEmbed = item_embed[self.item_id_neg_list[index]]  # torch.Size([8192, 16])

                pred_i, pred_j = 0, 0
                pred_i, pred_j = self.innerProduct(userEmbed, posEmbed, negEmbed)  # torch.Size([8192])

                self.behavior_loss_list[index] = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log()   # torch.Size([8145])

            infoNCELoss_list, SSL_user_step_index = self.SSL(user_embeds, item_embeds, user_embed, item_embed, self.user_step_index)

            with t.no_grad():
                infoNCELoss_list_weights, behavior_loss_list_weights = self.meta_weight_net(\
                                                                            infoNCELoss_list, \
                                                                            self.behavior_loss_list, \
                                                                            SSL_user_step_index, \
                                                                            self.user_id_list, \
                                                                            user_embeds, \
                                                                            user_embed)


            for i in range(len(self.behaviors)):
                infoNCELoss_list[i] = (infoNCELoss_list[i]*infoNCELoss_list_weights[i]).sum()
                self.behavior_loss_list[i] = (self.behavior_loss_list[i]*behavior_loss_list_weights[i]).sum()  
                

            bprloss = sum(self.behavior_loss_list) / len(self.behavior_loss_list)
            infoNCELoss = sum(infoNCELoss_list) / len(infoNCELoss_list)
            regLoss = (t.norm(userEmbed) ** 2 + t.norm(posEmbed) ** 2 + t.norm(negEmbed) ** 2)
            
            # TODO 이게 BEHAVIOR에 관한 전체 ROUND의 최종 loss ==> 
            
            loss = (bprloss + args.reg * regLoss + args.beta*infoNCELoss) / args.batch

            epoch_loss = epoch_loss + loss.item()

            self.opt.zero_grad(set_to_none=True)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            self.opt.step()


#---round three---------------------------------------------------------------------------------------------
            cnt+=1

        return epoch_loss, user_embed, item_embed, user_embeds, item_embeds


    def testEpoch(self, data_loader, save=False): #[1]
        #print("test!!!!")
        epochHR, epochNDCG = [0]*2
        with t.no_grad():
            user_embed, item_embed, user_embeds, item_embeds = self.model()

        cnt = 0
        tot = 0
        for user, item_i in data_loader: #batch_size:8192
            user_compute, item_compute, user_item1, user_item100 = self.sampleTestBatch(user, item_i)  
            userEmbed = user_embed[user_compute]  #torch.Size([819200, 16])
            itemEmbed = item_embed[item_compute]  #shape:torch.Size([819200, 16])
           
            pred_i = t.sum(t.mul(userEmbed, itemEmbed), dim=1)   # shape:torch.Size([819200])

            hit, ndcg = self.calcRes(t.reshape(pred_i, [user.shape[0], 100]), user_item1, user_item100)  
            epochHR = epochHR + hit  # 1100
            epochNDCG = epochNDCG + ndcg  # 556.6269754434703
            cnt += 1 
            tot += user.shape[0]
            print("tot:",tot)
        
        result_HR = epochHR / tot
        result_NDCG = epochNDCG / tot
        print(f"Step {cnt}:  hit:{result_HR}, ndcg:{result_NDCG}")

        return result_HR, result_NDCG
   

    def calcRes(self, pred_i, user_item1, user_item100):  #[8192, 100] [8192] [8192, (ndarray:(100,))]
     
        hit = 0
        ndcg = 0

    
        for j in range(pred_i.shape[0]): #pred_i.torch.Size([8192, 100])

            _, shoot_index = t.topk(pred_i[j], args.shoot) # 샘플 100개중에 topk 10개 추출 
            shoot_index = shoot_index.cpu() #tensor([20, 63,  7, 99, 45, 90, 88,  2,  4, 93])
            shoot = user_item100[j][shoot_index] #array([ 2464,  1213,  7979,   392, 20147,   436,  5978,  8548, 10801,
            shoot = shoot.tolist()

            if type(shoot)!=int and (user_item1[j] in shoot):  # j ; 3884  user_item1[3884]: tensor(23981)  shoot: [5551, 30001, 23820, 23981, 28126, 23483, 30983, 29971, 1857, 27479]
                hit += 1                                       # j : 3885  tensor(29582)    shoot: [9091, 15193, 16908, 4457, 29582, 7019, 13553, 22792, 6564, 14152]
                ndcg += np.reciprocal( np.log2( shoot.index( user_item1[j])+2))  # shoot.index( user_item1[j])+2 = 5 , np.log2(5) =2.321928094887362, np.reciprocal( np.log2(5)) = 0.43067655807339306
            elif type(shoot)==int and (user_item1[j] == shoot):   # user_item1[j] : torch.Size([8192])
                hit += 1  
                ndcg += np.reciprocal( np.log2( 0+2))
    
        return hit, ndcg  #int, float     # j: 8191번 돌고 hit 1100, ndcg  556.6269754434703  


    def sampleTestBatch(self, batch_user_id, batch_item_id):  # batch size 8192개만큼 샘플데이터를 만듦.
       
        batch = len(batch_user_id) #8192  batch_user_id:torch.Size([8192])  2174
        tmplen = (batch*100) #819200

        sub_trainMat = self.trainMat[batch_user_id].toarray()   #shape:(8192, 31232)  (2174, 30113)
        user_item1 = batch_item_id # torch.Size([8192]) 
        user_compute = [None] * tmplen  #[None, 을 819200개 만큼 만들어]
        item_compute = [None] * tmplen
        user_item100 = [None] * (batch)  # [None, 을 8192개 만큼 만들어]

        cur = 0
        #batch만큼 positem과 neg 설정  8192개의  100개 씩 추출
        for i in range(batch): #batch:8192
            pos_item = user_item1[i]  #data:tensor([  253,   392,   462,  ..., 24187, 25701, 16801])
            negset = np.reshape(np.argwhere(sub_trainMat[i]==0), [-1])   #shape:(31229,) array([    0,     1,     2, ..., 31229, 31230, 31231])  array([    1,     2,     3, ..., 30110, 30111, 30112])
            pvec = self.labelP[negset]  #shape:(31229,),  array([19,  2,  3, ...,  0,  0,  0])    #[0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 2, 0, ...]
            pvec = pvec / np.sum(pvec)  #shape:(31229,) array([1.13196306e-04, 1.19154007e-05, 1.78731010e-05, ...,
                                                                  # 랜덤으로 100 neg 샘플 추출 permutation:모양은 그대로 유지한 무작위 배열
            random_neg_sam = np.random.permutation(negset)[:99]   #shape:(99,)  array([ 5573, 22024,  8937,   293, 27507, 20736,  4024, 30301,  5778,
            user_item100_one_user = np.concatenate(( random_neg_sam, np.array([pos_item]))) #  random_ng sample 99개 concat pos_item 1개  /// shape:(100,)array([ 5573, 22024,  8937,   293, 27507, 20736,  4024, 30301,  5778
            user_item100[i] = user_item100_one_user      # random_ng sample 99개 concat pos_item 1개 concat 해서 100 개의 user_item100 샘플 생성

            for j in range(100):
                user_compute[cur] = batch_user_id[i]  # user_compute len():819200 // batch_user_id: tensor([    6,     9,    11,  ..., 26099, 26101, 26103])
                item_compute[cur] = user_item100_one_user[j] # item_compute len():819200 //user_item100_one_user: array([ 5573, 22024,  8937,   293, 27507, 20736,  4024, 30301,  5778
                cur += 1

        return user_compute, item_compute, user_item1, user_item100


    def setRandomSeed(self):
        np.random.seed(args.seed)
        t.manual_seed(args.seed)
        t.cuda.manual_seed(args.seed)
        random.seed(args.seed)

    def getModelName(self):  
        title = args.title
        ModelName = \
        args.point + \
        "_" + title + \
        "_" +  args.dataset +\
        "_" + modelTime + \
        "_lr_" + str(args.lr) + \
        "_reg_" + str(args.reg) + \
        "_batch_size_" + str(args.batch) + \
        "_gnn_layer_" + str(args.gnn_layer)

        return ModelName

    def saveHistory(self):  
        history = dict()
        history['loss'] = self.train_loss  
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        ModelName = self.modelName
        
        # with open(r'C:\\Users\choi\Desktop\\HY-CML\\History\\' + args.dataset + r'\\' + ModelName + '.his', 'wb') as fs:
        with open(args.History_path + args.dataset + '\\' + ModelName + '.his', 'wb') as fs: 
            pickle.dump(history, fs)
        
    def saveModel(self):  
        ModelName = self.modelName

        history = dict()
        history['loss'] = self.train_loss
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        savePath = args.Model_path + args.dataset + '\\' + ModelName + '.pth'
        
        params = {
            'epoch': self.curEpoch,
            # 'lr': self.lr,
            'model': self.model,
            # 'reg': self.reg,
            'history': history,
            'user_embed': self.user_embed,
            'user_embeds': self.user_embeds,
            'item_embed': self.item_embed,
            'item_embeds': self.item_embeds,
        }
        t.save(params, savePath)

    def loadModel(self, loadPath):      
        ModelName = self.modelName
        # loadPath = r'./Model/' + args.dataset + r'/' + ModelName + r'.pth'
        loadPath = loadPath
        checkpoint = t.load(loadPath)
        self.model = checkpoint['model']

        self.curEpoch = checkpoint['epoch'] + 1
        # self.lr = checkpoint['lr']
        # self.args.reg = checkpoint['reg']
        history = checkpoint['history']
        self.train_loss = history['loss']
        self.his_hr = history['HR']
        self.his_ndcg = history['NDCG']
        # log("load model %s in epoch %d"%(modelPath, checkpoint['epoch']))


if __name__ == '__main__':
    print(args)
    my_model = Model()  

    my_model.run()
    # my_model.test()

