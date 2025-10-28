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
from torchmetrics.regression import CosineSimilarity

import graph_utils
import DataHandler
#import Gru

import AGNN
import HY_GNN
import hyper_graph_utils
# import SE_NET

from Params import args
from Utils.TimeLogger import log
from tqdm import tqdm


import wandb
import warnings
warnings.filterwarnings("ignore")


t.backends.cudnn.benchmark=True
t.autograd.set_detect_anomaly(True)

device = t.device(f"cuda:{args.cuda_num}" if t.cuda.is_available() else "cpu")

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


        self.meta_multi_single_file = args.path + args.dataset + '/meta_multi_single_beh_user_index_shuffle'
        self.meta_multi_single = pickle.load(open(self.meta_multi_single_file, 'rb')) 

        self.gru_user_embeddings = pickle.load(open('./data/Tmall/only_user_gru.pkl', 'rb'))  
        self.gru_item_embeddings = pickle.load(open('./data/Tmall/all_item_gru.pkl', 'rb'))

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


        if args.dataset == 'Tmall':
            self.predir = '/home/joo/JOOCML/data/Tmall/'
            self.behaviors_SSL = ['pv','fav', 'cart', 'buy']
            self.behaviors = ['pv','fav', 'cart', 'buy']


        elif args.dataset == 'IJCAI_15':
            self.predir = './data/IJCAI_15/'
            self.behaviors = ['click','fav', 'cart', 'buy']
            self.behaviors_SSL = ['click','fav', 'cart', 'buy']


        elif args.dataset == 'beibei':
            self.predir = './data/beibei/'
            self.behaviors = ['pv', 'cart', 'buy']
            self.behaviors_SSL = ['pv', 'cart', 'buy']


        elif args.dataset == 'retail_rocket':
            self.predir = './data/retail_rocket/'
            self.behaviors = ['view','cart', 'buy']
            self.behaviors_SSL = ['view','cart', 'buy']
        
        elif args.data == 'taobao':
            self.predir = './data/taobao/'
            self.behaviors = ['pv', 'cart', 'buy']


        for i in range(0, len(self.behaviors)): #[5]
            with open(self.trn_file + self.behaviors[i], 'rb') as fs:
                data = pickle.load(fs) 
                
                self.behaviors_data[i] = data
                if data.get_shape()[0] > self.user_num:  
                    self.user_num = data.get_shape()[0]  
                if data.get_shape()[1] > self.item_num: 
                    self.item_num = data.get_shape()[1]  

             
                if data.data.max() > self.t_max:  
                    self.t_max = data.data.max() 
                if data.data.min() < self.t_min:
                    self.t_min = data.data.min()

        
                if self.behaviors[i]==args.target: 
                    self.trainMat = data  
                    self.trainLabel = 1*(self.trainMat != 0)   
                    self.labelP = np.squeeze(np.array(np.sum(self.trainLabel, axis=0)))   
        self.test_mat = pickle.load(open(self.test_file, 'rb')) 
        self.userNum = self.behaviors_data[0].shape[0]  
        self.itemNum = self.behaviors_data[0].shape[1]  

        time = datetime.datetime.now()
        
        print("Start BEHAVIOR_building:  ", time)
        for i in range(0, len(self.behaviors_data)):          
            self.behaviors_data[i] = 1*(self.behaviors_data[i]!=0) 
            self.behavior_mats[i] = graph_utils.get_use(self.behaviors_data[i]) #  

            self.hyper_behavior_Adj[i] = hyper_graph_utils.get_hyper_use(self.behaviors_data[i])
         
        time = datetime.datetime.now()
        print("End BEHAVIOR_building:", time)

        print("user_num: ", self.user_num) 
        print("item_num: ", self.item_num) 
        print("\n")
        
        train_u, train_v = self.trainMat.nonzero()  
        train_data = np.hstack((train_u.reshape(-1,1), train_v.reshape(-1,1))).tolist() 

        train_dataset = DataHandler.RecDataset_beh(self.behaviors, train_data, self.item_num, self.behaviors_data, True)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

        with open(self.tst_file, 'rb') as fs:   
            data = pickle.load(fs) 
        test_user = np.array([idx for idx, i in enumerate(data) if i is not None]) 
        test_item = np.array([i for idx, i in enumerate(data) if i is not None]) 
        test_data = np.hstack((test_user.reshape(-1,1), test_item.reshape(-1,1))).tolist() 
        test_dataset = DataHandler.RecDataset(test_data, self.item_num, self.trainMat, 0, False)  
        self.test_loader = dataloader.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)  
        # -------------------------------------------------------------------------------------------------->>>>>

    def prepareModel(self):
        self.modelName = self.getModelName()  
        self.hidden_dim = args.hidden_dim 
        

        if args.isload == True:
            self.loadModel(args.loadModelPath)
        else:
            self.model = AGNN.myModel(self.user_num, self.item_num, self.behaviors, self.behavior_mats).to(device)

            


        # #IJCAI_15
        # self.opt = t.optim.AdamW(self.model.parameters(), lr = args.lr, weight_decay = args.opt_weight_decay)
        # self.scheduler = t.optim.lr_scheduler.CyclicLR(self.opt, args.opt_base_lr, args.opt_max_lr, step_size_up=5, step_size_down=10, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)


        #Tmall
        self.opt = t.optim.AdamW(self.model.parameters(), lr = args.lr, weight_decay = args.opt_weight_decay)
        self.scheduler = t.optim.lr_scheduler.CyclicLR(self.opt, args.opt_base_lr, args.opt_max_lr, step_size_up=5, step_size_down=10, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)

        # retailrocket
        # self.opt = t.optim.AdamW(self.model.parameters(), lr = args.lr, weight_decay = args.opt_weight_decay)
        # self.scheduler = t.optim.lr_scheduler.CyclicLR(self.opt, args.opt_base_lr, args.opt_max_lr, step_size_up=1, step_size_down=3, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
                                                                                                                                   
        # if use_cuda:
        self.model = self.model.to(device)

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
                index_set_neg = t.as_tensor(np.array(list(index_set))).long().to(device)  

                x_pos = x1[i].repeat(x1.shape[0]-1, 1)
                x_neg = x2[index_set]  
                
                if i==0:
                    x_pos_all = x_pos
                    x_neg_all = x_neg
                else:
                    x_pos_all = t.cat((x_pos_all, x_pos), 0)
                    x_neg_all = t.cat((x_neg_all, x_neg), 0)
            x_pos_all = t.as_tensor(x_pos_all) 
            x_neg_all = t.as_tensor(x_neg_all)  

            return x_pos_all, x_neg_all

        def multi_neg_sample_pair_index(batch_index, step_index, embedding1, embedding2):  

            index_set = set(np.array(step_index.cpu())) 
            batch_index_set = set(np.array(batch_index.cpu()))  
            neg2_index_set = index_set - batch_index_set                         
            neg2_index = t.as_tensor(np.array(list(neg2_index_set))).long().to(device)  
            neg2_index = t.unsqueeze(neg2_index, 0)                             
            neg2_index = neg2_index.repeat(len(batch_index), 1)             
            neg2_index = t.reshape(neg2_index, (1, -1))                       
            neg2_index = t.squeeze(neg2_index)                                 
                                                                               
            neg1_index = batch_index.long().to(device)                            
            neg1_index = t.unsqueeze(neg1_index, 1)                            
            neg1_index = neg1_index.repeat(1, len(neg2_index_set))              
            neg1_index = t.reshape(neg1_index, (1, -1))                              
            neg1_index = t.squeeze(neg1_index)                                 

            neg_score_pre = t.sum(compute(embedding1, embedding2, neg1_index, neg2_index).squeeze().view(len(batch_index), -1), -1)  
                                     
            return neg_score_pre  

        def compute(x1, x2, neg1_index=None, neg2_index=None, τ = 0.05): 

            if neg1_index!=None:
                x1 = x1[neg1_index] 
                x2 = x2[neg2_index] 
            N = x1.shape[0]   
            D = x1.shape[1]   

            x1 = x1  
            x2 = x2  

            scores = t.exp(t.div(t.bmm(x1.view(N, 1, D), x2.view(N, D, 1)).view(N, 1), np.power(D, 1)+1e-8)) 
            return scores  
        

        def single_infoNCE_loss_one_by_one(embedding1, embedding2, step_index):  
            N = step_index.shape[0]  
            D = embedding1.shape[1]  

            pos_score = compute(embedding1[step_index], embedding2[step_index]).squeeze()  
            neg_score = t.zeros((N,), dtype = t.float64).to(device)  

            steps = int(np.ceil(N / args.SSL_batch))  
            for i in range(steps):
                st = i * args.SSL_batch
                ed = min((i+1) * args.SSL_batch, N)
                batch_index = step_index[st: ed] 

                neg_score_pre = multi_neg_sample_pair_index(batch_index, step_index, embedding1, embedding2) 
                if i ==0:
                    neg_score = neg_score_pre
                else:
                    neg_score = t.cat((neg_score, neg_score_pre), 0)
            con_loss = -t.log(1e-8 +t.div(pos_score, neg_score+1e-8))  


            assert not t.any(t.isnan(con_loss))
            assert not t.any(t.isinf(con_loss))

            return t.where(t.isnan(con_loss), t.full_like(con_loss, 0+1e-8), con_loss)

        user_con_loss_list = []  
        item_con_loss_list = []

        SSL_len = int(user_step_index.shape[0]/10)  
        user_step_index = t.as_tensor(np.random.choice(user_step_index.cpu(), size=SSL_len, replace=False, p=None)).to(device) 

        for i in range(len(self.behaviors_SSL)):  

            user_con_loss_list.append(single_infoNCE_loss_one_by_one(user_embeddings[-1], user_embeddings[i], user_step_index))
        user_con_losss = t.stack(user_con_loss_list, dim=0)  

        return user_con_loss_list, user_step_index  

    def run(self):
        
        

        wandb.init(project="TempSSL")
        wandb.run.name = args.wandb
        wandb.config.update(args)
                
        self.prepareModel()

        cvWait = 0  
        self.best_HR = 0 
        self.best_NDCG = 0
        flag = 0

        self.user_embed = None 
        self.item_embed = None
        self.user_embeds = None
        self.item_embeds = None


        print("Test before train:")

        for e in tqdm(range(self.curEpoch, args.epoch+1)):  
            self.curEpoch = e

            self.meta_flag = 0
            if e%args.meta_slot == 0:
                self.meta_flag=1


            log("*****************Start epoch: %d ************************"%e)  
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

            if HR > self.best_HR:
                self.best_HR = HR
                self.best_epoch = self.curEpoch 
                cvWait = 0
                print("--------------------------------------------------------------------------------------------------------------------------best_HR", self.best_HR)
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
        while cur < sampSize:  
            rdmItm = np.random.choice(nodeNum)   
            if temLabel[rdmItm] == 0:     
                negset[cur] = rdmItm   
                cur += 1
        return negset 

    def sampleTrainBatch(self, batIds, labelMat):
        temLabel = labelMat[batIds.cpu()].toarray()  
        batch = len(batIds)
        user_id = []      
        item_id_pos = []  
        item_id_neg = [] 
 
        cur = 0
        for i in range(batch): 
            posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])  
            sampNum = min(args.sampNum, len(posset))   
            if sampNum == 0:
                poslocs = [np.random.choice(labelMat.shape[1])]  
                neglocs = [poslocs[0]]                             
            else:
                poslocs = np.random.choice(posset, sampNum)    
                neglocs = self.negSamp(temLabel[i], sampNum, labelMat.shape[1])  

            for j in range(sampNum):
                user_id.append(batIds[i].item())  
                item_id_pos.append(poslocs[j].item())  
                item_id_neg.append(neglocs[j])         
                cur += 1

        return t.as_tensor(np.array(user_id)).to(device), t.as_tensor(np.array(item_id_pos)).to(device), t.as_tensor(np.array(item_id_neg)).to(device) 


    def trainEpoch(self):   
        train_loader = self.train_loader
        train_meta_loader = self.train_loader
        time = datetime.datetime.now()

        print("start_beh_ng_samp:  ", time)
        train_loader.dataset.ng_sample()
        print("end__beh_ng_samp:  ", time)


        epoch_loss = 0
        
#-----------------------------------------------------------------------------------
        self.behavior_loss_list = [None]*len(self.behaviors)  
        self.user_id_list = [None]*len(self.behaviors)        
        self.item_id_pos_list = [None]*len(self.behaviors)    
        self.item_id_neg_list = [None]*len(self.behaviors)     

        self.meta_start_index = 0   
        self.meta_end_index = self.meta_start_index + args.meta_batch  

        self.mp_behavior_loss_list = [None]*len(self.behaviors)   
        self.mp_user_id_list = [None]*len(self.behaviors)       
        self.mp_item_id_pos_list = [None]*len(self.behaviors)     
        self.mp_item_id_neg_list = [None]*len(self.behaviors)     

        self.mp_meta_start_index = 0  
        self.mp_meta_end_index = self.mp_meta_start_index + args.meta_batch 
#----------------------------------------------------------------------------------
        
        cnt = 0
        args.isJustbeh == True

        for user, item_i, item_j in train_loader:   
        
            user = user.long().to(device)
            self.user_step_index = user
            #print(len(user)) #8192

            self.meta_user = t.as_tensor(self.meta_multi_single[self.meta_start_index:self.meta_end_index]).to(device)  
            
            if self.meta_end_index == self.meta_multi_single.shape[0]:
                self.meta_start_index = 0  
            else:
                self.meta_start_index = (self.meta_start_index + args.meta_batch) % (self.meta_multi_single.shape[0] - 1) 
            self.meta_end_index = min(self.meta_start_index + args.meta_batch, self.meta_multi_single.shape[0])


#---round one---------------------------------------------------------------------------------------------
            # print("round_1")
            meta_behavior_loss_list = [None]*len(self.behaviors)  
            meta_user_index_list = [None]*len(self.behaviors)   


            meta_model = AGNN.myModel(self.user_num, self.item_num, self.behaviors, self.behavior_mats).to(device) 
            meta_opt = t.optim.AdamW(meta_model.parameters(), lr = args.lr, weight_decay = args.opt_weight_decay)
            meta_model.load_state_dict(self.model.state_dict())

            
            meta_user_embed, meta_item_embed, meta_user_embeds, meta_item_embeds = meta_model()
        

            for index in range(len(self.behaviors)):

                not_zero_index = np.where(item_i[index].cpu().numpy()!=-1)[0]  
                self.user_id_list[index] = user[not_zero_index].long().to(device)
                meta_user_index_list[index] = self.user_id_list[index]
                self.item_id_pos_list[index] = item_i[index][not_zero_index].long().to(device)
                self.item_id_neg_list[index] = item_j[index][not_zero_index].long().to(device)

                meta_userEmbed = meta_user_embed[self.user_id_list[index]]   
                meta_posEmbed = meta_item_embed[self.item_id_pos_list[index]] 
                meta_negEmbed = meta_item_embed[self.item_id_neg_list[index]] 

                meta_pred_i, meta_pred_j = 0, 0
                meta_pred_i, meta_pred_j = self.innerProduct(meta_userEmbed, meta_posEmbed, meta_negEmbed)
                
                meta_behavior_loss_list[index] = - (meta_pred_i.view(-1) - meta_pred_j.view(-1)).sigmoid().log()


            meta_infoNCELoss_list, SSL_user_step_index = self.SSL(meta_user_embeds, meta_item_embeds, meta_user_embed, meta_item_embed, self.user_step_index)


            for i in range(len(self.behaviors)):  
                meta_infoNCELoss_list[i] = (meta_infoNCELoss_list[i]).sum()   
                meta_behavior_loss_list[i] = (meta_behavior_loss_list[i]).sum()  


            meta_bprloss = sum(meta_behavior_loss_list) / len(meta_behavior_loss_list)  
            meta_infoNCELoss = sum(meta_infoNCELoss_list) / len(meta_infoNCELoss_list) 
            meta_regLoss = (t.norm(meta_userEmbed) ** 2 + t.norm(meta_posEmbed) ** 2 + t.norm(meta_negEmbed) ** 2) 
            meta_model_loss = (meta_bprloss + args.reg * meta_regLoss + args.beta*meta_infoNCELoss) / args.batch    
            meta_opt.zero_grad(set_to_none=True)
            meta_model_loss.backward()
            nn.utils.clip_grad_norm_(meta_model.parameters(), max_norm=20, norm_type=2)
            meta_opt.step()
            
#---round one---------------------------------------------------------------------------------------------



#---round two---------------------------------------------------------------------------------------------
            # print("round_2")
            behavior_loss_list = [None]*len(self.behaviors)
            user_index_list = [None]*len(self.behaviors)  #---

            user_embed, item_embed, user_embeds, item_embeds = meta_model()


            for index in range(len(self.behaviors)):

                user_id, item_id_pos, item_id_neg = self.sampleTrainBatch(t.as_tensor(self.meta_user), self.behaviors_data[i])

                user_index_list[index] = user_id  
                userEmbed = user_embed[user_id] 

                posEmbed = item_embed[item_id_pos] 
                negEmbed = item_embed[item_id_neg] 

                pred_i, pred_j = self.innerProduct(userEmbed, posEmbed, negEmbed)
                behavior_loss_list[index] = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log()  
            


            self.infoNCELoss_list, SSL_user_step_index = self.SSL(user_embeds, item_embeds, user_embed, item_embed, self.meta_user)

 
            for i in range(len(self.behaviors)):
                self.infoNCELoss_list[i] = (self.infoNCELoss_list[i]).sum()
                behavior_loss_list[i] = (behavior_loss_list[i]).sum()   


            bprloss = sum(behavior_loss_list) / len(self.behavior_loss_list)
            infoNCELoss = sum(self.infoNCELoss_list) / len(self.infoNCELoss_list)
            round_two_regLoss = (t.norm(userEmbed) ** 2 + t.norm(posEmbed) ** 2 + t.norm(negEmbed) ** 2)

            meta_loss = (bprloss + args.reg * round_two_regLoss  + args.beta*infoNCELoss) / args.batch
            meta_loss.backward()

#---round three---------------------------------------------------------------------------------------------

            # print("round_3")
            user_embed, item_embed, user_embeds, item_embeds = self.model()
 

            for index in range(len(self.behaviors)):


                userEmbed = user_embed[self.user_id_list[index]]  
                posEmbed = item_embed[self.item_id_pos_list[index]]  
                negEmbed = item_embed[self.item_id_neg_list[index]]  

                pred_i, pred_j = 0, 0
                pred_i, pred_j = self.innerProduct(userEmbed, posEmbed, negEmbed)  

                self.behavior_loss_list[index] = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log()  
            infoNCELoss_list, SSL_user_step_index = self.SSL(user_embeds, item_embeds, user_embed, item_embed, self.user_step_index)


            for i in range(len(self.behaviors)):
                infoNCELoss_list[i] = (infoNCELoss_list[i]).sum()
                self.behavior_loss_list[i] = (self.behavior_loss_list[i]).sum()  
                

            bprloss = sum(self.behavior_loss_list) / len(self.behavior_loss_list)
            infoNCELoss = sum(infoNCELoss_list) / len(infoNCELoss_list)
            regLoss = (t.norm(userEmbed) ** 2 + t.norm(posEmbed) ** 2 + t.norm(negEmbed) ** 2)
            
           
            loss = (bprloss + args.reg * regLoss + args.beta*infoNCELoss) / args.batch
            epoch_loss = epoch_loss + loss.item()

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            self.opt.step()


#=========== HYPER-GRAPH LEARNING  ===========================================================================
        args.is_Meta_Path == True
        for mp_user, mp_item_i, mp_item_j in train_loader:  
        
            mp_user = mp_user.long().to(device)
            self.mp_user_step_index = mp_user

            self.mp_meta_user = t.as_tensor(self.meta_multi_single[self.mp_meta_start_index:self.mp_meta_end_index]).to(device)  
            
            if self.mp_meta_end_index == self.meta_multi_single.shape[0]:
                self.mp_meta_start_index = 0  
            else:
                self.mp_meta_start_index = (self.mp_meta_start_index + args.meta_batch) % (self.meta_multi_single.shape[0] - 1) 
            self.mp_meta_end_index = min(self.mp_meta_start_index + args.meta_batch, self.meta_multi_single.shape[0])


#--- HYPER-GRAPH round one---------------------------------------------------------------------------------------------
            # print("round_1")
            mp_meta_behavior_loss_list = [None]*len(self.behaviors)   
            mp_meta_user_index_list = [None]*len(self.behaviors)    

            mp_meta_model = HY_GNN.myModel(self.user_num, self.item_num, self.behaviors,self.hyper_behavior_Adj).to(device)
            hyper_meta_opt = t.optim.AdamW(mp_meta_model.parameters(), lr=args.hyper_lr, weight_decay=0)

            mp_meta_model.load_state_dict(self.model.state_dict(), strict=False)

            mp_meta_user_embed, mp_meta_item_embed, mp_meta_user_embeds, mp_meta_item_embeds = mp_meta_model()


            for index in range(len(self.behaviors)):

                mp_not_zero_index = np.where(mp_item_i[index].cpu().numpy()!=-1)[0] 
                self.mp_user_id_list[index] = mp_user[mp_not_zero_index].long().to(device)
                mp_meta_user_index_list[index] = self.mp_user_id_list[index]
                self.mp_item_id_pos_list[index] = mp_item_i[index][mp_not_zero_index].long().to(device)
                self.mp_item_id_neg_list[index] = mp_item_j[index][mp_not_zero_index].long().to(device)

                mp_meta_userEmbed = mp_meta_user_embed[self.mp_user_id_list[index]]   
                mp_meta_posEmbed = mp_meta_item_embed[self.mp_item_id_pos_list[index]] 
                mp_meta_negEmbed = mp_meta_item_embed[self.mp_item_id_neg_list[index]] 

                mp_meta_pred_i, mp_meta_pred_j = 0, 0
                mp_meta_pred_i, mp_meta_pred_j = self.innerProduct(mp_meta_userEmbed, mp_meta_posEmbed, mp_meta_negEmbed)
                
                mp_meta_behavior_loss_list[index] = - (mp_meta_pred_i.view(-1) - mp_meta_pred_j.view(-1)).sigmoid().log()

            
            mp_meta_infoNCELoss_list, mp_SSL_user_step_index = self.SSL(mp_meta_user_embeds, mp_meta_item_embeds, mp_meta_user_embed, mp_meta_item_embed, self.mp_user_step_index)


            for i in range(len(self.behaviors)):   
                mp_meta_infoNCELoss_list[i] = (mp_meta_infoNCELoss_list[i]).sum()  
                mp_meta_behavior_loss_list[i] = (mp_meta_behavior_loss_list[i]).sum()    


            mp_meta_bprloss = sum(mp_meta_behavior_loss_list) / len(mp_meta_behavior_loss_list)
            mp_meta_infoNCELoss = sum(mp_meta_infoNCELoss_list) / len(mp_meta_infoNCELoss_list) 
            mp_meta_regLoss = (t.norm(mp_meta_userEmbed) ** 2 + t.norm(mp_meta_posEmbed) ** 2 + t.norm(mp_meta_negEmbed) ** 2)  

            mp_meta_model_loss = (mp_meta_bprloss + args.reg * mp_meta_regLoss + args.beta*mp_meta_infoNCELoss) / args.batch   
            
            hyper_meta_opt.zero_grad(set_to_none=True)
            mp_meta_model_loss.backward()
            nn.utils.clip_grad_norm_(mp_meta_model.parameters(), max_norm=20, norm_type=2)
            hyper_meta_opt.step()
#---HYPER-GRAPH round one---------------------------------------------------------------------------------------------



#---HYPER-GRAPH round two---------------------------------------------------------------------------------------------
            # print("round_2")
            mp_behavior_loss_list = [None]*len(self.behaviors)
            mp_user_index_list = [None]*len(self.behaviors)  #---

            mp_user_embed, mp_item_embed, mp_user_embeds, mp_item_embeds = mp_meta_model()
            
            for index in range(len(self.behaviors)):

                mp_user_id, mp_item_id_pos, mp_item_id_neg = self.sampleTrainBatch(t.as_tensor(self.mp_meta_user), self.behaviors_data[i])

                mp_user_index_list[index] = mp_user_id 
                mp_userEmbed = mp_user_embed[mp_user_id]  

                mp_posEmbed = mp_item_embed[mp_item_id_pos]  
                mp_negEmbed = mp_item_embed[mp_item_id_neg]  

                mp_pred_i, mp_pred_j = self.innerProduct(mp_userEmbed, mp_posEmbed, mp_negEmbed)
                mp_behavior_loss_list[index] = - (mp_pred_i.view(-1) - mp_pred_j.view(-1)).sigmoid().log()  
            


            self.mp_infoNCELoss_list, mp_SSL_user_step_index = self.SSL(mp_user_embeds, mp_item_embeds, mp_user_embed, mp_item_embed, self.mp_meta_user)

            for i in range(len(self.behaviors)): 
                self.mp_infoNCELoss_list[i] = (self.mp_infoNCELoss_list[i]).sum()
                mp_behavior_loss_list[i] = (mp_behavior_loss_list[i]).sum()   

            mp_bprloss = sum(mp_behavior_loss_list) / len(self.mp_behavior_loss_list)
            mp_infoNCELoss = sum(self.mp_infoNCELoss_list) / len(self.mp_infoNCELoss_list)
            mp_round_two_regLoss = (t.norm(mp_userEmbed) ** 2 + t.norm(mp_posEmbed) ** 2 + t.norm(mp_negEmbed) ** 2)

            mp_meta_loss = (mp_bprloss + args.reg * mp_round_two_regLoss  + args.beta*mp_infoNCELoss) / args.batch
            mp_meta_loss.backward()
#---HYPER-GRAPH round two-----------------------------------------------------------------------------------------------



#---HYPER-GRAPH round three---------------------------------------------------------------------------------------------

            # print("round_3")
            mp_user_embed, mp_item_embed, mp_user_embeds, mp_item_embeds = mp_meta_model()


            for index in range(len(self.behaviors)):
                mp_userEmbed = mp_user_embed[self.mp_user_id_list[index]]  
                mp_posEmbed = mp_item_embed[self.mp_item_id_pos_list[index]]  
                mp_negEmbed = mp_item_embed[self.mp_item_id_neg_list[index]] 

                mp_pred_i, mp_pred_j = 0, 0
                mp_pred_i, mp_pred_j = self.innerProduct(mp_userEmbed, mp_posEmbed, mp_negEmbed)  

                self.mp_behavior_loss_list[index] = - (mp_pred_i.view(-1) - mp_pred_j.view(-1)).sigmoid().log()   
            
            with t.no_grad():
                mp_infoNCELoss_list, mp_SSL_user_step_index = self.SSL(mp_user_embeds, mp_item_embeds, mp_user_embed, mp_item_embed, self.mp_user_step_index)


            for i in range(len(self.behaviors)):
                mp_infoNCELoss_list[i] = (mp_infoNCELoss_list[i]).sum()
                self.mp_behavior_loss_list[i] = (self.mp_behavior_loss_list[i]).sum()  
                

            mp_bprloss = sum(self.mp_behavior_loss_list) / len(self.mp_behavior_loss_list)
            mp_infoNCELoss = sum(mp_infoNCELoss_list) / len(mp_infoNCELoss_list)
            mp_regLoss = (t.norm(mp_userEmbed) ** 2 + t.norm(mp_posEmbed) ** 2 + t.norm(mp_negEmbed) ** 2)          
            mp_loss = (mp_bprloss + args.reg * mp_regLoss + args.beta*mp_infoNCELoss) / args.batch           
            epoch_loss = epoch_loss + mp_loss.item()

            self.opt.zero_grad(set_to_none=True)
            mp_loss.backward()

            nn.utils.clip_grad_norm_(mp_meta_model.parameters(), max_norm=20, norm_type=2)
            self.opt.step()


#---round three---------------------------------------------------------------------------------------------
            cnt+=1

        return epoch_loss, user_embed, item_embed, user_embeds, item_embeds


    def testEpoch(self, data_loader, save=False): 
        #print("test!!!!")
        epochHR, epochNDCG = [0]*2
        with t.no_grad():
            user_embed, item_embed, user_embeds, item_embeds = self.model()

        cnt = 0
        tot = 0
        for user, item_i in data_loader:
            user_compute, item_compute, user_item1, user_item100 = self.sampleTestBatch(user, item_i)  
            userEmbed = user_embed[user_compute]  
            itemEmbed = item_embed[item_compute]  
           
            pred_i = t.sum(t.mul(userEmbed, itemEmbed), dim=1)  

            hit, ndcg = self.calcRes(t.reshape(pred_i, [user.shape[0], 100]), user_item1, user_item100)  
            epochHR = epochHR + hit  
            epochNDCG = epochNDCG + ndcg  
            cnt += 1 
            tot += user.shape[0]
            print("tot:",tot)
        
        result_HR = epochHR / tot
        result_NDCG = epochNDCG / tot
        print(f"Step {cnt}:  hit:{result_HR}, ndcg:{result_NDCG}")

        return result_HR, result_NDCG
   

    def calcRes(self, pred_i, user_item1, user_item100):  
        hit = 0
        ndcg = 0

    
        for j in range(pred_i.shape[0]): 
            _, shoot_index = t.topk(pred_i[j], args.shoot) 
            shoot_index = shoot_index.cpu() 
            shoot = user_item100[j][shoot_index]
            shoot = shoot.tolist()
            if type(shoot)!=int and (user_item1[j] in shoot): 
                hit += 1                                       
                ndcg += np.reciprocal( np.log2( shoot.index( user_item1[j])+2))  
            elif type(shoot)==int and (user_item1[j] == shoot):  
                hit += 1  
                ndcg += np.reciprocal( np.log2( 0+2))
    
        return hit, ndcg  

    def sampleTestBatch(self, batch_user_id, batch_item_id): 
        batch = len(batch_user_id) 
        tmplen = (batch*100)
        sub_trainMat = self.trainMat[batch_user_id].toarray()  
        user_item1 = batch_item_id
        user_compute = [None] * tmplen 
        item_compute = [None] * tmplen
        user_item100 = [None] * (batch)  

        cur = 0
        for i in range(batch): 
            pos_item = user_item1[i]  
            negset = np.reshape(np.argwhere(sub_trainMat[i]==0), [-1])   
            pvec = self.labelP[negset]  
            pvec = pvec / np.sum(pvec)  
            random_neg_sam = np.random.permutation(negset)[:99]   
            user_item100_one_user = np.concatenate(( random_neg_sam, np.array([pos_item]))) 
            user_item100[i] = user_item100_one_user 

            for j in range(100):
                user_compute[cur] = batch_user_id[i]  
                item_compute[cur] = user_item100_one_user[j] 
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
        
        with open(args.History_path + args.dataset + '/' + ModelName + '.his', 'wb') as fs: 
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
            'model': self.model,
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

        history = checkpoint['history']
        self.train_loss = history['loss']
        self.his_hr = history['HR']
        self.his_ndcg = history['NDCG']
        # log("load model %s in epoch %d"%(modelPath, checkpoint['epoch']))


if __name__ == '__main__':
    print(args)
    my_model = Model()  
    my_model.run()


