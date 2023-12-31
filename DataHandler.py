import torch
import torch.utils.data as data
import pickle
import numpy as np
import scipy.sparse as sp
from math import ceil
import datetime

from Params import args
import graph_utils

#https://lovit.github.io/nlp/machine%20learning/2018/04/09/sparse_mtarix_handling/

class RecDataset(data.Dataset):
    def __init__(self, data, num_item, train_mat=None, num_ng=1, is_training=True):  
        super(RecDataset, self).__init__()

        self.data = np.array(data) # data:: [[6, 253], [9, 392], [11, 462], [17, 726], [18, 738], [24, 1008], [28, 1142], [30, 1210],..., ]
                                   # [array([  6, 253]), array([  9, 392]), 
        self.num_item = num_item
        self.train_mat = train_mat
        self.is_training = is_training

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'  
        dok_trainMat = self.train_mat.todok()  
        length = self.data.shape[0]  
        self.neg_data = np.random.randint(low=0, high=self.num_item, size=length)  

        for i in range(length):  #
            uid = self.data[i][0]
            iid = self.neg_data[i]
            if (uid, iid) in dok_trainMat:
                while (uid, iid) in dok_trainMat:  
                    iid = np.random.randint(low=0, high=self.num_item)
                    self.neg_data[i] = iid
                self.neg_data[i] = iid  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        item_i = self.data[idx][1]

        if self.is_training:  
            neg_data = self.neg_data
            item_j = neg_data[idx]
            return user, item_i, item_j
        else:  
            return user, item_i

    def getMatrix(self):
        pass
    
    def getAdj(self):
        pass
    
    def sampleLargeGraph(self):
   
    
        def makeMask():
            pass
    
        def updateBdgt():
            pass
    
        def sample():
            pass
    
    def constructData(self):
        pass




class RecDataset_beh(data.Dataset):
    def __init__(self, beh, data, num_item, behaviors_data=None, num_ng=1, is_training=True):  
        super(RecDataset_beh, self).__init__()

        self.data = np.array(data) #  len(167862) [7] 전체 데이터 [[0,2],[0,9],[0,10],[0,12],...,[22436, 18816],[22437, 6871], [22437,7438],[22437,3116],[22437, 33318]]
        self.num_item = num_item  # 31232
        self.is_training = is_training
        self.beh = beh   # ['pv', 'fav', 'cart', 'buy'] 0: 'pv', 1: 'fav', 2: 'cart', 3:'buy' 
        self.behaviors_data = behaviors_data
        #csr_matrix의 경우 데이터를 data, indptr, indices로 관리
        # Compressed Sparse Row 의 약자입니다. Row 순서대로 데이터를 저장
        # sparse matrix 의 구성요소는 정확히는 (data, indices, indptr)
        # indptr[i]는 i번째 행의 원소가 data의 어느 인덱스에서 시작되는지 나타낸다.
        # indices[i]는 data[i]의 열 번호를 저장한다. 
        # 0 'click': time stamp array([1432396800, 1432396800,1436889600,...,]) 
        # 1 'fav'  : time stamp array([1446739200, 1446739200, 1440000000, ...,]) 158119


        self.length = self.data.shape[0]  #shape:(167862, 2)
        self.neg_data = [None]*self.length  #[None]* 167862
        self.pos_data = [None]*self.length  

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'  
 
        for i in range(self.length):  #self.length 167862
            self.neg_data[i] = [None]*len(self.beh) # [None, None, None, None]
            self.pos_data[i] = [None]*len(self.beh) # [None, None, None, None]

        for index in range(len(self.beh)):


            train_u, train_v = self.behaviors_data[index].nonzero()
            beh_dok = self.behaviors_data[index].todok()  #len(beh_dok)= 1330, shape:(22438, 35573)
            # dok(Dictionary of Keys)는 좌표가 key이고 원소 값이 value인 딕셔너리 구조

            set_pos = np.array(list(set(train_v))) #shape:(35573,)

            self.pos_data_index = np.random.choice(set_pos, size=self.length, replace=True, p=None) #shape:(199654,)
            self.neg_data_index = np.random.randint(low=0, high=self.num_item, size=self.length)  #shape:(199654,)


            for i in range(self.length):  #

                uid = self.data[i][0]
                iid_neg = self.neg_data[i][index] = self.neg_data_index[i] #29579
                iid_pos = self.pos_data[i][index] = self.pos_data_index[i] #2644

                if (uid, iid_neg) in beh_dok:
                    while (uid, iid_neg) in beh_dok:
                        iid_neg = np.random.randint(low=0, high=self.num_item)  
                        self.neg_data[i][index] = iid_neg
                    self.neg_data[i][index] = iid_neg

                if index == (len(self.beh)-1):
                    self.pos_data[i][index] = train_v[i]
                elif (uid, iid_pos) not in beh_dok:
                    if len(self.behaviors_data[index][uid].data)==0:
                        self.pos_data[i][index] = -1
                    else:
                        t_array = self.behaviors_data[index][uid].toarray()
                        pos_index = np.where(t_array!=0)[1]
                        iid_pos = np.random.choice(pos_index, size = 1, replace=True, p=None)[0]   
                        self.pos_data[i][index] = iid_pos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        item_i = self.pos_data[idx]

        if self.is_training:  
            item_j = self.neg_data[idx]
            return user, item_i, item_j
        else:  
            return user, item_i



# class RecDataset_beh_path(data.Dataset):
#     def __init__(self, beh, data, num_item, beh_meta_path =None,behaviors_data=None, num_ng=1, is_training=True):  
#         super(RecDataset_beh_path, self).__init__()

#         self.data = np.array(data) # [7] 전체 데이터 [[0,2],[0,9],[0,10],[0,12],...,[22436, 18816],[22437, 6871], [22437,7438],[22437,3116],[22437, 33318]]
#         self.num_item = num_item  # 35573
#         self.is_training = is_training
#         self.beh = beh   # ['click', 'fav', 'cart', 'buy'] 0: 'click', 1: 'fav', 2: 'cart', 3:'buy' 
#         self.behaviors_data = behaviors_data
#         self.beh_meta_path = beh_meta_path
#         #csr_matrix의 경우 데이터를 data, indptr, indices로 관리
#         # Compressed Sparse Row 의 약자입니다. Row 순서대로 데이터를 저장
#         # sparse matrix 의 구성요소는 정확히는 (data, indices, indptr)
#         # indptr[i]는 i번째 행의 원소가 data의 어느 인덱스에서 시작되는지 나타낸다.
#         # indices[i]는 data[i]의 열 번호를 저장한다. 
#         # 0 'click': time stamp array([1432396800, 1432396800,1436889600,...,]) 
#         # 1 'fav'  : time stamp array([1446739200, 1446739200, 1440000000, ...,]) 158119


#         self.length = self.data.shape[0]  #(199654,2)
#         self.neg_data = [None]*self.length  #[None]* 199654
#         self.pos_data = [None]*self.length  

#     def ng_sample(self):
#         assert self.is_training, 'no need to sampling when testing'  
 
#         for i in range(self.length):  #self.length 199654
#             self.neg_data[i] = [None]*len(self.beh_meta_path) # [None, None, None, None]
#             self.pos_data[i] = [None]*len(self.beh_meta_path) # [None, None, None, None]

#         for index in range(len(self.beh)):


#             train_u, train_v = self.beh_meta_path_data[index].nonzero()
#             beh_path_dok = self.beh_meta_path_data[index].todok()  #len(beh_dok)= 1330, shape:(22438, 35573)
#             # dok(Dictionary of Keys)는 좌표가 key이고 원소 값이 value인 딕셔너리 구조

#             set_pos = np.array(list(set(train_v))) #shape:(35573,)

#             self.pos_data_index = np.random.choice(set_pos, size=self.length, replace=True, p=None) #shape:(199654,)
#             self.neg_data_index = np.random.randint(low=0, high=self.num_item, size=self.length)  #shape:(199654,)


#             for i in range(self.length):  #

#                 uid = self.data[i][0]
#                 iid_neg = self.neg_data[i][index] = self.neg_data_index[i] #29579
#                 iid_pos = self.pos_data[i][index] = self.pos_data_index[i] #2644

#                 if (uid, iid_neg) in beh_path_dok:
#                     while (uid, iid_neg) in beh_path_dok:
#                         iid_neg = np.random.randint(low=0, high=self.num_item)  
#                         self.neg_data[i][index] = iid_neg
#                     self.neg_data[i][index] = iid_neg

#                 if index == (len(self.beh)-1):
#                     self.pos_data[i][index] = train_v[i]
#                 elif (uid, iid_pos) not in beh_path_dok:
#                     if len(self.beh_meta_path_data[index][uid].data)==0:
#                         self.pos_data[i][index] = -1
#                     else:
#                         t_array = self.beh_meta_path_data[index][uid].toarray()
#                         pos_index = np.where(t_array!=0)[1]
#                         iid_pos = np.random.choice(pos_index, size = 1, replace=True, p=None)[0]   
#                         self.pos_data[i][index] = iid_pos

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         user = self.data[idx][0]
#         item_i = self.pos_data[idx]

#         if self.is_training:  
#             item_j = self.neg_data[idx]
#             return user, item_i, item_j
#         else:  
#             return user, item_i

