import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import pickle# CSV 파일 로드


df = pd.read_csv('/home/joo/JOOCML/data/new_IJCAI_15/data_format1  (1)/data_format1/user_log_format1.csv')
# df = pd.read_csv('/home/joo/JOOCML/data/new_IJCAI_15/new_0.csv')
# print(df['user_id'].values)
# unique_users = df['user_id'].unique()
# user_id_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
# df['user_id'] = df['user_id'].map(user_id_mapping)
# df.to_csv("/home/joo/JOOCML/data/new_IJCAI_15/all_new_0.csv")
# df
df = df.sort_values(by=['user_id','time_stamp'], ascending=[ True, True])
print(df)
# 'item_id' 컬럼의 고유값을 얻어서 순서대로 새로운 인덱스 부여
unique_items = df['item_id'].unique()
unique_users = df['user_id'].unique()
user_id_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
item_id_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}
df['user_id'] = df['user_id'].map(user_id_mapping)
# 'item_id' 컬럼의 값을 새로운 인덱스로 매핑
df['item_id'] = df['item_id'].map(item_id_mapping)
for i in range(4):
    sample = df.loc[df['action_type']==i]
    # 결과 확인
    print(df)
    sample.to_csv("/home/joo/JOOCML/data/new_IJCAI_15/new_{}.csv".format(i))

print("finish")


# # 예시: 4개의 다른 행동에 대한 CSV 파일 경로
# behavior_files = ['/home/joo/JOOCML/data/new_IJCAI_15/buy.csv', '/home/joo/JOOCML/data/new_IJCAI_15/cart.csv',
#                    '/home/joo/JOOCML/data/new_IJCAI_15/click.csv', '/home/joo/JOOCML/data/new_IJCAI_15/fav.csv']

# # 전체 사용자 및 항목 ID를 저장할 집합
# all_users = set()
# all_items = set()

# # Step 1: 각 행동에 대한 사용자 및 항목 ID를 모두 추출
# for behavior_file in behavior_files:
#     df = pd.read_csv(behavior_file)
#     all_users.update(df['user_id'].unique())
#     all_items.update(df['item_id'].unique())

# # 전체 사용자 및 항목 ID를 기반으로 행렬의 shape 설정
# num_users = len(all_users)
# num_items = len(all_items)

# # 각 사용자 및 항목의 인덱스를 만듦
# user_to_index = {user: idx for idx, user in enumerate(all_users)}
# item_to_index = {item: idx for idx, item in enumerate(all_items)}

# # 각 행동에 대한 CSR 행렬을 담을 리스트
# csr_matrices = []

# # Step 2: 각 행동에 대한 CSR 행렬 생성
# for behavior_file in behavior_files:
#     df = pd.read_csv(behavior_file)
#     # df['user_id'].sort_values(ascending=[True])
#     df = df.sort_values(by=['user_id'], ascending=[ True])
#     df = df.sort_values(by=[ 'time_stamp'], ascending=[ True])
#     # CSR 행렬을 만들기 위한 데이터, 행 및 열의 리스트 생성
#     print(df)
    
# #     data = np.ones(len(df))
# #     row = [user_to_index[user] for user in df['user_id']]
# #     col = [item_to_index[item] for item in df['item_id']]
    
# #     # CSR 행렬 생성
# #     csr_mat = csr_matrix((data, (row, col)), shape=(num_users, num_items))
    
# #     csr_matrices.append(csr_mat)

# # # Step 3: 각 행동에 대한 CSR 행렬의 shape를 (424170, 1090390)으로 조정
# # for i in range(len(csr_matrices)):
# #     csr_matrices[i] = csr_matrices[i][:424170, :1090390]

# # # 각 CSR 행렬을 pickle 파일로 저장
# # for i, csr_mat in enumerate(csr_matrices):
# #     output_file_path = f'/home/joo/JOOCML/data/new_IJCAI_15/csr_matrix_behavior_{i + 1}.pkl'
# #     with open(output_file_path, 'wb') as output_file:
# #         pickle.dump(csr_mat, output_file)
# #     print(f"CSR Matrix for Behavior {i+1} 저장 완료: {output_file_path}")


# # # import numpy as np
# # # from scipy import sparse
# # # import pickle
# # # import numpy as np
# # # from scipy.sparse import csr_array
# # # import torch


# # # # trn_buy = pickle.load(open("/home/joo/JOOCML/data/retail_rocket/trn_buy", 'rb'))
# # # # print(trn_buy)
# # # with open('/home/joo/JOOCML/data/new_IJCAI_15/trn_mat_buy.pkl', 'rb') as fs:
# # #             # with open(self.train_file + self.beh_meta_path[i] + '.pkl', 'rb') as fs:  #/home/joo/CML/data/retail_rocket/train_mat_view_cart_buy.pkl
# # #     data = pickle.load(fs)   # '/home/joo/CML/data/Tmall/trn_pv'  trn_fav' trn_fav' trn_buy'>
# # # data.indptr    
# # # print(data)   