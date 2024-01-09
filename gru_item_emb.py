import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# GRU 모델 정의
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        self.embedding_user = nn.Embedding(input_size['user'], hidden_size)
        self.embedding_behavior = nn.Embedding(input_size['behavior'], hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.gate_layer = torch.nn.Linear(32, 16)
        self.gate_sigmoid = torch.nn.Sigmoid()

    def gate(self,A_embedding, B_embedding):

        AB_concat = torch.cat((A_embedding,B_embedding),-1)   # torch.Size([44, 16])  
        context_gate = self.gate_sigmoid(self.gate_layer(AB_concat))  # torch.Size([44, 16])  
        return torch.add(context_gate * A_embedding, (1.- context_gate) * B_embedding)  

    def forward(self, user, behavior):
        user_embed = self.embedding_user(user)  # torch.Size([1, 64])  torch.Size([44, 64])
        behavior_embed = self.embedding_behavior(behavior) # torch.Size([1, 64])   torch.Size([44, 64])
        x = self.gate(user_embed, behavior_embed)   # torch.Size([44, 64])
        _, hidden = self.gru(x)  # [4, 16]
        return hidden#[-1, :, :]


def item_gru(self):
    # df = pd.read_csv("/home/joo/JOOCML/data/Tmall/real_gru_data.csv")  

    max_user_index = max(df['user'])
    max_beh_index = max(df['behavior'])

    item_data = {}

    # 데이터를 딕셔너리에 저장
    for item_id, group in df.groupby('item'):
        user_tensor = torch.tensor(group['user'].tolist())
        behavior_tensor = torch.tensor(group['behavior'].tolist())

        # 딕셔너리에 사용자 아이디를 키로 하여 데이터 저장
        item_data[item_id] = {'user': user_tensor, 'behavior': behavior_tensor}


    item_patterns = []
    # Hyperparameters
    for i in tqdm(range(item_id )):

        input_size = {'user': max_user_index, 'behavior': max_beh_index}
        hidden_size = 16
        num_layers = 1

        model = GRUModel(input_size, hidden_size, num_layers)
        user_id = item_data[i]['user']
        behavior = item_data[i]['behavior']
        hidden = model(torch.tensor(user_id), torch.tensor(behavior))
        item_patterns.append(hidden)


    output = torch.stack(item_patterns , 1)
    output.squeeze()
    print(output.shape) # torch.Size([1, 31231, 16])
    return output


if __name__ == "__main__":
    df = pd.read_csv("/home/joo/JOOCML/data/Tmall/real_gru_data.csv")  
    item_gru(df)

    

    
