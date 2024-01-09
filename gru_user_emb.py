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
        self.embedding_item = nn.Embedding(input_size['item'], hidden_size)
        self.embedding_behavior = nn.Embedding(input_size['behavior'], hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.gate_layer = torch.nn.Linear(32, 16)
        self.gate_sigmoid = torch.nn.Sigmoid()

    def gate(self,A_embedding, B_embedding):

        AB_concat = torch.cat((A_embedding,B_embedding),-1)   # torch.Size([44, 16])  
        context_gate = self.gate_sigmoid(self.gate_layer(AB_concat))  # torch.Size([44, 16])  
        return torch.add(context_gate * A_embedding, (1.- context_gate) * B_embedding)  

    def forward(self, item, behavior):
        item_embed = self.embedding_item(item)  # torch.Size([1, 64])  torch.Size([44, 64])
        behavior_embed = self.embedding_behavior(behavior) # torch.Size([1, 64])   torch.Size([44, 64])
        x = self.gate(item_embed, behavior_embed)   # torch.Size([44, 64])
        _, hidden = self.gru(x)  # [4, 16]
        return hidden#[-1, :, :]


def user_gru(self):
    # df = pd.read_csv("/home/joo/JOOCML/data/Tmall/real_gru_data.csv")  

    max_item_index = max(df['item'])
    max_beh_index = max(df['behavior'])

    user_data = {}

    # 데이터를 딕셔너리에 저장
    for user_id, group in df.groupby('user'):
        item_tensor = torch.tensor(group['item'].tolist())
        behavior_tensor = torch.tensor(group['behavior'].tolist())

        # 딕셔너리에 사용자 아이디를 키로 하여 데이터 저장
        user_data[user_id] = {'item': item_tensor, 'behavior': behavior_tensor}


    user_patterns = []
    # Hyperparameters
    for i in tqdm(range(user_id+1 )):

        input_size = {'item': max_item_index, 'behavior': max_beh_index}
        hidden_size = 16
        num_layers = 1

        model = GRUModel(input_size, hidden_size, num_layers)
        item_id = user_data[i]['item']
        behavior = user_data[i]['behavior']
        hidden = model(torch.tensor(item_id), torch.tensor(behavior))
        user_patterns.append(hidden)


    output = torch.stack(user_patterns , 1)
    print(output.shape) # torch.Size([1, 31881, 16])
    return output


if __name__ == "__main__":
    df = pd.read_csv("/home/joo/JOOCML/data/Tmall/real_gru_data.csv")  
    user_gru(df)

    

    
