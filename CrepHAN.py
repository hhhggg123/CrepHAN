
# coding: utf-8

# In[1]:



import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
import torch 
from torch.utils import data 
import torch.nn.functional as F
import linecache 
import sys
"""
def vec_():
    with open("total_10MerVector.txt") as itf:
        vec_lists = itf.readlines()
    return vec_lists
"""


Max_length = 800
Batch_size = 1
num_layer = 3

def vec_index(base):
    list_bases = ["A","C","G","T"]
    num = np.zeros(10)
    for j in range(10):
        num[j] = list_bases.index(base[j])
    index_num = 0
    for j in range(10):
        index_num += num[j]*(4**(9-j))
    return int(index_num)-1


# In[2]:


def seq_split(seq):
    seqs = []
    for i in range(int(len(seq)/10)):
        seqs.append(seq[i*10:i*10+10])
    return seqs


# In[4]:


def transform_data(seq):
    bases = seq_split(seq)
    data_200 = np.zeros([200,30])
    data_400 = np.zeros([200,30])
    data_600 = np.zeros([200,30])
    data_800 = np.zeros([200,30])
    test_data = []
    for i in range(len(bases)):
        try:
            if(i<200):
                index = vec_index(bases[i])
                data_200[i] = linecache.getline("total_10MerVector.txt",index+1).strip().split(" ")[1:]
            if(i>=200 and i<400):
                index = vec_index(bases[i])
                data_400[i-200] = linecache.getline("total_10MerVector.txt",index+1).strip().split(" ")[1:]
            if(i>=400 and i<600):
                index = vec_index(bases[i])
                data_600[i-400] = linecache.getline("total_10MerVector.txt",index+1).strip().split(" ")[1:]
            if(i>=600 and i<800):
                index = vec_index(bases[i])
                data_800[i-600] = linecache.getline("total_10MerVector.txt",index+1).strip().split(" ")[1:]
        except:
            print(1)
            continue
    test_data.append([data_200,data_400,data_600,data_800,0,int((len(bases)-1))])
    return test_data

class GeneDataset(Dataset):
    """
     root：图像存放地址根路径
     augment：是否需要图像增强
    """
    def __init__(self,sams,augment=None):
        # 这个list存放所有图像的地址
        self.data_files = sams

    def __getitem__(self, index):
        # 读取图像数据并返回
        # 这里的open_image是读取图像函数，可以用PIL、opencv等库进行读取
        data_200 = self.data_files[index][0]
        data_400 = self.data_files[index][1]
        data_600 = self.data_files[index][2]
        data_800 = self.data_files[index][3]
        label = self.data_files[index][4]
        length = self.data_files[index][5]
        return data_200,data_400,data_600,data_800,label,length

    def __len__(self):
        # 返回图像的数量
        return len(self.data_files)
    



# In[ ]:



# 定义网络结构
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.hidden_size = 32
        self.rnn_one=torch.nn.LSTM(
            input_size=30,
            hidden_size=self.hidden_size,
            num_layers=num_layer,
            batch_first=True,
            bidirectional = True,
            dropout = 0.5
        )
        self.rnn_two=torch.nn.LSTM(
            input_size=30,
            hidden_size=self.hidden_size,
            num_layers=num_layer,
            batch_first=True,
            bidirectional = True,
            dropout = 0.5
        )
        self.rnn_three=torch.nn.LSTM(
            input_size=30,
            hidden_size=self.hidden_size,
            num_layers=num_layer,
            batch_first=True,
            bidirectional = True,
            dropout = 0.5
        )
        self.rnn_four=torch.nn.LSTM(
            input_size=30,
            hidden_size=self.hidden_size,
            num_layers=num_layer,
            batch_first=True,
            bidirectional = True,
            dropout = 0.5
        )
        self.projection_200 = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            #nn.ReLU(True),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1,bias=False)
        )
        self.projection_400 = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            #nn.ReLU(True),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1,bias=False)
        )
        self.projection_600 = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            #nn.ReLU(True),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1,bias=False)
        )
        self.projection_800 = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.Tanh(),
            #nn.ReLU(True),
            nn.Linear(self.hidden_size, 1,bias=False)
        )
        
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.Tanh(),
            #nn.ReLU(True),
            nn.Linear(self.hidden_size, 1,bias=False)
        )
        self.out_1=nn.Sequential(nn.Linear(in_features=self.hidden_size*2,out_features=self.hidden_size),
                                 nn.BatchNorm1d(self.hidden_size)
                                 #nn.ReLU6()
                                 #nn.Dropout(0.5)
                                 )
        self.out_2=nn.Linear(in_features=self.hidden_size,out_features=2)
        self.soft = nn.Softmax(dim=1)
    def forward(self, x1,x2,x3,x4,l1):
        ###对四个句子分别做LSTM
        output_200,(h_n,c_n)=self.rnn_one(x1)
        output_400,(h_n,c_n)=self.rnn_two(x2)
        output_600,(h_n,c_n)=self.rnn_three(x3)
        output_800,(h_n,c_n)=self.rnn_four(x4)
        
        output = torch.cat([output_200,output_400,output_600,output_800],dim=1)
        #mask掉多于长度的部分
        ones = torch.zeros([len(output),Max_length,self.hidden_size*2])
        for i in range(len(l1)):
            for j in range(l1[i]):
                if(l1[i]>=800):
                    break
                ones[i][j] = torch.ones(self.hidden_size*2)
        output = ones.mul(output)
        ####重新分成四个句子
        output_200 = output[:,0:200,:]
        output_400 = output[:,200:400,:]
        output_600 = output[:,400:600,:]
        output_800 = output[:,600:800,:]
        
        ###对前200进行attention
        result_200 = self.projection_200(output_200)
        weights_200 = F.softmax(result_200.squeeze(-1), dim=1)
        output_200 = (output_200 * weights_200.unsqueeze(-1)).sum(dim=1)
        
        ###对前400进行attention
        result_400 = self.projection_400(output_400)
        weights_400 = F.softmax(result_400.squeeze(-1), dim=1)
        output_400 = (output_400 * weights_400.unsqueeze(-1)).sum(dim=1)
        
        ###对前600进行attention
        result_600 = self.projection_600(output_600)
        weights_600 = F.softmax(result_600.squeeze(-1), dim=1)
        output_600 = (output_600 * weights_600.unsqueeze(-1)).sum(dim=1)
        
        ###对前800进行attention
        result_800 = self.projection_800(output_800)
        weights_800 = F.softmax(result_800.squeeze(-1), dim=1)
        output_800 = (output_800 * weights_800.unsqueeze(-1)).sum(dim=1)
        
        ###合并四个部分
        output = torch.cat([output_200.unsqueeze(-1),output_400.unsqueeze(-1),
                            output_600.unsqueeze(-1),output_800.unsqueeze(-1)],dim=2).permute(0,2,1)
        #print(output.size())
        
        ###做句子的attention
        result = self.projection(output)
        weights = F.softmax(result.squeeze(-1), dim=1)
        output = (output * weights.unsqueeze(-1)).sum(dim=1)
        
        ###全连接层算概率
        output=self.out_1(output)
        output=self.out_2(output)
        out_soft = self.soft(output)
        return output,out_soft


# In[11]:


def prediction(seq="ACTGGGTCAGTGCTA"):
    model = CNNnet()
    model.load_state_dict(torch.load("hierarchical.pt"))
    test_data = transform_data(seq)
    test_dataset = GeneDataset(test_data)
    # 利用dataloader读取我们的数据对象，并设定batch-size和工作现场
    test_loader = DataLoader(test_dataset, batch_size=Batch_size, num_workers=0, shuffle=False)
    for j in test_loader:
        model.eval()
        X_200 = j[0]
        X_400 = j[1]
        X_600 = j[2]
        X_800 = j[3]
        y = j[4]
        length = j[5]
        batch_x1 = Variable(X_200)
        batch_x2 = Variable(X_400)
        batch_x3 = Variable(X_600)
        batch_x4 = Variable(X_800) 
        batch_l1 = Variable(length) 
        out,out_soft = model(batch_x1.float(),batch_x2.float(),batch_x3.float(),batch_x4.float(),batch_l1) # torch.Size([16,2])
    return out_soft.detach().numpy()[0][1]

def main():
    if(len(sys.argv)==2):
        print(prediction(sys.argv[1]))
    else:
        print("InputError")
if __name__ == '__main__':
    main()