
import torch
import torch.nn as nn
import torch.nn.functional as F
class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat


        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    # def forward(self, input, adj):
    #     #h = torch.mm(input, self.W) # shape [N, out_features]
    #     h = torch.matmul(input,self.W)   #[10,128]->>[10,32]
    #     N = h.size()[0]    #=[10]  
    #     #torch.ones_like函数和torch.zeros_like函数的基本功能是根据给定张量，生成与其形状相同的全1张量或全0张量

    #     # 沿着 水平方向和垂直方向分别赋值n次，再水平拼接，就可以得到 n*n个样本,每个样本的向量是由原来 的2个向量拼接而成
    #     a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features) # [10,10,64],shape[N, N, 2*out_features]
    #     e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [10,10],[N,N,1] -> [N,N]

    #     #torch.ones_like函数生成与e的形状相同的全为1的矩阵
    #     zero_vec = -9e15*torch.ones_like(e)  #[10,10]

    #     #合并e,zero_vec两个tensor，如果adj中元素大于0，则c中与a对应的位置取e的值，否则取zero_vec的值
    #     attention = torch.where(adj > 0, e, zero_vec)  #[10,10]

    #     #使用softmax()函数进行归一化处理
    #     attention = F.softmax(attention, dim=1)  #[10,10]

        
    #     attention = F.dropout(attention, self.dropout, training=self.training)
    #     h_prime = torch.matmul(attention, h)  # [N,N], [N, out_features] --> [N, out_features]

    #     if self.concat:
    #         return F.elu(h_prime)
    #     else:
    #         return h_prime



    def forward(self, input, adj):
        #h = torch.mm(input, self.W) # shape [N, out_features]
        h = torch.matmul(input,self.W)   #[10,10,128]->>[10,10,32]
        #N = h.size()[0]    #=[10]  

        batch_size = h.size()[0] #[10]
        seq_len = h.size()[1] #[12]

        hidden_size = h.size()[2] #[32]

        #torch.ones_like函数和torch.zeros_like函数的基本功能是根据给定张量，生成与其形状相同的全1张量或全0张量

        # 沿着 水平方向和垂直方向分别赋值n次，再水平拼接，就可以得到 n*n个样本,每个样本的向量是由原来 的2个向量拼接而成
        # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features) # shape[N, N, 2*out_features]
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N,N,1] -> [N,N]

        a = h.repeat(1,1, seq_len) #[10,12,384]
        b = a.view(batch_size,seq_len*seq_len,-1) #[10,144,32]
 
        c = h.repeat(1,seq_len, 1) #[10,144,32]

        d = torch.cat([b,c],dim=2) #[10,144,64]


        e = d.view(batch_size,seq_len,-1, 2 * self.out_features) #[10,12,12,64]

        a_input = torch.cat([h.repeat(1,1, seq_len).view(batch_size,seq_len*seq_len,-1), h.repeat(1,seq_len, 1)], dim=2).view(batch_size,seq_len,-1, 2 * self.out_features) #shape[N, N, 2*out_features]
        

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # [N,N,1] -> [N,N]



        #torch.ones_like函数生成与e的形状相同的全为1的矩阵
        zero_vec = -9e15*torch.ones_like(e)

        #合并e,zero_vec两个tensor，如果adj中元素大于0，则c中与a对应的位置取e的值，否则取zero_vec的值
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # [N,N], [N, out_features] --> [N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime