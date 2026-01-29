#!/usr/bin/python3
#通过在代码的顶部导入 __future__ 模块，你可以确保代码在将来的 Python 版本中也能正常运行
#可以理解为统一不同版本的一个模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
#这一行代码导入了 PyTorch 库的主要模块，包括张量操作、神经网络构建、自动求导等功能。
# nn 模块包括了各种神经网络层（如全连接层、卷积层、循环层等）以及其他与神经网络相关的功能，例如损失函数。
#F模块包括了诸如激活函数如 ReLU、Sigmoid、Softmax 等，池化操作、卷积操作以及其他一些与神经网络有关的函数。

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, house_dim, house_num, housd_num, thred,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = int(hidden_dim / house_dim)
        self.house_dim = house_dim
        self.house_num = house_num
        self.epsilon = 2.0
        #self.domain_num=121
        self.rthred=0.8
        self.ntype=571
        self.type_num=1
        self.weight=0.9
        self.housd_num = housd_num
        self.thred = thred
        if model_name == 'HousE' or model_name == 'HousE_plus':
            self.house_num = house_num + (2*self.housd_num)
        else:
            self.house_num = house_num
        #nn.Parameter 是 nn.Module 的子类，它表示一个可训练的参数。在这里，self.gamma 被定义为一个可训练参数，因为它是 nn.Parameter 的实例。
        #torch.Tensor([gamma])：这一部分代码创建了一个包含单个值 gamma 的 PyTorch 张量。
        #requires_grad=False：这个参数指定了 self.gamma 是否需要梯度计算和更新
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / (self.hidden_dim * (self.house_dim ** 0.5))]),
            requires_grad=False
        )
        
        # self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        # self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        self.entity_dim = self.hidden_dim
        self.relation_dim = self.hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim, self.house_dim))
        #zai nn.paremeter 属性下，默认require——grad是ture，即需要梯度。
        #均值初始化，初始化的计算方式可以进一步的思考一下
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        #model 初始化以后就再也不进来了
        self.head_type_vec = nn.Parameter(torch.zeros(self.nentity),requires_grad=False)
        # self.tail_type_vec = nn.Parameter(torch.zeros(self.nrelation),requires_grad=False)
        #self.head_domain_vec = nn.Parameter(torch.zeros(self.nrelation),requires_grad=False)
        #self.tail_domain_vec = nn.Parameter(torch.zeros(self.nrelation),requires_grad=False)
        # dict = "C:/Users/lenovo/Desktop/Sportdata/sports/create-res/"
        dict = "/home/25171213997/ITI/data/YAGO3-10/res/"
        f1 = open(os.path.join(dict, 'entity2type.txt'), 'r')
        #f3 = open(os.path.join(dict, 'head_domain_vec.txt'), 'r')
        #f4 = open(os.path.join(dict, 'tail_domain_vec.txt'), 'r')
        for line in f1.readlines():
            i, j = line.split()
            self.head_type_vec[int(i)] = int(j)
        # for line in f2.readlines():
        #     i, j = line.split()
        #     self.tail_type_vec[int(i)] = int(j)
        #for line in f3.readlines():
         #   i, j = line.split()
         #   self.head_domain_vec[int(i)] = int(j)
        #for line in f4.readlines():
         #   i, j = line.split()
         #   self.tail_domain_vec[int(i)] = int(j)
        f1.close()
        #f2.close()
        #f3.close()
        #f4.close()

        # 复制单位矩阵来初始化 self.domain_mat,解决了叶子节点的问题
        '''
        unit_matrix = torch.eye(self.entity_dim)
        self.domain_mat = nn.Parameter(unit_matrix.repeat(self.domain_num, 1, 1))
        self.type_mat =nn.Parameter(unit_matrix.repeat(self.ntype, 1, 1))
        #self.domain_mat = nn.Parameter(torch.zeros(self.domain_num, self.entity_dim, self.entity_dim))
        '''
        self.head_type_mat = nn.Parameter(torch.zeros(self.ntype, self.relation_dim, self.house_dim*self.type_num))
        nn.init.uniform_(
            tensor=self.head_type_mat,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.tail_type_mat = nn.Parameter(torch.zeros(self.ntype, self.relation_dim, self.house_dim*self.type_num))
        nn.init.uniform_(
            tensor=self.tail_type_mat,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim, self.house_dim*self.house_num))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )



        self.r1_dir_head = nn.Parameter(torch.zeros(self.ntype, 1, self.type_num))
        nn.init.uniform_(
            tensor=self.r1_dir_head,
            a=-0.01,
            b=+0.01
        )
        self.r2_dir_tail = nn.Parameter(torch.zeros(self.ntype, 1, self.type_num))
        with torch.no_grad():
            self.r2_dir_tail.data = - self.r1_dir_head.data

        self.r1_scale_head = nn.Parameter(torch.zeros(self.ntype, self.relation_dim, self.type_num))
        nn.init.uniform_(
            tensor=self.r1_scale_head,
            a=-1,
            b=+1
        )
        self.r2_scale_tail = nn.Parameter(torch.zeros(self.ntype, self.relation_dim, self.type_num))
        nn.init.uniform_(
            tensor=self.r2_scale_tail,
            a=-1,
            b=+1
        )

        self.k_dir_head = nn.Parameter(torch.zeros(nrelation, 1, self.housd_num))
        nn.init.uniform_(
            tensor=self.k_dir_head,
            a=-0.01,
            b=+0.01
        )
        self.k_dir_tail = nn.Parameter(torch.zeros(nrelation, 1, self.housd_num))
        with torch.no_grad():
            self.k_dir_tail.data = - self.k_dir_head.data

        self.k_scale_head = nn.Parameter(torch.zeros(nrelation, self.relation_dim, self.housd_num))
        nn.init.uniform_(
            tensor=self.k_scale_head,
            a=-1,
            b=+1
        )
        self.k_scale_tail = nn.Parameter(torch.zeros(nrelation, self.relation_dim, self.housd_num))
        nn.init.uniform_(
            tensor=self.k_scale_tail,
            a=-1,
            b=+1
        )

        self.relation_weight = nn.Parameter(torch.zeros(nrelation, self.relation_dim, self.house_dim))
        nn.init.uniform_(
            tensor=self.relation_weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['HousE_r', 'HousE', 'HousE_r_plus', 'HousE_plus','TransE','rotate','ITI_DistMult']:
            raise ValueError('model %s not supported' % model_name)

    def norm_embedding(self, mode):
        entity_embedding = self.entity_embedding
        r_list = torch.chunk(self.relation_embedding, self.house_num, 2)
        normed_r_list = []
        for i in range(self.house_num):
            r_i = torch.nn.functional.normalize(r_list[i], dim=2, p=2)
            normed_r_list.append(r_i)
        # 归一化后又和成原来的样子，r就是我们要求的单位向量 关系r矩阵
        r = torch.cat(normed_r_list, dim=2)

        r1_list = torch.chunk(self.head_type_mat, self.type_num, 2)
        normed_r1_list = []
        for i in range(self.type_num):
            r1_i = torch.nn.functional.normalize(r1_list[i], dim=2, p=2)
            normed_r1_list.append(r1_i)
        # 归一化后又和成原来的样子，r就是我们要求的单位向量 关系r矩阵
        r1 = torch.cat(normed_r1_list, dim=2)

        r2_list = torch.chunk(self.tail_type_mat, self.type_num, 2)
        normed_r2_list = []
        for i in range(self.type_num):
            r2_i = torch.nn.functional.normalize(r2_list[i], dim=2, p=2)
            normed_r2_list.append(r2_i)
        # 归一化后又和成原来的样子，r就是我们要求的单位向量 关系r矩阵
        r2 = torch.cat(normed_r2_list, dim=2)

        '''
        r1_list = torch.nn.functional.normalize(self.head_type_mat, dim=2, p=2)
        r2_list = torch.nn.functional.normalize(self.tail_type_mat, dim=2, p=2)#(250*2)
        # 归一化后又和成原来的样子，r就是我们要求的单位向量 关系r矩阵
        '''
        self.k_head = self.k_dir_head * torch.abs(self.k_scale_head)
        self.k_head[self.k_head>self.thred] = self.thred
        self.k_tail = self.k_dir_tail * torch.abs(self.k_scale_tail)
        self.k_tail[self.k_tail>self.thred] = self.thred

        self.r1_head = self.r1_dir_head * torch.abs(self.r1_scale_head)
        self.r1_head[self.r1_head>self.rthred] = self.rthred
        self.r2_tail = self.r2_dir_tail * torch.abs(self.r2_scale_tail)
        self.r2_tail[self.r2_tail>self.rthred] = self.rthred
        return entity_embedding, r, r1 ,r2

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''
        entity_embedding, r , r1_list , r2_list  = self.norm_embedding(mode)
        #这里的r就是单位向量

        if mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            #index = tail_part[:, 1]
            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, self.entity_dim, -1)
        #view一般是重塑操作，view括号里的就是重塑的张量
            #head——part 应该是输入的索引
            k_head = torch.index_select(
                self.k_head,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            k_tail = torch.index_select(
                self.k_tail,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)


            re_weight = torch.index_select(
                self.relation_weight,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            #从这里开始的投影
            head_type_id=torch.index_select(
                self.head_type_vec,
                dim=0,
                index=tail_part[:, 0]
            ).squeeze(0)

            tail_type_id= torch.index_select(
                self.head_type_vec,
                dim=0,
                index=tail_part[:, 2]
            ).squeeze(0)

            head_type_id=head_type_id.long()
            tail_type_id=tail_type_id.long()

            r1_head = torch.index_select(
                self.r1_head,
                dim=0,
                index=head_type_id
            ).unsqueeze(1)

            r2_tail = torch.index_select(
                self.r2_tail,
                dim=0,
                index=tail_type_id
            ).unsqueeze(1)

            r1=torch.index_select(
                r1_list,
                dim=0,
                index=head_type_id
            ).unsqueeze(1)

            r2=torch.index_select(
                r2_list,
                dim=0,
                index=tail_type_id
            ).unsqueeze(1)

            relation = torch.index_select(
                r,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            #head-part是正例三元组，tail-part是他的负样本个数挑出来的实体，
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)
            #挑出来的系数呀
            k_head = torch.index_select(
                self.k_head,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            k_tail = torch.index_select(
                self.k_tail,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            re_weight = torch.index_select(
                self.relation_weight,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            head_type_id=torch.index_select(
                self.head_type_vec,
                dim=0,
                index=head_part[:, 0]#这里就是头实体的type-id
            ).squeeze(0)

            tail_type_id= torch.index_select(
                self.head_type_vec,
                dim=0,
                index=head_part[:, 2]#这里就是尾实体挑出来的type-id
            ).squeeze(0)

            head_type_id=head_type_id.long()
            tail_type_id=tail_type_id.long()
            #挑出来的系数
            r1_head = torch.index_select(
                self.r1_head,
                dim=0,
                index=head_type_id
            ).unsqueeze(1)

            r2_tail = torch.index_select(
                self.r2_tail,
                dim=0,
                index=tail_type_id
            ).unsqueeze(1)

            r1=torch.index_select(
                r1_list,#挑头实体type旋转的单位向量
                dim=0,
                index=head_type_id
            ).unsqueeze(1)

            r2=torch.index_select(
                r2_list,#挑尾实体type旋转的单位向量
                dim=0,
                index=tail_type_id
            ).unsqueeze(1)
            #行不行

            relation = torch.index_select(
                r,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            #(尾，旋转，头)
            #relation=torch.cat((r2[:,:,:,:2],relation[:,:,:,2:6],r1[:,:,:,6:8]), dim=-1)

            tail = torch.index_select(
                entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, self.entity_dim, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        if self.model_name == 'HousE_r':
            score = self.HousE_r(head, relation, tail, mode)
        elif self.model_name == 'HousE':
            score = self.HousE(head, relation, k_head, k_tail, tail, r1 , r2 ,r1_head,r2_tail, mode)
        elif self.model_name == 'HousE_r_plus':
            score = self.HousE_r_plus(head, relation, re_weight, tail, mode)
        elif self.model_name == 'rotate':
            score = self.HousE(head, relation, k_head, k_tail, tail, r1, r2, r1_head, r2_tail, mode)
        elif self.model_name == 'HousE_plus':
            score = self.HousE_plus(head, relation, re_weight, k_head, k_tail, tail, mode)
        elif self.model_name == 'TransE':
            score = self.HousE(head, relation, k_head, k_tail, tail, r1, r2, r1_head, r2_tail, mode)
        elif self.model_name == 'ITI_DistMult':
            score = self.ITI_DistMult(head, relation, tail,mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score
    def ITI_DistMult(self, head, relation, tail, mode):
        # head, tail 同样是经过 ITI 类型投影后的
        print(head.shape, relation.shape, tail.shape)

        # DistMult 评分: <h, r, t>
        if mode == 'head-batch':
            score = head * relation * tail
        else:
            score = head * relation * tail

        score = torch.sum(score, dim=-1)
        return score

    def HousE_r(self, head, relation, tail, mode):

        r_list = torch.chunk(relation, self.house_num, 3)
        if mode == 'head-batch':
            for i in range(self.house_num):
                tail = tail - 2 * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]
            cos_score = tail - head
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)
        else:
            for i in range(self.house_num):
                j = self.house_num - 1 - i
                head = head - 2 * (r_list[j] * head).sum(dim=-1, keepdim=True) * r_list[j]
            cos_score = head - tail
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)
            
        score = self.gamma.item() - (cos_score)
        return score

    def HousE_r_plus(self, head, relation, re_weight, tail, mode):

        r_list = torch.chunk(relation, self.house_num, 3)

        if mode == 'head-batch':
            for i in range(self.house_num):
                tail = tail - 2 * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]
            tail = tail - re_weight
            cos_score = tail - head
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)
        else:
            head = head + re_weight
            for i in range(self.house_num):
                j = self.house_num - 1 - i
                head = head - 2 * (r_list[j] * head).sum(dim=-1, keepdim=True) * r_list[j]
            cos_score = head - tail
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)

        score = self.gamma.item() - (cos_score)
        return score


    def HousE(self, head, relation, k_head, k_tail, tail,r1,r2,r1_head,r2_tail, mode):

        r_list = torch.chunk(relation, self.house_num, 3)
        r1_list= torch.chunk(r1, self.type_num, 3)
        r2_list= torch.chunk(r2, self.type_num, 3)
        if mode == 'head-batch':
           #tail同理
            for i in range(self.type_num):
               r2_tail_i = r2_tail[:, :, :, i].unsqueeze(dim=3)
               tail = tail - (0 + r2_tail_i) * (r2_list[i] * tail).sum(dim=-1, keepdim=True) * r2_list[i]

            for i in range(self.type_num):
               r1_head_i = r1_head[:, :, :, i].unsqueeze(dim=3)
               head = head - (0 + r1_head_i) * (r1_list[i] * head).sum(dim=-1, keepdim=True) * r1_list[i]

            for i in range(self.housd_num):
                k_tail_i = k_tail[:, :, :, i].unsqueeze(dim=3)#系数 r—list是向量
                tail = tail - (0 + k_tail_i) * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]

            for i in range(self.housd_num, self.house_num-self.housd_num):
                tail = tail - 2 * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]
            
            for i in range(self.housd_num):
                k_head_i = k_head[:, :, :, i].unsqueeze(dim=3)
                head = head - (0 + k_head_i) * (r_list[self.house_num-1-i] * head).sum(dim=-1, keepdim=True) * r_list[self.house_num-1-i]
            cos_score = tail - head
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)
        else:
            #mode=tail时对头实体的处理
            for i in range(self.type_num):
                r2_tail_i = r2_tail[:, :, :, i].unsqueeze(dim=3)
                tail = tail - (0 + r2_tail_i) * (r2_list[i] * tail).sum(dim=-1, keepdim=True) * r2_list[i]

            for i in range(self.type_num):
                r1_head_i = r1_head[:, :, :, i].unsqueeze(dim=3)
                head = head - (0 + r1_head_i) * (r1_list[i] * head).sum(dim=-1, keepdim=True) * r1_list[i]

            for i in range(self.housd_num):
                k_head_i = k_head[:, :, :, i].unsqueeze(dim=3)
                head = head - (0 + k_head_i) * (r_list[self.house_num-1-i] * head).sum(dim=-1, keepdim=True) * r_list[self.house_num-1-i]
            
            for i in range(self.housd_num, self.house_num-self.housd_num):
                j = self.house_num - 1 - i
                head = head - 2 * (r_list[j] * head).sum(dim=-1, keepdim=True) * r_list[j]
            
            for i in range(self.housd_num):
                k_tail_i = k_tail[:, :, :, i].unsqueeze(dim=3)
                tail = tail - (0 + k_tail_i) * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]

            cos_score = head - tail
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)

        score = self.gamma.item() - (cos_score)
        return score

    def TransE(self, head, relation, k_head, k_tail, tail, r1, r2, r1_head, r2_tail, mode):

        r_list = torch.chunk(relation, self.house_num, 3)
        r1_list = torch.chunk(r1, self.type_num, 3)
        r2_list = torch.chunk(r2, self.type_num, 3)
        if mode == 'head-batch':
            # tail同理
            for i in range(self.type_num):
                r2_tail_i = r2_tail[:, :, :, i].unsqueeze(dim=3)
                tail = tail - (0 + r2_tail_i) * (r2_list[i] * tail).sum(dim=-1, keepdim=True) * r2_list[i]

            for i in range(self.type_num):
                r1_head_i = r1_head[:, :, :, i].unsqueeze(dim=3)
                head = head - (0 + r1_head_i) * (r1_list[i] * head).sum(dim=-1, keepdim=True) * r1_list[i]

            for i in range(self.housd_num):
                k_tail_i = k_tail[:, :, :, i].unsqueeze(dim=3)#系数 r—list是向量
                tail = tail - (0 + k_tail_i) * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]

            for i in range(self.housd_num):
                k_head_i = k_head[:, :, :, i].unsqueeze(dim=3)
                head = head - (0 + k_head_i) * (r_list[self.house_num - 1 - i] * head).sum(dim=-1, keepdim=True) * \
                       r_list[self.house_num - 1 - i]
            cos_score = tail - head
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)
        else:
            # mode=tail时对头实体的处理
            for i in range(self.type_num):
                r2_tail_i = r2_tail[:, :, :, i].unsqueeze(dim=3)
                tail = tail - (0 + r2_tail_i) * (r2_list[i] * tail).sum(dim=-1, keepdim=True) * r2_list[i]

            for i in range(self.type_num):
                r1_head_i = r1_head[:, :, :, i].unsqueeze(dim=3)
                head = head - (0 + r1_head_i) * (r1_list[i] * head).sum(dim=-1, keepdim=True) * r1_list[i]

            for i in range(self.housd_num):
                k_head_i = k_head[:, :, :, i].unsqueeze(dim=3)
                head = head - (0 + k_head_i) * (r_list[self.house_num - 1 - i] * head).sum(dim=-1, keepdim=True) * \
                       r_list[self.house_num - 1 - i]

            for i in range(self.housd_num):
                k_tail_i = k_tail[:, :, :, i].unsqueeze(dim=3)
                tail = tail - (0 + k_tail_i) * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]

            cos_score = head - tail
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)

        score = self.gamma.item() - (cos_score)
        return score



    def HousE_plus(self, head, relation, re_weight, k_head, k_tail, tail, mode):

        r_list = torch.chunk(relation, self.house_num, 3)

        if mode == 'head-batch':
            for i in range(self.housd_num):
                k_tail_i = k_tail[:, :, :, i].unsqueeze(dim=3)
                tail = tail - (0 + k_tail_i) * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]
            
            for i in range(self.housd_num, self.house_num-self.housd_num):
                tail = tail - 2 * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]
            
            tail = tail - re_weight
            for i in range(self.housd_num):
                k_head_i = k_head[:, :, :, i].unsqueeze(dim=3)
                head = head - (0 + k_head_i) * (r_list[self.house_num-1-i] * head).sum(dim=-1, keepdim=True) * r_list[self.house_num-1-i]
            
            cos_score = tail - head
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)
        else:
            for i in range(self.housd_num):
                k_head_i = k_head[:, :, :, i].unsqueeze(dim=3)
                head = head - (0 + k_head_i) * (r_list[self.house_num-1-i] * head).sum(dim=-1, keepdim=True) * r_list[self.house_num-1-i]
            head = head + re_weight
            
            for i in range(self.housd_num, self.house_num-self.housd_num):
                j = self.house_num - 1 - i
                head = head - 2 * (r_list[j] * head).sum(dim=-1, keepdim=True) * r_list[j]
            
            for i in range(self.housd_num):
                k_tail_i = k_tail[:, :, :, i].unsqueeze(dim=3)
                tail = tail - (0 + k_tail_i) * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]
            
            cos_score = head - tail
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)
        score = self.gamma.item() - (cos_score)
        return score
    def rotate(self, head, relation, k_head, k_tail, tail, r1, r2, r1_head, r2_tail, mode):
        r_list = torch.chunk(relation, self.house_num, 3)
        r1_list = torch.chunk(r1, self.type_num, 3)
        r2_list = torch.chunk(r2, self.type_num, 3)

        if mode == 'head-batch':
            # 类型投影
            for i in range(self.type_num):
                r2_tail_i = r2_tail[:, :, :, i].unsqueeze(dim=3)
                tail = tail - (r2_tail_i) * (r2_list[i] * tail).sum(dim=-1, keepdim=True) * r2_list[i]

            for i in range(self.type_num):
                r1_head_i = r1_head[:, :, :, i].unsqueeze(dim=3)
                head = head - (r1_head_i) * (r1_list[i] * head).sum(dim=-1, keepdim=True) * r1_list[i]

            # Householder投影部分
            for i in range(self.housd_num):
                k_tail_i = k_tail[:, :, :, i].unsqueeze(dim=3)
                tail = tail - (k_tail_i) * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]

            #house旋转
            for i in range(self.housd_num, self.house_num - self.housd_num):
                tail = tail - 2 * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]

            #改为Hadamard旋转（尾实体参与）
            #h_head = head
            #h_relation = torch.cat(r_list[self.housd_num:self.house_num - self.housd_num], dim=-1)
            #head = h_head * h_relation  # Hadamard乘积（元素乘）

            # 剩下部分
            for i in range(self.housd_num):
                k_head_i = k_head[:, :, :, i].unsqueeze(dim=3)
                head = head - (k_head_i) * (r_list[self.house_num - 1 - i] * head).sum(dim=-1, keepdim=True) * r_list[
                    self.house_num - 1 - i]

            cos_score = tail - head
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)

        else:
            # 类型投影
            for i in range(self.type_num):
                r2_tail_i = r2_tail[:, :, :, i].unsqueeze(dim=3)
                tail = tail - (r2_tail_i) * (r2_list[i] * tail).sum(dim=-1, keepdim=True) * r2_list[i]

            for i in range(self.type_num):
                r1_head_i = r1_head[:, :, :, i].unsqueeze(dim=3)
                head = head - (r1_head_i) * (r1_list[i] * head).sum(dim=-1, keepdim=True) * r1_list[i]

            # Householder投影部分
            for i in range(self.housd_num):
                k_head_i = k_head[:, :, :, i].unsqueeze(dim=3)
                head = head - (k_head_i) * (r_list[self.house_num - 1 - i] * head).sum(dim=-1, keepdim=True) * r_list[
                    self.house_num - 1 - i]
            #house旋转
            for i in range(self.housd_num, self.house_num - self.housd_num):
                j = self.house_num - 1 - i
                head = head - 2 * (r_list[j] * head).sum(dim=-1, keepdim=True) * r_list[j]

            # 改为Hadamard旋转（头实体参与）
            #h_tail = tail
            #h_relation = torch.cat(r_list[self.housd_num:self.house_num - self.housd_num], dim=-1)
            #tail = h_tail * h_relation  # Hadamard乘积（元素乘）

            # 剩下部分
            for i in range(self.housd_num):
                k_tail_i = k_tail[:, :, :, i].unsqueeze(dim=3)
                tail = tail - (k_tail_i) * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]

            cos_score = head - tail
            cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)

        score = self.gamma.item() - cos_score
        return score
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()
        #model.train()的作用是启用 Batch Normalization 和 Dropout

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
        #next 是python的迭代器
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)
        #负例三元组等分
        #上面写的那个类是返回score的

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        if mode == 'head-batch':
            pos_part = positive_sample[:, 0].unsqueeze(dim=1)#把头实体拿了出来
        else:
            pos_part = positive_sample[:, 2].unsqueeze(dim=1)#把尾实体拿了出来
            #正例三元组等分，笨
        positive_score = model((positive_sample, pos_part), mode=mode)

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()
        #计算总损失
        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            regularization = args.regularization * (
                model.entity_embedding.norm(dim=2, p=2).norm(dim=1, p=2).mean()
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
        #误差反向传播
        loss.backward()

        optimizer.step()
        #梯度更新

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log




    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        #model.train()的作用是启用 Batch Normalization 和 Dropout
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
