import torch
import torch.nn as nn
import torch.nn.functional as F
import rsgd
import HIME_dataset
from torch.utils.data import DataLoader
import pickle
import time
import os
import hyper_math as hm
from torch.autograd import Variable
import argparse
import utils


class HIME(nn.Module):
    def __init__(self, node_num, tag_num, alpha=1, emb_num=4, emb_dim=32, lru_period=5, init=1e-3):
        super(HIME, self).__init__()
        self.node_num = node_num
        self.tag_num = tag_num
        self.emb_num = emb_num
        self.emb_dim = emb_dim
        self.lru_period = lru_period
        self.alpha = alpha

        init_emb = init * torch.randn(tag_num, emb_dim)
        self.tag_embeddings = nn.Embedding(tag_num, emb_dim)
        self.tag_embeddings.weight.data.copy_(init_emb)

        self.node_embeddings = nn.ModuleList()
        for i in range(emb_num):
            init_emb = init * torch.randn(node_num, emb_dim)
            node_embedding = nn.Embedding(node_num, emb_dim)
            node_embedding.weight.data.copy_(init_emb)
            self.node_embeddings.append(node_embedding)

        self.weight = nn.Parameter(torch.zeros(node_num, 1, emb_num))
        self.bias = nn.Parameter(torch.zeros(node_num, emb_dim))

        self.hit = torch.zeros([self.node_num, self.emb_num])
        self.final_branch_embedding = []

    def init_before_node_node(self):
        for i in self.node_embeddings:
            self.final_branch_embedding.append(i.weight)
        self.final_branch_embedding = torch.stack(self.final_branch_embedding, 1).detach()
        for i in range(self.node_num):
            for j in range(self.emb_num):
                if self.hit[i][j] == 0:
                    self.final_branch_embedding[i][j][:] = 0

    def tag_tag_dist(self, u_list, v_list):
        emb_u = self.tag_embeddings(u_list)
        emb_v = self.tag_embeddings(v_list)
        return hm.hyp_dist(emb_u, emb_v)

    def node_tag_dist(self, node_list, tag_list, pos=False):
        temp_result = []
        tag_emb = self.tag_embeddings(tag_list)
        for i in range(self.emb_num):
            node_emb_i = self.node_embeddings[i](node_list)
            #temp_result.append((1-self.alpha*tag_emb.norm(dim=-1, p=2, keepdim=True))*hm.hyp_dist(node_emb_i, tag_emb))
            temp_result.append(hm.hyp_dist(node_emb_i, tag_emb))

        temp = torch.stack(temp_result, dim=1)
        dist, min_dices = torch.min(temp, 1)
        if pos:
            for i in range(len(node_list)):
                self.hit[node_list[i]][min_dices[i]] += 1
        return dist

    def node_node_dist(self, u_list, v_list):
        u_weight = torch.index_select(self.weight, 0, u_list)
        u_softmax = F.softmax(u_weight, 2)
        u_emb = torch.index_select(self.final_branch_embedding, 0, u_list)
        log_u_emb = rsgd.p_log_map(u_emb)
        u_bias = torch.index_select(self.bias, 0, u_list)
        log_u_root = torch.bmm(u_softmax, log_u_emb).squeeze(dim=1) + u_bias
        u_root_emb = rsgd.p_exp_map(log_u_root)

        v_weight = torch.index_select(self.weight, 0, v_list)
        v_softmax = F.softmax(v_weight, 2)
        v_emb = torch.index_select(self.final_branch_embedding, 0, v_list)
        log_v_emb = rsgd.p_log_map(v_emb)
        v_bias = torch.index_select(self.bias, 0, v_list)
        log_v_root = torch.bmm(v_softmax, log_v_emb).squeeze(dim=1) + v_bias
        v_root_emb = rsgd.p_exp_map(log_v_root)

        return hm.hyp_dist(u_root_emb, v_root_emb)

    def tag_tag_forward(self, pos_u, pos_v, neg_u, neg_v):
        losses = []
        score = self.tag_tag_dist(pos_u, pos_v)
        score = F.logsigmoid(-score)
        score = -torch.sum(score)
        losses.append(score)
        neg_score = self.tag_tag_dist(neg_u, neg_v)
        neg_score = F.logsigmoid(neg_score)
        neg_score = -torch.sum(neg_score)
        losses.append(neg_score)
        return sum(losses)

    def node_tag_forward(self, pos_node, pos_tag, neg_node, neg_tag):
        losses = []
        score = self.node_tag_dist(pos_node, pos_tag, pos=True)
        score = F.logsigmoid(-score)
        score = -torch.sum(score)
        losses.append(score)
        neg_score = self.node_tag_dist(neg_node, neg_tag)
        neg_score = F.logsigmoid(neg_score)
        neg_score = -torch.sum(neg_score)
        losses.append(neg_score)
        return sum(losses)

    def node_node_forward(self, pos_u, pos_v, neg_u, neg_v):
        losses = []
        score = self.node_node_dist(pos_u, pos_v)
        score = F.logsigmoid(-score)
        score = -torch.sum(score)
        losses.append(score)
        neg_score = self.node_node_dist(neg_u, neg_v)
        neg_score = F.logsigmoid(neg_score)
        neg_score = -torch.sum(neg_score)
        losses.append(neg_score)
        return sum(losses)

    def LRU(self, use_cuda):
        old_embedding = []
        for i in range(self.emb_num):
            old_embedding.append(self.node_embeddings[i].weight)
        old_embedding = torch.stack(old_embedding, dim=1)
        min_value, min_index = torch.min(self.hit, 1)
        max_value, max_index = torch.max(self.hit, 1)
        paste = []
        emb = []
        delta = []
        for node in range(self.node_num):
            if min_value[node] == 0:
                paste.append((node, min_index[node]))
                emb.append(old_embedding[node][max_index[node]])
                delta.append(1e-3*torch.randn(self.emb_dim))
        if len(emb) == 0:
            return

        emb = torch.stack(emb)
        delta = torch.stack(delta)
        if use_cuda:
            emb = emb.cuda()
            delta = delta.cuda()

        copy_results = rsgd.full_p_exp_map(emb, delta)
        for i in range(len(paste)):
            old_embedding[paste[i]] = copy_results[i]
        for i in range(self.emb_num):
            self.node_embeddings[i].weight.data.copy_(old_embedding[:, i, :])

    def convert(self, pairs, labels, use_cuda):
        pos_u = []
        pos_v = []
        neg_u = []
        neg_v = []
        u_list = pairs[0]
        v_list = pairs[1]

        for i in range(len(labels)):
            u = u_list[i]
            v = v_list[i]
            label = labels[i]
            if label == 1:
                pos_u.append(u)
                pos_v.append(v)
            else:
                neg_u.append(u)
                neg_v.append(v)

        pos_u = Variable(torch.LongTensor(pos_u))
        pos_v = Variable(torch.LongTensor(pos_v))
        neg_u = Variable(torch.LongTensor(neg_u))
        neg_v = Variable(torch.LongTensor(neg_v))

        if use_cuda:
            pos_u = pos_u.cuda()
            pos_v = pos_v.cuda()
            neg_u = neg_u.cuda()
            neg_v = neg_v.cuda()
        return pos_u, pos_v, neg_u, neg_v


def train(dataset, epoch, neg_num, emb_num, emb_dim, batch_size, lru_period, lda, alpha):
    data_file = ""
    if dataset == "dblp":
        data_file = 'dataset/dblp/train/dblp.txt'
    if dataset == "protein_go":
        data_file = 'dataset/protein_go/train/protein_go.txt'
    if dataset == "gene_pathway":
        data_file = 'dataset/gene_pathway/train/gene_pathway.txt'

    save_dir = "saved_model/" + dataset + "_emb_num_" + str(emb_num) + "_emb_dim_" + str(emb_dim)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(data_file, 'r') as f:
        items = f.readline().strip().split()
        node_num = int(items[0])
        tag_num = int(items[1])
    use_cuda = False
    if torch.cuda.is_available():
       use_cuda = True

    model = HIME(node_num=node_num, tag_num=tag_num, emb_num=emb_num, emb_dim=emb_dim, lru_period=lru_period, alpha=alpha)
    print("use cuda: ", use_cuda)
    if use_cuda:
        model = model.cuda()

    rsgd_opt = rsgd.RiemannianSGD([
        {'params': model.node_embeddings.parameters(), 'lr': 0.02},
        {'params': model.tag_embeddings.parameters(), 'lr': 0.01},
        ])
    EU_opt = torch.optim.SGD([
        {'params': model.weight, 'lr': 0.1},
        {'params': model.bias, 'lr': 0.01},
    ], lr=0.01)
    data = HIME_dataset.HIME_Dataset(data_file, neg_num)


    for ep in range(epoch):
        time_start = time.time()
        print("Sampling dataset...")
        tag_tag_data, node_tag_data, node_node_data = data.sample_dataset()
        tag_tag_loader = DataLoader(tag_tag_data, batch_size=batch_size, shuffle=True)
        node_tag_loader = DataLoader(node_tag_data, batch_size=batch_size, shuffle=True)

        tt_loss = []
        for i, tag_tag_data in enumerate(tag_tag_loader):
            pairs, labels = tag_tag_data
            pos_u, pos_v, neg_u, neg_v = model.convert(pairs, labels, use_cuda)
            loss = model.tag_tag_forward(pos_u, pos_v, neg_u, neg_v)
            rsgd_opt.zero_grad()
            loss.backward()
            with torch.no_grad():
                rsgd_opt.step()
            tt_loss.append(loss)
        tt_loss = sum(tt_loss).item()
        print("tt_loss: ", tt_loss)

        nt_loss = []
        for i, node_tag_data in enumerate(node_tag_loader):
            edges, labels = node_tag_data
            pos_u, pos_v, neg_u, neg_v = model.convert(edges, labels, use_cuda)
            loss = model.node_tag_forward(pos_u, pos_v, neg_u, neg_v)
            rsgd_opt.zero_grad()
            loss.backward()
            with torch.no_grad():
                rsgd_opt.step()
            nt_loss.append(loss)
        nt_loss = sum(nt_loss).item()
        print("nt_loss: ", nt_loss)

        print("epoch " + str(ep) + ': ' + str(tt_loss + nt_loss))

        if ep % model.lru_period == 0 and ep != epoch:
            with torch.no_grad():
                model.LRU(use_cuda)
            print("hit matrix:")
            print(model.hit)
            model.hit = torch.zeros(model.node_num, model.emb_num)

        time_end = time.time()
        print('epoch time cost: ' + str(time_end - time_start)+'\n')

    model.init_before_node_node()
    for ep in range(epoch):
        nn_loss = []
        tag_tag_data, node_tag_data, node_node_data = data.sample_dataset()
        node_node_loader = DataLoader(node_node_data, batch_size=batch_size, shuffle=True)
        for i, node_node_data in enumerate(node_node_loader):
            pairs, labels = node_node_data
            pos_u, pos_v, neg_u, neg_v = model.convert(pairs, labels, use_cuda)
            loss = model.node_node_forward(pos_u, pos_v, neg_u, neg_v)
            EU_opt.zero_grad()
            loss.backward()
            with torch.no_grad():
                EU_opt.step()
            nn_loss.append(loss)
        nn_loss = sum(nn_loss).item()

        bias_loss = lda * utils.l2_loss(model.bias)
        EU_opt.zero_grad()
        bias_loss.backward()
        bias_loss = bias_loss.item()
        with torch.no_grad():
            EU_opt.step()
        print("epoch " + str(ep) + ": nn_loss: ", nn_loss+bias_loss)
    pickle.dump(model, open(save_dir + "/epoch_" + str(epoch) + '.pkl', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dblp', type=str, help='dblp, protein_go, gene_pathway')
    parser.add_argument('--neg_num', default=5, type=int, help='negative sampling number')
    parser.add_argument('--emb_num', default=8, type=int, help='branch vector number')
    parser.add_argument('--emb_dim', default=32, type=int, help='embedding dimension')
    parser.add_argument('--epoch_num', default=50, type=int, help='epoch number')
    parser.add_argument('--batch_size', default=1000, type=int, help='batch size')
    parser.add_argument('--LRU_period', default=5, type=int, help='LRU period')
    parser.add_argument('--lda', default=1, type=int, help='lambda')
    parser.add_argument('--alpha', default=1, type=int, help='alpha')

    args = parser.parse_args()
    train(dataset=args.dataset, neg_num=args.neg_num, emb_num=args.emb_num, emb_dim=args.emb_dim,
          epoch=args.epoch_num, batch_size=args.batch_size, lru_period=args.LRU_period, lda=args.lda, alpha=1)

