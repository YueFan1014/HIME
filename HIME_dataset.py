import torch
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import random as rd


class SubDataset(Dataset):
    def __init__(self, edges, labels):
        self.edges = edges
        self.labels = labels

    def __getitem__(self, index):
        return self.edges[index], self.labels[index]

    def __len__(self):
        return len(self.edges)




class HIME_Dataset():
    def __init__(self, file, neg_num=5):
        openfile = open(file, 'r')
        line = openfile.readline()
        items = line.strip().split()
        self.node_num = int(items[0])
        self.tag_num = int(items[1])
        self.neg_num = neg_num

        lines = openfile.readlines()
        self.graph = nx.Graph()
        for i in range(self.node_num+self.tag_num):
            self.graph.add_node(int(i))

        for line in lines:
            items = line.strip().split()
            self.graph.add_edge(int(items[0]), int(items[1]))

    def sample_dataset(self):
        tt_edge = []
        tt_label = []
        nt_edge = []
        nt_label = []
        nn_edge = []
        nn_label = []

        for i in range(self.node_num):
            neighbor_tags = set()
            neighbor_nodes = set()
            neighbor_list = list(self.graph.neighbors(i))
            for j in neighbor_list:
                if j >= self.node_num:
                    neighbor_tags.add(j - self.node_num)
                else:
                    neighbor_nodes.add(j)
            n_negative_pool = set(range(self.node_num)).difference(neighbor_nodes)
            t_negative_pool = set(range(self.tag_num)).difference(neighbor_tags)
            for j in neighbor_tags:
                nt_edge.append((i, j))
                nt_label.append(1)
                negative_samples = rd.sample(t_negative_pool, self.neg_num)
                for k in negative_samples:
                    nt_edge.append((i, k))
                    nt_label.append(0)

            for j in neighbor_nodes:
                nn_edge.append((i, j))
                nn_label.append(1)
                negative_samples = rd.sample(n_negative_pool, self.neg_num)
                for k in negative_samples:
                    nn_edge.append((i, k))
                    nn_label.append(0)

        for i in range(self.tag_num):
            neighbor_tags = set()
            neighbor_list = list(self.graph.neighbors(i + self.node_num))
            for j in range(len(neighbor_list)):
                if neighbor_list[j] >= self.node_num:
                    neighbor_tags.add(neighbor_list[j] - self.node_num)
            negative_pool = set(range(self.tag_num)).difference(neighbor_tags)
            for j in neighbor_tags:
                tt_edge.append((i, j))
                tt_label.append(1)
                negative_samples = rd.sample(negative_pool, self.neg_num)
                for k in negative_samples:
                    tt_edge.append((i, k))
                    tt_label.append(0)

        tt_dataset = SubDataset(tt_edge, tt_label)
        nt_dataset = SubDataset(nt_edge, nt_label)
        nn_dataset = SubDataset(nn_edge, nn_label)
        return tt_dataset, nt_dataset, nn_dataset