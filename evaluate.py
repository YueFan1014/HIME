import numpy as np
import pickle
from HIME import HIME
import hyper_math as hm
import torch
from HIME_dataset import HIME_Dataset
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import networkx as nx
import argparse
import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dblp', type=str, help='dblp, protein_go, gene_pathway')
    parser.add_argument('--emb_num', default=8, type=int, help='branch vector number')
    parser.add_argument('--emb_dim', default=32, type=int, help='embedding dimension')
    parser.add_argument('--epoch_num', default=50, type=int, help='epoch number')
    parser.add_argument('--use_cuda', default=True, type=bool, help='Is the model trained by GPU?')
    return parser.parse_args()


def read_labeled_data(use_cuda, dataset):
    if dataset == "dblp":
        file = open("dataset/dblp/test/label.lp.dat", 'r')
    if dataset == "protein_go":
        file = open("dataset/protein_go/test/label.lp.dat", 'r')
    if dataset == "gene_pathway":
        file = open("dataset/gene_pathway/test/label.lp.dat", 'r')

    left_nodes = []
    right_nodes = []
    labels = []
    for line in file.readlines():
        items = line.strip().split('.')
        node1 = int(items[0])
        node2 = int(items[1])
        label = int(items[2])
        left_nodes.append(node1)
        right_nodes.append(node2)
        labels.append(label)
    left_nodes = torch.LongTensor(left_nodes)
    right_nodes = torch.LongTensor(right_nodes)
    labels = torch.LongTensor(labels)
    if use_cuda:
        return left_nodes.cuda(), right_nodes.cuda(), labels
    else:
        return left_nodes, right_nodes, labels


def node_pair_retrieval(left_nodes, right_nodes, labels):
    x=[]
    length = len(left_nodes)
    print(length)
    i = 0
    while 1000 * i < length:
        batch_left = left_nodes[1000*i:1000*(i+1)]
        batch_right = right_nodes[1000 * i:1000 * (i + 1)]
        x.append(-model.node_node_dist(batch_left, batch_right).data.cpu())
        i = i + 1

    x=np.concatenate(x, axis=0)
    labels = labels.reshape(-1, 1)
    data = np.concatenate((x, labels), axis=1)

    fpr, tpr, thresholds = roc_curve(y_true=data[:, 1],
                                     y_score=data[:, 0], pos_label=1)
    prec, recall, thresholds = precision_recall_curve(y_true=data[:, 1],
                                                      probas_pred=data[:, 0], pos_label=1)
    AUPRC = auc(recall, prec)*100
    AUROC = auc(fpr, tpr)*100
    print("AUPRC: ", AUPRC, "AUROC: ", AUROC)
    return auc


def label_path_retrieval(use_cuda, dataset, taxo_child2parents, level_by_level=True):
    correct = []
    if dataset == "dblp":
        test_data = r"dataset/dblp/test/label.taxo.dat"
    if dataset == "protein_go":
        test_data = r"dataset/protein_go/test/label.taxo.dat"
    if dataset == "gene_pathway":
        test_data = r"dataset/gene_pathway/test/label.taxo.dat"

    file = open(test_data, 'r')
    for line in file.readlines():
        items = line.strip().split('.')
        node = int(items[0])
        category = items[1]
        category_path = utils.find_path(taxo_child2parents, category2id[category])[:-1]

        p = category2id["root"]
        for true_pos in category_path[::-1]:
            candidates = taxo_parent2children.get(p, None)
            if candidates is None:
                continue
            node_list = []
            for i in range(len(candidates)):
                node_list.append(node)
            if use_cuda:
                node_list = torch.tensor(node_list, dtype=torch.long).cuda()
                tag_list = torch.tensor(candidates, dtype=torch.long).cuda()
            else:
                node_list = torch.tensor(node_list, dtype=torch.long)
                tag_list = torch.tensor(candidates, dtype=torch.long)
            pred = model.node_tag_dist(node_list, tag_list).argmin()
            pred = candidates[pred]
            correct.append(pred == true_pos)
            if level_by_level:
                p = true_pos
            else:
                p = pred
    acc = np.sum(correct) * 100 / len(correct)
    print("mean ACC: ", acc)
    return acc


def get_node_tag(use_cuda, node_id):
    node_list = []
    for i in range(tag_num):
        node_list.append(node_id)
    tag_list = range(0, tag_num)

    if use_cuda:
        tag_list = torch.tensor(tag_list, dtype=torch.long).cuda()
        node_list = torch.tensor(node_list, dtype=torch.long).cuda()
    else:
        tag_list = torch.tensor(tag_list, dtype=torch.long)
        node_list = torch.tensor(node_list, dtype=torch.long)

    score = model.node_tag_dist(node_list, tag_list).cpu()
    rank_score = dict()
    for k in range(tag_num):
        rank_score[k] = score[k]
    sorted_score = sorted(rank_score, key=lambda x:rank_score[x])
    ground_truth_tag = nodeid2category[node_id]
    ground_truth_label = []
    for i in range(tag_num):
        if i in ground_truth_tag:
            ground_truth_label.append(1)
        else:
            ground_truth_label.append(0)

    avg_rank = 0
    for i in ground_truth_tag:
        avg_rank += sorted_score.index(i) + 1
    avg_rank /= len(ground_truth_tag)
    return avg_rank, len(ground_truth_tag)


def get_tag_node(use_cuda, tag_id, node_num=1000):
    if tag_id not in category2nodeid:
        return -1
    ground_truth_node = category2nodeid[tag_id]
    if len(ground_truth_node) < 10:
        return -1

    ground_truth_label = []
    cnt = 0
    for i in range(node_num):
        if i in ground_truth_node:
            ground_truth_label.append(1)
            cnt += 1
        else:
            ground_truth_label.append(0)
    if cnt == 0:
        return -1
    tag_list = []
    for i in range(node_num):
        tag_list.append(tag_id)
    node_list = range(0, node_num)

    if use_cuda:
        tag_list = torch.tensor(tag_list, dtype=torch.long).cuda()
        node_list = torch.tensor(node_list, dtype=torch.long).cuda()
    else:
        tag_list = torch.tensor(tag_list, dtype=torch.long)
        node_list = torch.tensor(node_list, dtype=torch.long)

    score = model.node_tag_dist(node_list, tag_list).cpu()
    rank_score = dict()
    for k in range(node_num):
        rank_score[k] = score[k]

    prec, recall, thresholds = precision_recall_curve(y_true=ground_truth_label,
                                                      probas_pred=-score.detach().numpy(), pos_label=1)
    AUPRC = auc(recall, prec)
    #print("AUPRC:", AUPRC)

    return AUPRC


def node_retrieval(use_cuda, tag_num):
    mtAUPRC = 0
    tot = 0
    for i in range(tag_num):
        AUPRC = get_tag_node(use_cuda, i)
        if AUPRC != -1:
            mtAUPRC += AUPRC
            tot += 1
    print("mean AUPRC: ", 100*mtAUPRC/tot)


def label_retrieval(use_cuda):
    mMR = 0
    mTN = 0
    for i in range(100):
        MR, TN = get_node_tag(use_cuda, i)
        mMR += MR
        mTN += TN
    mMR /= 100
    mTN /= 100
    print("mean MR: ", mMR)


def get_label_label(use_cuda, data, label):
    source_tag = []
    for i in range(tag_num):
        source_tag.append(label)
    tag_list = range(tag_num)
    if use_cuda:
        source_tag = torch.tensor(source_tag, dtype=torch.long).cuda()
        tag_list = torch.tensor(tag_list, dtype=torch.long).cuda()
    else:
        source_tag = torch.tensor(source_tag, dtype=torch.long)
        tag_list = torch.tensor(tag_list, dtype=torch.long)

    score = model.tag_tag_dist(source_tag, tag_list).cpu()
    rank_score = dict()
    for k in range(tag_num):
        rank_score[k] = score[k]
    sorted_score = sorted(rank_score, key=lambda x: rank_score[x])

    label_neighbor = list(data.graph[data.node_num + label])
    ground_truth_tag = set()
    for i in label_neighbor:
        if i >= node_num:
            ground_truth_tag.add(i-node_num)
    #print(ground_truth_tag)
    avg_rank = 0
    for i in ground_truth_tag:
        avg_rank += sorted_score.index(i) + 1
    avg_rank /= len(ground_truth_tag)
    #print(avg_rank, len(ground_truth_tag))
    return avg_rank, len(ground_truth_tag)


def label_label_relationship(tag_num, use_cuda, dataset):
    datafile = ""
    if dataset == "dblp":
        datafile = "dataset/dblp/train/dblp.txt"
    if dataset == "protein_go":
        datafile = "dataset/protein_go/train/protein_go.txt"
    if dataset == "gene_pathway":
        datafile = "dataset/gene_pathway/train/gene_pathway.txt"

    data = HIME_Dataset(datafile, 5)
    mMR = 0
    mTN = 0
    for i in range(200):
        MR, TN = get_label_label(use_cuda, data, i)
        mMR += MR
        mTN += TN
    mMR /= 200
    mTN /= 200
    print("mean MR: ", mMR, " average_degree:", mTN)


if __name__ == '__main__':
    args = parse_args()
    save_dir = "saved_model/" + args.dataset + "_emb_num_" + str(args.emb_num) + "_emb_dim_" + str(args.emb_dim)
    model_path = save_dir + "/epoch_" + str(args.epoch_num) + '.pkl'
    if args.dataset == "dblp":
        file = open("dataset/dblp/train/dblp.txt")
    if args.dataset == "protein_go":
        file = open("dataset/protein_go/train/protein_go.txt")
    if args.dataset == "gene_pathway":
        file = open("dataset/gene_pathway/train/gene_pathway.txt")
    items = file.readline().strip().split()
    node_num, tag_num = int(items[0]), int(items[1])
    tot_num = node_num + tag_num

    if args.dataset == "dblp":
        json_file = "dataset/dblp/taxo.json"
        taxo_file = "dataset/dblp/taxo.dat"
    if args.dataset == "protein_go":
        json_file = "dataset/protein_go/taxo.json"
        taxo_file = "dataset/protein_go/taxo.dat"
    if args.dataset == "gene_pathway":
        json_file = "dataset/gene_pathway/taxo.json"
        taxo_file = "dataset/gene_pathway/taxo.dat"
    taxo_parent2children, taxo_child2parents, nodeid2category, category2nodeid, category2id, nodeid2path = utils.read_taxos(json_file,
           taxo_file, extend_label=True)

    with open(model_path, 'rb') as f:
        if args.use_cuda:
            model = pickle.loads(f.read()).cuda()
        else:
            model = pickle.loads(f.read())

    a, b, c = read_labeled_data(args.use_cuda, args.dataset)
    print("Link Prediction:")
    node_pair_retrieval(a, b, c)
    print("Hierarchical Classification:")
    label_path_retrieval(args.use_cuda, args.dataset, taxo_child2parents)
    print("Node Search:")
    node_retrieval(args.use_cuda, tag_num)
    print("Label Search:")
    label_retrieval(args.use_cuda)
    print("Child Search:")
    label_label_relationship(tag_num, args.use_cuda, args.dataset)
