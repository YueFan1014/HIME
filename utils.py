import json
import collections

def l2_loss(tensor):
    return 0.5*((tensor ** 2).sum())

def sublist(lst1, lst2):
    return set(lst1) <= set(lst2)


def read_taxos(taxo_file, taxo_assign_file, extend_label=True):
    taxo_parent2children = json.load(open(taxo_file))['parent2children']
    categories = set()
    for p, cs in taxo_parent2children.items():
        categories.add(p)
        categories.update(cs)
    category2id = {c: i for i, c in enumerate(sorted(list(categories)))}
    taxo_parent2children = {category2id[p]:[category2id[c] for c in cs] for p, cs in taxo_parent2children.items()}
    taxo_child2parents = {c: p for p, cs in taxo_parent2children.items() for c in cs}
    #print(taxo_child2parents)

    nodeid2category = collections.defaultdict(list)
    with open(taxo_assign_file, "r") as fin:
        for l in fin:
            #print(l)
            nodeid, category = l.strip().split('.')
            nodeid = int(nodeid)
            cateid = category2id[category]
            nodeid2category[nodeid].append(cateid)
    category2id = dict(category2id)

    nodeid2path = {}
    for node, categorys in nodeid2category.items():
        paths = []
        for c in categorys:
            path = collections.deque()
            while True:
                p = taxo_child2parents.get(c, None)
                if p is None:
                    break
                path.appendleft(c)
                c = p
            paths.append(list(path))
        nodeid2path[node] = paths

    new_nodeid2path = {}
    for nodeid, paths in nodeid2path.items():
        if len(paths) == 1:
            unique_paths = paths
        else:
            unique_paths = []
            for i, path in enumerate(paths):
                flag = 1
                for j, other_path in enumerate(paths):
                    if i==j:
                        continue
                    if sublist(path, other_path):
                        flag = 0
                        break
                if flag:
                    unique_paths.append(path)
        new_nodeid2path[nodeid] = unique_paths
    nodeid2path = new_nodeid2path

    category2nodeid = collections.defaultdict(list)
    if extend_label:
        nodeid2category = {}
        for nodeid, paths in nodeid2path.items():
            cs = list(set([i for p in paths for i in p]))
            nodeid2category[nodeid] = cs
            for c in cs:
                category2nodeid[c].append(nodeid)
    else:
        for nodeid, category in nodeid2category.items():
            for c in category:
                category2nodeid[c].append(nodeid)
    category2nodeid = dict(category2nodeid)
    return taxo_parent2children, taxo_child2parents, nodeid2category, category2nodeid, category2id, nodeid2path


def find_path(child2parents, c):
    path = [c]
    while True:
        p = child2parents.get(c, None)
        if p is None:
            return path
        else:
            path.append(p)
            c = p
