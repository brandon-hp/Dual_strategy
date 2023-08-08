import json
import os
import random
import string

import numpy as np
import scipy.sparse as sp
import sys
sys.path.append('..')
from utils.data_utils import *

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
def normalize_adj1(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T
def row_normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -1).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return r_mat_inv_sqrt.dot(mx)
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def get_nadj(ent_size,rel_size,triples,direction=True,norm=True):
    print('getting a sparse tensor r_adj...')
    edge_dict = {}
    for (h, r, t) in triples:
        if h != t:
            if (h, t) not in edge_dict:
                edge_dict[(h, t)] = []
                edge_dict[(t, h)] = []
            edge_dict[(h, t)].append(r)
            if direction:
                edge_dict[(t, h)].append(rel_size + r)
    rel_size= rel_size*2-1 if direction else rel_size
    edges = [[h, t] for (h, t) in edge_dict for r in edge_dict[(h, t)]]
    values = [1 for (h, t) in edge_dict for r in edge_dict[(h, t)]]
    r_ij = [abs(r) for (h, t) in edge_dict for r in edge_dict[(h, t)]]

    r_sq = torch.tensor(r_ij)
    ind = np.array(edges, dtype=np.int32)
    val = np.array(values, dtype=np.float32)
    adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size, ent_size), dtype=np.float32)
    nadj=sparse_mx_to_torch_sparse_tensor(normalize_adj(adj) if norm else adj)
    return rel_size,r_sq, nadj
def get_adj(ent_size,rel_size,triples,align_rel,norm=True,direction=True):
    print('getting a sparse tensor r_adj...')
    edge_dict = {}
    for (h, r, t) in triples:
        if h != t:
            if r in align_rel:
                r=align_rel[r]
            if (h, t) not in edge_dict:
                edge_dict[(h, t)] = []
            edge_dict[(h, t)].append(r)
            if direction:
                if (t, h) not in edge_dict:
                    edge_dict[(t, h)] = []
                edge_dict[(t, h)].append(rel_size + r)
    self_r=rel_size-1
    rel_size= rel_size*2-1 if direction else rel_size
    e2e_edges=[]
    e2e_values=[]
    e2eself_edges=[]
    e2eself_values=[]
    d2r_edges=[]
    d2r_values=[]
    e2r_edges=[]
    e2r_values=[]
    e2d_edges=[]
    e2d_values=[]
    r2e_edges=[]
    r2e_values=[]
    r2t_edges=[]
    r2t_values=[]
    h2t_edges=[]
    h2t_values=[]
    t2t_edges=[]
    t2t_values=[]

    r2tself_edges=[]
    r2tself_values=[]
    h2tself_edges=[]
    h2tself_values=[]
    t2tself_edges=[]
    t2tself_values=[]

    e2ei_edges=[]
    e2ei_values=[]
    e2eo_edges=[]
    e2eo_values=[]
    e2ri_edges=[]
    e2ri_values=[]
    e2ro_edges=[]
    e2ro_values=[]
    r2ei_edges = []
    r2ei_values=[]
    r2eo_edges=[]
    r2eo_values=[]
    i=0
    tr_n=0
    for (h, t) in edge_dict:
        e2d_edges.append((h, i))
        e2d_values.append(1)
        e2ei_edges.append((t,h))
        e2ei_values.append(1)
        e2eo_edges.append((h,t))
        e2eo_values.append(1)
        for r in edge_dict[(h,t)]:
            e2e_edges.append((h, t))
            e2e_values.append(1)
            e2eself_edges.append((h, t))
            e2eself_values.append(1)
            d2r_edges.append((i,r))
            d2r_values.append(1)
            e2r_edges.append((h,r))
            e2r_values.append(1)
            r2e_edges.append((r,t))
            r2e_values.append(1)
            r2t_edges.append((r,tr_n))
            r2t_values.append(1)
            h2t_edges.append((h,tr_n))
            h2t_values.append(1)
            t2t_edges.append((t,tr_n))
            t2t_values.append(1)
            e2ri_edges.append((t,r))
            e2ri_values.append(1)
            e2ro_edges.append((h,r))
            e2ro_values.append(1)
            r2ei_edges.append((r,t))
            r2ei_values.append(1)
            r2eo_edges.append((r,h))
            r2eo_values.append(1)
            tr_n=tr_n+1
        i=i+1

    e2eself_edges=e2eself_edges+[(e,e) for e in range(ent_size)]
    e2eself_values=e2eself_values+[1 for e in range(ent_size)]
    r2tself_edges=r2t_edges+[(self_r,tr_n+e) for e in range(ent_size)]
    r2tself_values=r2t_values+[1 for e in range(ent_size)]
    h2tself_edges=h2t_edges+[(e,tr_n+e) for e in range(ent_size)]
    h2tself_values=h2t_values+[1 for e in range(ent_size)]
    t2tself_edges=t2t_edges+[(e,tr_n+e) for e in range(ent_size)]
    t2tself_values=t2t_values+[1 for e in range(ent_size)]





    ind = np.array(e2e_edges, dtype=np.int32)
    val = np.array(e2e_values, dtype=np.float32)
    e2e_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size, ent_size), dtype=np.float32)
    e2e_adj=sparse_mx_to_torch_sparse_tensor(normalize_adj(e2e_adj) if norm else e2e_adj)

    ind = np.array(e2eself_edges, dtype=np.int32)
    val = np.array(e2eself_values, dtype=np.float32)
    e2eself_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size, ent_size), dtype=np.float32)
    e2eself_adj=sparse_mx_to_torch_sparse_tensor(normalize_adj(e2eself_adj) if norm else e2eself_adj)

    ind = np.array(d2r_edges, dtype=np.int32)
    val = np.array(d2r_values, dtype=np.float32)
    d2r_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(i, rel_size), dtype=np.float32)
    d2r_adj=sparse_mx_to_torch_sparse_tensor(row_normalize_adj(d2r_adj) if norm else d2r_adj)

    ind = np.array(e2r_edges, dtype=np.int32)
    val = np.array(e2r_values, dtype=np.float32)
    e2r_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size,rel_size), dtype=np.float32)
    e2r_adj=sparse_mx_to_torch_sparse_tensor(row_normalize_adj(e2r_adj) if norm else e2r_adj)

    ind = np.array(e2d_edges, dtype=np.int32)
    val = np.array(e2d_values, dtype=np.float32)
    e2d_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size,i), dtype=np.float32)
    e2d_adj=sparse_mx_to_torch_sparse_tensor(row_normalize_adj(e2d_adj) if norm else e2d_adj)

    ind = np.array(r2e_edges, dtype=np.int32)
    val = np.array(r2e_values, dtype=np.float32)
    r2e_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(rel_size,ent_size), dtype=np.float32)
    r2e_adj=sparse_mx_to_torch_sparse_tensor(row_normalize_adj(r2e_adj) if norm else r2e_adj)

    ind = np.array(r2t_edges, dtype=np.int32)
    val = np.array(r2t_values, dtype=np.float32)
    r2t_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(rel_size,tr_n), dtype=np.float32)
    r2t_adj=sparse_mx_to_torch_sparse_tensor(r2t_adj)

    ind = np.array(h2t_edges, dtype=np.int32)
    val = np.array(h2t_values, dtype=np.float32)
    h2t_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size,tr_n), dtype=np.float32)
    h2t_adj=sparse_mx_to_torch_sparse_tensor(h2t_adj)

    ind = np.array(t2t_edges, dtype=np.int32)
    val = np.array(t2t_values, dtype=np.float32)
    t2t_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size,tr_n), dtype=np.float32)
    t2t_adj=sparse_mx_to_torch_sparse_tensor(t2t_adj)

    ind = np.array(r2tself_edges, dtype=np.int32)
    val = np.array(r2tself_values, dtype=np.float32)
    r2t_adj_self = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(rel_size,tr_n+ent_size), dtype=np.float32)
    r2t_adj_self=sparse_mx_to_torch_sparse_tensor(r2t_adj_self)

    ind = np.array(h2tself_edges, dtype=np.int32)
    val = np.array(h2tself_values, dtype=np.float32)
    h2t_adj_self = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size,tr_n+ent_size), dtype=np.float32)
    h2t_adj_self=sparse_mx_to_torch_sparse_tensor(h2t_adj_self)

    ind = np.array(t2tself_edges, dtype=np.int32)
    val = np.array(t2tself_values, dtype=np.float32)
    t2t_adj_self = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size,tr_n+ent_size), dtype=np.float32)
    t2t_adj_self=sparse_mx_to_torch_sparse_tensor(t2t_adj_self)

    ind = np.array(e2ei_edges, dtype=np.int32)
    val = np.array(e2ei_values, dtype=np.float32)
    e2ei_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size, ent_size), dtype=np.float32)
    e2ei_adj = sparse_mx_to_torch_sparse_tensor(row_normalize_adj(e2ei_adj) if norm else e2ei_adj)

    ind = np.array(e2eo_edges, dtype=np.int32)
    val = np.array(e2eo_values, dtype=np.float32)
    e2eo_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size, ent_size), dtype=np.float32)
    e2eo_adj = sparse_mx_to_torch_sparse_tensor(row_normalize_adj(e2eo_adj) if norm else e2eo_adj)

    ind = np.array(e2ri_edges, dtype=np.int32)
    val = np.array(e2ri_values, dtype=np.float32)
    e2ri_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size, rel_size), dtype=np.float32)
    e2ri_adj = sparse_mx_to_torch_sparse_tensor(row_normalize_adj(e2ri_adj) if norm else e2ri_adj)

    ind = np.array(e2ro_edges, dtype=np.int32)
    val = np.array(e2ro_values, dtype=np.float32)
    e2ro_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size, rel_size), dtype=np.float32)
    e2ro_adj = sparse_mx_to_torch_sparse_tensor(row_normalize_adj(e2ro_adj) if norm else e2ro_adj)

    ind = np.array(r2ei_edges, dtype=np.int32)
    val = np.array(r2ei_values, dtype=np.float32)
    r2ei_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(rel_size, ent_size), dtype=np.float32)
    r2ei_adj = sparse_mx_to_torch_sparse_tensor(row_normalize_adj(r2ei_adj) if norm else r2ei_adj)

    ind = np.array(r2eo_edges, dtype=np.int32)
    val = np.array(r2eo_values, dtype=np.float32)
    r2eo_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(rel_size, ent_size), dtype=np.float32)
    r2eo_adj = sparse_mx_to_torch_sparse_tensor(row_normalize_adj(r2eo_adj) if norm else r2eo_adj)

    return rel_size,e2e_adj,e2eself_adj, e2r_adj,e2d_adj,d2r_adj,r2e_adj,r2t_adj,h2t_adj,t2t_adj,e2ei_adj,e2eo_adj,e2ri_adj,e2ro_adj,r2ei_adj,r2eo_adj,r2t_adj_self,t2t_adj_self,h2t_adj_self
def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
def get_name(name):
    if r"resource/" in name:
        sub_string = name.split(r"resource/")[-1]
    elif r"property/" in name:
        sub_string = name.split(r"property/")[-1]
    else:
        sub_string = name.split(r"/")[-1]
    sub_string = sub_string.replace('_',' ')
    re_string=remove_punc(sub_string)
    sub_string=sub_string if re_string=='' else re_string
    #sub_string=re.sub(' +',' ',sub_string)
    return sub_string
def read_data(args):
    prefix = '../../data/'+ args.dataset + '/' + args.lang
    ent_ill_path=prefix+'/ent_links_id'
    ent_ids_1_path=prefix+'/ent_ids_1'
    ent_ids_2_path = prefix + '/ent_ids_2'
    rel_ids_1_path=prefix+'/rel_ids_1'
    rel_ids_2_path = prefix + '/rel_ids_2'
    kg1_path = prefix + '/triples_1'
    kg2_path = prefix + '/triples_2'


    ent_ill=load_file(ent_ill_path,2)
    if args.random_ill:
        print("Random divide train/valid/test ILLs!")
        random.shuffle(ent_ill)
        train_valid_ill = random.sample(ent_ill, int(len(ent_ill) * (args.train_ill_rate+args.valid_ill_rate)))
        test_ill = list(set(ent_ill) - set(train_valid_ill))
        train_ill=random.sample(train_valid_ill,int(len(train_valid_ill) * (args.train_ill_rate)))
        valid_ill = list(set(train_valid_ill) - set(train_ill))
    else:
        train_ill_path=prefix+'/train_links'
        valid_ill_path = prefix + '/valid_links'
        test_ill_path=prefix+'/test_links'
        test_ill = load_file(test_ill_path,2)
        train_ill=load_file(train_ill_path,2)
        valid_ill = load_file(valid_ill_path,2)
    print("train ILL num: {},valid ILL num: {}, test ILL num: {}".format(len(train_ill), len(valid_ill),len(test_ill)))
    print("train ILL | valid ILL |test ILL num:", len(set(train_ill) | set(valid_ill)|set(test_ill)))
    print("train ILL & valid ILL &test ILL num:", len(set(train_ill)& set(valid_ill) & set(test_ill)))



    kg1 = load_file(kg1_path, 3)  # KG1
    kg2 = load_file(kg2_path, 3)  # KG2
    print(len(kg1),len(kg2))
    all_triple=kg1+kg2

    index2name1 = load_id2object([ent_ids_1_path],"entity")
    index2name2 = load_id2object([ent_ids_2_path], "entity")
    index2name={}
    index2name.update(index2name1)
    index2name.update(index2name2)
    ent_ids_1=list(index2name1.keys())
    ent_ids_2=list(index2name2.keys())
    index2rel = load_id2object([rel_ids_1_path, rel_ids_2_path])
    align_rel={}
    name2rel={}
    if args.use_ra:
        for id in index2rel:
            name=get_name(index2rel[id])
            if name in name2rel:
                align_rel[id]=name2rel[name]
            else:
                name2rel[name]=id
    ent_size=max(index2name.keys())+1
    rel_size=max(index2rel.keys())+1+1
    if args.model=='RAGA' or args.model=='RAGA_literal' or args.model=='RREAN_literal' or args.model=='ERMC_adapt'or args.model=='Test':
        rel_size, e2e_adj,e2eself_adj, e2r_adj,e2d_adj,d2r_adj,r2e_adj,r2t_adj_o,h2t_adj_o,t2t_adj_o,e2ei_adj,e2eo_adj,e2ri_adj,e2ro_adj,r2ei_adj,r2eo_adj,_,_,_=get_adj(ent_size, rel_size, all_triple,align_rel, args.normalize_adj, False)
    else:
        r2t_adj_o, h2t_adj_o, t2t_adj_o=None,None,None
    rel_size, e2e_adj,e2eself_adj, e2r_adj,e2d_adj,d2r_adj,r2e_adj,r2t_adj,h2t_adj,t2t_adj,e2ei_adj,e2eo_adj,e2ri_adj,e2ro_adj,r2ei_adj,r2eo_adj,r2t_adj_self,t2t_adj_self,h2t_adj_self=get_adj(ent_size,rel_size,all_triple,align_rel,args.normalize_adj,args.direction)
    if args.swap==True:
        align={}
        for e1,e2 in train_ill:
            align[e1]=e2
            align[e2]=e1
        addset=[]
        for h,r,t in kg1:
            if h in align:
                addset.append((align[h],r,t))
            if t in align:
                addset.append((h, r, align[t]))
        kg1+=addset
        addset = []
        for h,r,t in kg2:
            if h in align:
                addset.append((align[h],r,t))
            if t in align:
                addset.append((h, r, align[t]))
        kg2+=addset


    data={
        'train_ill':train_ill,
        'valid_ill':valid_ill,
        'test_ill':test_ill,
        'rel_sizes':(rel_size,args.feat_dim),
        'ent_sizes': (ent_size, args.feat_dim),
        'e2e_adj':e2e_adj,
        'd2r_adj':d2r_adj,
        'e2eself_adj': e2eself_adj,
        'e2r_adj': e2r_adj,
        'e2d_adj':e2d_adj,
        'r2e_adj': r2e_adj,
        'r2t_adj': r2t_adj,
        'h2t_adj': h2t_adj,
        't2t_adj': t2t_adj,
        'r2t_adj_self': r2t_adj_self,
        'h2t_adj_self': h2t_adj_self,
        't2t_adj_self': t2t_adj_self,
        'r2t_adj_o': r2t_adj_o,
        'h2t_adj_o': h2t_adj_o,
        't2t_adj_o': t2t_adj_o,
        'e2ei_adj': e2ei_adj,
        'e2eo_adj': e2eo_adj,
        'e2ri_adj': e2ri_adj,
        'e2ro_adj': e2ro_adj,
        'r2ei_adj': r2ei_adj,
        'r2eo_adj': r2eo_adj,
        'ent_ids_1':ent_ids_1,
        'ent_ids_2':ent_ids_2,
        'kg1':kg1,
        'kg2':kg2

    }

    return data

def get_e_init_layer(args,sizes,train_ill,normalize_feats=True):
    file_path=args.ent_vec_path
    print('entity:adding the primal init layer')
    if file_path=='random':
        embedding = sizes
    elif file_path=='lightea':
        random_vec=F.normalize(torch.FloatTensor(np.random.normal(0,1,(len(train_ill),sizes[1]))), p=2, dim=1)
        embedding=torch.zeros(sizes)
        embedding[torch.tensor(train_ill).view(-1)]=random_vec.repeat(1,2).reshape(-1,sizes[1])
    elif 'vectorList.json' in file_path:
        file_path = args.save_path1 + '/' + file_path
        with open(file=file_path, mode='r', encoding='utf-8') as f:
            embedding_list = json.load(f)
        input_embeddings = F.normalize(torch.tensor(embedding_list), p=2, dim=1)
        embedding = input_embeddings
        embedding[torch.isnan(embedding)] = 0

    else:
        file_path = args.save_path1 + '/' + file_path
        embedding=load_embedding(file_path,normalize_feats)
    return embedding
def get_r_init_layer(args,sizes,normalize_feats=True):
    file_path=args.rel_vec_path
    print('relation:adding the primal init layer')
    if file_path=='random':
        embedding = sizes
    elif file_path=='lightea':
        embedding=torch.zeros(sizes)
    else:
        file_path=args.save_path1+'/'+file_path
        embedding=load_embedding(file_path,normalize_feats)
        if embedding.shape[0]+1<sizes[0]:
            embedding=torch.cat([embedding,torch.zeros(1,embedding.shape[1]),embedding],dim=0)
        elif embedding.shape[0]+1==sizes[0]:
            embedding=torch.cat([embedding,torch.zeros(1,embedding.shape[1])],dim=0)
        else:
            pass
    return embedding