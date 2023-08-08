import json
import os
import random
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
def get_att_adj(triples,ent_size):
    print('getting a sparse tensor r_adj...')
    edge_dict = {}
    ent_attr_dict={}
    attribute_number=0
    attr_len=10
    val_number=0
    attr_value_dict={}
    ent_attr_value_dict={}
    value_attr_concate={}
    for (e, a, v) in triples:
        aids=ent_attr_dict.get(e,set())
        aids.add(a)
        attribute_number=max(a,attribute_number)
        val_number=max(v,val_number)
        ent_attr_dict[e]=aids
        vids=attr_value_dict.get(a,set())
        vids.add(v)
        attr_value_dict[a]=vids

        ent_attr = ent_attr_value_dict.get(e, set())
        ent_attr.add(v)
        ent_attr_value_dict[e] = ent_attr
        value_attr_concate[v]=a
    attr_value=list()
    attribute_number+=1
    val_number+=1
    print('val:',val_number)
    ent_init_list=[]
    sorted_value_list=[]
    for i in range(ent_size):
        val=[val_number]
        if i in ent_attr_value_dict:
            val=ent_attr_value_dict[i]
        sort_val=sorted(val)
        ent_init_list.append(sort_val[0])
        if len(val) > attr_len:
            val = sort_val[1:attr_len + 1]
        else:
            temp_val = [val_number] * (attr_len + 1 - len(val))
            val = sort_val[1:] + temp_val
        sorted_value_list.append(val)

    for i in range(attribute_number):
        vs=list(attr_value_dict[i])
        random.shuffle(vs)
        pad=max(0,100-len(vs))
        vs+=[val_number]*pad
        attr_value.append(vs[:100])
    value_attr_concate_list=[]
    for i in range(val_number):
        value_attr_concate_list.append(value_attr_concate[i])

    attr_pair = {}
    attr_count={}
    for id in ent_attr_dict.keys():
        attr_set = ent_attr_dict[id]
        attr_num = len(attr_set)
        temp_attr_list = list(attr_set)
        for i in range(attr_num):
            for j in range(i + 1, attr_num):
                temp_attr_pair = attr_pair.get((temp_attr_list[i], temp_attr_list[j]), 0)
                temp_attr_pair += 1
                attr_pair[(temp_attr_list[i], temp_attr_list[j])] = temp_attr_pair

                temp_attr_pair = attr_pair.get((temp_attr_list[j], temp_attr_list[i]), 0)
                temp_attr_pair += 1
                attr_pair[(temp_attr_list[j], temp_attr_list[i])] = temp_attr_pair
            temp_count = attr_count.get(temp_attr_list[i], 0)
            temp_count += (attr_num - 1)
            attr_count[temp_attr_list[i]] = temp_count
    row = list()
    col = list()
    data = list()
    for item in attr_pair:
        row.append(item[0])
        col.append(item[1])
        temp_all = attr_count[item[0]] + attr_count[item[1]]
        data.append(2 * attr_pair[item] / temp_all)

    adj = sp.coo_matrix((data, (row, col)), shape=(attribute_number, attribute_number))
    adj = adj + sp.eye(adj.shape[0])
    adj_normalized = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj))
    return adj_normalized,torch.tensor(attr_value),torch.tensor(value_attr_concate_list),torch.tensor(sorted_value_list),torch.tensor(ent_init_list)
def get_val_adj(triples,ent_size,norm=True):
    print('getting a sparse tensor val_adj...')
    e2val_edges=[]
    e2val_values=[]
    val_number = 0
    ev2a={}

    att_number=0
    e2att_edges=[]
    e2att_values=[]
    for (e, a, v) in triples:
        e2val_edges.append((e,v))
        e2val_values.append(1)
        val_number=max(v+1,val_number)
        ev2a[(e,v)]=a
        e2att_edges.append((e,a))
        e2att_values.append(1)
        att_number = max(a + 1, att_number)

    print('att',att_number)
    ind = np.array(e2val_edges, dtype=np.int32)
    val = np.array(e2val_values, dtype=np.float32)
    e2v_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size,val_number), dtype=np.float32)
    e2v_adj=sparse_mx_to_torch_sparse_tensor(row_normalize_adj(e2v_adj) if norm else e2v_adj)

    ind = np.array(e2att_edges, dtype=np.int32)
    val = np.array(e2att_values, dtype=np.float32)
    e2a_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size,att_number), dtype=np.float32)
    e2a_adj=sparse_mx_to_torch_sparse_tensor(row_normalize_adj(e2a_adj))

    return e2v_adj,ev2a,e2a_adj
def get_adj(ent_size,rel_size,triples,norm=True,direction=True):
    print('getting a sparse tensor r_adj...')
    edge_dict = {}
    all_neb={}
    max_nb=0
    for (h, r, t) in triples:
        if h != t:
            if (h, t) not in edge_dict:
                edge_dict[(h, t)] = []
            edge_dict[(h, t)].append(r)
            nebs=all_neb.get(h,set())
            nebs.add(t)
            max_nb=max(max_nb,len(nebs))
            all_neb[h]=nebs
            if direction:
                if (t, h) not in edge_dict:
                    edge_dict[(t, h)] = []
                edge_dict[(t, h)].append(rel_size + r)
                nebs = all_neb.get(t, set())
                nebs.add(h)
                max_nb = max(max_nb, len(nebs))
                all_neb[t] = nebs
    max_nb=min(max_nb,1000)
    all_nebs={}
    for key in all_neb:
        neighbors=list(all_neb[key])
        if len(neighbors) > max_nb:
            neighbors = neighbors[:max_nb]
        all_nebs[key] = neighbors + [ent_size] * (max_nb - len(neighbors))
    for i in range(ent_size):
        if i not in all_nebs:
            all_nebs[i] =[i]+[ent_size] * (max_nb-1)



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

    return rel_size,e2e_adj,e2eself_adj, e2r_adj,e2d_adj,d2r_adj,r2e_adj,r2t_adj,h2t_adj,t2t_adj,e2ei_adj,e2eo_adj,e2ri_adj,e2ro_adj,r2ei_adj,r2eo_adj,all_nebs
def read_data(args):
    prefix = '../../data/'+ args.dataset + '/' + args.lang
    ent_ill_path=prefix+'/ent_links_id'
    ent_ids_1_path=prefix+'/ent_ids_1'
    ent_ids_2_path = prefix + '/ent_ids_2'
    rel_ids_1_path=prefix+'/rel_ids_1'
    rel_ids_2_path = prefix + '/rel_ids_2'
    kg1_path = prefix + '/triples_1'
    kg2_path = prefix + '/triples_2'
    att_kg1_path=prefix+'/attr_triple_1'
    att_kg2_path=prefix+'/attr_triple_2'
    val_ids_path = prefix + '/val_ids'


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
    all_triple = kg1 + kg2
    index2name1 = load_id2object([ent_ids_1_path],"entity")
    index2name2 = load_id2object([ent_ids_2_path], "entity")
    index2name={}
    index2name.update(index2name1)
    index2name.update(index2name2)
    ent_ids_1=list(index2name1.keys())
    ent_ids_2=list(index2name2.keys())
    index2rel = load_id2object([rel_ids_1_path, rel_ids_2_path])
    ent_size=max(index2name.keys())+1
    rel_size=max(index2rel.keys())+1+1
    if args.model=='roadEA':
        attr_kg1 = load_file(att_kg1_path, 3)
        attr_kg2 = load_file(att_kg2_path, 3)
        all_att_triple = attr_kg1 + attr_kg2
        attr_adj,attr_value,value_attr_concate,sorted_value_list,ent_init_list=get_att_adj(all_att_triple,ent_size)
        val_adj,ev2a,e2a_adj=None,None,None
        vid2type=None
    elif args.model=='TestM':
        attr_kg1 = load_file(att_kg1_path, 3)
        attr_kg2 = load_file(att_kg2_path, 3)
        all_att_triple = attr_kg1 + attr_kg2
        val_adj,ev2a,e2a_adj=get_val_adj(all_att_triple,ent_size,args.normalize_adj)
        attr_adj, attr_value, value_attr_concate, sorted_value_list, ent_init_list = None, None, None, None, None
        val_type = load_file(val_ids_path, 2)
        vid2type={vid:t for vid,t in val_type }
    else:
        attr_adj,attr_value,value_attr_concate,sorted_value_list,ent_init_list=None,None,None,None,None
        val_adj,ev2a,e2a_adj=None,None,None
        vid2type=None
    if args.model=='RAGA':
        rel_size, e2e_adj,e2eself_adj, e2r_adj,e2d_adj,d2r_adj,r2e_adj,r2t_adj_o,h2t_adj_o,t2t_adj_o,e2ei_adj,e2eo_adj,e2ri_adj,e2ro_adj,r2ei_adj,r2eo_adj,all_nebs=get_adj(ent_size, rel_size, all_triple, args.normalize_adj, False)
    else:
        r2t_adj_o, h2t_adj_o, t2t_adj_o=None,None,None
    rel_size, e2e_adj,e2eself_adj, e2r_adj,e2d_adj,d2r_adj,r2e_adj,r2t_adj,h2t_adj,t2t_adj,e2ei_adj,e2eo_adj,e2ri_adj,e2ro_adj,r2ei_adj,r2eo_adj,all_nebs=get_adj(ent_size,rel_size,all_triple,args.normalize_adj,args.direction)
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
        'val_adj':val_adj,
        'att_adj': attr_adj,
        'attr_value':attr_value,
        'value_attr_concate':value_attr_concate,
        'sorted_value_list':sorted_value_list,
        'ent_init_list':ent_init_list,
        'kg1':kg1,
        'kg2':kg2,
        'ev2a':ev2a,
        'vid2type':vid2type,
        'e2a_adj':e2a_adj,
        'all_nebs':all_nebs

    }

    return data

def get_e_init_layer(args,file_path,sizes,train_ill,normalize_feats=True):
    print('entity:adding the primal init layer')
    if file_path=='random':
        embedding = sizes
    elif file_path=='lightea':
        random_vec=F.normalize(torch.FloatTensor(np.random.normal(0,1,(len(train_ill),sizes[1]))), p=2, dim=1)
        embedding=torch.zeros(sizes)
        embedding[torch.tensor(train_ill).view(-1)]=random_vec.repeat(1,2).reshape(-1,sizes[1])
    elif file_path=='vectorList.json':
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
def get_r_init_layer(args,file_path,sizes,normalize_feats=True):
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
def get_v_init_layer(args,file_path,normalize_feats=True):
    print('value:adding the primal init layer')
    if file_path=='random':
        return []
    file_path = args.save_path1 + '/' + file_path
    embedding = load_embedding(file_path, normalize_feats)
    return embedding
def get_a_init_layer(args,file_path,normalize_feats=True):
    print('attribute:adding the primal init layer')
    if file_path=='random':
        return []
    file_path = args.save_path1 + '/' + file_path
    embedding = load_embedding(file_path, normalize_feats)
    return embedding