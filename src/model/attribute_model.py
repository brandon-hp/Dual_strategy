import os
import pickle

import torch
import torch.nn as nn
import numpy as np
import model.encoders as encoders
import model.decoders as decoders
import torch.nn.functional as F
import scipy.sparse as sp
import copy
from tqdm import tqdm
import sys
sys.path.append('..')
import manifolds
from utils.data_utils import candidate_generate,all_entity_pairs_gene,ent2attributeValues_gene,neigh_ent_dict_gene
def get_bert_int_data(args,data,ent_emb,val_emb):
    train_candidates, valid_candidates, test_candidates=get_candidates(args,data,ent_emb)
    if os.path.exists('../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/' + args.entity_pairs_path):
        entity_pairs = pickle.load(open(
            '../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/' + args.entity_pairs_path, "rb"))
        train_candidates = dict()
        valid_candidates = dict()
        test_candidates = dict()
        test_ids_1 = [int(e1) for e1, e2 in data['test_ill']]
        valid_ids_1 = [int(e1) for e1, e2 in data['valid_ill']]
        train_ids_1 = [int(e1) for e1, e2 in data['train_ill']]
        for e1, e2 in entity_pairs:
            if e1 in test_ids_1:
                elist=test_candidates.get(e1,[])
                elist.append(e2)
                test_candidates[e1]=elist
            elif e1 in valid_ids_1:
                elist=valid_candidates.get(e1,[])
                elist.append(e2)
                valid_candidates[e1]=elist
            elif e1 in train_ids_1:
                elist=train_candidates.get(e1,[])
                elist.append(e2)
                train_candidates[e1]=elist
        for e in test_candidates:
            test_candidates[e]=np.array(list(set(test_candidates[e])))
        for e in train_candidates:
            train_candidates[e]=np.array(list(set(train_candidates[e])))
        for e in valid_candidates:
            valid_candidates[e]=np.array(list(set(valid_candidates[e])))
    else:
        entity_pairs = all_entity_pairs_gene([ train_candidates,valid_candidates,test_candidates ],[ data['train_ill'] ])
        pickle.dump(entity_pairs, open('../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/bert_int_entity_pairs', "wb"))
    att_datas=pickle.load(open('../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/att_datas', "rb"))
    ent2valueids=ent2attributeValues_gene(att_datas, args.val_max, entity_pairs,len(val_emb))
    sim=pickle.load(open('../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/'+args.relation_model+'_cosin_sim', "rb"))
    cosin_features = [sim[(e1,e2)] for e1, e2 in entity_pairs]
    return entity_pairs,ent2valueids,cosin_features,train_candidates,valid_candidates,test_candidates
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def get_new_att_triple(args,data,ent_emb,val_emb,a_emb):
    train_candidates, valid_candidates, test_candidates=get_candidates(args,data,ent_emb)
    entity_pairs = all_entity_pairs_gene([ train_candidates,valid_candidates,test_candidates ],[ data['train_ill'] ])
    #pickle.dump(entity_pairs, open('../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/bert_int_entity_pairs', "wb"))
    att_datas=pickle.load(open('../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/att_datas', "rb"))
    ent2valueids=ent2attributeValues_gene(att_datas, args.val_max, entity_pairs,len(val_emb))
    pair_ev={}
    vid2type=data['vid2type']
    pair_ev=set()
    e2val_edges_set=set()
    print(len(entity_pairs))
    val_emb1=torch.cat((val_emb, torch.zeros(1, val_emb.shape[1])),dim=0)
    id2sim={}
    id2sim_100 = {}
    default=torch.zeros(args.val_max,1)
    for pair in tqdm(entity_pairs):
        if (pair[0],pair[1]) in pair_ev or (pair[1],pair[0]) in pair_ev:
            continue
        pair_ev.add((pair[0],pair[1]))
        i=pair[0]
        j=pair[1]
        e1_values = ent2valueids[i]  # size: [B(Batchsize), ne1(e1_attributeValue_max_num)]
        e2_values = ent2valueids[j] # [B,ne2
        e1_masks = np.ones(np.array(e1_values).shape)
        e2_masks = np.ones(np.array(e2_values).shape)
        e1_masks[np.array(e1_values) == len(val_emb)] = 0
        e2_masks[np.array(e2_values) == len(val_emb)] = 0
        e1_masks = torch.FloatTensor(e1_masks.tolist()).unsqueeze(-1)  # [B,ne1,1]
        e2_masks = torch.FloatTensor(e2_masks.tolist()).unsqueeze(-1)  # [B,ne2,1]

        e1_values = torch.LongTensor(e1_values)  # [B,ne1]
        e2_values = torch.LongTensor(e2_values)  # [B,ne2]
        e1_values_emb = val_emb1[e1_values]  # [ne1,embedding_dim]
        e2_values_emb = val_emb1[e2_values]  # [ne2,embedding_dim]
        sim_matrix = torch.mm(e1_values_emb, torch.transpose(e2_values_emb, 0, 1))  # [ne1,ne2]
        sim_maxpooing_1, _ = sim_matrix.topk(k=1, dim=-1)  # [ne1,1] #get max value.
        sim=sim_maxpooing_1*e1_masks
        sim[sim<0]=0
        id2sim[i]=id2sim.get(i,default)+sim
        sim[sim<0.99]=0
        id2sim_100[i]=id2sim_100.get(i,default)+sim
        sim_maxpooing_2, _ = torch.transpose(sim_matrix, 0, 1).topk(k=1, dim=-1)
        sim = sim_maxpooing_2 * e2_masks
        sim[sim<0]=0
        id2sim[j]=id2sim.get(j,default)+sim
        sim[sim<0.99]=0
        id2sim_100[i]=id2sim_100.get(i,default)+sim
    e2val_edges1=[]
    for i in id2sim:
        sim=id2sim[i]
        e1_values = ent2valueids[i]
        e1_masks = np.ones(np.array(e1_values).shape)
        e1_masks[np.array(e1_values) == len(val_emb)] = 0
        for k in range(args.val_max):
            if e1_masks[k] == 0:
                break
            if sim[k] > 0:
                e2val_edges1.append((i, e1_values[k]))
    for i in id2sim_100:
        sim=id2sim_100[i]
        e1_values = ent2valueids[i]
        e1_masks = np.ones(np.array(e1_values).shape)
        e1_masks[np.array(e1_values) == len(val_emb)] = 0
        for k in range(args.val_max):
            if e1_masks[k] == 0:
                break
            if sim[k] > 0:
                for _ in range(9):
                    e2val_edges1.append((i, e1_values[k]))

    e2val_values = [1] * len(e2val_edges1)
    ind = np.array(e2val_edges1, dtype=np.int32)
    val = np.array(e2val_values, dtype=np.float32)
    e2v_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(len(ent_emb), len(val_emb)), dtype=np.float32)
    e2v_adj = sparse_mx_to_torch_sparse_tensor(e2v_adj)
    '''
    if os.path.exists('../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/e2v_adj7'):
        e2v_adj = pickle.load(
            open('../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/e2v_adj5', "rb"))
    else:
        e2val_edges_set=set()
        e2val_edges_set=e2val_edges_set|get_candidates_att(args,train_candidates,ent2valueids,len(val_emb),torch.cat((val_emb, torch.zeros(1, val_emb.shape[1])), dim=0),pair_ev)
        e2val_edges_set = e2val_edges_set|get_candidates_att(args, valid_candidates, ent2valueids, len(val_emb),torch.cat((val_emb, torch.zeros(1, val_emb.shape[1])), dim=0),pair_ev)
        e2val_edges_set =e2val_edges_set| get_candidates_att(args, test_candidates, ent2valueids, len(val_emb),torch.cat((val_emb, torch.zeros(1, val_emb.shape[1])), dim=0),pair_ev)
        e2val_edges=list(e2val_edges_set)
        e2val_edges.sort(key=lambda x: x[0])
        e2val_edges1=[]
        for tr in e2val_edges:
            if tr[2]==1:
                if (tr[0],tr[1],2) not in e2val_edges_set:
                    e2val_edges1.append((tr[0],tr[1]))
                else:
                    pass
            else:
                e2val_edges1.append((tr[0], tr[1]))
                e2val_edges1.append((tr[0], tr[1]))
                e2val_edges1.append((tr[0], tr[1]))
                e2val_edges1.append((tr[0], tr[1]))
                e2val_edges1.append((tr[0], tr[1]))
                e2val_edges1.append((tr[0], tr[1]))
                e2val_edges1.append((tr[0], tr[1]))
                e2val_edges1.append((tr[0], tr[1]))
                e2val_edges1.append((tr[0], tr[1]))
                e2val_edges1.append((tr[0], tr[1]))
        e2val_values = [1] * len(e2val_edges1)
        ind = np.array(e2val_edges1, dtype=np.int32)
        val = np.array(e2val_values, dtype=np.float32)
        e2v_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(len(ent_emb),len(val_emb)), dtype=np.float32)
        e2v_adj=sparse_mx_to_torch_sparse_tensor(e2v_adj)
        pickle.dump(e2v_adj,
                    open('../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/e2v_adj5',
                         "wb"))
    edge=e2v_adj._indices()
    ev2a=data['ev2a']
    e2a_edges=[]
    e2t_edges = []
    a2t_edges = []
    v2t_edges = []
    triple_n=0
    acc_ev=set()
    pos_ev=set()
    for i in range(edge.shape[1]):
        e=int(edge[0][i])
        v=int(edge[1][i])
        a=ev2a[(e,v)]
        e2a_edges.append((e,a))
        e2t_edges.append((e,triple_n))
        a2t_edges.append((a,triple_n))
        v2t_edges.append((v,triple_n))
        triple_n+=1
        if (e,v) in pos_ev:
            acc_ev.add((e,v))
        else:
            pos_ev.add((e,v))
    e2a_values=[1]*len(e2a_edges)
    ind = np.array(e2a_edges, dtype=np.int32)
    val = np.array(e2a_values, dtype=np.float32)
    e2a_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(len(ent_emb), len(a_emb)), dtype=np.float32)
    e2a_adj = sparse_mx_to_torch_sparse_tensor(e2a_adj)

    e2t_values=[1]*len(e2t_edges)
    ind = np.array(e2t_edges, dtype=np.int32)
    val = np.array(e2t_values, dtype=np.float32)
    e2t_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(len(ent_emb),triple_n), dtype=np.float32)
    e2t_adj = sparse_mx_to_torch_sparse_tensor(e2t_adj)

    a2t_values=[1]*len(a2t_edges)
    ind = np.array(a2t_edges, dtype=np.int32)
    val = np.array(a2t_values, dtype=np.float32)
    a2t_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(len(a_emb),triple_n), dtype=np.float32)
    a2t_adj = sparse_mx_to_torch_sparse_tensor(a2t_adj)

    v2t_values=[1]*len(v2t_edges)
    ind = np.array(v2t_edges, dtype=np.int32)
    val = np.array(v2t_values, dtype=np.float32)
    v2t_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(len(val_emb),triple_n), dtype=np.float32)
    v2t_adj = sparse_mx_to_torch_sparse_tensor(v2t_adj)
    '''
    ev2a = data['ev2a']
    edge = e2v_adj._indices()
    e2a_edges=[]
    e2t_edges = []
    a2t_edges = []
    v2t_edges = []
    triple_n=0
    acc_ev=set()
    pos_ev=set()
    for i in range(edge.shape[1]):
        e=int(edge[0][i])
        v=int(edge[1][i])
        a=ev2a[(e,v)]
        e2a_edges.append((e,a))
        e2t_edges.append((e,triple_n))
        a2t_edges.append((a,triple_n))
        v2t_edges.append((v,triple_n))
        triple_n+=1
        if (e,v) in pos_ev:
            acc_ev.add((e,v))
        else:
            pos_ev.add((e,v))
    e2a_values=[1]*len(e2a_edges)
    ind = np.array(e2a_edges, dtype=np.int32)
    val = np.array(e2a_values, dtype=np.float32)
    e2a_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(len(ent_emb), len(a_emb)), dtype=np.float32)
    e2a_adj = sparse_mx_to_torch_sparse_tensor(e2a_adj)

    e2t_values=[1]*len(e2t_edges)
    ind = np.array(e2t_edges, dtype=np.int32)
    val = np.array(e2t_values, dtype=np.float32)
    e2t_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(len(ent_emb),triple_n), dtype=np.float32)
    e2t_adj = sparse_mx_to_torch_sparse_tensor(e2t_adj)

    a2t_values=[1]*len(a2t_edges)
    ind = np.array(a2t_edges, dtype=np.int32)
    val = np.array(a2t_values, dtype=np.float32)
    a2t_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(len(a_emb),triple_n), dtype=np.float32)
    a2t_adj = sparse_mx_to_torch_sparse_tensor(a2t_adj)

    v2t_values=[1]*len(v2t_edges)
    ind = np.array(v2t_edges, dtype=np.int32)
    val = np.array(v2t_values, dtype=np.float32)
    v2t_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(len(val_emb),triple_n), dtype=np.float32)
    v2t_adj = sparse_mx_to_torch_sparse_tensor(v2t_adj)



    return e2v_adj,e2a_adj,e2t_adj,a2t_adj,v2t_adj,pair_ev,vid2type
def get_candidates_att(args,train_candidates,ent2valueids,value_pad_id,val_emb,pair_ev):
    e2val_edges = set()
    for i in tqdm(train_candidates):
        clist=train_candidates[i].tolist()
        for j in clist:
            if (i,j) in pair_ev:
                e2val_edges|=pair_ev[(i,j)]
                continue
            p_ev=set()
            e1_values = ent2valueids[i]  # size: [B(Batchsize), ne1(e1_attributeValue_max_num)]
            e2_values = ent2valueids[j] # [B,ne2
            e1_masks = np.ones(np.array(e1_values).shape)
            e2_masks = np.ones(np.array(e2_values).shape)
            e1_masks[np.array(e1_values) == value_pad_id] = 0
            e2_masks[np.array(e2_values) == value_pad_id] = 0
            e1_masks = torch.FloatTensor(e1_masks.tolist()).unsqueeze(-1)  # [B,ne1,1]
            e2_masks = torch.FloatTensor(e2_masks.tolist()).unsqueeze(-1)  # [B,ne2,1]

            e1_values = torch.LongTensor(e1_values)  # [B,ne1]
            e2_values = torch.LongTensor(e2_values)  # [B,ne2]
            e1_values_emb = val_emb[e1_values]  # [ne1,embedding_dim]
            e2_values_emb = val_emb[e2_values]  # [ne2,embedding_dim]
            sim_matrix = torch.mm(e1_values_emb, torch.transpose(e2_values_emb, 0, 1))  # [ne1,ne2]
            sim_maxpooing_1, _ = sim_matrix.topk(k=1, dim=-1)  # [ne1,1] #get max value.
            sim=sim_maxpooing_1*e1_masks
            for k in range(args.val_max):
                if e1_masks[k]==0:
                    break
                if sim[k]>0.9999:
                    p_ev.add((i,int(e1_values[k]),2))
                elif sim[k]>0:
                    p_ev.add((i,int(e1_values[k]),1))
            sim_maxpooing_2, _ = torch.transpose(sim_matrix, 0, 1).topk(k=1, dim=-1)
            sim = sim_maxpooing_2 * e2_masks
            for k in range(args.val_max):
                if e2_masks[k]==0:
                    break
                if sim[k]>0.9999:
                    p_ev.add((j,int(e2_values[k]),2))
                elif sim[k]>0:
                    p_ev.add((j,int(e2_values[k]),1))
            pair_ev[(i, j)]=p_ev
            e2val_edges |= pair_ev[(i, j)]

    return e2val_edges
def get_res_att(args,res_pair,ent2valueids,value_pad_id,val_emb):
    e2val_edges = set()
    for tr in tqdm(res_pair):
        i=tr[0]
        j=tr[1]
        e1_values = ent2valueids[i]  # size: [B(Batchsize), ne1(e1_attributeValue_max_num)]
        e2_values = ent2valueids[j] # [B,ne2
        e1_masks = np.ones(np.array(e1_values).shape)
        e2_masks = np.ones(np.array(e2_values).shape)
        e1_masks[np.array(e1_values) == value_pad_id] = 0
        e2_masks[np.array(e2_values) == value_pad_id] = 0
        e1_masks = torch.FloatTensor(e1_masks.tolist()).unsqueeze(-1)  # [B,ne1,1]
        e2_masks = torch.FloatTensor(e2_masks.tolist()).unsqueeze(-1)  # [B,ne2,1]

        e1_values = torch.LongTensor(e1_values)  # [B,ne1]
        e2_values = torch.LongTensor(e2_values)  # [B,ne2]
        e1_values_emb = val_emb[e1_values]  # [ne1,embedding_dim]
        e2_values_emb = val_emb[e2_values]  # [ne2,embedding_dim]
        sim_matrix = torch.mm(e1_values_emb, torch.transpose(e2_values_emb, 0, 1))  # [ne1,ne2]
        sim_maxpooing_1, _ = sim_matrix.topk(k=1, dim=-1)  # [ne1,1] #get max value.
        sim=sim_maxpooing_1*e1_masks
        for k in range(args.val_max):
            if e1_masks[k]==0:
                break
            if sim[k]>0.9999:
                e2val_edges.add((i,int(e1_values[k]),2))
            elif sim[k]>0:
                e2val_edges.add((i,int(e1_values[k]),1))
        sim_maxpooing_2, _ = torch.transpose(sim_matrix, 0, 1).topk(k=1, dim=-1)
        sim = sim_maxpooing_2 * e2_masks
        for k in range(args.val_max):
            if e2_masks[k]==0:
                break
            if sim[k]>0.9999:
                e2val_edges.add((j,int(e2_values[k]),2))
            elif sim[k]>0:
                e2val_edges.add((j,int(e2_values[k]),1))

    return e2val_edges
def get_bert_int_all_data(args,data,ent_emb,val_emb):
    train_candidates, valid_candidates, test_candidates=get_candidates(args,data,ent_emb)
    entity_pairs = all_entity_pairs_gene([ train_candidates,valid_candidates,test_candidates ],[ data['train_ill'] ])
    #pickle.dump(entity_pairs, open('../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/bert_int_entity_pairs', "wb"))
    att_datas=pickle.load(open('../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/att_datas', "rb"))
    ent2valueids=ent2attributeValues_gene(att_datas, args.val_max, entity_pairs,len(val_emb))
    neigh_dict = neigh_ent_dict_gene(entity_pairs, data['e2e_adj']._indices().T, args.neigh_max, len(ent_emb))
    return entity_pairs,ent2valueids,neigh_dict,train_candidates,valid_candidates,test_candidates
def get_candidates(args,data,ent_emb):
    test_ids_1 = [e1 for e1, e2 in data['test_ill']]
    test_ids_2 = [e2 for e1, e2 in data['test_ill']]
    valid_ids_1 = [e1 for e1, e2 in data['valid_ill']]
    valid_ids_2 = [e2 for e1, e2 in data['valid_ill']]
    train_ids_1 = [e1 for e1, e2 in data['train_ill']]
    train_ids_2 = [e2 for e1, e2 in data['train_ill']]
    train_candidates= candidate_generate(train_ids_1, train_ids_2, ent_emb,
                                          args.candidate_num)
    valid_candidates = candidate_generate(valid_ids_1, valid_ids_2, ent_emb,
                                              args.candidate_num)
    test_candidates = candidate_generate(test_ids_1, test_ids_2, ent_emb,
                                             args.candidate_num)
    return train_candidates,valid_candidates,test_candidates

class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args, data):
        super(BaseModel, self).__init__()

        self.manifold_name = args.manifold
        self.modeltype = args.model
        self.manifold = getattr(manifolds, self.manifold_name)()


        if args.vec_type=='n':
            size, dim = data["ent_features_n"].shape
            self.ins_embeddings_n = nn.Embedding(size, dim, max_norm=1)
            self.ins_embeddings_n.weight.data.copy_(data["ent_features_n"])
            self.esize =size
            size, dim = data["rel_features_n"].shape
            self.rel_embeddings_n = nn.Embedding(size, dim, max_norm=1)
            self.rel_embeddings_n.weight.data.copy_(data["rel_features_n"])
            rdim = self.rel_embeddings_n.weight.shape[1]
        elif args.ent_vec_path_a=='random':
            size,dim=data["ent_features_a"]
            self.esize = size
            self.ins_embeddings_a = nn.Embedding(size,dim,max_norm=1)
            nn.init.kaiming_normal_(self.ins_embeddings_a.weight, mode='fan_out', nonlinearity='relu')
            size, dim = data["rel_features_a"]
            self.rel_embeddings_a = nn.Embedding(size,dim,max_norm=1)
            nn.init.kaiming_normal_(self.rel_embeddings_a.weight, mode='fan_out', nonlinearity='relu')
        else:
            size, dim = data["ent_features_a"].shape
            self.ins_embeddings_a = nn.Embedding(size, dim, max_norm=1)
            self.esize = size
            size, dim = data["rel_features_a"].shape
            self.rel_embeddings_a = nn.Embedding(size, dim, max_norm=1)

            self.ins_embeddings_a.weight.data.copy_(data["ent_features_a"])
            self.rel_embeddings_a.weight.data.copy_(data["rel_features_a"])
            rdim = self.rel_embeddings_a.weight.shape[1]
        if 'hybrid' in args.vec_type:
            size, dim = data["ent_features_n"].shape
            self.esize = size
            self.ins_embeddings_n = nn.Embedding(size, dim, max_norm=1)
            self.ins_embeddings_n.weight.data.copy_(data["ent_features_n"])
            size, dim = data["rel_features_n"].shape
            self.rel_embeddings_n = nn.Embedding(size, dim, max_norm=1)
            self.rel_embeddings_n.weight.data.copy_(data["rel_features_n"])
            if args.vec_type=='hybrid_cat':
                args.feat_dim=2*dim
                args.dim=2*dim
                rdim*=2
            elif args.vec_type=='hybrid_att':
                self.a = nn.Parameter(torch.zeros(size=(2*args.feat_dim,2)))
                nn.init.xavier_normal_(self.a.data, gain=1.414)
                self.a_r = nn.Parameter(torch.zeros(size=(2*rdim,2)))
                nn.init.xavier_normal_(self.a_r.data, gain=1.414)

        if self.modeltype == 'MRAEA_literal':
            self.encoder = getattr(encoders, args.model)(rdim,data['e2eself_adj'],data['e2r_adj'],data['r2t_adj'],data['h2t_adj'],data['t2t_adj'],self.manifold,args)
            self.decoder = getattr(decoders, args.model)(args)
        elif self.modeltype == 'DUAL_literal':
            args.feat_dim = args.dim
            self.encoder = getattr(encoders, args.model)(rdim, data['r2t_adj'],data['h2t_adj'],data['t2t_adj'],data['e2eself_adj'],data['e2r_adj'],self.manifold,args)
            self.decoder = getattr(decoders, args.model)(args)
        elif  self.modeltype == 'RREA_adapt':
            args.feat_dim = args.dim
            self.encoder = getattr(encoders, args.model)(rdim, data['r2t_adj'],data['h2t_adj'],data['t2t_adj'],data['e2eself_adj'],data['e2r_adj'],self.manifold,args)
            self.decoder = getattr(decoders, args.model)(args)
        elif self.modeltype == 'BERT_INT_A':
            size, dim = data["val_features"].shape
            self.val_embeddings = nn.Embedding(size, dim, max_norm=1)
            self.val_embeddings.weight.data.copy_(data["val_features"])
            entity_pairs, ent2valueids, cosin_features,self.train_candidates, self.valid_candidates, self.test_candidates = get_bert_int_data(
                args, data, self.ins_embeddings_a.weight.detach(),self.val_embeddings.weight.detach())
            self.encoder = getattr(encoders, args.model)(self.manifold,entity_pairs,ent2valueids,cosin_features,len(self.val_embeddings.weight.detach()),args)
            self.decoder = getattr(decoders, args.model)(43,11, args.gamma_margin, args.device)
            self.entpair2f_idx = {entpair: feature_idx for feature_idx, entpair in enumerate(entity_pairs)}
        elif self.modeltype == 'GRU':
            self.train_candidates, self.valid_candidates, self.test_candidates = get_candidates(args, data, self.ins_embeddings_a.weight.detach())
            self.encoder = getattr(encoders, args.model)(self.manifold,self.esize,args)
            self.decoder = getattr(decoders, args.model)(args.gamma_margin, args.device)
        elif self.modeltype == 'BERT_INT_ALL':
            size, dim = data["val_features"].shape
            self.val_embeddings = nn.Embedding(size, dim, max_norm=1)
            self.val_embeddings.weight.data.copy_(data["val_features"])
            entity_pairs, ent2valueids, neigh_dict, self.train_candidates, self.valid_candidates, self.test_candidates = get_bert_int_all_data(
                args, data, self.ins_embeddings_n.weight.detach(), self.val_embeddings.weight.detach())
            self.encoder = getattr(encoders, args.model)(self.manifold, entity_pairs, ent2valueids, neigh_dict,
                                                         len(self.val_embeddings.weight.detach()),len(self.ins_embeddings_n.weight.detach()), args)
            self.decoder = getattr(decoders, args.model)(43+42, 11, args.gamma_margin, args.device)
            self.entpair2f_idx = {entpair: feature_idx for feature_idx, entpair in enumerate(entity_pairs)}
        elif self.modeltype == 'BERT_INT_test':
            size, dim = data["val_features"].shape
            self.val_embeddings = nn.Embedding(size, dim, max_norm=1)
            self.val_embeddings.weight.data.copy_(data["val_features"])
            entity_pairs, ent2valueids, neigh_dict, self.train_candidates, self.valid_candidates, self.test_candidates = get_bert_int_all_data(
                args, data, self.ins_embeddings_a.weight.detach(), self.val_embeddings.weight.detach())
            self.encoder = getattr(encoders, 'BERT_INT_ALL')(self.manifold, entity_pairs, ent2valueids, neigh_dict,
                                                         len(self.val_embeddings.weight.detach()),len(self.ins_embeddings_a.weight.detach()), args)
            self.decoder = getattr(decoders, 'BERT_INT_ALL')(43+42, 11, args.gamma_margin, args.device)
            self.entpair2f_idx = {entpair: feature_idx for feature_idx, entpair in enumerate(entity_pairs)}
        elif self.modeltype=='roadEA':
            size, dim = data["val_features"].shape
            self.val_embeddings = nn.Embedding(size, dim, max_norm=1)
            self.val_embeddings.weight.data.copy_(data["val_features"])
            size, dim = data["att_features"].shape
            self.att_embeddings = nn.Embedding(size, dim, max_norm=1)
            self.att_embeddings.weight.data.copy_(data["att_features"])
            self.gen_emb= getattr(encoders, args.model)(data['sorted_value_list'], data['ent_init_list'],data['value_attr_concate'],data['attr_value'],data['att_adj'],args.batch_size,self.manifold,args)
            self.encoder = getattr(encoders, args.relation_model)(rdim, data['r2t_adj'],data['h2t_adj'],data['t2t_adj'],data['e2eself_adj'],data['e2r_adj'],self.manifold,args)
            self.decoder = getattr(decoders, args.model)(args)
        elif  self.modeltype == 'TestM':
            args.feat_dim = args.dim
            size, dim = data["val_features"].shape
            self.val_embeddings = nn.Embedding(size, dim, max_norm=1)
            self.val_embeddings.weight.data.copy_(data["val_features"])
            vdim=dim
            size, dim = data["att_features"].shape
            self.att_embeddings = nn.Embedding(size, dim, max_norm=1)
            self.att_embeddings.weight.data.copy_(data["att_features"])
            if args.vec_type == 'n':
                ins_embeddings= self.ins_embeddings_n.weight.detach()
            elif args.vec_type == 'a':
                ins_embeddings = self.ins_embeddings_a.weight.detach()
            elif args.vec_type=='hybrid_cat':
                ins_embeddings=torch.cat((self.ins_embeddings_a.weight.detach(), self.ins_embeddings_n.weight.detach()), dim=1)
            else:
                ins_embeddings =F.normalize((self.ins_embeddings_a.weight.detach()+ self.ins_embeddings_n.weight.detach())/2,p=2,dim=1)
            data['val_adj'],data['e2a_adj'],e2t_adj,a2t_adj,v2t_adj,self.pair_ev,self.vid2type=get_new_att_triple(args,data, ins_embeddings, self.val_embeddings.weight.detach(), self.att_embeddings.weight.detach())

            self.encoder = getattr(encoders, args.model)(rdim, vdim,data['r2t_adj'],data['h2t_adj'],data['t2t_adj'],data['e2eself_adj'],data['e2r_adj'],data['val_adj'],data['e2a_adj'],e2t_adj,a2t_adj,v2t_adj,self.manifold,args)
            self.decoder = getattr(decoders, args.model)(args)
        else:
            pass
        self.device=args.device
        self.vec_type=args.vec_type
        self.ev2a=data['ev2a']


    def encode(self):
        res = {}
        if self.vec_type=='a':
            x = self.ins_embeddings_a.weight
            r= self.rel_embeddings_a.weight
            if self.modeltype=='roadEA':
                v=self.val_embeddings.weight
                a=self.att_embeddings.weight
                x=self.gen_emb.encode((torch.cat((x,torch.zeros(1,x.shape[1]).to(self.device)),dim=0), torch.cat((a,torch.zeros(1,a.shape[1]).to(self.device)),dim=0),torch.cat((v,torch.zeros(1,v.shape[1]).to(self.device)),dim=0)))
        elif self.vec_type=='n':
            x = self.ins_embeddings_n.weight
            r= self.rel_embeddings_n.weight
            if self.modeltype == 'roadEA':
                v = self.val_embeddings.weight
                a = self.att_embeddings.weight
                x = self.gen_emb.encode((torch.cat((x, torch.zeros(1, x.shape[1]).to(self.device)), dim=0),
                                         torch.cat((a, torch.zeros(1, a.shape[1]).to(self.device)), dim=0),
                                         torch.cat((v, torch.zeros(1, v.shape[1]).to(self.device)), dim=0)))
        elif self.vec_type=='hybrid_mean':
            x = self.ins_embeddings_a.weight
            r= F.normalize((self.rel_embeddings_a.weight+self.rel_embeddings_n.weight)/2,p=2,dim=1)
            if self.modeltype == 'roadEA':
                v = self.val_embeddings.weight
                a = self.att_embeddings.weight
                x = self.gen_emb.encode((torch.cat((x, torch.zeros(1, x.shape[1]).to(self.device)), dim=0),
                                         torch.cat((a, torch.zeros(1, a.shape[1]).to(self.device)), dim=0),
                                         torch.cat((v, torch.zeros(1, v.shape[1]).to(self.device)), dim=0)))
                r = self.rel_embeddings_n.weight
            x = F.normalize((x+ self.ins_embeddings_n.weight)/2,p=2,dim=1)
        elif self.vec_type=='hybrid_cat':
            x = self.ins_embeddings_a.weight
            r= torch.cat((self.rel_embeddings_a.weight,self.rel_embeddings_n.weight),dim=1)
            if self.modeltype == 'roadEA':
                v = self.val_embeddings.weight
                a = self.att_embeddings.weight
                x = self.gen_emb.encode((torch.cat((x, torch.zeros(1, x.shape[1]).to(self.device)), dim=0),
                                         torch.cat((a, torch.zeros(1, a.shape[1]).to(self.device)), dim=0),
                                         torch.cat((v, torch.zeros(1, v.shape[1]).to(self.device)), dim=0)))
                r = self.rel_embeddings_n.weight
            x = torch.cat((x, self.ins_embeddings_n.weight), dim=1)
        elif self.vec_type=='hybrid_att':
            x =torch.cat((self.ins_embeddings_a.weight,self.ins_embeddings_n.weight),dim=1)
            r= torch.cat((self.rel_embeddings_a.weight,self.rel_embeddings_n.weight),dim=1)
            weight = torch.nn.functional.softmax(x.mm(self.a), 1)
            x=weight[:, 0].unsqueeze(1) *self.ins_embeddings_a.weight+weight[:, 1].unsqueeze(1)*self.ins_embeddings_n.weight
            weight_r = torch.nn.functional.softmax(r.mm(self.a_r), 1)
            r=weight_r[:, 0].unsqueeze(1) *self.rel_embeddings_a.weight+weight_r[:, 1].unsqueeze(1)*self.rel_embeddings_n.weight
        if self.modeltype in ['roadEA','MRAEA','DUAL','DUAL_literal','RREA','RAGA','RREA_adapt','ERMC','MRAEA_literal','KECG_literal','LightEA_literal','ICLEA_literal']:
            outputs,r_hyp,c = self.encoder.encode((x, r))
        elif self.modeltype=='BERT_INT_A':
            x=self.val_embeddings.weight
            outputs, r_hyp, c = self.encoder.encode((torch.cat((x,torch.zeros(1,x.shape[1]).to(self.device)),dim=0), r))
        elif self.modeltype in ['BERT_INT_ALL','BERT_INT_test'] :
            x1=self.val_embeddings.weight
            outputs, r_hyp, c = self.encoder.encode((torch.cat((x,torch.zeros(1,x.shape[1]).to(self.device)),dim=0),torch.cat((x1,torch.zeros(1,x1.shape[1]).to(self.device)),dim=0)))
        elif self.modeltype=='TestM':
            v=self.val_embeddings.weight
            a=self.att_embeddings.weight
            outputs, r_hyp, c = self.encoder.encode((x,r,v,a))
        else:
            outputs=[x]
            r_hyp=[r]
            c=[1]
        manifolds = [self.manifold] * len(outputs)
        res['M0'] = (outputs,r_hyp,c)
        return res,manifolds

    def encoder1(self,input):
        if self.vec_type == 'a':
            x = self.ins_embeddings_a.weight
            r = self.rel_embeddings_a.weight
            if self.modeltype == 'roadEA':
                v = self.val_embeddings.weight
                a = self.att_embeddings.weight
                x = self.gen_emb.encode((torch.cat((x, torch.zeros(1, x.shape[1]).to(self.device)), dim=0),
                                         torch.cat((a, torch.zeros(1, a.shape[1]).to(self.device)), dim=0),
                                         torch.cat((v, torch.zeros(1, v.shape[1]).to(self.device)), dim=0)))
        elif self.vec_type == 'n':
            x = self.ins_embeddings_n.weight
            r = self.rel_embeddings_n.weight
            if self.modeltype == 'roadEA':
                v = self.val_embeddings.weight
                a = self.att_embeddings.weight
                x = self.gen_emb.encode((torch.cat((x, torch.zeros(1, x.shape[1]).to(self.device)), dim=0),
                                         torch.cat((a, torch.zeros(1, a.shape[1]).to(self.device)), dim=0),
                                         torch.cat((v, torch.zeros(1, v.shape[1]).to(self.device)), dim=0)))
        elif self.vec_type == 'hybrid_mean':
            x = self.ins_embeddings_a.weight
            r = F.normalize((self.rel_embeddings_a.weight + self.rel_embeddings_n.weight) / 2, p=2, dim=1)
            if self.modeltype == 'roadEA':
                v = self.val_embeddings.weight
                a = self.att_embeddings.weight
                x = self.gen_emb.encode((torch.cat((x, torch.zeros(1, x.shape[1]).to(self.device)), dim=0),
                                         torch.cat((a, torch.zeros(1, a.shape[1]).to(self.device)), dim=0),
                                         torch.cat((v, torch.zeros(1, v.shape[1]).to(self.device)), dim=0)))
                r = self.rel_embeddings_n.weight
            x = F.normalize((x + self.ins_embeddings_n.weight) / 2, p=2, dim=1)
        elif self.vec_type == 'hybrid_cat':
            x = self.ins_embeddings_a.weight
            r = torch.cat((self.rel_embeddings_a.weight, self.rel_embeddings_n.weight), dim=1)
            if self.modeltype == 'roadEA':
                v = self.val_embeddings.weight
                a = self.att_embeddings.weight
                x = self.gen_emb.encode((torch.cat((x, torch.zeros(1, x.shape[1]).to(self.device)), dim=0),
                                         torch.cat((a, torch.zeros(1, a.shape[1]).to(self.device)), dim=0),
                                         torch.cat((v, torch.zeros(1, v.shape[1]).to(self.device)), dim=0)))
                r = self.rel_embeddings_n.weight
            x = torch.cat((x, self.ins_embeddings_n.weight), dim=1)
        elif self.vec_type == 'hybrid_att':
            x = torch.cat((self.ins_embeddings_a.weight, self.ins_embeddings_n.weight), dim=1)
            r = torch.cat((self.rel_embeddings_a.weight, self.rel_embeddings_n.weight), dim=1)
            weight = torch.nn.functional.softmax(x.mm(self.a), 1)
            x = weight[:, 0].unsqueeze(1) * self.ins_embeddings_a.weight + weight[:, 1].unsqueeze(
                1) * self.ins_embeddings_n.weight
            weight_r = torch.nn.functional.softmax(r.mm(self.a_r), 1)
            r = weight_r[:, 0].unsqueeze(1) * self.rel_embeddings_a.weight + weight_r[:, 1].unsqueeze(
                1) * self.rel_embeddings_n.weight

        return self.encoder.encode((input[0],input[1],input[2],input[3],torch.cat((x,torch.zeros(1,x.shape[1]).to(self.device)),dim=0)))
    def get_emb(self,input):
        if self.vec_type == 'a':
            x = self.ins_embeddings_a.weight
            r = self.rel_embeddings_a.weight
            if self.modeltype == 'roadEA':
                v = self.val_embeddings.weight
                a = self.att_embeddings.weight
                x = self.gen_emb.encode((torch.cat((x, torch.zeros(1, x.shape[1]).to(self.device)), dim=0),
                                         torch.cat((a, torch.zeros(1, a.shape[1]).to(self.device)), dim=0),
                                         torch.cat((v, torch.zeros(1, v.shape[1]).to(self.device)), dim=0)))
        elif self.vec_type == 'n':
            x = self.ins_embeddings_n.weight
            r = self.rel_embeddings_n.weight
            if self.modeltype == 'roadEA':
                v = self.val_embeddings.weight
                a = self.att_embeddings.weight
                x = self.gen_emb.encode((torch.cat((x, torch.zeros(1, x.shape[1]).to(self.device)), dim=0),
                                         torch.cat((a, torch.zeros(1, a.shape[1]).to(self.device)), dim=0),
                                         torch.cat((v, torch.zeros(1, v.shape[1]).to(self.device)), dim=0)))
        elif self.vec_type == 'hybrid_mean':
            x = self.ins_embeddings_a.weight
            r = F.normalize((self.rel_embeddings_a.weight + self.rel_embeddings_n.weight) / 2, p=2, dim=1)
            if self.modeltype == 'roadEA':
                v = self.val_embeddings.weight
                a = self.att_embeddings.weight
                x = self.gen_emb.encode((torch.cat((x, torch.zeros(1, x.shape[1]).to(self.device)), dim=0),
                                         torch.cat((a, torch.zeros(1, a.shape[1]).to(self.device)), dim=0),
                                         torch.cat((v, torch.zeros(1, v.shape[1]).to(self.device)), dim=0)))
                r = self.rel_embeddings_n.weight
            x = F.normalize((x + self.ins_embeddings_n.weight) / 2, p=2, dim=1)
        elif self.vec_type == 'hybrid_cat':
            x = self.ins_embeddings_a.weight
            r = torch.cat((self.rel_embeddings_a.weight, self.rel_embeddings_n.weight), dim=1)
            if self.modeltype == 'roadEA':
                v = self.val_embeddings.weight
                a = self.att_embeddings.weight
                x = self.gen_emb.encode((torch.cat((x, torch.zeros(1, x.shape[1]).to(self.device)), dim=0),
                                         torch.cat((a, torch.zeros(1, a.shape[1]).to(self.device)), dim=0),
                                         torch.cat((v, torch.zeros(1, v.shape[1]).to(self.device)), dim=0)))
                r = self.rel_embeddings_n.weight
            x = torch.cat((x, self.ins_embeddings_n.weight), dim=1)
        elif self.vec_type == 'hybrid_att':
            x = torch.cat((self.ins_embeddings_a.weight, self.ins_embeddings_n.weight), dim=1)
            r = torch.cat((self.rel_embeddings_a.weight, self.rel_embeddings_n.weight), dim=1)
            weight = torch.nn.functional.softmax(x.mm(self.a), 1)
            x = weight[:, 0].unsqueeze(1) * self.ins_embeddings_a.weight + weight[:, 1].unsqueeze(
                1) * self.ins_embeddings_n.weight
            weight_r = torch.nn.functional.softmax(r.mm(self.a_r), 1)
            r = weight_r[:, 0].unsqueeze(1) * self.rel_embeddings_a.weight + weight_r[:, 1].unsqueeze(
                1) * self.rel_embeddings_n.weight

        return self.encoder.get_emb(input[0],input[1],torch.cat((x,torch.zeros(1,x.shape[1]).to(self.device)),dim=0))
    def update_attribute(self,args,train_candidates, valid_candidates, test_candidates,train_ill):
        entity_pairs = all_entity_pairs_gene([train_candidates, valid_candidates, test_candidates], [train_ill])
        val_emb=self.val_embeddings.weight.detach()
        a_emb=self.att_embeddings.weight.detach()
        att_datas = pickle.load(
            open('../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/att_datas', "rb"))
        ent2valueids = ent2attributeValues_gene(att_datas, args.val_max, entity_pairs, len(val_emb))
        e2val_edges_set = set()
        e2val_edges_set = e2val_edges_set | get_candidates_att(args, train_candidates, ent2valueids, len(val_emb),
                                                       torch.cat((val_emb, torch.zeros(1, val_emb.shape[1])),
                                                                 dim=0), self.pair_ev)
        e2val_edges_set = e2val_edges_set | get_candidates_att(args, valid_candidates, ent2valueids, len(val_emb),
                                                       torch.cat((val_emb, torch.zeros(1, val_emb.shape[1])),
                                                                 dim=0), self.pair_ev)
        e2val_edges_set = e2val_edges_set | get_candidates_att(args, test_candidates, ent2valueids, len(val_emb),
                                                       torch.cat((val_emb, torch.zeros(1, val_emb.shape[1])),
                                                                 dim=0), self.pair_ev)
        e2val_edges = list(e2val_edges_set)
        e2val_edges.sort(key=lambda x: x[0])
        e2val_edges1 = []
        for tr in e2val_edges:
            if tr[2] == 1:
                if (tr[0], tr[1], 2) not in e2val_edges_set:
                    e2val_edges1.append((tr[0], tr[1]))
                else:
                    pass
            else:
                e2val_edges1.append((tr[0], tr[1]))
                e2val_edges1.append((tr[0], tr[1]))
                e2val_edges1.append((tr[0], tr[1]))
                e2val_edges1.append((tr[0], tr[1]))
                e2val_edges1.append((tr[0], tr[1]))
                e2val_edges1.append((tr[0], tr[1]))
                e2val_edges1.append((tr[0], tr[1]))
                e2val_edges1.append((tr[0], tr[1]))
        e2val_values = [1] * len(e2val_edges1)
        ind = np.array(e2val_edges1, dtype=np.int32)
        val = np.array(e2val_values, dtype=np.float32)
        e2v_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(self.esize, len(val_emb)), dtype=np.float32)
        e2v_adj = sparse_mx_to_torch_sparse_tensor(e2v_adj)

        edge = e2v_adj._indices()
        ev2a = self.ev2a
        e2a_edges = []
        e2t_edges = []
        a2t_edges = []
        v2t_edges = []
        triple_n = 0
        for i in range(edge.shape[1]):
            e = int(edge[0][i])
            v = int(edge[1][i])
            a = ev2a[(e, v)]
            e2a_edges.append((e, a))
            e2t_edges.append((e, triple_n))
            a2t_edges.append((a, triple_n))
            v2t_edges.append((v, triple_n))
            triple_n += 1
        e2a_values = [1] * len(e2a_edges)
        ind = np.array(e2a_edges, dtype=np.int32)
        val = np.array(e2a_values, dtype=np.float32)
        e2a_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(self.esize, len(a_emb)), dtype=np.float32)
        e2a_adj = sparse_mx_to_torch_sparse_tensor(e2a_adj)

        e2t_values = [1] * len(e2t_edges)
        ind = np.array(e2t_edges, dtype=np.int32)
        val = np.array(e2t_values, dtype=np.float32)
        e2t_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(self.esize, triple_n), dtype=np.float32)
        e2t_adj = sparse_mx_to_torch_sparse_tensor(e2t_adj)

        a2t_values = [1] * len(a2t_edges)
        ind = np.array(a2t_edges, dtype=np.int32)
        val = np.array(a2t_values, dtype=np.float32)
        a2t_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(len(a_emb), triple_n), dtype=np.float32)
        a2t_adj = sparse_mx_to_torch_sparse_tensor(a2t_adj)

        v2t_values = [1] * len(v2t_edges)
        ind = np.array(v2t_edges, dtype=np.int32)
        val = np.array(v2t_values, dtype=np.float32)
        v2t_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(len(val_emb), triple_n), dtype=np.float32)
        v2t_adj = sparse_mx_to_torch_sparse_tensor(v2t_adj)

        self.encoder.update_attribute(e2t_adj, a2t_adj, v2t_adj)





    def loss_func(self, res, manifolds,metrics_input):
        loss=self.decoder.decode(res, manifolds,metrics_input)
        return loss



    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError