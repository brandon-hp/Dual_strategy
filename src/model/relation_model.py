import os
import pickle

import torch
import torch.nn as nn
import numpy as np
import model.encoders as encoders
import model.decoders as decoders
import scipy.sparse as sp
from tqdm import tqdm
import torch.nn.functional as F
import sys
sys.path.append('..')
import manifolds
from utils.data_utils import candidate_generate,all_entity_pairs_gene,neigh_ent_dict_gene
def get_bert_int_data(args,data,ent_emb):
    train_candidates, valid_candidates, test_candidates=get_candidates(args,data,ent_emb)
    entity_pairs = all_entity_pairs_gene([ train_candidates,valid_candidates,test_candidates ],[ data['train_ill'] ])
    pickle.dump(entity_pairs, open('../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/bert_int_entity_pairs', "wb"))
    neigh_dict=neigh_ent_dict_gene(entity_pairs, data['e2e_adj']._indices().T, args.neigh_max, len(ent_emb))
    return entity_pairs,neigh_dict,train_candidates,valid_candidates,test_candidates
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
def get_candidates_att(args,train_candidates,neigh_dict,ent_pad_id,ent_emb,e2val_edges):
    zeros=torch.zeros((args.neigh_max,1))
    for i in tqdm(train_candidates):
        clist=train_candidates[i].tolist()
        for j in clist:
            e1_values = neigh_dict[i]  # size: [B(Batchsize), ne1(e1_attributeValue_max_num)]
            e2_values = neigh_dict[j] # [B,ne2
            e1_masks = np.ones(np.array(e1_values).shape)
            e2_masks = np.ones(np.array(e2_values).shape)
            e1_masks[np.array(e1_values) == ent_pad_id] = 0
            e2_masks[np.array(e2_values) == ent_pad_id] = 0
            e1_masks = torch.FloatTensor(e1_masks.tolist()).unsqueeze(-1)  # [B,ne1,1]
            e2_masks = torch.FloatTensor(e2_masks.tolist()).unsqueeze(-1)  # [B,ne2,1]

            e1_values = torch.LongTensor(e1_values)  # [B,ne1]
            e2_values = torch.LongTensor(e2_values)  # [B,ne2]
            e1_values_emb = ent_emb[e1_values]  # [ne1,embedding_dim]
            e2_values_emb = ent_emb[e2_values]  # [ne2,embedding_dim]
            sim_matrix = torch.mm(e1_values_emb, torch.transpose(e2_values_emb, 0, 1))  # [ne1,ne2]
            sim_maxpooing_1, _ = sim_matrix.topk(k=1, dim=-1)  # [ne1,1] #get max value.
            sim=sim_maxpooing_1*e1_masks
            sim1=e2val_edges.get(i,zeros)
            e2val_edges[i]=torch.max(sim,sim1)
            sim_maxpooing_2, _ = torch.transpose(sim_matrix, 0, 1).topk(k=1, dim=-1)
            sim = sim_maxpooing_2 * e2_masks
            sim1=e2val_edges.get(j,zeros)
            e2val_edges[j]=torch.max(sim,sim1)
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def get_new_rel_triple(args,data,ent_emb):
    train_candidates, valid_candidates, test_candidates=get_candidates(args,data,ent_emb)
    entity_pairs = all_entity_pairs_gene([ train_candidates,valid_candidates,test_candidates ],[ data['train_ill'] ])
    pickle.dump(entity_pairs, open('../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/bert_int_entity_pairs', "wb"))
    neigh_dict=neigh_ent_dict_gene(entity_pairs, data['e2e_adj']._indices().T, args.neigh_max, len(ent_emb))

    e2val_edges_set={}
    get_candidates_att(args,train_candidates,neigh_dict,len(ent_emb),torch.cat((ent_emb, torch.zeros(1, ent_emb.shape[1])), dim=0),e2val_edges_set)
    get_candidates_att(args,valid_candidates,neigh_dict,len(ent_emb),torch.cat((ent_emb, torch.zeros(1, ent_emb.shape[1])), dim=0),e2val_edges_set)
    get_candidates_att(args,test_candidates,neigh_dict,len(ent_emb),torch.cat((ent_emb, torch.zeros(1, ent_emb.shape[1])), dim=0),e2val_edges_set)
    e2e_edges=[]
    e2e_edges1 = []
    n=0
    r=[]
    for tr in tqdm(e2val_edges_set):
        e1_values = neigh_dict[tr]
        sim=e2val_edges_set[tr]
        e1_masks = np.ones(np.array(e1_values).shape)
        e1_masks[np.array(e1_values) == len(ent_emb)] = 0
        for i in range(args.neigh_max):
            if e1_masks[i]==0:
                break
            elif sim[i]>0:
                e2e_edges.append((tr,n))
                e2e_edges1.append((e1_values[i],n))
                r.append(sim[i])
                n+=1
    print(n)
    e2e_values=[1]*len(e2e_edges)
    ind = np.array(e2e_edges, dtype=np.int32)
    val = np.array(e2e_values, dtype=np.float32)
    e2e_adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(len(ent_emb), n), dtype=np.float32)
    e2e_adj = sparse_mx_to_torch_sparse_tensor(e2e_adj)

    ind = np.array(e2e_edges1, dtype=np.int32)
    val = np.array(e2e_values, dtype=np.float32)
    e2e_adj1 = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(len(ent_emb), n), dtype=np.float32)
    e2e_adj1 = sparse_mx_to_torch_sparse_tensor(e2e_adj1)
    return e2e_adj,e2e_adj1,torch.tensor(r)
class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args, data):
        super(BaseModel, self).__init__()

        self.manifold_name = args.manifold
        self.modeltype = args.model
        self.manifold = getattr(manifolds, self.manifold_name)()
        if isinstance(data["ent_features"],tuple):
            size,dim=data["ent_features"]
            self.ins_embeddings = nn.Embedding(size,dim,max_norm=1)
            nn.init.kaiming_normal_(self.ins_embeddings.weight, mode='fan_out', nonlinearity='relu')
        else:
            size, dim =data["ent_features"].shape
            self.ins_embeddings = nn.Embedding(size, dim,max_norm=1)
            self.ins_embeddings.weight.data.copy_(data["ent_features"])

        if isinstance(data["rel_features"],tuple):
            size, dim=data["rel_features"]
            self.rel_embeddings = nn.Embedding(size, dim,max_norm=1)
            nn.init.kaiming_normal_(self.rel_embeddings.weight, mode='fan_out', nonlinearity='relu')
        else:
            size, dim=data["rel_features"].shape
            self.rel_embeddings = nn.Embedding(size, dim,max_norm=1)
            self.rel_embeddings.weight.data.copy_(data["rel_features"])

        if self.modeltype == 'MRAEA':
            args.feat_dim = args.dim
            rdim=self.rel_embeddings.weight.shape[1]
            self.encoder = getattr(encoders, args.model)(rdim,data['e2eself_adj'],data['e2r_adj'],data['r2t_adj'],data['h2t_adj'],data['t2t_adj'],self.manifold,args)
            self.decoder = getattr(decoders, args.model)(args)
        elif self.modeltype == 'MRAEA_adapt':
            rdim=self.rel_embeddings.weight.shape[1]
            self.encoder = getattr(encoders, args.model)(rdim,data['e2eself_adj'],data['e2r_adj'],data['r2t_adj'],data['h2t_adj'],data['t2t_adj'],self.manifold,args)
            self.decoder = getattr(decoders, args.model)(args)
        elif self.modeltype == 'GAT':
            rdim=self.rel_embeddings.weight.shape[1]
            self.encoder = getattr(encoders, args.model)(rdim,data['e2eself_adj'],data['e2r_adj'],data['r2t_adj'],data['h2t_adj'],data['t2t_adj'],self.manifold,args)
            self.decoder = getattr(decoders, args.model)(args)
        elif self.modeltype == 'DUAL':
            args.feat_dim = args.dim
            rdim=self.rel_embeddings.weight.shape[1]
            self.encoder = getattr(encoders, args.model)(rdim, data['r2t_adj'],data['h2t_adj'],data['t2t_adj'],data['e2eself_adj'],data['e2r_adj'],self.manifold,args)
            self.decoder = getattr(decoders, args.model)(args)
        elif self.modeltype == 'DUAL_adapt':
            args.feat_dim = args.dim
            rdim=self.rel_embeddings.weight.shape[1]
            self.encoder = getattr(encoders, args.model)(rdim, data['r2t_adj'],data['h2t_adj'],data['t2t_adj'],data['e2eself_adj'],data['e2r_adj'],self.manifold,args)
            self.decoder = getattr(decoders, args.model)(args)
        elif self.modeltype == 'RREA' or self.modeltype == 'RREA_adapt' or self.modeltype == 'RREA_adapt1':
            args.feat_dim = args.dim
            rdim=self.rel_embeddings.weight.shape[1]
            self.encoder = getattr(encoders, args.model)(rdim, data['r2t_adj'],data['h2t_adj'],data['t2t_adj'],data['e2eself_adj'],data['e2r_adj'],self.manifold,args)
            self.decoder = getattr(decoders, args.model)(args)
        elif self.modeltype == 'RREAN_literal':
            args.feat_dim = args.dim
            rdim=self.rel_embeddings.weight.shape[1]
            self.encoder = getattr(encoders, args.model)(rdim, data['r2t_adj'],data['h2t_adj'],data['t2t_adj'],data['r2t_adj_o'],data['h2t_adj_o'],data['t2t_adj_o'],data['e2r_adj'],data['e2eself_adj'],self.manifold,args)
            self.decoder = getattr(decoders, args.model)(args)
        elif self.modeltype == 'ICLEA_literal' or self.modeltype == 'ICLEA':
            args.feat_dim = args.dim
            rdim = self.rel_embeddings.weight.shape[1]
            self.encoder = getattr(encoders, args.model)(rdim, data['r2t_adj_self'], data['h2t_adj_self'], data['t2t_adj_self'],data['r2t_adj'], data['h2t_adj'], data['t2t_adj'],
                                                         data['e2eself_adj'], data['e2r_adj'], self.manifold, args)
            self.decoder = getattr(decoders, args.model)(args)
        elif self.modeltype == 'LightEA_literal':
            args.feat_dim = args.dim
            rdim=self.rel_embeddings.weight.shape[1]
            self.encoder = getattr(encoders, args.model)(rdim, data['e2e_adj'],data['e2eself_adj'],data['e2r_adj'],data['r2e_adj'],self.manifold,args)
            self.decoder = getattr(decoders, args.model)(args)
        elif self.modeltype == 'RAGA':
            args.feat_dim = args.dim
            rdim=self.rel_embeddings.weight.shape[1]
            self.encoder = getattr(encoders, args.model)(rdim, data['e2e_adj'],data['e2eself_adj'],data['r2t_adj'],data['h2t_adj'],data['t2t_adj'],data['r2t_adj_o'],data['h2t_adj_o'],data['t2t_adj_o'],self.manifold,args)
            self.decoder = getattr(decoders, args.model)(args)
        elif self.modeltype == 'RAGA_literal':
            args.feat_dim = args.dim
            rdim=self.rel_embeddings.weight.shape[1]
            self.encoder = getattr(encoders, args.model)(rdim, data['e2e_adj'],data['e2eself_adj'],data['r2t_adj'],data['h2t_adj'],data['t2t_adj'],data['r2t_adj_o'],data['h2t_adj_o'],data['t2t_adj_o'],self.manifold,args)
            self.decoder = getattr(decoders, args.model)(args)
        elif self.modeltype == 'Test':
            args.feat_dim = args.dim
            rdim=self.rel_embeddings.weight.shape[1]
            #h2t_adj,t2t_adj,r=get_new_rel_triple(args, data, self.ins_embeddings.weight.detach())
            self.encoder = getattr(encoders, args.model)(rdim, data['e2r_adj'],data['e2e_adj'],data['e2eself_adj'],data['r2t_adj'],data['h2t_adj'],data['t2t_adj'],data['r2t_adj_o'],data['h2t_adj_o'],data['t2t_adj_o'],self.manifold,args)
            self.decoder = getattr(decoders,'RAGA_literal')(args)
        elif self.modeltype == 'ERMC':
            args.feat_dim = args.dim
            rdim=self.rel_embeddings.weight.shape[1]
            self.encoder = getattr(encoders, args.model)(rdim, data['e2ei_adj'],data['e2eo_adj'],data['e2ri_adj'],data['e2ro_adj'],data['r2ei_adj'],data['r2eo_adj'],self.manifold,args)
            self.decoder = getattr(decoders, args.model)(args)
        elif self.modeltype == 'ERMC_adapt':
            args.feat_dim = args.dim
            rdim=self.rel_embeddings.weight.shape[1]
            self.encoder = getattr(encoders, args.model)(rdim, data['e2ei_adj'],data['e2eo_adj'],data['e2ri_adj'],data['e2ro_adj'],data['r2ei_adj'],data['r2eo_adj'],data['r2t_adj_o'],data['h2t_adj_o'],data['t2t_adj_o'],self.manifold,args)
            self.decoder = getattr(decoders, args.model)(args)
        elif self.modeltype == 'BERT_INT_N':
            entity_pairs, neigh_dict, self.train_candidates, self.valid_candidates, self.test_candidates = get_bert_int_data(
                args, data, self.ins_embeddings.weight.detach())
            self.encoder = getattr(encoders, args.model)(self.manifold,entity_pairs,neigh_dict,len(self.ins_embeddings.weight.detach()),args)
            self.decoder = getattr(decoders, args.model)(43,11, args.gamma_margin, args.device)
            self.entpair2f_idx = {entpair: feature_idx for feature_idx, entpair in enumerate(entity_pairs)}
        elif self.modeltype == 'TransEdge':
            self.encoder = getattr(encoders, args.model)(self.manifold, args)
            output_dim = self.rel_embeddings.weight.shape[1]
            self.decoder = getattr(decoders, args.model)(args,output_dim,output_dim)
        elif self.modeltype == 'BootEA':
            self.encoder = getattr(encoders, args.model)(self.manifold, args)
            self.decoder = getattr(decoders, args.model)(args)
        elif self.modeltype == 'SSP':
            self.encoder = getattr(encoders, args.model)(data['e2eself_adj'],self.manifold, args)
            output_dim = self.rel_embeddings.weight.shape[1]
            self.decoder = getattr(decoders, args.model)(args,output_dim,output_dim)
        elif self.modeltype == 'SSP_adapt':
            self.encoder = getattr(encoders, args.model)(data['r2t_adj_self'], data['h2t_adj_self'], data['t2t_adj_self'],self.manifold, args)
            output_dim = self.rel_embeddings.weight.shape[1]
            self.decoder = getattr(decoders, args.model)(args,output_dim,output_dim)
        elif self.modeltype == 'KECG':
            self.encoder = getattr(encoders, args.model)(data['r2t_adj_self'], data['h2t_adj_self'], data['t2t_adj_self'],self.manifold, args)
            self.decoder = getattr(decoders, args.model)(args)
        elif self.modeltype == 'KECG_literal':
            self.encoder = getattr(encoders, args.model)(data['e2eself_adj'], data['h2t_adj'], data['t2t_adj'],
                                                         self.manifold, args)
            self.decoder = getattr(decoders, args.model)(args.k, args.gamma_margin, args.transe_k, args.pos_margin,
                                                         args.neg_margin, args.neg_param, args.bp_param, args.device)

        self.device=args.device


    def encode(self):
        res = {}
        x = self.ins_embeddings.weight
        r= self.rel_embeddings.weight
        if self.modeltype in ['GAT','MRAEA','DUAL','Test','DUAL_adapt','RREA','RAGA','RAGA_literal','RREAN_literal','RREA_adapt','RREA_adapt1','ERMC','ERMC_adapt','MRAEA_adapt','KECG_literal','LightEA_literal','ICLEA_literal','ICLEA']:
            outputs,r_hyp,c = self.encoder.encode((x, r))
        elif self.modeltype=='BERT_INT_N':
            outputs, r_hyp, c = self.encoder.encode((torch.cat((x,torch.zeros(1,x.shape[1]).to(self.device)),dim=0), r))
        elif self.modeltype in ['TransEdge','TransEdge_literal']:
            outputs, outputs1,r_hyp, c = self.encoder.encode((x, r))
            res['M1'] =(outputs1,r_hyp, c)
        elif self.modeltype == 'SSP' or 'SSP_adapt':
            outputs, r_hyp, c = self.encoder.encode((x, r))
            res['M1'] =(outputs, r_hyp, c)
        elif self.modeltype == 'KECG':
            outputs, r_hyp, c = self.encoder.encode((x, r))
        elif self.modeltype == 'BootEA':
            outputs, r_hyp, c = self.encoder.encode((x, r))
        manifolds = [self.manifold] * len(outputs)
        res['M0'] = (outputs,r_hyp,c)
        return res,manifolds
    def loss_func(self, res, manifolds,metrics_input):
        loss=self.decoder.decode(res, manifolds,metrics_input)
        return loss



    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError