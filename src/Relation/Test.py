import torch
import numpy as np
import sys
sys.path.append('..')
from utils.data_utils import list_index_select,all_entity_pairs_gene
def get_entity_pairs(res,manifolds, train_pairs,valid_pairs,test_pairs,candidate_num,device):
    output_layer, _, c = res['M0']
    cosin_sim = {}
    c = torch.FloatTensor(c).to(device)
    ###########add#############
    ents1=[e1 for e1, e2 in train_pairs]
    ents2=[e2 for e1, e2 in train_pairs]
    L = np.array(ents1)
    R = np.array(ents2)
    e2id = {}
    Lvec = list_index_select(output_layer, torch.tensor(L).to(device))
    Rvec = list_index_select(output_layer, torch.tensor(R).to(device))
    sim = 0
    for i in range(len(manifolds)):
        sim += manifolds[i].sqdist(Lvec[i], Rvec[i], c[i], True)
    traincandidates = dict()
    for i in range(len(train_pairs)):
        e1 = ents1[i]
        rank = sim[i, :].argsort().cpu()
        e2_list = np.array(ents2)[rank[:candidate_num]]
        traincandidates[e1] = e2_list
    ents1=[e1 for e1, e2 in valid_pairs]
    ents2=[e2 for e1, e2 in valid_pairs]
    L = np.array(ents1)
    R = np.array(ents2)
    e2id = {}
    Lvec = list_index_select(output_layer, torch.tensor(L).to(device))
    Rvec = list_index_select(output_layer, torch.tensor(R).to(device))
    sim = 0
    for i in range(len(manifolds)):
        sim += manifolds[i].sqdist(Lvec[i], Rvec[i], c[i], True)
    validcandidates = dict()
    for i in range(len(valid_pairs)):
        e1 = ents1[i]
        rank = sim[i, :].argsort().cpu()
        e2_list = np.array(ents2)[rank[:candidate_num]]
        validcandidates[e1] = e2_list
    ents1=[e1 for e1, e2 in test_pairs]
    ents2=[e2 for e1, e2 in test_pairs]
    L = np.array(ents1)
    R = np.array(ents2)
    e2id = {}
    Lvec = list_index_select(output_layer, torch.tensor(L).to(device))
    Rvec = list_index_select(output_layer, torch.tensor(R).to(device))
    sim = 0
    for i in range(len(manifolds)):
        sim += manifolds[i].sqdist(Lvec[i], Rvec[i], c[i], True)
    testcandidates = dict()
    for i in range(len(test_pairs)):
        e1 = ents1[i]
        rank = sim[i, :].argsort().cpu()
        e2_list = np.array(ents2)[rank[:candidate_num]]
        testcandidates[e1] = e2_list
    entity_pairs = all_entity_pairs_gene([traincandidates,validcandidates,testcandidates],[ train_pairs ])
    return entity_pairs
def get_hits(res,manifolds, test_pair, stop_mrr,device,top_k=[1]):
    # self.ws =torch.nn.functional.softmax(self.w.detach(),0)
    ########### 双曲空间 #############
    output_layer, _,c = res['M0']

    c = torch.FloatTensor(c).to(device)

    ###########add#############
    L = np.array([e1 for e1, e2 in test_pair])
    R = np.array([e2 for e1, e2 in test_pair])
    Lvec = list_index_select(output_layer,torch.tensor(L).to(device))
    Rvec = list_index_select(output_layer,torch.tensor(R).to(device))
    sim=0
    for i in range(len(manifolds)):
        sim += manifolds[i].sqdist(Lvec[i], Rvec[i], c[i], True)
    sim = sim.cpu().numpy()
    # 计算MRR数值情况
    mrr_l = []  # KG1
    # top_k=10
    top_lr = [0] * len(top_k)
    # shape[0]输出矩阵的行数，这里为10500
    for i in range(L.shape[0]):  #
        # for i in range(Lvec.size(0)):
        rank = sim[i, :].argsort()
        # 找出是在第几个位置匹配上的
        rank_index = np.where(rank == i)[0][0]
        mrr_l.append(1.0 / (rank_index + 1))
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100),end='')
    print('MRR: %.4f' % (np.mean(mrr_l)))

    if stop_mrr:
        return np.mean(mrr_l)
    else:
        return top_lr[0] /  len(test_pair) * 100
def save_sim(res,manifolds, pairs,entity_pairs,device):
    # self.ws =torch.nn.functional.softmax(self.w.detach(),0)
    ########### 双曲空间 #############
    output_layer, _,c = res['M0']
    cosin_sim={}
    c = torch.FloatTensor(c).to(device)
    ###########add#############
    L = np.array([e1 for e1, e2 in pairs])
    R = np.array([e2 for e1, e2 in pairs])
    e2id={}
    for i,pair in enumerate(pairs):
        e1,e2=pair
        e2id[e1]=i
        e2id[e2]=i
    Lvec = list_index_select(output_layer,torch.tensor(L).to(device))
    Rvec = list_index_select(output_layer,torch.tensor(R).to(device))
    sim=0
    for i in range(len(manifolds)):
        sim += manifolds[i].sqdist(Lvec[i], Rvec[i], c[i], True)
    sim = sim.cpu().numpy()
    for e1,e2 in entity_pairs:
        cosin_sim[(e1,e2)]=sim[e2id[e1]][e2id[e2]]

    return cosin_sim



def get_hits_init(output_layer, test_pair, stop_mrr,top_k=[1]):
    output_layer1=[output_layer]
    ###########add#############
    L = np.array([e1 for e1, e2 in test_pair])
    R = np.array([e2 for e1, e2 in test_pair])
    Lvec = list_index_select(output_layer1,torch.tensor(L))
    Rvec = list_index_select(output_layer1,torch.tensor(R))
    sim=0
    for i in range(len(output_layer1)):
        sim +=torch.cdist(Lvec[i], Rvec[i], 2).pow(2) #manifolds[i].sqdist(Lvec[i], Rvec[i], c[i], True)
    sim = sim.numpy()
    # 计算MRR数值情况
    mrr_l = []  # KG1
    # top_k=10
    top_lr = [0] * len(top_k)
    # shape[0]输出矩阵的行数，这里为10500
    for i in range(L.shape[0]):  #
        # for i in range(Lvec.size(0)):
        rank = sim[i, :].argsort()
        # 找出是在第几个位置匹配上的
        rank_index = np.where(rank == i)[0][0]
        mrr_l.append(1.0 / (rank_index + 1))
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100),end='')
    print('MRR: %.4f' % (np.mean(mrr_l)))

    if stop_mrr:
        return np.mean(mrr_l)
    else:
        return top_lr[0] /  len(test_pair) * 100
def get_hits_b(res,Model, test_candidate, test_ill, entpair2f_idx, batch_size, device, stop_mrr,top_k=[1]):
    test_topk=top_k[-1]
    outlayer, _, _ = res['M0']
    f_emb = torch.cat(outlayer, dim=-1)

    test_ill_set = set(test_ill)
    test_pairs = []#all candidate entity pairs of Test set.
    for e1 in [a for a, b in test_ill]:
        for e2 in test_candidate[e1]:
            test_pairs.append((e1, e2))
    np.random.shuffle(test_pairs)
    isin_test_ill_set_num = sum([pair in test_ill_set for pair in test_pairs])
    print("all test entity pair num {}/ max align entity pair num: {}".format(len(test_pairs), isin_test_ill_set_num))
    scores = []
    for start_pos in range(0, len(test_pairs), batch_size):
        batch_pair_ids = test_pairs[start_pos:start_pos + batch_size]
        batch_f_ids = [entpair2f_idx[pair_idx] for pair_idx in batch_pair_ids]
        batch_features = f_emb[torch.LongTensor(batch_f_ids)].to(device)  # [B,f]
        batch_scores = Model.decoder.mlp(batch_features)
        batch_scores = batch_scores.detach().cpu().tolist()
        scores.extend(batch_scores)
    assert len(test_pairs) == len(scores)
    # eval
    e1_to_e2andscores = dict()
    for i in range(len(test_pairs)):
        e1, e2 = test_pairs[i]
        score = scores[i]
        if (e1, e2) in test_ill_set:
            label = 1
        else:
            label = 0
        if e1 not in e1_to_e2andscores:
            e1_to_e2andscores[e1] = []
        e1_to_e2andscores[e1].append((e2, score, label))

    all_test_num = len(e1_to_e2andscores.keys()) # test set size.
    result_labels = []
    for e, value_list in e1_to_e2andscores.items():
        v_list = value_list
        v_list.sort(key=lambda x: x[1], reverse=True)
        label_list = [label for e2, score, label in v_list]
        label_list = label_list[:test_topk]
        result_labels.append(label_list)
    result_labels = np.array(result_labels)
    result_labels = result_labels.sum(axis=0).tolist()
    topk_list = []
    for i in range(test_topk):
        nums = sum(result_labels[:i + 1])
        topk_list.append(round(nums / all_test_num, 5))
    for i in top_k:
        print("Hits@{}: {:.5f}".format(i,topk_list[i - 1]), end="")
    MRR = 0
    for i in range(len(result_labels)):
        MRR += (1 / (i + 1)) * result_labels[i]
    MRR /= all_test_num
    print('MRR: %.4f' % (MRR))
    if stop_mrr:
        return MRR
    else:
        return topk_list[1 - 1]
def save_sim_b(res,Model, entpair2f_idx, batch_size, device):
    outlayer, _, _ = res['M0']
    f_emb = torch.cat(outlayer, dim=-1)
    test_pairs=list(entpair2f_idx.keys())
    scores = []
    cosin_sim = {}
    for start_pos in range(0, len(test_pairs), batch_size):
        batch_pair_ids = test_pairs[start_pos:start_pos + batch_size]
        batch_f_ids = [entpair2f_idx[pair_idx] for pair_idx in batch_pair_ids]
        batch_features = f_emb[torch.LongTensor(batch_f_ids)].to(device)  # [B,f]
        batch_scores = Model.decoder.mlp(batch_features)
        batch_scores = batch_scores.detach().cpu().tolist()
        scores.extend(batch_scores)
    assert len(test_pairs) == len(scores)
    for i in range(len(test_pairs)):
        e1, e2 = test_pairs[i]
        score = scores[i]
        cosin_sim[(e1,e2)]=score
    return cosin_sim
