import copy
import pickle
import numpy as np
import torch
import torch.nn.functional as F


def load_id2object(file_paths,cn=''):
    id2object = {}
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            print('loading a (id2object)file...  ' + file_path)
            for line in f:
                th = line.strip('\n').split('\t')
                id2object[int(th[0])] = th[1]
                if cn=='entity' and 'http://' not in th[1]:
                    id2object[int(th[0])] ='http://'+'e'+file_path[-1]+'/'+th[1]
    return id2object


def load_file(fn, num=1):
    print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def load_embedding(file_path,norm=True):
    with open(file=file_path, mode='rb') as f:
        embedding_list = pickle.load(f)
        print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
    input_embeddings = torch.FloatTensor(embedding_list)
    if norm:
        F.normalize(input_embeddings, p=2, dim=1)
        input_embeddings[torch.isnan(input_embeddings)] = 0
    return input_embeddings


def candidate_generate(ents1, ents2, ent_emb, candidate_num):
    emb1 = ent_emb[ ents1 ]
    emb2 = ent_emb[ ents2 ]
    print("Test(get candidate) embedding shape:",emb1.shape,emb2.shape)
    print("get candidate by cosine similartity.")
    sim=torch.cdist(emb1, emb2, 2).pow(2)
    ent2candidates = dict()
    for i in range(emb1.shape[0]):
        e1=ents1[i]
        rank = sim[i, :].argsort()
        e2_list = np.array(ents2)[rank[:candidate_num]]
        ent2candidates[e1] = e2_list
    return ent2candidates
def candidate_generate_sim(ents1, ents2, sim, candidate_num):
    sim=sim[ents1][ents2]#torch.cdist(emb1, emb2, 2).pow(2)
    ent2candidates = dict()
    for i in range(len(ents1)):
        e1=ents1[i]
        rank = sim[i, :].argsort()
        e2_list = np.array(ents2)[rank[:candidate_num]]
        ent2candidates[e1] = e2_list
    return ent2candidates
def all_entity_pairs_gene(candidate_dict_list, ill_pair_list):
    #generate list of all candidate entity pairs.
    entity_pairs_list = []
    for candidate_dict in candidate_dict_list:
        for e1 in candidate_dict.keys():
            for e2 in candidate_dict[e1]:
                entity_pairs_list.append((int(e1), int(e2)))
    for ill_pair in ill_pair_list:
        for e1, e2 in ill_pair:
            entity_pairs_list.append((int(e1), int(e2)))
    entity_pairs_list = list(set(entity_pairs_list))
    print("entity_pair (e1,e2) num is: {}".format(len(entity_pairs_list)))
    return entity_pairs_list
def neigh_ent_dict_gene(entity_pairs,rel_triples,max_length,pad_id = None):
    """
    get one hop neighbor of entity
    return a dict, key = entity, value = (padding) neighbors of entity
    """
    neigh_ent_dict = dict()
    for h,t in rel_triples:
        h=int(h)
        t=int(t)
        if h == t:
            continue
        if h not in neigh_ent_dict:
            neigh_ent_dict[h] = []
        if t not in neigh_ent_dict:
            neigh_ent_dict[t] = []
        neigh_ent_dict[h].append(t)
        neigh_ent_dict[t].append(h)
    for e in neigh_ent_dict.keys():
        np.random.shuffle(neigh_ent_dict[e])
        np.random.shuffle(neigh_ent_dict[e])
        np.random.shuffle(neigh_ent_dict[e])
    for e in neigh_ent_dict.keys():
        neigh_ent_dict[e] = neigh_ent_dict[e][:max_length]
    if pad_id != None:
        for e in neigh_ent_dict.keys():
            pad_list = [pad_id] * (max_length - len(neigh_ent_dict[e]))
            neigh_ent_dict[e] = neigh_ent_dict[e] + pad_list
        for e1,e2 in entity_pairs:
            if e1 not in neigh_ent_dict:
                neigh_ent_dict[e1] = [pad_id] * max_length
            if e2 not in neigh_ent_dict:
                neigh_ent_dict[e2] = [pad_id] * max_length
    return neigh_ent_dict
def ent2attributeValues_gene(att_datas,max_length,entity_pairs,pad_value_id = None):

    ent2attributevalues=att_datas
    # random choose attributeValue to maxlength.
    for e in ent2attributevalues.keys():
        np.random.shuffle(ent2attributevalues[e])
    for e in ent2attributevalues.keys():
        ent2attributevalues[e] = ent2attributevalues[e][:max_length]
    if pad_value_id != None:
        for e in ent2attributevalues.keys():
            pad_list = [pad_value_id] * (max_length - len(ent2attributevalues[e]))
            ent2attributevalues[e] = ent2attributevalues[e] + pad_list
        for e1,e2 in entity_pairs:
            if e1 not in ent2attributevalues:
                pad_list = [pad_value_id] * max_length
                ent2attributevalues[e1] =pad_list
            if e2 not in ent2attributevalues:
                pad_list = [pad_value_id] * max_length
                ent2attributevalues[e2] =pad_list

    return ent2attributevalues

def cos_sim_mat_generate(emb1,emb2,device,bs = 128):
    """
    return cosine similarity matrix of embedding1(emb1) and embedding2(emb2)
    """
    res_mat = batch_mat_mm(emb1,emb2.t(),device,bs=bs)
    return res_mat
def sort_a(data_list):
    """
    sort
    """
    new_data_list = []
    e2e_datas = dict()
    for e, a, l in data_list:
        if e not in e2e_datas:
            e2e_datas[e] = set()
        e2e_datas[e].add((e, a, l))
    for e in e2e_datas.keys():
        e2e_datas[e] = list(e2e_datas[e])
        e2e_datas[e].sort(key=lambda x: x[1])
        for one in e2e_datas[e]:
            new_data_list.append(one)
    return new_data_list
def remove_one_to_N_att_data_by_threshold(ori_keep_data,one2N_threshold):
    """
    Filter noise attribute triples based on threshold
    """
    att_data = copy.deepcopy(ori_keep_data)
    e_a2fre = dict()
    for e,a,l,l_type in att_data:
        if (e,a) not in e_a2fre:
            e_a2fre[(e,a)] = 0
        e_a2fre[(e,a)] += 1
    remove_set = set()
    for e_a in e_a2fre:
        if e_a2fre[e_a] > one2N_threshold:
            remove_set.add(e_a)
    remove_set = set()
    keep_datas = []
    remove_datas = []
    for e,a,l,l_type in att_data:
        if (e,a) in remove_set:
            remove_datas.append((e,a,l))
        else:
            keep_datas.append((e,a,l))
    keep_datas.sort(key=lambda x:x[0])
    keep_datas = sort_a(keep_datas)
    print("Before removing noisy attribute triples, attribute triples {}".format(len(att_data)))
    print("remaining attribute_triples num {} ; noisy attribute_triples num {}".format(len(keep_datas), len(remove_datas)))
    return keep_datas
def read_att_data(data_path,cn='1'):
    """
    load attribute triples file.
    """
    print("loading attribute triples file from: ",data_path)
    att_data = []
    with open(data_path,"r",encoding="utf-8") as f:
        for line in f:
            e,a,l = line.rstrip('\n').split('\t',2)
            e = e.strip('<>')
            a = a.strip('<>')
            if "/property/" in a:
                a = a.split(r'/property/')[-1]
            else:
                a = a.split(r'/')[-1]
            l = l.rstrip('@zhenjadefr .')
            if len(l.rsplit('^^',1)) == 2:
                l,l_type = l.rsplit("^^")
            else:
                l_type = 'string'
            l = l.strip("\"")
            if 'http://' not in e:
                e='http://'+'e'+cn+'/'+e
            att_data.append((e,a,l,l_type)) #(entity,attribute,value,value_type)
    return att_data
def filter_attributes(att_data,type):
    """
    load attribute triples file.
    """
    str_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '#', '(', ')', '*', '+', '-']
    number_cn = 0
    number_list =[]
    string_list =[]
    for e, a, l, l_type in att_data:
        total = len(l)
        if total == 0:
            continue
        cn = 0
        for sub in str_set:
            if sub in l:
                cn += l.count(sub)
        if cn / total > 0.6:
            number_cn += 1
            number_list.append((e,a,l,l_type))
        else:
            string_list.append((e,a,l,l_type))
    if type=='only_number':
        save_data=number_list
    else:
        save_data=att_data
    print('save attr:',len(save_data),'del attr:',len(att_data)-len(save_data))
    return save_data
def write_att_data(data_path,keep_data,name2ids,attribute2index,value2index,ent_ids_set,index2name,attribute_use_data):
    with open(data_path,'w') as fin:
        for e,a,v in keep_data:
            eids=name2ids[e]
            for eid in eids:
                fin.write(str(eid)+'\t'+str(attribute2index[a])+'\t'+str(value2index[v])+'\n')
        if attribute_use_data in ['add_name','add_relation']:
            for eid in ent_ids_set:
                fin.write(str(eid) + '\t' + str(attribute2index['addname']) + '\t' + str(value2index[index2name[eid]]) + '\n')
def write_val_data(data_path,index2value):
    str_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '#', '(', ')', '*', '+', '-']
    with open(data_path,'w') as fin:
        for vid in index2value:
            vname=index2value[vid]
            total = len(vname)
            if total == 0:
                fin.write(str(vid) + '\t' + str(0) + '\t' + str(vname) + '\n')
                continue
            cn = 0
            for sub in str_set:
                if sub in vname:
                    cn += vname.count(sub)
            if cn / total > 0.6:
                fin.write(str(vid) + '\t' + str(1) + '\t' + str(vname) + '\n')
            else:
                fin.write(str(vid) + '\t' + str(0) + '\t' + str(vname) + '\n')
def list_index_select(outputs,ILL):
    res=[]
    for vec in outputs:
        res.append(torch.index_select(vec,0,ILL))
    return res

def read_translation(data_path):
    """
    load attribute triples file.
    """
    print("loading translation file from: ",data_path)
    trans_data = {}
    with open(data_path,"r",encoding="utf-8") as f:
        for line in f:
            e,a,l = line.rstrip('\n').split('\t',2)
            trans_data[e]=l
    return trans_data

def batch_mat_mm(mat1,mat2,device,bs=128):
    #be equal to matmul, Speed up computing with GPU
    res_mat = []
    axis_0 = mat1.shape[0]
    for i in range(0,axis_0,bs):
        temp_div_mat_1 = mat1[i:min(i+bs,axis_0)].to(device)
        res = temp_div_mat_1.mm(mat2.to(device))
        res_mat.append(res.cpu())
    res_mat = torch.cat(res_mat,0)
    return res_mat

def batch_topk(mat,device,bs=128,topn = 50,largest = False):
    #be equal to topk, Speed up computing with GPU
    res_score = []
    res_index = []
    axis_0 = mat.shape[0]
    for i in range(0,axis_0,bs):
        temp_div_mat = mat[i:min(i+bs,axis_0)].to(device)
        score_mat,index_mat =temp_div_mat.topk(topn,largest=largest)
        res_score.append(score_mat.cpu())
        res_index.append(index_mat.cpu())
    res_score = torch.cat(res_score,0)
    res_index = torch.cat(res_index,0)
    return res_score,res_index
