import os
import random
import string
import time
import json

import joblib
import torch
from transformers import BertTokenizer
from Test import entlist2emb
import numpy as np
import sys
import re
sys.path.append('..')
from utils.data_utils import *






def generate_neg_candidates(Model, train_ent1s, train_ent2s, for_candidate_ent1s, for_candidate_ent2s,entid2data,device,nearest_sample_num=128, batch_size=128):
    start_time = time.time()
    Model.eval()
    torch.cuda.empty_cache()
    candidate_dict = dict()
    with torch.no_grad():
        train_emb1 = []
        for_candidate_emb1 = []
        for i in range(0, len(train_ent1s), batch_size):
            temp_emb = entlist2emb(Model, train_ent1s[i:i + batch_size], entid2data, device).cpu().tolist()
            train_emb1.extend(temp_emb)
        for i in range(0, len(for_candidate_ent2s), batch_size):
            temp_emb = entlist2emb(Model, for_candidate_ent2s[i:i + batch_size], entid2data,device).cpu().tolist()
            for_candidate_emb1.extend(temp_emb)

        # language2 (KG2)
        train_emb2 = []
        for_candidate_emb2 = []
        for i in range(0, len(train_ent2s), batch_size):
            temp_emb = entlist2emb(Model, train_ent2s[i:i + batch_size], entid2data, device).cpu().tolist()
            train_emb2.extend(temp_emb)
        for i in range(0, len(for_candidate_ent1s), batch_size):
            temp_emb = entlist2emb(Model, for_candidate_ent1s[i:i + batch_size], entid2data,device).cpu().tolist()
            for_candidate_emb2.extend(temp_emb)
        torch.cuda.empty_cache()

        # cos sim
        cos_sim_mat1 = cos_sim_mat_generate(torch.FloatTensor(train_emb1),torch.FloatTensor(for_candidate_emb1),device)
        cos_sim_mat2 = cos_sim_mat_generate(torch.FloatTensor(train_emb2),torch.FloatTensor(for_candidate_emb2),device)
        torch.cuda.empty_cache()
        # topk index
        _, topk_index_1 = batch_topk(cos_sim_mat1, device,topn=nearest_sample_num, largest=True)
        topk_index_1 = topk_index_1.tolist()
        _, topk_index_2 = batch_topk(cos_sim_mat2, device,topn=nearest_sample_num, largest=True)
        topk_index_2 = topk_index_2.tolist()
        # get candidate
        for x in range(len(topk_index_1)):
            e = train_ent1s[x]
            candidate_dict[e] = []
            for y in topk_index_1[x]:
                c = for_candidate_ent2s[y]
                candidate_dict[e].append(c)
        for x in range(len(topk_index_2)):
            e = train_ent2s[x]
            candidate_dict[e] = []
            for y in topk_index_2[x]:
                c = for_candidate_ent1s[y]
                candidate_dict[e].append(c)

    print("get candidate using time: {:.3f}".format(time.time() - start_time))
    torch.cuda.empty_cache()
    return candidate_dict
def generate_neg_candidates_gpt(Model, train_ent1s, train_ent2s, for_candidate_ent1s, for_candidate_ent2s,device,nearest_sample_num=128, batch_size=128):
    start_time = time.time()
    Model.eval()
    torch.cuda.empty_cache()
    candidate_dict = dict()
    with torch.no_grad():
        train_emb1 = []
        for_candidate_emb1 = []
        for i in range(0, len(train_ent1s), batch_size):
            temp_emb = Model(train_ent1s[i:i + batch_size]).cpu().tolist()
            train_emb1.extend(temp_emb)
        for i in range(0, len(for_candidate_ent2s), batch_size):
            temp_emb =  Model(for_candidate_ent2s[i:i + batch_size]).cpu().tolist()
            for_candidate_emb1.extend(temp_emb)

        # language2 (KG2)
        train_emb2 = []
        for_candidate_emb2 = []
        for i in range(0, len(train_ent2s), batch_size):
            temp_emb = Model(train_ent2s[i:i + batch_size]).cpu().tolist()
            train_emb2.extend(temp_emb)
        for i in range(0, len(for_candidate_ent1s), batch_size):
            temp_emb =Model(for_candidate_ent1s[i:i + batch_size]).cpu().tolist()
            for_candidate_emb2.extend(temp_emb)
        torch.cuda.empty_cache()

        # cos sim
        cos_sim_mat1 = cos_sim_mat_generate(torch.FloatTensor(train_emb1),torch.FloatTensor(for_candidate_emb1),device)
        cos_sim_mat2 = cos_sim_mat_generate(torch.FloatTensor(train_emb2),torch.FloatTensor(for_candidate_emb2),device)
        torch.cuda.empty_cache()
        # topk index
        _, topk_index_1 = batch_topk(cos_sim_mat1, device,topn=nearest_sample_num, largest=True)
        topk_index_1 = topk_index_1.tolist()
        _, topk_index_2 = batch_topk(cos_sim_mat2, device,topn=nearest_sample_num, largest=True)
        topk_index_2 = topk_index_2.tolist()
        # get candidate
        for x in range(len(topk_index_1)):
            e = train_ent1s[x]
            candidate_dict[e] = []
            for y in topk_index_1[x]:
                c = for_candidate_ent2s[y]
                candidate_dict[e].append(c)
        for x in range(len(topk_index_2)):
            e = train_ent2s[x]
            candidate_dict[e] = []
            for y in topk_index_2[x]:
                c = for_candidate_ent1s[y]
                candidate_dict[e].append(c)

    print("get candidate using time: {:.3f}".format(time.time() - start_time))
    torch.cuda.empty_cache()
    return candidate_dict
def read_bert_input(index2entity,bert_path,des_max_length=128):
    Tokenizer = BertTokenizer.from_pretrained(bert_path)
    ent2desTokens = dict()
    for id,des in index2entity.items():
        string = des
        encode_indexs = Tokenizer.encode(string)[:des_max_length-2]
        ent2desTokens[id] = encode_indexs

    ent2data = dict()
    pad_id = Tokenizer.pad_token_id
    for ent_id in ent2desTokens:
        ent2data[ent_id] = [[],[]]
        ent_token_id = ent2desTokens[ent_id]
        ent_token_ids = Tokenizer.build_inputs_with_special_tokens(ent_token_id)

        token_length = len(ent_token_ids)
        assert token_length <= des_max_length

        ent_token_ids = ent_token_ids + [pad_id] * max(0, des_max_length - token_length)

        ent_mask_ids = np.ones(np.array(ent_token_ids).shape)
        ent_mask_ids[np.array(ent_token_ids) == pad_id] = 0
        ent_mask_ids = ent_mask_ids.tolist()

        ent2data[ent_id][0] = ent_token_ids
        ent2data[ent_id][1] = ent_mask_ids
    return ent2data
def read_bert_input_mix(index2entity,bert_path,des_max_length=128):
    Tokenizer = BertTokenizer.from_pretrained(bert_path)
    ent2desTokens = dict()
    for id,txt in index2entity.items():
        string1 = txt[0]
        encode_indexs1 = Tokenizer.encode(string1)[:des_max_length-2]
        string2 = txt[1]
        encode_indexs2 = Tokenizer.encode(string2)[:des_max_length - 2]
        ent2desTokens[id] = (encode_indexs1,encode_indexs2)
    ent2data = dict()
    pad_id = Tokenizer.pad_token_id
    for ent_id in ent2desTokens:
        ent2data[ent_id] = [[],[]]
        ent_token_id = ent2desTokens[ent_id][0]
        ent_token_ids = Tokenizer.build_inputs_with_special_tokens(ent_token_id)

        token_length = len(ent_token_ids)
        assert token_length <= des_max_length

        ent_token_ids = ent_token_ids + [pad_id] * max(0, des_max_length - token_length)

        ent_mask_ids = np.ones(np.array(ent_token_ids).shape)
        ent_mask_ids[np.array(ent_token_ids) == pad_id] = 0
        ent_mask_ids = ent_mask_ids.tolist()

        ent2data[ent_id][0] += ent_token_ids
        ent2data[ent_id][1] += ent_mask_ids

        ent_token_id = ent2desTokens[ent_id][1]
        ent_token_ids = Tokenizer.build_inputs_with_special_tokens(ent_token_id)

        token_length = len(ent_token_ids)
        assert token_length <= des_max_length

        ent_token_ids = ent_token_ids + [pad_id] * max(0, des_max_length - token_length)

        ent_mask_ids = np.ones(np.array(ent_token_ids).shape)
        ent_mask_ids[np.array(ent_token_ids) == pad_id] = 0
        ent_mask_ids = ent_mask_ids.tolist()

        ent2data[ent_id][0] += ent_token_ids
        ent2data[ent_id][1] += ent_mask_ids
    return ent2data
def read_gpt_input(bert_path):
    # with open(file=bert_path, mode='r', encoding='utf-8') as f:
    #     embedding_list = json.load(f)
    if '100K' in bert_path:
        embedding_list = joblib.load(open(bert_path, 'rb'))
    else:
        embedding_list=pickle.load(open(bert_path, 'rb'))
    print(len(embedding_list),len(embedding_list[0]))
    input_embeddings =torch.tensor(embedding_list)
    embedding = input_embeddings
    embedding[torch.isnan(embedding)] = 0
    return embedding
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
    comment_1_path=prefix+'/comment_1'
    comment_2_path = prefix + '/comment_2'
    attr_1_path=prefix+'/attr_triples_1/2attr'
    attr_2_path = prefix + '/attr_triples_2/2attr'
    wattr_1_path=prefix+'/attr_triple_1'
    wattr_2_path = prefix + '/attr_triple_2'
    wval_ids_path=prefix+'/val_ids'
    translation_path=prefix+'/s_labels'
    rel_ids_1_path=prefix+'/rel_ids_1'
    rel_ids_2_path = prefix + '/rel_ids_2'
    kg1_path = prefix + '/triples_1'
    kg2_path = prefix + '/triples_2'


    ent_ill = load_file(ent_ill_path, 2)
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

    index2name1 = load_id2object([ent_ids_1_path, ent_ids_2_path],"entity")
    ent_ids_set1=load_id2object([ent_ids_1_path],"entity").keys()
    ent_ids_set2 = load_id2object([ent_ids_2_path], "entity").keys()
    name2ids={}
    for id,name in index2name1.items():
        ids=name2ids.get(name,set())
        ids.add(id)
        name2ids[name]=ids

    index2name={id:get_name(name)for id,name in index2name1.items()}
    index2entity={}
    index2V, value2index,index2A,attribute2index=None,None,None,None
    if args.data_type=='entity':
        if args.model=='bert_int_des' and os.path.exists(comment_1_path) and os.path.exists(comment_2_path):
            index2des = load_id2object([comment_1_path, comment_2_path])
            for id in index2name:
                index2entity[id]=index2des[id] if id in index2des else index2name[id]
        elif args.model=='bert_mix' and os.path.exists(comment_1_path) and os.path.exists(comment_2_path):
            index2des = load_id2object([comment_1_path, comment_2_path])
            keep_data_1 = read_att_data(attr_1_path,'1')
            keep_data_2 = read_att_data(attr_2_path,'2')
            value2id, id2value, attr2id, id2attr, eid2facts, vid2eids, vword2eids=generate_att_data(keep_data_1+keep_data_2,name2ids)
            aid_func=calculate_func(id2attr,attr2id,eid2facts,index2name)
            header = ['eid', 'ent_name']
            for p, pid in attr2id.items():
                header.append(p)
            dicts=get_property_table_line(args,index2name,eid2facts,id2attr,id2value,aid_func)
            dicts = filter(lambda dic: dic is not None, dicts)
            dicts = list(dicts)
            header1 = header.copy()
            header1.remove('eid')
            seqs = [get_seq(dic,header1) for dic in dicts]
            index2attr={eid:text for eid,text in seqs}
            for id in index2name:
                index2entity[id] = (index2des[id] if id in index2des else index2name[id],index2attr[id] if id in index2attr else '')

        elif 'SDEA' in args.model:
            keep_data_1 = read_att_data(attr_1_path,'1')
            keep_data_2 = read_att_data(attr_2_path,'2')
            keep_data_1=filter_attributes(keep_data_1,args.attribute_use_data)
            keep_data_2 = filter_attributes(keep_data_2, args.attribute_use_data)
            value2id, id2value, attr2id, id2attr, eid2facts, vid2eids, vword2eids=generate_att_data(keep_data_1+keep_data_2,name2ids)
            aid_func=calculate_func(id2attr,attr2id,eid2facts,index2name)
            header = ['eid', 'ent_name']
            for p, pid in attr2id.items():
                header.append(p)
            dicts=get_property_table_line(args,index2name,eid2facts,id2attr,id2value,aid_func)
            dicts = filter(lambda dic: dic is not None, dicts)
            dicts = list(dicts)
            header1 = header.copy()
            header1.remove('eid')
            if args.attribute_use_data!='add_name':
                header1.remove('ent_name')
            seqs = [get_seq(dic,header1) for dic in dicts]
            index2entity={eid:text for eid,text in seqs}
        elif args.model=='word-based':
            index2entity = index2name
            if os.path.exists(translation_path):
                name2trans=read_translation(translation_path)
                name2trans={get_name(name):trans for name,trans in name2trans.items()}
                index2entity={id:name2trans[name] if name in name2trans else name for id,name in index2name.items()}
        else:
            index2entity = index2name
    elif args.data_type=='relation':
        index2rel = load_id2object([rel_ids_1_path, rel_ids_2_path])
        index2entity={id:get_name(name)for id,name in index2rel.items()}
    elif args.data_type=='attribute':
        keep_data_1 = read_att_data(attr_1_path, '1')
        keep_data_2 = read_att_data(attr_2_path, '2')
        keep_data_1 = filter_attributes(keep_data_1, args.attribute_use_data)
        keep_data_2 = filter_attributes(keep_data_2, args.attribute_use_data)

        keep_data_1 = remove_one_to_N_att_data_by_threshold(keep_data_1,one2N_threshold=3)
        keep_data_2= remove_one_to_N_att_data_by_threshold(keep_data_2,one2N_threshold=3)
        if args.attribute_use_data == 'add_relation':
            print('load relation triplet')
            index2rel = load_id2object([rel_ids_1_path, rel_ids_2_path])
            kg1 = load_file(kg1_path, 3)  # KG1
            kg2 = load_file(kg2_path, 3)  # KG2
            kg1+=[(tr[2],tr[1],tr[0]) for tr in kg1]
            kg2 += [(tr[2], tr[1], tr[0]) for tr in kg2]
            for tr in kg1:
                e=index2name1[tr[0]]
                e = e.strip('<>')
                if 'http://' not in e:
                    e = 'http://' + 'e' + '1'+ '/' + e
                a=index2rel[tr[1]]
                a = a.strip('<>')
                if "/property/" in a:
                    a = a.split(r'/property/')[-1]
                else:
                    a = a.split(r'/')[-1]
                keep_data_1.append((e,a,index2name[tr[2]]))
            for tr in kg2:
                e=index2name1[tr[0]]
                e = e.strip('<>')
                if 'http://' not in e:
                    e = 'http://' + 'e' + '2'+ '/' + e
                a=index2rel[tr[1]]
                a = a.strip('<>')
                if "/property/" in a:
                    a = a.split(r'/property/')[-1]
                else:
                    a = a.split(r'/')[-1]
                keep_data_2.append((e,a,index2name[tr[2]]))

        index2entity={}
        attribute_set1=set()
        attribute_set2 = set()
        attr_dict={}
        for e, a, l in keep_data_1+keep_data_2:
            attr_set = attr_dict.get(a, set())
            attr_set.add(l)
            attr_dict[a] = attr_set
            eids=name2ids[e]
            for eid in eids:
                vs=index2entity.get(eid,[])
                vs.append(l)
                index2entity[eid]=vs
        for e,a,l in keep_data_1:
            attribute_set1.add(a)
        for e,a,l in keep_data_2:
            attribute_set2.add(a)
        exist_values =[]
        values_set=set()
        if args.attribute_use_data in ['add_name','add_relation']:
            for eid,name in index2name.items():
                vs = index2entity.get(eid, [])
                vs.append(name)
                index2entity[eid] = vs
                if name not in values_set:
                    exist_values.append(name)
                    values_set.add(name)
                    # print(eid,len(exist_values),name)
        sorted_attr = sorted(attr_dict.items(), key=lambda item: len(item[1]), reverse=True)
        for item in sorted_attr:
            value_list = item[1]
            for v in value_list:
                if v not in values_set:
                    exist_values.append(v)
                    values_set.add(v)
        value_set=exist_values
        print('value num:'+str(len(value_set)))
        attribute_set=list(attribute_set1)+list(attribute_set2)
        if args.attribute_use_data in ['add_name','add_relation']:
            attribute_set.append('addname')
        index2A={id:a for id,a in enumerate(attribute_set)}
        index2V={id:v for id,v in enumerate(value_set)}
        value2index = {value: v_id for v_id, value in index2V.items()}
        attribute2index1={index2A[id]:id for id in range(len(list(attribute_set1)))}
        attribute2index2 = {index2A[id+len(list(attribute_set1))]: id+len(list(attribute_set1)) for id in range(len(list(attribute_set2)))}
        if args.attribute_use_data:
            attribute2index1['addname']=len(attribute_set)-1
            attribute2index2['addname'] = len(attribute_set) - 1
        write_val_data(wval_ids_path,index2V)
        write_att_data(wattr_1_path,keep_data_1,name2ids,attribute2index1,value2index,ent_ids_set1,index2name,args.attribute_use_data)
        write_att_data(wattr_2_path, keep_data_2, name2ids, attribute2index2, value2index,ent_ids_set2,index2name,args.attribute_use_data)

    return ent_ill, train_ill, valid_ill,test_ill,  index2entity,index2V,value2index,index2A
def text_to_word_sequence(text,
                          filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if text is None:
        return []
    if lower: text = text.lower()
    translate_table = {ord(c): ord(t) for c, t in zip(filters, split * len(filters))}
    text = text.translate(translate_table)
    seq = text.split(split)
    return [i for i in seq if i]
def generate_att_data(att_tuples,name2ids):
    value2id={}
    id2value={}
    attr2id={}
    id2attr={}
    eid2facts={}
    vid2eids={}
    vword2eids={}
    for e,a,l,l_type in att_tuples:
        ename=e
        vid=value2id[l] if l in value2id else len(value2id)
        value2id[l]=vid
        id2value[vid]=l
        aid=attr2id[a] if a in attr2id else len(attr2id)
        attr2id[a]=aid
        id2attr[aid]=a
        eids = name2ids[ename]
        words = text_to_word_sequence(l)
        for eid in eids:
            facts=eid2facts.get(eid,[])
            facts.append((aid,vid))
            eid2facts[eid]=facts
            eids=vid2eids.get(vid,set())
            eids.add(eid)
            vid2eids[vid]=eids
            for wd in words:
                eids=vword2eids.get(wd,set())
                eids.add(eid)
                vword2eids[wd]=eids
    return value2id,id2value,attr2id,id2attr,eid2facts,vid2eids,vword2eids
def calculate_func(id2attr,attr2id,eid2facts,index2name):
    num_occurrences = [0] * len(id2attr)
    func = [0.] * len(id2attr)
    num_subjects_per_relation = [0] * len(id2attr)
    last_subject = [-1] * len(id2attr)

    for sbj_id in index2name.keys():
        facts = eid2facts.get(sbj_id)
        if facts is None:
            continue
        for fact in facts:
            num_occurrences[fact[0]] += 1
            if last_subject[fact[0]] != sbj_id:
                last_subject[fact[0]] = sbj_id
                num_subjects_per_relation[fact[0]] += 1

    for r_name, rid in attr2id.items():
        func[rid] = num_subjects_per_relation[rid] / num_occurrences[rid]
    return func
def get_property_table_line(args,index2name,eid2facts,id2attr,id2value,aid_func):
    dicts=[]
    for ei,ename in index2name.items():
        dic = {'eid': ei, 'ent_name': ename}
        facts = eid2facts.get(ei)
        if facts is not None:
            fact_aggregation = {}
            for fact in facts:
                # 过滤函数性低的
                if args.func_control and aid_func[fact[0]] <= args.func_threshold:
                    continue
                factslist=fact_aggregation.get(fact[0],[])
                factslist.append(id2value[fact[1]])
                fact_aggregation[fact[0]]=factslist
            for pid, objs in fact_aggregation.items():
                pred = id2attr[pid]
                obj = ' '.join(objs)
                dic[pred] = obj
        dicts.append(dic)
    return dicts


def get_seq(dic,header):
    eid = dic['eid']
    values = [dic[key] for key in header if key in dic]
    seq = ' '.join(values)
    if len(seq)==0:
        seq=' '
    try:
        assert len(seq) > 0
    except AssertionError:
        print(dic)
        exit(1)
    return eid, seq


