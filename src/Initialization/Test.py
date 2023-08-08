import time
import numpy as np
import torch
import sys
sys.path.append('..')
from utils.data_utils import cos_sim_mat_generate,batch_topk
from utils.test_utils import hit_res

def entlist2emb(Model,entids,entid2data,device):
    """
    return basic bert unit output embedding of entities
    """
    batch_token_ids = []
    batch_mask_ids = []
    for eid in entids:
        temp_token_ids = entid2data[int(eid)][0]
        temp_mask_ids = entid2data[int(eid)][1]

        batch_token_ids.append(temp_token_ids)
        batch_mask_ids.append(temp_mask_ids)

    batch_token_ids = torch.LongTensor(batch_token_ids).to(device)
    batch_mask_ids = torch.FloatTensor(batch_mask_ids).to(device)

    batch_emb = Model(batch_token_ids,batch_mask_ids)
    del batch_token_ids
    del batch_mask_ids
    return batch_emb

def test_data_gpt(Model,ent_ill,batch_size,device):
    start_time = time.time()
    Model.eval()
    with torch.no_grad():
        ents_1 = [e1 for e1,e2 in ent_ill]
        ents_2 = [e2 for e1,e2 in ent_ill]

        batch_ents_1 = ents_1
        batch_emb_1 = Model(batch_ents_1)#entlist2emb(Model,batch_ents_1,entid2data,device).detach().cpu().tolist()
        print(batch_emb_1.shape)
        emb1=batch_emb_1

        batch_ents_2 = ents_2
        batch_emb_2 = Model(batch_ents_2)#entlist2emb(Model,batch_ents_2,entid2data,device).detach().cpu().tolist()
        emb2=batch_emb_2
        res_mat = cos_sim_mat_generate(torch.FloatTensor(emb1.cpu()),torch.FloatTensor(emb2.cpu()),device,batch_size)
        score,top_index = batch_topk(res_mat,device,batch_size,topn=len(ent_ill),largest=True)
        res=hit_res(top_index,True)
    print("test using time: {:.3f}".format(time.time()-start_time))
    print("--------------------")
    return res
def test_data(Model,ent_ill,entid2data,batch_size,device):
    start_time = time.time()
    Model.eval()
    with torch.no_grad():
        ents_1 = [e1 for e1,e2 in ent_ill]
        ents_2 = [e2 for e1,e2 in ent_ill]

        emb1 = []
        for i in range(0,len(ents_1),batch_size):
            batch_ents_1 = ents_1[i: i+batch_size]
            batch_emb_1 = entlist2emb(Model,batch_ents_1,entid2data,device).detach().cpu().tolist()
            emb1.extend(batch_emb_1)
            del batch_emb_1

        emb2 = []
        for i in range(0,len(ents_2),batch_size):
            batch_ents_2 = ents_2[i: i+batch_size]
            batch_emb_2 = entlist2emb(Model,batch_ents_2,entid2data,device).detach().cpu().tolist()
            emb2.extend(batch_emb_2)
            del batch_emb_2
        res_mat = cos_sim_mat_generate(torch.FloatTensor(emb1),torch.FloatTensor(emb2),device,batch_size)
        score,top_index = batch_topk(res_mat,device,batch_size,topn=len(ent_ill),largest=True)
        res=hit_res(top_index,True)
    print("test using time: {:.3f}".format(time.time()-start_time))
    print("--------------------")
    return res

def test_datasets(entity_embbedding,ent_ill,batch_size,device,context=''):
    start_time = time.time()
    L = np.array([e1 for e1, e2 in ent_ill])
    R = np.array([e2 for e1, e2 in ent_ill])
    Lvec =torch.FloatTensor(entity_embbedding[L])
    Rvec = torch.FloatTensor(entity_embbedding[R])
    res_mat = cos_sim_mat_generate(Lvec, Rvec, device,batch_size)
    score, top_index = batch_topk(res_mat, device, batch_size,topn=len(ent_ill), largest=True)
    res = hit_res(top_index)
    print(context+" test using time: {:.3f}".format(time.time()-start_time))
    print("--------------------")
    return res
def check_test(entity_embbedding,word_ids,index2entity,ent_ill,batch_size,device,context=''):
    start_time = time.time()
    L = np.array([e1 for e1, e2 in ent_ill])
    R = np.array([e2 for e1, e2 in ent_ill])
    Lvec =torch.FloatTensor(entity_embbedding[L])
    Rvec = torch.FloatTensor(entity_embbedding[R])
    res_mat = cos_sim_mat_generate(Lvec, Rvec, device,batch_size)
    score, top_index = batch_topk(res_mat, device, batch_size,topn=len(ent_ill), largest=True)
    ent_num=top_index.shape[0]
    mrr_l=[]
    # shape[0]输出矩阵的行数，这里为10500
    for i in range(ent_num):  #np.where(rank == i)[0][0]
        e1=L[i]
        e2=R[i]
        e3=R[top_index[i][0]]
        # if e2!=e3:
        #     print(index2entity[e1],'|',index2entity[e2],'|',index2entity[e3])
        #     print(word_ids[e1],word_ids[e2],word_ids[e3])
        #     print(entity_embbedding[e1][:10],entity_embbedding[e2][:10],entity_embbedding[e3][:10])

    res = hit_res(top_index)
    print(context+" test using time: {:.3f}".format(time.time()-start_time))
    print("--------------------")
    return res

def get_bert_initemb(args,Model,indx2data):

    if args.is_load_BERT:Model.load_state_dict(torch.load(args.save_bert_path, map_location='cpu'))
    Model.eval()
    start_time = time.time()
    ent_emb = []
    with torch.no_grad():
        for eid in range(0, len(indx2data.keys()), args.test_batch_size): #eid == [0,n)
            token_inputs = []
            mask_inputs = []
            for i in range(eid, min(eid +  args.test_batch_size, len(indx2data.keys()))):
                token_input = indx2data[i][0]
                mask_input = indx2data[i][1]
                token_inputs.append(token_input)
                mask_inputs.append(mask_input)
            vec = Model(torch.LongTensor(token_inputs).to(args.device),
                        torch.FloatTensor(mask_inputs).to(args.device))
            ent_emb.extend(vec.detach().cpu().tolist())
        print("get entity embedding using time {:.3f}".format(time.time() - start_time))
    return ent_emb



