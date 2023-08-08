import time

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from load_data import generate_neg_candidates,generate_neg_candidates_gpt
from Test import *

def train_n_epoch_cross(args,Model,Criterion,optimizer,Train_gene,indx2data,train_ill,valid_ill,test_ill):
    print("start training...")
    best_res=0
    cur_patience=0
    test_data(Model, test_ill, indx2data, args.test_batch_size, args.device)
    for epoch in range(args.epochs):
        print("+++++++++++")
        print("Epoch: ",epoch)
        train_ent1s = [e1 for e1,e2 in train_ill]
        train_ent2s = [e2 for e1,e2 in train_ill]
        for_candidate_ent1s = Train_gene.ent_ids1
        for_candidate_ent2s = Train_gene.ent_ids2

        candidate_dict = generate_neg_candidates(Model,train_ent1s,train_ent2s,for_candidate_ent1s,for_candidate_ent2s,indx2data,args.device)
        Train_gene.train_index_gene(candidate_dict)
        training_data_iter = DataLoader(Train_gene, batch_size=Train_gene.batch_size, shuffle=True)
        epoch_loss,epoch_train_time = train_1_epoch_cross(Model,Criterion,optimizer,training_data_iter,indx2data,args.device)
        optimizer.zero_grad()
        print("Epoch {}: loss {:.3f}, using time {:.3f}".format(epoch,epoch_loss,epoch_train_time))
        cur_res=test_data(Model, valid_ill, indx2data, args.test_batch_size,args.device)
        if cur_res>best_res:
            best_res=cur_res
            test_data(Model, test_ill, indx2data, args.test_batch_size, args.device)
            Model.eval()
            torch.save(Model.state_dict(), args.save_bert_path)
            cur_patience = 0
        else:
            cur_patience+=1
            if cur_patience>args.patience:
                break
        torch.cuda.empty_cache()

def train_1_epoch_cross(Model,Criterion,optimizer,training_data_iter,indx2data,device):
    start_time = time.time()
    all_loss = 0
    Model.train()
    for pe1s,pe2s,ne1s,ne2s in training_data_iter:
        optimizer.zero_grad()
        pos_emb1 = entlist2emb(Model,pe1s,indx2data,device)
        pos_emb2 = entlist2emb(Model,pe2s,indx2data,device)
        batch_length = pos_emb1.shape[0]
        pos_score = F.pairwise_distance(pos_emb1,pos_emb2,p=1,keepdim=True)#L1 distance
        del pos_emb1
        del pos_emb2

        neg_emb1 = entlist2emb(Model,ne1s,indx2data,device)
        neg_emb2 = entlist2emb(Model,ne2s,indx2data,device)
        neg_score = F.pairwise_distance(neg_emb1,neg_emb2,p=1,keepdim=True)
        del neg_emb1
        del neg_emb2

        label_y = -torch.ones(pos_score.shape).to(device) #pos_score < neg_score
        batch_loss = Criterion( pos_score , neg_score , label_y )
        del pos_score
        del neg_score
        del label_y
        batch_loss.backward()
        optimizer.step()
        all_loss += batch_loss.item() * batch_length
    all_using_time = time.time()-start_time
    return all_loss,all_using_time
def train_n_epoch_gpt(args,Model,Criterion,optimizer,Train_gene,train_ill,valid_ill,test_ill):
    print("start training...")
    best_res=0
    cur_patience=0
    test_data_gpt(Model, test_ill, args.test_batch_size, args.device)
    for epoch in range(args.epochs):
        print("+++++++++++")
        print("Epoch: ",epoch)
        train_ent1s = [e1 for e1,e2 in train_ill]
        train_ent2s = [e2 for e1,e2 in train_ill]
        for_candidate_ent1s = Train_gene.ent_ids1
        for_candidate_ent2s = Train_gene.ent_ids2

        candidate_dict = generate_neg_candidates_gpt(Model,train_ent1s,train_ent2s,for_candidate_ent1s,for_candidate_ent2s,args.device)
        Train_gene.train_index_gene(candidate_dict)
        training_data_iter = DataLoader(Train_gene, batch_size=Train_gene.batch_size, shuffle=True)
        epoch_loss,epoch_train_time = train_1_epoch_gpt(Model,Criterion,optimizer,training_data_iter,args.device)
        optimizer.zero_grad()
        print("Epoch {}: loss {:.3f}, using time {:.3f}".format(epoch,epoch_loss,epoch_train_time))
        cur_res=test_data_gpt(Model, valid_ill, args.test_batch_size,args.device)
        if cur_res>best_res:
            best_res=cur_res
            test_data_gpt(Model, test_ill, args.test_batch_size, args.device)
            Model.eval()
            torch.save(Model.state_dict(), args.save_bert_path)
            cur_patience = 0
        else:
            cur_patience+=1
            if cur_patience>args.patience:
                break
        torch.cuda.empty_cache()

def train_1_epoch_gpt(Model,Criterion,optimizer,training_data_iter,device):
    start_time = time.time()
    all_loss = 0
    Model.train()
    for pe1s,pe2s,ne1s,ne2s in training_data_iter:
        optimizer.zero_grad()
        pos_emb1 = Model(pe1s)#entlist2emb(Model,pe1s,indx2data,device)
        pos_emb2 = Model(pe2s)#entlist2emb(Model,pe2s,indx2data,device)
        batch_length = pos_emb1.shape[0]
        pos_score = F.pairwise_distance(pos_emb1,pos_emb2,p=1,keepdim=True)#L1 distance
        del pos_emb1
        del pos_emb2

        neg_emb1 = Model(ne1s)#entlist2emb(Model,ne1s,indx2data,device)
        neg_emb2 = Model(ne2s)#entlist2emb(Model,ne2s,indx2data,device)
        neg_score = F.pairwise_distance(neg_emb1,neg_emb2,p=1,keepdim=True)
        del neg_emb1
        del neg_emb2

        label_y = -torch.ones(pos_score.shape).to(device) #pos_score < neg_score
        batch_loss = Criterion( pos_score , neg_score , label_y )
        del pos_score
        del neg_score
        del label_y
        batch_loss.backward()
        optimizer.step()
        all_loss += batch_loss.item() * batch_length
    all_using_time = time.time()-start_time
    return all_loss,all_using_time