import sys
import numpy as np
import torch
import copy

from tqdm import tqdm

from Test import get_hits,get_hits_b,get_hits_g1,update_candidates
sys.path.append('..')
from utils.data_utils import list_index_select
class RelationDataset(object):
    def __init__(self, train_ill, train_candidate, all_neighbor, neg_num, batch_size):
        self.train_ill = train_ill
        self.train_candidate = copy.deepcopy(train_candidate)
        self.iter_count = 0
        self.all_neighbor=all_neighbor
        self.batch_size = batch_size
        self.neg_num = neg_num
        print("In Train_batch_index_generator, train_ILL num : {}".format(len(self.train_ill)))
        print("In Train_batch_index_generator, Batch size: {}".format(self.batch_size))
        print("In Train_batch_index_generator, Negative sampling num: {}".format(self.neg_num))
        for e in self.train_candidate.keys():
            self.train_candidate[e] = np.array(self.train_candidate[e])
        self.train_pair_indexs, self.batch_num = self.train_pair_index_gene()

    def train_pair_index_gene(self):
        """
        generate training data (entity_index).
        """
        train_pair_indexs = []
        for pe1, pe2 in self.train_ill:
            neg_indexs = np.random.randint(len(self.train_candidate[pe1]),size=self.neg_num)
            ne2_list = self.train_candidate[pe1][neg_indexs].tolist()
            for ne2 in ne2_list:
                if ne2 == pe2:
                    continue
                ne1 = pe1
                train_pair_indexs.append((pe1, pe2, ne1, ne2))
                #(pe1,pe2) is aligned entity pair, (ne1,ne2) is negative sample
        np.random.shuffle(train_pair_indexs)
        np.random.shuffle(train_pair_indexs)
        np.random.shuffle(train_pair_indexs)
        batch_num = int(np.ceil(len(train_pair_indexs) * 1.0 / self.batch_size))
        return train_pair_indexs, batch_num
    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_count < self.batch_num:
            batch_index = self.iter_count
            self.iter_count += 1
            batch_ids = self.train_pair_indexs[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
            pe1=[pe1 for pe1, pe2, ne1, ne2 in batch_ids]
            pe2 = [pe2 for pe1, pe2, ne1, ne2 in batch_ids]
            ne1 = [ne1 for pe1, pe2, ne1, ne2 in batch_ids]
            ne2 = [ne2 for pe1, pe2, ne1, ne2 in batch_ids]
            pn1 =[self.all_neighbor[p] for p in pe1 ]
            pn2 = [self.all_neighbor[p] for p in pe2 ]
            nn1 = [self.all_neighbor[p] for p in ne1 ]
            nn2 = [self.all_neighbor[p] for p in ne2 ]

            return pe1,pe2,ne1,ne2,pn1,pn2,nn1,nn2
        else:
            self.iter_count = 0
            self.train_pair_indexs, self.batch_num = self.train_pair_index_gene()
            raise StopIteration()
class Train_index_generator(object):
    def __init__(self, train_ill, train_candidate, entpair2f_idx, neg_num, batch_size):
        self.train_ill = train_ill
        self.train_candidate = copy.deepcopy(train_candidate)
        self.entpair2f_idx = entpair2f_idx
        self.iter_count = 0
        self.batch_size = batch_size
        self.neg_num = neg_num
        print("In Train_batch_index_generator, train_ILL num : {}".format(len(self.train_ill)))
        print("In Train_batch_index_generator, Batch size: {}".format(self.batch_size))
        print("In Train_batch_index_generator, Negative sampling num: {}".format(self.neg_num))
        for e in self.train_candidate.keys():
            self.train_candidate[e] = np.array(self.train_candidate[e])
        self.train_pair_indexs, self.batch_num = self.train_pair_index_gene()

    def train_pair_index_gene(self):
        """
        generate training data (entity_index).
        """
        train_pair_indexs = []
        for pe1, pe2 in self.train_ill:
            neg_indexs = np.random.randint(len(self.train_candidate[pe1]),size=self.neg_num)
            ne2_list = self.train_candidate[pe1][neg_indexs].tolist()
            for ne2 in ne2_list:
                if ne2 == pe2:
                    continue
                ne1 = pe1
                train_pair_indexs.append((pe1, pe2, ne1, ne2))
                #(pe1,pe2) is aligned entity pair, (ne1,ne2) is negative sample
        np.random.shuffle(train_pair_indexs)
        np.random.shuffle(train_pair_indexs)
        np.random.shuffle(train_pair_indexs)
        batch_num = int(np.ceil(len(train_pair_indexs) * 1.0 / self.batch_size))
        return train_pair_indexs, batch_num

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_count < self.batch_num:
            batch_index = self.iter_count
            self.iter_count += 1
            batch_ids = self.train_pair_indexs[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
            pos_pairs = [(pe1, pe2) for pe1, pe2, ne1, ne2 in batch_ids]
            neg_pairs = [(ne1, ne2) for pe1, pe2, ne1, ne2 in batch_ids]

            pos_f_ids = [self.entpair2f_idx[pair_id] for pair_id in pos_pairs]
            neg_f_ids = [self.entpair2f_idx[pair_id] for pair_id in neg_pairs]
            return pos_f_ids, neg_f_ids
        else:
            self.iter_count = 0
            self.train_pair_indexs, self.batch_num = self.train_pair_index_gene()
            raise StopIteration()
def get_cross_predata(ILL,k):
    np.random.shuffle(ILL)
    t = len(ILL)  # 4500
    L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
    neg_left = L.reshape((t * k,))
    neg_left = torch.LongTensor(neg_left)
    L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
    neg2_right = L.reshape((t * k,))
    neg2_right = torch.LongTensor(neg2_right)
    return ILL, neg_left, neg2_right
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
def train_n_epoch_cross(args,model,optimizer,train_ill,valid_ill,test_ill,kg1,kg2,ent_ids_1,ent_ids_2):
    print("start training...")
    train_ills=np.array(train_ill)
    np.random.shuffle(train_ills)
    metrics_input = {'train_ill':train_ills,'kg1':kg1,'kg2':kg2,'ent_ids_1':ent_ids_1,'ent_ids_2':ent_ids_2}
    best_hit = 0
    update_hit=0
    fg = 0
    torch.save(model.state_dict(), args.save_model_path)
    for epoch in range(args.epochs):
        if epoch % args.eval_freq == 0:
            print("######################## epoch: ", epoch, " ###########################")
        model.train()
        np.random.shuffle(metrics_input['kg1'])
        np.random.shuffle(metrics_input['kg2'])
        metrics_input['epoch'] =epoch
        torch.cuda.empty_cache()
        train_1epoch_cross(args,model,  optimizer, metrics_input)
        torch.cuda.empty_cache()
        if epoch % args.eval_freq == 0:
            with torch.no_grad():
                model.eval()
                res,manifolds = model.encode()
                print('Valid Entity Alignment:',end='')
                mrr = get_hits(res, manifolds,valid_ill,args.stop_mrr ,args.device,[1])
                #print('Test Entity Alignment:', end='')
                #get_hits(res, manifolds, test_ill, args.stop_mrr, args.device, [1, 10, 50])
                if mrr > best_hit:
                    fg = 0
                    best_hit = mrr
                    print('Test Entity Alignment:', end='')
                    get_hits(res,manifolds ,test_ill, args.stop_mrr,args.device, [1, 10, 50])
                    torch.save(model.state_dict(), args.save_model_path)
                else:
                    fg+=1
                    if fg>args.patience:
                        break
                '''
                if args.model=='TestM' and epoch%10==9 and mrr>update_hit:
                    update_hit=mrr
                    train2candidates,valid2candidates,test2candidates=update_candidates(res, manifolds, args, train_ill,valid_ill,test_ill)
                    model.update_attribute(args,train2candidates,valid2candidates,test2candidates,train_ill)
                '''

def train_1epoch_cross(args,model,optimizer,metrics_input):
    t = metrics_input['train_ill'].shape[0]
    t1=len(metrics_input['kg1'])
    t2=len(metrics_input['kg2'])
    batch_num= args.b_n
    for i in range(batch_num):
        beg = int(t / batch_num * i)
        if i == batch_num - 1:
            end = t
        else:
            end = int(t / batch_num * (i + 1))
        beg1 = int(t1 / batch_num * i)
        if i == batch_num - 1:
            end1 = t1
        else:
            end1 = int(t1 / batch_num * (i + 1))
        beg2 = int(t2 / batch_num * i)
        if i == batch_num - 1:
            end2 = t2
        else:
            end2 = int(t2 / batch_num * (i + 1))
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        res,manifolds = model.encode()
        metrics_input['index']=(beg,end)
        metrics_input['index1'] = (beg1, end1)
        metrics_input['index2'] = (beg2, end2)
        loss = model.loss_func(res,manifolds, metrics_input)
        loss.backward()
        optimizer.step()
def n_step_train(args,model,optimizer,train_ill,valid_ill,test_ill):
    print("start training...")

    Train_gene = Train_index_generator(train_ill, model.train_candidates, model.entpair2f_idx, neg_num=args.k,
                                       batch_size=args.batch_size)
    best_hit = 0
    fg = 0
    res, manifolds = model.encode()
    for epoch in range(args.epochs):
        if epoch % args.eval_freq == 0:
            print("######################## epoch: ", epoch, " ###########################")
        model.train()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        one_step_train(model, res,optimizer,Train_gene,manifolds)
        torch.cuda.empty_cache()
        if epoch % args.eval_freq == 0:
            with torch.no_grad():
                model.eval()
                res,manifolds = model.encode()
                print('Valid Entity Alignment:',end='')
                mrr =get_hits_b(res,model, model.valid_candidates, valid_ill, model.entpair2f_idx, args.batch_size, args.device, args.stop_mrr,[1])
                if mrr > best_hit:
                    fg = 0
                    best_hit = mrr
                    print('Test Entity Alignment:', end='')
                    get_hits_b(res,model, model.test_candidates, test_ill, model.entpair2f_idx, args.batch_size, args.device, args.stop_mrr,[1,10,50])
                    torch.save(model.state_dict(), args.save_model_path)
                else:
                    fg+=1
                    if fg>args.patience:
                        break
def one_step_train(Model, res,Optimizer,Train_gene,manifolds):
    epoch_loss = 0
    for pos_f_ids, neg_f_ids in Train_gene:
        Optimizer.zero_grad()
        metrics_input = {}
        metrics_input['pos_f_ids']=pos_f_ids
        metrics_input['neg_f_ids'] = neg_f_ids
        loss=Model.loss_func(res,manifolds, metrics_input)
        loss.backward()
        Optimizer.step()
    return epoch_loss

def n_step_train1(args,model,optimizer,train_ill,valid_ill,test_ill,all_neb):
    print("start training...")

    Train_gene = RelationDataset(train_ill, model.train_candidates, all_neb, neg_num=args.k,
                                       batch_size=args.batch_size)
    best_hit = 0
    fg = 0
    res, manifolds = model.encode()
    for epoch in range(args.epochs):
        if epoch % args.eval_freq == 0:
            print("######################## epoch: ", epoch, " ###########################")
        model.train()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        one_step_train1(model, res,optimizer,Train_gene,manifolds,args.device)
        torch.cuda.empty_cache()
        if epoch % args.eval_freq == 0:
            with torch.no_grad():
                model.eval()
                res,manifolds = model.encode()
                #print('train Entity Alignment:',end='')
                #get_hits_g1(res,model,  train_ill,all_neb, args.batch_size*3,args.stop_mrr,args.device,[1])
                print('Valid Entity Alignment:',end='')
                mrr =get_hits_g1(res,model,  valid_ill,all_neb, args.batch_size*3,args.stop_mrr,args.device,[1])
                if mrr > best_hit:
                    fg = 0
                    best_hit = mrr
                    #print('Test Entity Alignment:', end='')
                    #get_hits_g1(res,model,test_ill, all_neb, args.batch_size*4,args.stop_mrr,args.device, [1,10,50])
                    torch.save(model.state_dict(), args.save_model_path)
                else:
                    fg+=1
                    if fg>args.patience:
                        break
def one_step_train1(Model, res,Optimizer,Train_gene,manifolds,device):
    epoch_loss = 0
    for pe1,pe2,ne1,ne2,pn1,pn2,nn1,nn2 in Train_gene:
        Optimizer.zero_grad()
        metrics_input = {}
        metrics_input['p_score']=Model.encoder1((torch.tensor(pn1).to(device),torch.tensor(pe1).to(device),torch.tensor(pn2).to(device),torch.tensor(pe2).to(device)))
        metrics_input['n_score'] = Model.encoder1((torch.tensor(nn1).to(device), torch.tensor(ne1).to(device), torch.tensor(nn2).to(device), torch.tensor(ne2).to(device)))
        loss=Model.loss_func(res,manifolds, metrics_input)
        loss.backward()
        Optimizer.step()
    return epoch_loss