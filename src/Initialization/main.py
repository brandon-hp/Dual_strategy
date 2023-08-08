from __future__ import division
from __future__ import print_function

import pickle

import torch
from load_data import *
import sys
import torch.nn as nn
from config import parser
from data_generator import Batch_TrainData_Generator
from train import train_n_epoch_cross,train_n_epoch_gpt
from Test import *
sys.path.append('..')
from model.Init_model import *
import optimizers

torch.autograd.set_detect_anomaly(True)

def train(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else int(args.patience)
    args.granularity=True if args.model in args.bert_type else False
    ent_ill, train_ill, valid_ill,test_ill,  index2entity,index2V, value2index,index2A=read_data(args)
    if args.data_type=='entity':
        if args.granularity:
            if args.model not in ['bert_all','bert_mix','GPT']:
                args.save_bert_path = '../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/bert_model_best_' + args.model + '.p'
                indx2data=read_bert_input(index2entity, args.bert_path, args.des_max_length)
                Model = Basic_Bert_Unit_model(args.input_dim,args.output_dim,args.bert_path)
                Model.to(args.device)
                Criterion = nn.MarginRankingLoss(args.gamma_margin,size_average=True)
                optimizer = getattr(optimizers, args.optimizer)(params=Model.parameters(),lr=args.lr)

                ent1 = [e1 for e1,e2 in ent_ill]
                ent2 = [e2 for e1,e2 in ent_ill]

                Train_gene = Batch_TrainData_Generator(train_ill, ent1, ent2, indx2data, batch_size=args.train_batch_size, neg_num=args.k)
                train_n_epoch_cross(args,Model,Criterion,optimizer,Train_gene,indx2data,train_ill,valid_ill,test_ill)
                #test(Model, test_ill, indx2data, args.test_batch_size, args.device)
                ent_emb=np.array(get_bert_initemb(args,Model,indx2data))
            elif args.model == 'bert_mix':
                args.save_bert_path = '../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/bert_model_best_' + args.model + '.p'
                indx2data=read_bert_input_mix(index2entity, args.bert_path, args.des_max_length)
                Model = Basic_Bert_Unit_model_mix(args.input_dim,args.output_dim,args.bert_path)
                Model.to(args.device)
                Criterion = nn.MarginRankingLoss(args.gamma_margin,size_average=True)
                optimizer = getattr(optimizers, args.optimizer)(params=Model.parameters(),lr=args.lr)

                ent1 = [e1 for e1,e2 in ent_ill]
                ent2 = [e2 for e1,e2 in ent_ill]

                Train_gene = Batch_TrainData_Generator(train_ill, ent1, ent2, indx2data, batch_size=args.train_batch_size, neg_num=args.k)
                train_n_epoch_cross(args,Model,Criterion,optimizer,Train_gene,indx2data,train_ill,valid_ill,test_ill)
                #test(Model, test_ill, indx2data, args.test_batch_size, args.device)
                ent_emb=np.array(get_bert_initemb(args,Model,indx2data))
            elif args.model == 'GPT':
                args.save_gpt_path = '../../data/'+ args.dataset + '/' + args.lang+ '/'+args.gpt_emb_path
                args.save_bert_path = '../../data/' + args.dataset + '/' + args.lang + '/mlp'
                chat_vec=read_gpt_input(args.save_gpt_path)#read_bert_input_mix(index2entity, args.bert_path, args.des_max_length)
                Model = GPT(args.input_dim,args.output_dim,chat_vec)
                Model.to(args.device)
                Criterion = nn.MarginRankingLoss(args.gamma_margin,size_average=True)
                optimizer = getattr(optimizers, args.optimizer)(params=Model.parameters(),lr=args.lr)

                ent1 = [e1 for e1,e2 in ent_ill]
                ent2 = [e2 for e1,e2 in ent_ill]

                Train_gene = Batch_TrainData_Generator(train_ill, ent1, ent2, ent1, batch_size=args.train_batch_size, neg_num=args.k)
                train_n_epoch_gpt(args,Model,Criterion,optimizer,Train_gene,train_ill,valid_ill,test_ill)
                #test(Model, test_ill, indx2data, args.test_batch_size, args.device)
                ent_emb=np.array(Model.embedding.weight.detach())
            else:
                prefix = '../../' + args.save_path + '/' + args.dataset + '/' + args.lang
                if not os.path.exists(prefix+ '/init_entity_embedding_bert_int_des.p'):
                    if not os.path.exists(prefix+ '/init_entity_embedding_bert_int_name.p'):
                        args.model='bert_int_name'
                        train(args)
                        ent_emb1 = pickle.load(open(prefix + '/init_entity_embedding_bert_int_name.p', "rb"))
                    else:
                        ent_emb1 = pickle.load(open(prefix + '/init_entity_embedding_bert_int_name.p', "rb"))
                else:
                    ent_emb1 = pickle.load(open(prefix+ '/init_entity_embedding_bert_int_des.p', "rb"))
                if not os.path.exists(prefix+ '/init_entity_embedding_SDEA.p'):
                    args.model='SDEA'
                    train(args)
                ent_emb2 = pickle.load(open(prefix + '/init_entity_embedding_SDEA.p', "rb"))
                ent_emb=(ent_emb2+ent_emb1)/2
                args.model = 'bert_all'
        else:
            start_time = time.time()
            Model=Basic_Word_model(args.word_emb_path,index2entity)
            ent_emb=Model.name_embeds
            #word_ids = Model.word_ids
            #check_test(ent_emb, word_ids, index2entity, ent_ill, args.test_batch_size, args.device)

            print("get entity embedding using time {:.3f}".format(time.time() - start_time))
        print("entity embedding shape: ", ent_emb.shape)
        pickle.dump(ent_emb, open('../../'+args.save_path + '/'+args.dataset+'/'+args.lang + '/init_entity_embedding_'+args.model+'.p', "wb"))
        test_datasets(ent_emb,train_ill,args.test_batch_size, args.device,'train_ill')
        test_datasets(ent_emb, valid_ill,args.test_batch_size, args.device,'valid_ill')
        test_datasets(ent_emb, test_ill,args.test_batch_size, args.device,'test_ill')
    elif args.data_type=='relation':
        if args.granularity:
            args.save_bert_path = '../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/bert_model_best_' + args.model + '.p'
            indx2data = read_bert_input(index2entity, args.bert_path, args.des_max_length)
            Model = Basic_Bert_Unit_model(args.input_dim, args.output_dim, args.bert_path)
            Model.to(args.device)
            relation_emb = np.array(get_bert_initemb(args, Model, indx2data))
            pickle.dump(relation_emb, open(
                '../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/init_relation_embedding_' + args.model + '.p',
                "wb"))
        else:
            Model=Basic_Word_model(args.word_emb_path,index2entity)
            relation_emb=Model.name_embeds
            pickle.dump(relation_emb, open(
                '../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/init_relation_embedding_' + args.model + '.p',
                "wb"))
    elif args.data_type == 'attribute':

        args.save_bert_path = '../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/bert_model_best_' + args.model + '.p'
        indx2data = read_bert_input(index2A, args.bert_path, 64)
        Model = Basic_Bert_Unit_model(args.input_dim, args.output_dim, args.bert_path)
        Model.to(args.device)
        attribute_emb = np.array(get_bert_initemb(args, Model, indx2data))
        pickle.dump(attribute_emb, open(
            '../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/init_attribute_embedding_' + args.model + '.p',
            "wb"))

        indx2data = read_bert_input(index2V, args.bert_path, 64)
        Model = Basic_Bert_Unit_model(args.input_dim, args.output_dim, args.bert_path)
        Model.to(args.device)
        value_emb = np.array(get_bert_initemb(args, Model, indx2data))
        pickle.dump(value_emb, open(
            '../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/init_value_embedding_' + args.model + '.p',
            "wb"))
        print(len(value_emb))
        att_datas={}
        for eid, vs in index2entity.items():
            vids=[]
            for v in vs:
                vids.append(value2index[v])
            att_datas[eid]=vids
        pickle.dump(att_datas, open(
            '../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/att_datas',
            "wb"))

if __name__ == '__main__':
    args = parser.parse_args()
    epoch=0
    '''
    for i in range(epoch):
        if i == 0:
            args.lang = 'zh_en'
        elif i == 1:
            args.lang = 'ja_en'
        elif i == 2:
            args.lang = 'fr_en'
        train(args)
    for i in range(epoch):
        if i == 0:
            args.lang = 'zh_en_sp'
        elif i == 1:
            args.lang = 'ja_en_sp'
        elif i == 2:
            args.lang = 'fr_en_sp'
        train(args)
    for i in range(epoch):
        if i == 0:
            args.lang = 'zh_en_50'
        elif i == 1:
            args.lang = 'ja_en_50'
        elif i == 2:
            args.lang = 'fr_en_50'
        train(args)
    for i in range(3):
        if i == 0:
            args.lang = 'zh_en_no'
        elif i == 1:
            args.lang = 'ja_en_no'
        elif i == 2:
            args.lang = 'fr_en_no'
        train(args)
    #train(args)
    
    for i in range(3):
        if i==0:
            args.lang='zh_en'
        elif i==1:
            args.lang = 'ja_en'
        elif i==2:
            args.lang = 'fr_en'
        train(args)
    '''
    '''
    args.dataset='SRPRS'
    for i in range(4):
        if i==0:
            args.lang='en_de_15k_V1'
        elif i==1:
            args.lang = 'en_fr_15k_V1'
        elif i==2:
            args.lang = 'dbp_yg_15k_V1'
        elif i==3:
            args.lang = 'dbp_wd_15k_V1'
        train(args)
    '''
    args.dataset = 'Openea'
    args.lang = 'D_W_15K_V1'
    train(args)
    '''
    args.dataset='Openea'
    for i in range(8):
        if i==0:
            args.lang='EN_DE_15K_V1'
        elif i==1:
            args.lang = 'EN_FR_15K_V1'
        elif i==2:
            args.lang = 'D_W_15K_V1'
        elif i==3:
            args.lang = 'D_Y_15K_V1'
        elif i==4:
            args.lang = 'EN_DE_100K_V1'
        elif i==5:
            args.lang = 'EN_FR_100K_V1'
        elif i==6:
            args.lang = 'D_W_100K_V1'
        elif i==7:
            args.lang = 'D_Y_100K_V1'
        train(args)
    '''
    '''
    #train(args)
    args.dataset='DWY100K'
    for i in range(2):
        if i==1:
            args.lang='dbp_yg_dwy'
        elif i==0:
            args.lang = 'dbp_wd_dwy'
        train(args)
    '''
