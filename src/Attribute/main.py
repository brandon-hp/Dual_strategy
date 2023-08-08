import numpy as np
import torch
import logging
import os
import sys
from config import parser
from load_data import *
from train import train_n_epoch_cross,n_step_train,n_step_train1
from Test import get_hits,get_hits_b,get_hits_init,save_sim,get_entity_pairs,get_hits_g,get_hits_g1,save_sim1

sys.path.append('..')
import optimizers
from model.attribute_model import BaseModel
torch.autograd.set_detect_anomaly(True)


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda' if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    logging.info('Using:{args.device}')
    logging.info("Using seed {}.".format(args.seed))
    args.save_path1='../../' + args.save_path + '/' + args.dataset + '/' + args.lang
    data=read_data(args)
    data["ent_features_n"]=get_e_init_layer(args,args.ent_vec_path_n,data['ent_sizes'],data['train_ill'],args.normalize_feats)
    data["ent_features_a"] = get_e_init_layer(args, args.ent_vec_path_a, data['ent_sizes'], data['train_ill'],
                                              args.normalize_feats)

    #get_hits_init(data["ent_features"], data['test_ill'], args.stop_mrr)
    data["rel_features_n"]=get_r_init_layer(args,args.rel_vec_path_n,data['rel_sizes'],args.normalize_feats)
    data["rel_features_a"] = get_r_init_layer(args, args.rel_vec_path_a, data['rel_sizes'], args.normalize_feats)
    data["val_features"] = get_v_init_layer(args, args.val_vec_path,args.normalize_feats)
    data["att_features"] = get_a_init_layer(args, args.att_vec_path, args.normalize_feats)

    args.save_model_path =args.save_path1 + '/attribute_model_best_' + args.model+'_'+args.vec_type + '.p'

    logging.info("Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0 :
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)
        model = BaseModel(args, data)

        logging.info(str(model))
        model = model.to(args.device)
    else:
        model = BaseModel(args, data)
        logging.info(str(model))
        model = model.to(args.device)
    for n, p in model.named_parameters():
        print(n)

    optimizer_grouped_parameters = [
        # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
        {'params': [p for n, p in model.named_parameters() if 'decoder' in n.strip()],
         'lr': args.lr,'weight_decay':args.weight_decay},
        # Filter for parameters which *do* include those.
        {'params': [p for n, p in model.named_parameters() if'decoder' not in n.strip()],
         'lr': args.lr,'weight_decay':0}
        ]
    optimizer = getattr(optimizers, args.optimizer)(optimizer_grouped_parameters)

    if 'BERT_INT' in args.model:
        '''
        with torch.no_grad():
            res, manifolds = model.encode()
            
            print('Train Entity Alignment:', end='')
            get_hits_b(res, model, model.train_candidates, data['train_ill'], model.entpair2f_idx, args.batch_size, args.device,
                       args.stop_mrr, [1,10,50])

            print('Valid Entity Alignment:', end='')
            get_hits_b(res, model, model.valid_candidates, data['valid_ill'], model.entpair2f_idx, args.batch_size,
                       args.device,
                       args.stop_mrr, [1, 10, 50])
            print('Test Entity Alignment:', end='')
            get_hits_b(res, model, model.test_candidates, data['test_ill'], model.entpair2f_idx, args.batch_size,
                       args.device,
                       args.stop_mrr, [1, 10, 50])
        '''
        n_step_train(args, model, optimizer, data['train_ill'], data['valid_ill'], data['test_ill'])
        model.load_state_dict(torch.load(args.save_model_path))
        model.eval()
        with torch.no_grad():
            res, manifolds = model.encode()
            print('Train Entity Alignment:', end='')
            get_hits_b(res, model, model.train_candidates, data['train_ill'], model.entpair2f_idx, args.batch_size, args.device,
                       args.stop_mrr, [1,10,50])

            print('Valid Entity Alignment:', end='')
            get_hits_b(res, model, model.valid_candidates, data['valid_ill'], model.entpair2f_idx, args.batch_size,
                       args.device,
                       args.stop_mrr, [1, 10, 50])
            print('Test Entity Alignment:', end='')
            get_hits_b(res, model, model.test_candidates, data['test_ill'], model.entpair2f_idx, args.batch_size,
                       args.device,
                       args.stop_mrr, [1, 10, 50])
            pickle.dump(model.encoder.entity_pairs, open(
                '../../' + args.save_path + '/' + args.dataset + '/' + args.lang +'/'+args.entity_pairs_path, "wb"))
            sim=save_sim1(res, model, model.encoder.entity_pairs, model.entpair2f_idx, args.batch_size,
                       args.device,
                       args.stop_mrr, [1, 10, 50])
            pickle.dump(sim, open(
                '../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/'+args.model+'_cosin_sim', "wb"))
    elif 'GRU' in args.model:
        '''
        with torch.no_grad():
            res, manifolds = model.encode()
            print('Train Entity Alignment:', end='')
            get_hits_g1(res, model,  data['train_ill'], data['all_nebs'], args.batch_size*3,
                       args.stop_mrr,args.device, [1,10,50])

            print('Valid Entity Alignment:', end='')
            get_hits_g1(res, model,  data['valid_ill'], data['all_nebs'], args.batch_size*3,
                       args.stop_mrr,args.device, [1,10,50])
            print('Test Entity Alignment:', end='')
            get_hits_g1(res, model,  data['test_ill'], data['all_nebs'], args.batch_size*3,
                       args.stop_mrr,args.device, [1,10,50])
        '''

        n_step_train1(args, model, optimizer, data['train_ill'], data['valid_ill'], data['test_ill'],data['all_nebs'])
        model.load_state_dict(torch.load(args.save_model_path))
        model.eval()
        with torch.no_grad():
            res, manifolds = model.encode()
            print('Train Entity Alignment:', end='')
            get_hits_g1(res, model,  data['train_ill'], data['all_nebs'], args.batch_size*3,
                       args.stop_mrr,args.device, [1,10,50])

            print('Valid Entity Alignment:', end='')
            get_hits_g1(res, model,  data['valid_ill'], data['all_nebs'], args.batch_size*3,
                       args.stop_mrr,args.device, [1,10,50])
            print('Test Entity Alignment:', end='')
            get_hits_g1(res, model,  data['test_ill'], data['all_nebs'], args.batch_size*3,
                       args.stop_mrr,args.device, [1,10,50])
    else:

        with torch.no_grad():
            model.eval()
            res, manifolds = model.encode()
            #print('Train Entity Alignment:', end='')
            #get_hits(res,manifolds ,data['train_ill'], args.stop_mrr,args.device, [1, 10, 50])
            #print('Valid Entity Alignment:', end='')
            #get_hits(res, manifolds,data['valid_ill'], args.stop_mrr,args.device, [1, 10, 50])
            print('Test Entity Alignment:', end='')
            get_hits(res,manifolds ,data['test_ill'], args.stop_mrr,args.device, [1, 10, 50])

        train_n_epoch_cross(args, model, optimizer, data['train_ill'], data['valid_ill'], data['test_ill'],data['kg1'],data['kg2'],data['ent_ids_1'],data['ent_ids_2'])
        model.load_state_dict(torch.load(args.save_model_path))
        model.eval()
        with torch.no_grad():
            res, manifolds = model.encode()
            '''
            print('Train Entity Alignment:', end='')
            get_hits(res,manifolds ,data['train_ill'], args.stop_mrr, args.device,[1, 10, 50])
            print('Valid Entity Alignment:', end='')
            get_hits(res, manifolds,data['valid_ill'], args.stop_mrr,args.device, [1, 10, 50])
            print('Test Entity Alignment:', end='')
            get_hits(res,manifolds ,data['test_ill'], args.stop_mrr, args.device,[1, 10, 50])
            '''
            # if os.path.exists('../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/'+args.entity_pairs_path):
            #     entity_pairs=pickle.load(open(
            #         '../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/'+args.entity_pairs_path, "rb"))
            #
            #     sim=save_sim(res,manifolds ,data['train_ill']+data['valid_ill']+data['test_ill'],entity_pairs, args.device)
            #     pickle.dump(sim, open(
            #         '../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/'+args.model+'_cosin_sim', "wb"))
            # else:
            entity_pairs=get_entity_pairs(res,manifolds, data['train_ill'],data['valid_ill'],data['test_ill'],args.candidate_num,args.device)
            pickle.dump(entity_pairs, open(
                '../../' + args.save_path + '/' + args.dataset + '/' + args.lang +'/'+args.entity_pairs_path, "wb"))
            sim=save_sim(res,manifolds ,data['train_ill']+data['valid_ill']+data['test_ill'],entity_pairs, args.device)
            print((8748, 37647) in sim,(8748, 37647) in entity_pairs)
            pickle.dump(sim, open(
                '../../' + args.save_path + '/' + args.dataset + '/' + args.lang + '/'+args.model+'_cosin_sim', "wb"))


if __name__ == '__main__':
    args = parser.parse_args()
    epoch=3
    '''
    for i in range(epoch):
        if i==0:
            args.lang='zh_en'
        elif i==1:
            args.lang = 'ja_en'
        elif i==2:
            args.lang = 'fr_en'
        train(args)
    
    
    for i in range(epoch):
        if i==0:
            args.lang='zh_en_sp'
        elif i==1:
            args.lang = 'ja_en_sp'
        elif i==2:
            args.lang = 'fr_en_sp'
        train(args)

    for i in range(epoch):
        if i==0:
            args.lang='zh_en_50'
        elif i==1:
            args.lang = 'ja_en_50'
        elif i==2:
            args.lang = 'fr_en_50'
        train(args)
    
    for i in range(epoch):
        if i==0:
            args.lang='zh_en_no'
        elif i==1:
            args.lang = 'ja_en_no'
        elif i==2:
            args.lang = 'fr_en_no'
        train(args)
    '''
    #train(args)
    args.lang = 'ja_en_no'
    train(args)

    '''
    args.lang = 'dbp_yg_dwy'
    train(args)
    args.dataset = 'Openea'
    args.lang = 'D_Y_100K_V1'
    train(args)
    
    args.dataset = 'Openea'
    args.lang = 'EN_FR_100K_V1'
    train(args)
    '''
    '''
    args.dataset='Openea'
    for i in range(3):
        if i==0:
            args.lang='EN_DE_15K_V1'
        elif i==1:
            args.lang = 'EN_FR_15K_V1'
        elif i==2:
            args.lang = 'D_Y_15K_V1'
        elif i==3:
            args.lang = 'D_W_15K_V1'
        train(args)
    '''
    '''
    args.dataset = 'Openea'
    for i in range(2):
        if i == 1:
            args.lang = 'EN_DE_100K_V1'
        elif i == 0:
            args.lang = 'EN_FR_100K_V1'
        train(args)
    '''
    '''
    args.dataset = 'Openea'
    for i in range(1):
        if i == 1:
            args.lang = 'EN_DE_100K_V1'
        elif i == 0:
            args.lang = 'D_W_100K_V1'
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
    args.dataset = 'DWY100K'
    for i in range(2):
        if i == 1:
            args.lang = 'dbp_yg_dwy'
        elif i == 0:
            args.lang = 'dbp_wd_dwy'
        train(args)
    '''
    '''
    args.dataset='Openea'
    for i in range(3):
        if i==0:
            args.lang='EN_DE_15K_V1'
        elif i==1:
            args.lang = 'EN_FR_15K_V1'
        elif i==2:
            args.lang = 'D_Y_15K_V1'
        elif i==3:
            args.lang = 'D_W_15K_V1'
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

    # train(args)
    args.dataset = 'DWY100K'
    for i in range(1):
        if i == 1:
            args.lang = 'dbp_yg_dwy'
        elif i == 0:
            args.lang = 'dbp_wd_dwy'
        train(args)
    '''

