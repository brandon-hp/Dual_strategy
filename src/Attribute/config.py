import argparse
import sys
sys.path.append('..')
from utils.train_utils import add_flags_from_config

config_args_TransEdge_literal = {
    'training_config': {
        'lr': (0.001, 'learning rate'),  #########
        'dropout': (0.1, 'dropout probability'),  # 原本是0.1
        'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30000, 'maximum number of epochs to train for'),  ###############
        'pre_epochs': (30000, 'entity train num'),
        'weight-decay': (0, 'l2 regularization strength'),  # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'pos_margin': (0.2, 'margin based loss'),
        'neg_param': (0.8, 'margin based loss'),
        'bp_param': (100, 'margin based loss'),
        'neg_margin': (2.0, 'margin based loss'),
        'gamma_margin': (3.0, 'margin based loss'),
        'patience': (5, 'patience for early stopping'),
        'eval-freq': (5, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('TransEdge', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'),
        #################
        'dim': (300, 'layer dimension'),  ##############
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        ##################
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'),  ###################
        'num-layers': (1, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('tanh', 'which activation function to use (or None for no activation)'),
        'n_heads': (1, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim': (300, ''),
        'direction': (False, ''),
        'swap': (False, ''),
        'multi_hop': ('mlp', 'concat,hgate'),
        'loss_t': ('sum', 'mean,sum'),
        'ent_vec_path_n':('init_entity_embedding_bert_int_name.p',''),
        'ent_vec_path_a': ('init_entity_embedding_SDEA_noN.p', ''),
        'rel_vec_path_n':('init_relation_embedding_bert_int_des.p',''),
        'rel_vec_path_a': ('init_relation_embedding_SDEA_noN.p', ''),
        'val_vec_path': ('random', ''),
        'att_vec_path': ('random', ''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (True, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n':(100,''),
        'use_ra':(False,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
        'k':(20,'number of negative samples for each positive one'),
        'transe_k':(20,'number of negative samples for each positive one'),
        'normalize_feats': (True, 'whether to normalize input node features'),
        'normalize_adj': (True, 'whether to row-normalize the adjacency matrix'),
        'random_ill': (False, 'True or False'),
        'train_ill_rate': (0.2, ''),
        'valid_ill_rate': (0.1, ''),
        'stop_mrr': (True, ''),
        'candidate_num': (50, ''),
        'neigh_max': (50, '')
    }

}

config_args_RREA_literal = {
    'training_config': {
        'lr': (0.001, 'learning rate'), #########
        'dropout': (0.1, 'dropout probability'), #原本是0.1
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(30000,'entity train num'),
        'weight-decay': (0.00, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'pos_margin':(0.2,'margin based loss'),
        'neg_param':(0.8,'margin based loss'),
        'bp_param': (1, 'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (5, 'patience for early stopping'),
        'eval-freq': (5, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('RREA_adapt', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'dim': (300, 'layer dimension'), ##############
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'), ##################
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'), ###################
        'num-layers': (2, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n_heads': (1, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim':(300,''),
        'direction':(True,''),
        'swap': (False, ''),
        'multi_hop':('concat','concat,hgate'),
        'ent_vec_path_n':('init_entity_embedding_bert_int_name.p',''),
        'ent_vec_path_a': ('init_entity_embedding_SDEA.p', ''),
        'rel_vec_path_n':('init_relation_embedding_bert_int_des.p',''),
        'rel_vec_path_a': ('init_relation_embedding_SDEA.p', ''),
        'val_vec_path': ('random', ''),
        'att_vec_path': ('random', ''),
        'vec_type': ('hybrid_cat', ''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
        'entity_pairs_path': ('bert_int_entity_pairs', '[bert_int_entity_pairs,entity_pairs]'),
        'k':(125,'number of negative samples for each positive one'),
        'transe_k':(20,'number of negative samples for each positive one'),
        'normalize_feats': (True, 'whether to normalize input node features'),
        'normalize_adj': (True, 'whether to row-normalize the adjacency matrix'),
        'random_ill': (False, 'True or False'),
        'train_ill_rate': (0.2, ''),
        'valid_ill_rate': (0.1, ''),
        'stop_mrr': (True, ''),
        'candidate_num': (50, ''),
        'neigh_max': (50, '')
    }
}
config_args_TestM = {
    'training_config': {
        'lr': (0.001, 'learning rate'), #########
        'dropout': (0.1, 'dropout probability'), #原本是0.1
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(30000,'entity train num'),
        'weight-decay': (0.00, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'pos_margin':(0.2,'margin based loss'),
        'neg_param':(0.8,'margin based loss'),
        'bp_param': (1, 'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (5, 'patience for early stopping'),
        'eval-freq': (5, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('TestM', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'dim': (300, 'layer dimension'), ##############
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'), ##################
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'), ###################
        'num-layers': (2, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n_heads': (1, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim':(300,''),
        'direction':(True,''),
        'swap': (False, ''),
        'multi_hop':('concat','concat,hgate'),
        'ent_vec_path_n':('init_entity_embedding_bert_int_des.p',''),
        'ent_vec_path_a': ('init_entity_embedding_SDEA.p', ''),
        'rel_vec_path_n':('init_relation_embedding_bert_int_des.p',''),
        'rel_vec_path_a': ('init_relation_embedding_SDEA.p', ''),
        'val_vec_path': ('init_value_embedding_bert_int_des.p', ''),
        'att_vec_path': ('init_attribute_embedding_bert_int_des.p', ''),
        'vec_type': ('hybrid_cat', ''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
        'entity_pairs_path': ('entity_pairs_h', '[bert_int_entity_pairs,entity_pairs]'),
        'k':(125,'number of negative samples for each positive one'),
        'transe_k':(20,'number of negative samples for each positive one'),
        'normalize_feats': (True, 'whether to normalize input node features'),
        'normalize_adj': (False, 'whether to row-normalize the adjacency matrix'),
        'random_ill': (False, 'True or False'),
        'train_ill_rate': (0.2, ''),
        'valid_ill_rate': (0.1, ''),
        'stop_mrr': (True, ''),
        'candidate_num': (50, ''),
        'neigh_max': (50, ''),
        'val_max': (100, '')
    }
}
config_args_NoS = {
    'training_config': {
        'lr': (0.001, 'learning rate'), #########
        'dropout': (0.1, 'dropout probability'), #原本是0.1
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (0, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(0,'entity train num'),
        'weight-decay': (0.00, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'pos_margin':(0.2,'margin based loss'),
        'neg_param':(0.8,'margin based loss'),
        'bp_param': (1, 'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (15, 'patience for early stopping'),
        'eval-freq': (10, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('NoS', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'dim': (300, 'layer dimension'), ##############
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'), ##################
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'), ###################
        'num-layers': (2, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n_heads': (1, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim':(300,''),
        'direction':(True,''),
        'swap': (False, ''),
        'multi_hop':('concat','concat,hgate'),
        'ent_vec_path_n':('init_entity_embedding_SDEA_noN.p',''),
        'ent_vec_path_a': ('init_entity_embedding_SDEA.p', ''),
        'rel_vec_path_n':('init_relation_embedding_SDEA.p',''),
        'rel_vec_path_a': ('init_relation_embedding_SDEA.p', ''),
        'val_vec_path': ('init_value_embedding_SDEA.p', ''),
        'att_vec_path': ('init_attribute_embedding_SDEA.p', ''),
        'vec_type': ('n', ''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
        'entity_pairs_path': ('entity_pairs_noN', '[bert_int_entity_pairs,entity_pairs]'),
        'k':(125,'number of negative samples for each positive one'),
        'transe_k':(20,'number of negative samples for each positive one'),
        'normalize_feats': (True, 'whether to normalize input node features'),
        'normalize_adj': (False, 'whether to row-normalize the adjacency matrix'),
        'random_ill': (False, 'True or False'),
        'train_ill_rate': (0.2, ''),
        'valid_ill_rate': (0.1, ''),
        'stop_mrr': (True, ''),
        'candidate_num': (50, ''),
        'neigh_max': (50, ''),
        'val_max': (50, '')
    }
}
config_args_BERT_INT_A = {
    'training_config': {
        'lr': (0.001, 'learning rate'), #########
        'dropout': (0.3, 'dropout probability'), #原本是0.1
        'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (10, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(10,'entity train num'),
        'weight-decay': (0.00, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'pos_margin':(0.2,'margin based loss'),
        'neg_param':(0.8,'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (2, 'patience for early stopping'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('BERT_INT_A', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'relation_model': ('TransEdge', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'dim': (300, 'layer dimension'), ##############
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'), ##################
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'), ###################
        'num-layers': (1, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n_heads': (1, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim':(300,''),
        'direction':(False,''),
        'swap':(False,''),
        'multi_hop':('mlp','concat,hgate'),
        'ent_vec_path_n':('init_entity_embedding_SDEA.p',''),
        'ent_vec_path_a': ('init_entity_embedding_SDEA.p', ''),
        'rel_vec_path_n':('init_relation_embedding_SDEA.p',''),
        'rel_vec_path_a': ('init_relation_embedding_SDEA.p', ''),
        'val_vec_path': ('init_value_embedding_SDEA_noN.p', ''),
        'att_vec_path': ('init_attribute_embedding_SDEA_noN.p', ''),
        'vec_type': ('a', ''),
        'sim_reverse': (True, ''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (True, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n':(1,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
        'entity_pairs_path':('bert_int_entity_pairs','[bert_int_entity_pairs,entity_pairs]'),
        'k':(125,'number of negative samples for each positive one'),
        'transe_k':(20,'number of negative samples for each positive one'),
        'normalize_feats': (True, 'whether to normalize input node features'),
        'normalize_adj': (True, 'whether to row-normalize the adjacency matrix'),
        'random_ill': (False, 'True or False'),
        'train_ill_rate': (0.2, ''),
        'valid_ill_rate': (0.1, ''),
        'stop_mrr': (True, ''),
        'candidate_num': (50, ''),
        'val_max': (100, '')
    }
}
config_args_GRU = {
    'training_config': {
        'lr': (0.001, 'learning rate'), #########
        'dropout': (0.1, 'dropout probability'), #原本是0.1
        'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (200, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(2,'entity train num'),
        'weight-decay': (0.00, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'pos_margin':(0.2,'margin based loss'),
        'neg_param':(0.8,'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (5, 'patience for early stopping'),
        'eval-freq': (5, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('GRU', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'relation_model': ('GRU', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'dim': (300, 'layer dimension'), ##############
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'), ##################
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'), ###################
        'num-layers': (1, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n_heads': (1, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim':(300,''),
        'direction':(True,''),
        'swap':(False,''),
        'multi_hop':('mlp','concat,hgate'),
        'ent_vec_path_n':('random',''),
        'ent_vec_path_a': ('random', ''),
        'rel_vec_path_n':('random',''),
        'rel_vec_path_a': ('random', ''),
        'val_vec_path': ('random', ''),
        'att_vec_path': ('random', ''),
        'vec_type': ('a', ''),
        'sim_reverse': (False, ''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (True, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(45,''),
        'b_n':(1,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
        'entity_pairs_path':('entity_pairs_all1','[bert_int_entity_pairs,entity_pairs]'),
        'k':(2,'number of negative samples for each positive one'),
        'transe_k':(20,'number of negative samples for each positive one'),
        'normalize_feats': (True, 'whether to normalize input node features'),
        'normalize_adj': (True, 'whether to row-normalize the adjacency matrix'),
        'random_ill': (False, 'True or False'),
        'train_ill_rate': (0.2, ''),
        'valid_ill_rate': (0.1, ''),
        'stop_mrr': (True, ''),
        'candidate_num': (50, ''),
        'val_max': (100, '')
    }
}
config_args_BERT_INT_test = {
    'training_config': {
        'lr': (0.001, 'learning rate'), #########
        'dropout': (0.3, 'dropout probability'), #原本是0.1
        'cuda': (1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (300000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(300000,'entity train num'),
        'weight-decay': (0.00, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'pos_margin':(0.2,'margin based loss'),
        'neg_param':(0.8,'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (1, 'patience for early stopping'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('BERT_INT_test', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'dim': (300, 'layer dimension'), ##############
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'), ##################
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'), ###################
        'num-layers': (1, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n_heads': (1, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim':(300,''),
        'direction':(False,''),
        'swap':(False,''),
        'multi_hop':('mlp','concat,hgate'),
        'ent_vec_path_n':('init_entity_embedding_bert_int_des.p',''),
        'ent_vec_path_a': ('init_entity_embedding_SDEA_noN.p', ''),
        'rel_vec_path_n':('init_relation_embedding_bert_int_des.p',''),
        'rel_vec_path_a': ('init_relation_embedding_SDEA_noN.p', ''),
        'val_vec_path': ('init_value_embedding_bert_int_des.p', ''),
        'att_vec_path': ('init_attribute_embedding_bert_int_des.p', ''),
        'vec_type': ('a', ''),
        'sim_reverse': (False, ''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (True, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n':(1,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
        'entity_pairs_path': ('entity_pairs_noS4', '[bert_int_entity_pairs,entity_pairs]'),
        'k':(125,'number of negative samples for each positive one'),
        'transe_k':(20,'number of negative samples for each positive one'),
        'normalize_feats': (True, 'whether to normalize input node features'),
        'normalize_adj': (True, 'whether to row-normalize the adjacency matrix'),
        'random_ill': (False, 'True or False'),
        'train_ill_rate': (0.2, ''),
        'valid_ill_rate': (0.1, ''),
        'stop_mrr': (True, ''),
        'candidate_num': (50, ''),
        'neigh_max': (50, ''),
        'val_max': (50, '')
    }
}

config_args_BERT_INT_ALL = {
    'training_config': {
        'lr': (0.01, 'learning rate'), #########
        'dropout': (0.1, 'dropout probability'), #原本是0.1
        'cuda': (3, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (300000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(300000,'entity train num'),
        'weight-decay': (0.00, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'pos_margin':(0.2,'margin based loss'),
        'neg_param':(0.8,'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (3, 'patience for early stopping'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('BERT_INT_ALL', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'dim': (300, 'layer dimension'), ##############
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'), ##################
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'), ###################
        'num-layers': (1, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n_heads': (1, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim':(300,''),
        'direction':(False,''),
        'swap':(False,''),
        'multi_hop':('mlp','concat,hgate'),
        'ent_vec_path_n':('init_entity_embedding_SDEA_noN.p',''),
        'ent_vec_path_a': ('init_entity_embedding_SDEA_noN.p', ''),
        'rel_vec_path_n':('init_relation_embedding_SDEA.p',''),
        'rel_vec_path_a': ('init_relation_embedding_SDEA.p', ''),
        'val_vec_path': ('init_value_embedding_SDEA_noN.p', ''),
        'att_vec_path': ('init_attribute_embedding_SDEA_noN.p', ''),
        'vec_type': ('n', ''),
        'sim_reverse': (False, ''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (True, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n':(1,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
        'entity_pairs_path': ('entity_pairs_noN', '[bert_int_entity_pairs,entity_pairs]'),
        'k':(125,'number of negative samples for each positive one'),
        'transe_k':(20,'number of negative samples for each positive one'),
        'normalize_feats': (True, 'whether to normalize input node features'),
        'normalize_adj': (True, 'whether to row-normalize the adjacency matrix'),
        'random_ill': (False, 'True or False'),
        'train_ill_rate': (0.2, ''),
        'valid_ill_rate': (0.1, ''),
        'stop_mrr': (True, ''),
        'candidate_num': (50, ''),
        'neigh_max': (50, ''),
        'val_max': (50, '')
    }
}
config_args_ROADEA = {
    'training_config': {
        'lr': (0.001, 'learning rate'), #########
        'dropout': (0.1, 'dropout probability'), #原本是0.1
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (300000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(300000,'entity train num'),
        'weight-decay': (0.00, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'pos_margin':(0.2,'margin based loss'),
        'neg_param':(0.8,'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (50, 'patience for early stopping'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('roadEA', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'relation_model': ('RREA_adapt', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'),
        #################
        'dim': (300, 'layer dimension'), ##############
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'), ##################
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'), ###################
        'num-layers': (2, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n_heads': (1, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim':(300,''),
        'direction':(False,''),
        'swap':(False,''),
        'multi_hop':('concat','concat,hgate'),
        'ent_vec_path_n':('init_entity_embedding_bert_int_des.p',''),
        'ent_vec_path_a': ('init_entity_embedding_SDEA.p', ''),
        'rel_vec_path_n':('init_relation_embedding_bert_int_des.p',''),
        'rel_vec_path_a': ('init_relation_embedding_SDEA.p', ''),
        'val_vec_path': ('init_value_embedding_bert_int_des.p', ''),
        'att_vec_path': ('init_attribute_embedding_bert_int_des.p', ''),
        'vec_type': ('n', ''),
        'sim_reverse': (False, ''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n':(1,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
        'entity_pairs_path': ('bert_int_entity_pairs', '[bert_int_entity_pairs,entity_pairs]'),
        'k':(125,'number of negative samples for each positive one'),
        'transe_k':(20,'number of negative samples for each positive one'),
        'normalize_feats': (True, 'whether to normalize input node features'),
        'normalize_adj': (True, 'whether to row-normalize the adjacency matrix'),
        'random_ill': (False, 'True or False'),
        'train_ill_rate': (0.2, ''),
        'valid_ill_rate': (0.1, ''),
        'stop_mrr': (True, ''),
        'candidate_num': (50, ''),
        'neigh_max': (50, ''),
        'val_max': (50, '')
    }
}
parser = argparse.ArgumentParser()
for _, config_dict in config_args_RREA_literal.items():
    parser = add_flags_from_config(parser, config_dict)
