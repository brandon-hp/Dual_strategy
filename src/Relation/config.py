import argparse
import sys
sys.path.append('..')
from utils.train_utils import add_flags_from_config

config_args_MRAEA = {
    'training_config': {
        'lr': (0.001, 'learning rate'), #########
        'dropout': (0.1, 'dropout probability'), #原本是0.1
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(30000,'entity train num'),
        'weight-decay': (0.00, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (5, 'patience for early stopping'),
        'eval-freq': (3, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': { 
        'model': ('MRAEA', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'dim': (300, 'layer dimension'), ##############
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'), ##################
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'), ###################
        'num-layers': (2, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n_heads': (2, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim':(300,''),
        'direction': (True, ''),
        'swap': (False, ''),
        'multi_hop':('concat','concat,hgate'),
        'ent_vec_path':('init_entity_embedding_bert_int_des.p',''),
        'rel_vec_path':('init_relation_embedding_bert_int_des.p',''),
        'use-att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local-agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'b_n': (1, ''),
        'use_ra':(False,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
        'k':(125,'number of negative samples for each positive one'),
        'normalize_feats': (True, 'whether to normalize input node features'),
        'normalize_adj': (True, 'whether to row-normalize the adjacency matrix'),
        'random_ill': (False, 'True or False'),
        'train_ill_rate': (0.2, ''),
        'valid_ill_rate': (0.1, ''),
        'stop_mrr': (True, '')
    }
}
config_args_GAT = {
    'training_config': {
        'lr': (0.001, 'learning rate'), #########
        'dropout': (0.3, 'dropout probability'), #原本是0.1
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(30000,'entity train num'),
        'weight-decay': (0.00, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (5, 'patience for early stopping'),
        'eval-freq': (5, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('GAT', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'dim': (300, 'layer dimension'), ##############
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'), ##################
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'), ###################
        'num-layers': (2, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n_heads': (2, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim':(300,''),
        'direction': (True, ''),
        'swap': (False, ''),
        'multi_hop':('hgate','concat,hgate'),
        'ent_vec_path':('random',''),
        'rel_vec_path':('random',''),
        'use-att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local-agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'b_n': (1, ''),
        'use_ra':(False,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
        'k':(125,'number of negative samples for each positive one'),
        'normalize_feats': (True, 'whether to normalize input node features'),
        'normalize_adj': (True, 'whether to row-normalize the adjacency matrix'),
        'random_ill': (False, 'True or False'),
        'train_ill_rate': (0.2, ''),
        'valid_ill_rate': (0.1, ''),
        'candidate_num': (50, ''),
        'stop_mrr': (True, '')
    }
}
config_args_GAT_literal = {
    'training_config': {
        'lr': (0.001, 'learning rate'), #########
        'dropout': (0.1, 'dropout probability'), #原本是0.1
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(30000,'entity train num'),
        'weight-decay': (0.00, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (30, 'patience for early stopping'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('GAT', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'direction': (True, ''),
        'swap': (False, ''),
        'multi_hop':('hgate','concat,hgate'),
        'ent_vec_path':('init_entity_embedding_word-based.p',''),
        'rel_vec_path':('init_relation_embedding_word-based.p',''),
        'use-att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local-agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'b_n': (1, ''),
        'use_ra':(False,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
        'k':(125,'number of negative samples for each positive one'),
        'normalize_feats': (True, 'whether to normalize input node features'),
        'normalize_adj': (True, 'whether to row-normalize the adjacency matrix'),
        'random_ill': (False, 'True or False'),
        'train_ill_rate': (0.2, ''),
        'valid_ill_rate': (0.1, ''),
        'candidate_num': (50, ''),
        'stop_mrr': (True, '')
    }
}
config_args_MRAEA_literal = {
    'training_config': {
        'lr': (0.001, 'learning rate'), #########
        'dropout': (0.1, 'dropout probability'), #原本是0.1
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(30000,'entity train num'),
        'weight-decay': (0.00, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (30, 'patience for early stopping'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('MRAEA', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'direction': (True, ''),
        'swap': (False, ''),
        'multi_hop':('concat','concat,hgate'),
        'ent_vec_path':('init_entity_embedding_word-based.p',''),
        'rel_vec_path':('init_relation_embedding_word-based.p',''),
        'use-att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local-agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'b_n': (1, ''),
        'use_ra':(False,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
        'k':(125,'number of negative samples for each positive one'),
        'normalize_feats': (True, 'whether to normalize input node features'),
        'normalize_adj': (True, 'whether to row-normalize the adjacency matrix'),
        'random_ill': (False, 'True or False'),
        'train_ill_rate': (0.2, ''),
        'valid_ill_rate': (0.1, ''),
        'candidate_num': (50, ''),
        'stop_mrr': (True, '')
    }
}
config_args_MRAEA_adapt = {
    'training_config': {
        'lr': (0.001, 'learning rate'), #########
        'dropout': (0.1, 'dropout probability'), #原本是0.1
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(30000,'entity train num'),
        'weight-decay': (0.00, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (5, 'patience for early stopping'),
        'eval-freq': (5, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('MRAEA_adapt', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'direction': (True, ''),
        'swap': (False, ''),
        'multi_hop':('concat','concat,hgate'),
        'ent_vec_path':('init_entity_embedding_word-based.p',''),
        'rel_vec_path':('init_relation_embedding_word-based.p',''),
        'use-att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local-agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'b_n': (1, ''),
        'use_ra':(False,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
        'k':(125,'number of negative samples for each positive one'),
        'normalize_feats': (True, 'whether to normalize input node features'),
        'normalize_adj': (True, 'whether to row-normalize the adjacency matrix'),
        'random_ill': (False, 'True or False'),
        'train_ill_rate': (0.2, ''),
        'valid_ill_rate': (0.1, ''),
        'stop_mrr': (True, '')
    }
}
config_args_MRAEA_adapt_r = {
    'training_config': {
        'lr': (0.001, 'learning rate'), #########
        'dropout': (0.1, 'dropout probability'), #原本是0.1
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(30000,'entity train num'),
        'weight-decay': (0.00, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (30, 'patience for early stopping'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('MRAEA_adapt', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'direction': (True, ''),
        'swap': (False, ''),
        'multi_hop':('concat','concat,hgate'),
        'ent_vec_path':('init_entity_embedding_bert_int_des.p',''),
        'rel_vec_path':('init_relation_embedding_bert_int_des.p',''),
        'use-att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local-agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'b_n': (1, ''),
        'use_ra':(True,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
        'k':(125,'number of negative samples for each positive one'),
        'normalize_feats': (True, 'whether to normalize input node features'),
        'normalize_adj': (True, 'whether to row-normalize the adjacency matrix'),
        'random_ill': (False, 'True or False'),
        'train_ill_rate': (0.2, ''),
        'valid_ill_rate': (0.1, ''),
        'stop_mrr': (True, '')
    }
}
config_args_DUAL = {
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
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (10, 'patience for early stopping'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('DUAL', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'dim': (300, 'layer dimension'), ##############
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'), ##################
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'), ###################
        'num-layers': (1, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n_heads': (2, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim':(300,''),
        'direction': (True, ''),
        'swap': (False, ''),
        'multi_hop':('concat','concat,hgate'),
        'ent_vec_path':('init_entity_embedding_bert_int_des.p',''),
        'rel_vec_path':('init_relation_embedding_bert_int_des.p',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra': (False, ''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
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
config_args_DUAL_literal = {
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
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (30, 'patience for early stopping'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('DUAL', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'direction': (True, ''),
        'swap': (False, ''),
        'multi_hop':('concat','concat,hgate'),
        'ent_vec_path':('init_entity_embedding_word-based.p',''),
        'rel_vec_path':('init_relation_embedding_word-based.p',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra':(False,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
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
config_args_DUAL_adapt = {
    'training_config': {
        'lr': (0.001, 'learning rate'), #########
        'dropout': (0.3, 'dropout probability'), #原本是0.1
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(30000,'entity train num'),
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
        'model': ('DUAL_adapt', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'dim': (300, 'layer dimension'), ##############
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'), ##################
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'), ###################
        'num-layers': (2, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n_heads': (2, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim':(300,''),
        'direction': (True, ''),
        'swap': (False, ''),
        'multi_hop':('concat','concat,hgate'),
        'ent_vec_path':('init_entity_embedding_word-based.p',''),
        'rel_vec_path':('init_relation_embedding_word-based.p',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra': (False, ''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
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
config_args_DUAL_adapt_r = {
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
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (30, 'patience for early stopping'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('DUAL_adapt', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'direction': (True, ''),
        'swap': (False, ''),
        'multi_hop':('concat','concat,hgate'),
        'ent_vec_path':('init_entity_embedding_bert_int_des.p',''),
        'rel_vec_path':('init_relation_embedding_bert_int_des.p',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra':(True,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
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
config_args_RREA = {
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
        'bp_param': (0.2, 'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (10, 'patience for early stopping'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('RREA', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'dim': (300, 'layer dimension'), ##############
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'), ##################
        'c': (1, 'hyperbolic radius, set to None for trainable curvature'), ###################
        'num-layers': (2, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n_heads': (2, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim':(300,''),
        'direction': (True, ''),
        'swap':(False,''),
        'multi_hop':('concat','concat,hgate'),
        'ent_vec_path':('init_entity_embedding_bert_int_des.p',''),
        'rel_vec_path':('init_relation_embedding_bert_int_des.p',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra':(False,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
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
        'bp_param': (0.2, 'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (30, 'patience for early stopping'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('RREA', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'direction': (True, ''),
        'swap':(False,''),
        'multi_hop':('concat','concat,hgate'),
        'ent_vec_path':('random',''),
        'rel_vec_path':('random',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra':(False,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
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
config_args_RREA_literal_r = {
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
        'bp_param': (0.2, 'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (30, 'patience for early stopping'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('RREA', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'direction': (True, ''),
        'swap':(False,''),
        'multi_hop':('concat','concat,hgate'),
        'ent_vec_path':('init_entity_embedding_bert_int_des.p',''),
        'rel_vec_path':('init_relation_embedding_bert_int_des.p',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra':(True,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
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
config_args_RREA_adapt = {
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
        'bp_param': (0.2, 'margin based loss'),
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
        'direction': (True, ''),
        'swap':(False,''),
        'multi_hop':('concat','concat,hgate'),
        'ent_vec_path':('random',''),
        'rel_vec_path':('random',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra':(False,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
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
config_args_RREA_adapt_r = {
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
        'bp_param': (0.2, 'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (30, 'patience for early stopping'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
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
        'direction': (True, ''),
        'swap':(False,''),
        'multi_hop':('concat','concat,hgate'),
        'ent_vec_path':('init_entity_embedding_bert_int_des.p',''),
        'rel_vec_path':('init_relation_embedding_bert_int_des.p',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra':(True,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
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
config_args_RREAN_literal = {
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
        'model': ('RREAN_literal', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'ent_vec_path':('init_entity_embedding_bert_int_des.p',''),
        'rel_vec_path':('init_relation_embedding_bert_int_des.p',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra':(False,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
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
config_args_RAGA_adapt = {
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
        'bp_param':(0.2,'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (5, 'patience for early stopping'),
        'eval-freq': (5, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('RAGA_literal', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'swap':(False,''),
        'multi_hop':('concat','concat,hgate'),
        'ent_vec_path':('random',''),
        'rel_vec_path':('random',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra':(False,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
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
config_args_Test = {
    'training_config': {
        'lr': (0.001, 'learning rate'), #########
        'dropout': (0.3, 'dropout probability'), #原本是0.1
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(30000,'entity train num'),
        'weight-decay': (0.00, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'pos_margin':(0.2,'margin based loss'),
        'neg_param':(0.8,'margin based loss'),
        'bp_param':(0.2,'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (5, 'patience for early stopping'),
        'eval-freq': (5, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('Test', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'dim': (300, 'layer dimension'), ##############
        'manifold': ('PoincareBall', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'), ##################
        'c': (1, 'hyperbolic radius, set to None for trainable curvature'), ###################
        'num-layers': (2, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n_heads': (1, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim':(300,''),
        'direction':(True,''),
        'swap':(False,''),
        'multi_hop':('concat','concat,hgate'),
        'ent_vec_path':('random',''),
        'rel_vec_path':('random',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra':(False,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
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
config_args_RAGA = {
    'training_config': {
        'lr': (0.001, 'learning rate'), #########
        'dropout': (0.3, 'dropout probability'), #原本是0.1
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(30000,'entity train num'),
        'weight-decay': (0.00, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'pos_margin':(0.2,'margin based loss'),
        'neg_param':(0.8,'margin based loss'),
        'bp_param':(0.2,'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (5, 'patience for early stopping'),
        'eval-freq': (5, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('RAGA', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'multi_hop':('hgate','concat,hgate'),
        'ent_vec_path':('random',''),
        'rel_vec_path':('random',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra':(False,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
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
config_args_RAGA_literal = {
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
        'bp_param':(0.2,'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (5, 'patience for early stopping'),
        'eval-freq': (5, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('RAGA', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'multi_hop':('hgate','concat,hgate'),
        'ent_vec_path':('init_entity_embedding_word-based.p',''),
        'rel_vec_path':('init_relation_embedding_word-based.p',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra':(False,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
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
config_args_ERMC_literal = {
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
        'bp_param':(1,'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (5, 'patience for early stopping'),
        'eval-freq': (5, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('ERMC', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'multi_hop':('mlp','concat,hgate'),
        'ent_vec_path':('init_entity_embedding_bert_int_des.p',''),
        'rel_vec_path':('init_relation_embedding_bert_int_des.p',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (True, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra':(True,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
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
config_args_ERMC_literal_r = {
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
        'bp_param':(1,'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (30, 'patience for early stopping'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('ERMC', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'ent_vec_path':('init_entity_embedding_bert_int_des.p',''),
        'rel_vec_path':('init_relation_embedding_bert_int_des.p',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (True, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra':(True,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
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
config_args_ERMC = {
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
        'bp_param':(1,'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (5, 'patience for early stopping'),
        'eval-freq': (5, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('ERMC', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'multi_hop':('mlp','concat,hgate'),
        'ent_vec_path':('random',''),
        'rel_vec_path':('random',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (True, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra':(True,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
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
config_args_ERMC_adapt = {
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
        'bp_param':(1,'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (5, 'patience for early stopping'),
        'eval-freq': (5, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('ERMC_adapt', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'ent_vec_path':('random',''),
        'rel_vec_path':('random',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (True, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra':(True,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
        'k':(125,'number of negative samples for each positive one'),
        'transe_k':(20,'number of negative samples for each positive one'),
        'normalize_feats': (True, 'whether to normalize input node features'),
        'normalize_adj': (False, 'whether to row-normalize the adjacency matrix'),
        'random_ill': (False, 'True or False'),
        'train_ill_rate': (0.2, ''),
        'valid_ill_rate': (0.1, ''),
        'stop_mrr': (True, ''),
        'candidate_num': (50, ''),
        'neigh_max': (50, '')
    }
}
config_args_BERT_INT_N = {
    'training_config': {
        'lr': (0.001, 'learning rate'), #########
        'dropout': (0.3, 'dropout probability'), #原本是0.1
        'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (300000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(300000,'entity train num'),
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
        'model': ('BERT_INT_N', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'ent_vec_path': ('init_entity_embedding_SDEA_noN.p', ''),
        'rel_vec_path':('init_relation_embedding_SDEA_noN.p',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (True, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128*5,''),
        'b_n':(1,''),
        'use_ra':(False,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
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
config_args_BootEA_literal = {
    'training_config': {
        'lr': (0.001, 'learning rate'), #########
        'dropout': (0, 'dropout probability'), #原本是0.1
        'cuda': (2, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(30000,'entity train num'),
        'weight-decay': (0.00, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'pos_margin':(0.2,'margin based loss'),
        'neg_param':(0.8,'margin based loss'),
        'bp_param':(1,'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (5, 'patience for early stopping'),
        'eval-freq': (5, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('BootEA', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'loss_t':('mean','mean,sum'),
        'ent_vec_path':('init_entity_embedding_word-based.p',''),
        'rel_vec_path':('init_relation_embedding_word-based.p',''),
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
config_args_BootEA = {
    'training_config': {
        'lr': (0.01, 'learning rate'), #########
        'dropout': (0.0, 'dropout probability'), #原本是0.1
        'cuda': (2, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(30000,'entity train num'),
        'weight-decay': (0.00, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'pos_margin':(0.2,'margin based loss'),
        'neg_param':(0.8,'margin based loss'),
        'bp_param':(100,'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (5, 'patience for early stopping'),
        'eval-freq': (5, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('BootEA', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'loss_t':('sum','mean,sum'),
        'ent_vec_path':('random',''),
        'rel_vec_path':('random',''),
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
config_args_TransEdge = {
    'training_config': {
        'lr': (0.01, 'learning rate'), #########
        'dropout': (0.3, 'dropout probability'), #原本是0.1
        'cuda': (2, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(30000,'entity train num'),
        'weight-decay': (0, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'pos_margin':(0.2,'margin based loss'),
        'neg_param':(0.8,'margin based loss'),
        'bp_param':(100,'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (5, 'patience for early stopping'),
        'eval-freq': (5, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('TransEdge', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'dim': (300, 'layer dimension'), ##############
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'), ##################
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'), ###################
        'num-layers': (1, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('tanh', 'which activation function to use (or None for no activation)'),
        'n_heads': (1, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim':(300,''),
        'direction':(False,''),
        'swap':(False,''),
        'multi_hop':('mlp','concat,hgate'),
        'loss_t':('sum','mean,sum'),
        'ent_vec_path':('random',''),
        'rel_vec_path':('random',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (True, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n':(100,''),
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
        'patience': (10, 'patience for early stopping'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
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
        'ent_vec_path':('init_entity_embedding_SDEA_noN.p',''),
        'rel_vec_path':('init_relation_embedding_SDEA_noN.p',''),
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
config_args_KECG = {
    'training_config': {
        'lr': (0.01, 'learning rate'), #########
        'dropout': (0.3, 'dropout probability'), #原本是0.1
        'cuda': (1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(30000,'entity train num'),
        'weight-decay': (0, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'pos_margin':(0.2,'margin based loss'),
        'neg_param':(0.8,'margin based loss'),
        'bp_param':(100,'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (5, 'patience for early stopping'),
        'eval-freq': (5, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('KECG', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'dim': (300, 'layer dimension'), ##############
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'), ##################
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'), ###################
        'num-layers': (2, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n_heads': (2, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim':(300,''),
        'direction':(True,''),
        'swap': (False, ''),
        'multi_hop':('hgate','concat,hgate'),
        'loss_t': ('sum', 'mean,sum'),
        'ent_vec_path':('random',''),
        'rel_vec_path':('random',''),
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
        'normalize_adj': (False, 'whether to row-normalize the adjacency matrix'),
        'random_ill': (False, 'True or False'),
        'train_ill_rate': (0.2, ''),
        'valid_ill_rate': (0.1, ''),
        'stop_mrr': (True, ''),
        'candidate_num': (50, ''),
        'neigh_max': (50, '')
    }

}
config_args_KECG_literal = {
    'training_config': {
        'lr': (0.001, 'learning rate'), #########
        'dropout': (0.1, 'dropout probability'), #原本是0.1
        'cuda': (1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(30000,'entity train num'),
        'weight-decay': (0, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'pos_margin':(0.2,'margin based loss'),
        'neg_param':(0.8,'margin based loss'),
        'bp_param':(1,'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (30, 'patience for early stopping'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('KECG', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'dim': (300, 'layer dimension'), ##############
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'), ##################
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'), ###################
        'num-layers': (2, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('elu', 'which activation function to use (or None for no activation)'),
        'n_heads': (1, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim':(300,''),
        'direction':(True,''),
        'swap': (False, ''),
        'multi_hop':('hgate','concat,hgate'),
        'loss_t': ('mean', 'mean,sum'),
        'ent_vec_path':('init_entity_embedding_word-based.p',''),
        'rel_vec_path':('init_relation_embedding_word-based.p',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (True, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n':(100,''),
        'use_ra': (False, ''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
        'k':(20,'number of negative samples for each positive one'),
        'transe_k':(20,'number of negative samples for each positive one'),
        'normalize_feats': (True, 'whether to normalize input node features'),
        'normalize_adj': (False, 'whether to row-normalize the adjacency matrix'),
        'random_ill': (False, 'True or False'),
        'train_ill_rate': (0.2, ''),
        'valid_ill_rate': (0.1, ''),
        'stop_mrr': (True, ''),
        'candidate_num': (50, ''),
        'neigh_max': (50, '')
    }

}
config_args_SSP = {
    'training_config': {
        'lr': (0.01, 'learning rate'), #########
        'dropout': (0.3, 'dropout probability'), #原本是0.1
        'cuda': (2, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(30000,'entity train num'),
        'weight-decay': (0.1, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'pos_margin':(0.2,'margin based loss'),
        'neg_param':(0.8,'margin based loss'),
        'bp_param':(100,'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (5, 'patience for early stopping'),
        'eval-freq': (5, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('SSP', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'dim': (300, 'layer dimension'), ##############
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'), ##################
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'), ###################
        'num-layers': (2, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n_heads': (2, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim':(300,''),
        'direction':(True,''),
        'swap':(False,''),
        'multi_hop':('hgate','concat,hgate'),
        'loss_t': ('sum', 'mean,sum'),
        'ent_vec_path':('random',''),
        'rel_vec_path':('random',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
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
        'k':(25,'number of negative samples for each positive one'),
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
config_args_SSP_adapt = {
    'training_config': {
        'lr': (0.01, 'learning rate'), #########
        'dropout': (0.1, 'dropout probability'), #原本是0.1
        'cuda': (2, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(30000,'entity train num'),
        'weight-decay': (0.1, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'pos_margin':(0.2,'margin based loss'),
        'neg_param':(0.8,'margin based loss'),
        'bp_param':(100,'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (5, 'patience for early stopping'),
        'eval-freq': (5, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('SSP', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'dim': (300, 'layer dimension'), ##############
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'), ##################
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'), ###################
        'num-layers': (2, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n_heads': (2, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim':(300,''),
        'direction':(True,''),
        'swap':(False,''),
        'multi_hop':('concat','concat,hgate'),
        'loss_t': ('sum', 'mean,sum'),
        'ent_vec_path':('random',''),
        'rel_vec_path':('random',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
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
        'k':(25,'number of negative samples for each positive one'),
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
config_args_SSP_literal = {
    'training_config': {
        'lr': (0.001, 'learning rate'), #########
        'dropout': (0.1, 'dropout probability'), #原本是0.1
        'cuda': (2, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30000, 'maximum number of epochs to train for'), ###############
        'pre_epochs':(30000,'entity train num'),
        'weight-decay': (0.1, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'pos_margin':(0.2,'margin based loss'),
        'neg_param':(0.8,'margin based loss'),
        'bp_param':(100,'margin based loss'),
        'neg_margin':(2.0,'margin based loss'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (5, 'patience for early stopping'),
        'eval-freq': (5, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('SSP', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'swap':(False,''),
        'multi_hop':('hgate','concat,hgate'),
        'loss_t': ('sum', 'mean,sum'),
        'ent_vec_path':('init_entity_embedding_word-based.p',''),
        'rel_vec_path':('init_relation_embedding_word-based.p',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n':(100,''),
        'use_ra': (False, ''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
        'k':(25,'number of negative samples for each positive one'),
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
config_args_LightEA_literal = {
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
        'patience': (30, 'patience for early stopping'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('LightEA_literal', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'dim': (300, 'layer dimension'), ##############
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'), ##################
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'), ###################
        'num-layers': (2, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('tanh', 'which activation function to use (or None for no activation)'),
        'n_heads': (1, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim':(300,''),
        'direction':(True,''),
        'swap': (False, ''),
        'multi_hop':('concat','concat,hgate'),
        'ent_vec_path':('init_entity_embedding_bert_int_des.p',''),
        'rel_vec_path':('init_relation_embedding_bert_int_des.p',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (True, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra': (False, ''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
        'k':(125,'number of negative samples for each positive one'),
        'transe_k':(20,'number of negative samples for each positive one'),
        'normalize_feats': (True, 'whether to normalize input node features'),
        'normalize_adj': (False, 'whether to row-normalize the adjacency matrix'),
        'random_ill': (False, 'True or False'),
        'train_ill_rate': (0.2, ''),
        'valid_ill_rate': (0.1, ''),
        'stop_mrr': (True, ''),
        'candidate_num': (50, ''),
        'neigh_max': (50, '')
    }
}
config_args_LightEA = {
    'training_config': {
        'lr': (0.001, 'learning rate'), #########
        'dropout': (0.0, 'dropout probability'), #原本是0.1
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
        'patience': (4, 'patience for early stopping'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
    'model_config': {
        'model': ('LightEA_literal', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
        'dim': (1024, 'layer dimension'), ##############
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'), ##################
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'), ###################
        'num-layers': (2, 'number of layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('tanh', 'which activation function to use (or None for no activation)'),
        'n_heads': (1, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'heads_concat': (False, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'feat_dim':(1024,''),
        'direction':(True,''),
        'swap': (False, ''),
        'multi_hop':('concat','concat,hgate'),
        'ent_vec_path':('lightea',''),
        'rel_vec_path':('lightea',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (True, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
        'k':(125,'number of negative samples for each positive one'),
        'transe_k':(20,'number of negative samples for each positive one'),
        'normalize_feats': (True, 'whether to normalize input node features'),
        'normalize_adj': (False, 'whether to row-normalize the adjacency matrix'),
        'random_ill': (False, 'True or False'),
        'train_ill_rate': (0.2, ''),
        'valid_ill_rate': (0.1, ''),
        'stop_mrr': (True, ''),
        'candidate_num': (50, ''),
        'neigh_max': (50, '')
    }
}
config_args_ICLEA_literal = {
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
        'model': ('ICLEA', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'swap': (False, ''),
        'multi_hop':('mlp','concat,hgate'),
        'ent_vec_path':('init_entity_embedding_word-based.p',''),
        'rel_vec_path':('init_relation_embedding_word-based.p',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (True, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra':(False,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
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
config_args_ICLEA = {
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
        'model': ('ICLEA', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'swap': (False, ''),
        'multi_hop':('mlp','concat,hgate'),
        'ent_vec_path':('init_entity_embedding_bert_int_des.p',''),
        'rel_vec_path':('init_relation_embedding_bert_int_des.p',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (True, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra':(False,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
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
config_args_ICLEA_adapt = {
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
        'model': ('ICLEA', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'), #################
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
        'ent_vec_path':('init_entity_embedding_bert_int_des.p',''),
        'rel_vec_path':('init_relation_embedding_bert_int_des.p',''),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_w': (False, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'kernel_num':(21,''),
        'batch_size':(128,''),
        'b_n': (1, ''),
        'use_ra':(False,''),
    },
    'data_config': {
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
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
parser = argparse.ArgumentParser()
for _, config_dict in config_args_RREA.items():
    parser = add_flags_from_config(parser, config_dict)
