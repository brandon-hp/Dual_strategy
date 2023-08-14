import argparse
import sys
sys.path.append('..')
from utils.train_utils import add_flags_from_config
config_args = {
    'training_config': {
        'lr': (1e-5, 'learning rate'), #########
        'dropout': (0.1, 'dropout probability'), #原本是0.1
        'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (30, 'maximum number of epochs to train for'), ###############
        'weight-decay': (0.00, 'l2 regularization strength'), # 原本是0.
        'optimizer': ('AdamW', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'seed': (1234, 'seed for training'),
        'gamma': (0.1, 'gamma for lr scheduler'),
        'gamma_margin':(3.0,'margin based loss'),
        'patience': (3, 'patience for early stopping'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),

    },
    'model_config': { 
        'model': ('bert_int_des', 'orign,SDEA_numb,SDEA_noN,word-based,bert_int_name,bert_int_des,SDEA,bert_all,bert_mix'), #################
        'bert_type':(['GPT','SDEA_noN','SDEA_numb','bert_int_des','SDEA','bert_all','bert_mix','bert_int_name'],''),
        'is_load_BERT': (False, ''),
        'input_dim': (768, 'embedding dimension,768,1536'), ##############MODEL_INPUT_DIM  = 768MODEL_OUTPUT_DIM = 300
        'output_dim':(300,''),
        'bert_path':('../../bert-base/bert-base-multilingual-cased','bert-base-multilingual-cased'),
        'data_type': ('entity', 'relation,entity,attribute'),

    },
    'data_config': {
        'save_path': ('save_data', 'which dataset to use,ea use DBP15k'),
        'word_emb_path': ('../../data/wiki-news-300d-1M.vec', 'wiki-news-300d-1M.vec'),
        'gpt_emb_path': ('chatgpt_vectorList.pl', 'wiki-news-300d-1M.vec'),
        'dataset': ('DBP15k', 'which dataset to use,ea use DBP15k'),
        'lang':('zh_en','[ja_en,ja_en,fr_en]'), #############
        'k':(3,'number of negative samples for each positive one'),
        'random_ill':(False,'True or False'),
        'train_ill_rate':(0.2,''),
        'valid_ill_rate':(0.1,''),
        'des_max_length':(128,''),
        'train_batch_size': (14, ''),
        'cand_batch_size': (1024, ''),
        'test_batch_size': (512, ''),
        'nearest_sample_num': (128, ''),
        'func_control': (True, ''),
        'func_threshold': (0.9, ''),
        'attribute_use_data':('add_name','add_name,no_name,only_number,add_relation'),

        
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
