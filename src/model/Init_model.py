import string

from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

class Basic_Bert_Unit_model(nn.Module):
    def __init__(self,input_size,result_size,bert_path,dropout=0.1):
        super(Basic_Bert_Unit_model,self).__init__()
        self.result_size = result_size
        self.input_size = input_size
        self.bert_model = BertModel.from_pretrained(bert_path)
        self.out_linear_layer = nn.Linear(self.input_size,self.result_size)
        self.dropout = nn.Dropout(p =dropout)



    def forward(self,batch_word_list,attention_mask):
        x = self.bert_model(input_ids = batch_word_list,attention_mask = attention_mask)#token_type_ids =token_type_ids
        sequence_output, pooled_output = x
        cls_vec = sequence_output[:,0]
        output = self.dropout(cls_vec)
        output = self.out_linear_layer(output)
        return F.normalize(output, p=2,dim=1)
class GPT(nn.Module):
    def __init__(self,input_size,result_size,chat_vec,dropout=0.1):
        super(GPT,self).__init__()
        self.result_size = result_size
        self.input_size = input_size
        size, dim=chat_vec.shape
        self.embeddings=nn.Embedding(size, dim)
        self.embeddings.weight.data.copy_(chat_vec)
        #self.weight = nn.Parameter(torch.Tensor(self.input_size, self.result_size))
        self.out_linear_layer = nn.Linear(self.input_size,self.result_size)
        #self.out_linear_layer1 = nn.Linear( self.result_size*3, self.result_size,bias=False)
        #self.dropout = nn.Dropout(p =dropout)
    def forward(self,batch_word_list):
        output=self.embeddings.weight.detach()
        output =F.dropout(self.out_linear_layer(output),p=0.3, training=self.training)
        return F.normalize(output, p=2,dim=1)[batch_word_list]
class Basic_Bert_Unit_model_mix(nn.Module):
    def __init__(self,input_size,result_size,bert_path,dropout=0.1):
        super(Basic_Bert_Unit_model_mix,self).__init__()
        self.result_size = result_size
        self.input_size = input_size
        self.bert_model = BertModel.from_pretrained(bert_path)
        self.out_linear_layer = nn.Linear(self.input_size,self.result_size)
        self.dropout = nn.Dropout(p =dropout)



    def forward(self,batch_word_list,attention_mask):
        lent=batch_word_list.shape[-1]//2
        x1 = self.bert_model(input_ids = batch_word_list[:,:lent],attention_mask = attention_mask[:,:lent])#token_type_ids =token_type_ids
        sequence_output1, pooled_output1 = x1
        x2 = self.bert_model1(input_ids = batch_word_list[:,lent:],attention_mask = attention_mask[:,lent:])#token_type_ids =token_type_ids
        sequence_output2, pooled_output2 = x2
        cls_vec = sequence_output1[:,0]#(sequence_output1[:,0]+sequence_output2[:,0])/2
        output = self.dropout(cls_vec)
        output = self.out_linear_layer(output)

        cls_vec2 = sequence_output2[:,0]#(sequence_output1[:,0]+sequence_output2[:,0])/2
        output2 = self.dropout(cls_vec2)
        output2 = self.out_linear_layer(output2)
        return F.normalize(torch.cat([output,output2],dim=1), p=2,dim=1)
class Basic_Word_model:
    def __init__(self,word_emb_path,index2entity):
        super(Basic_Word_model,self).__init__()
        self.name=[(eid,index2entity[eid]) for eid in range(0, len(index2entity.keys()))]
        self.entities_num=len(self.name)
        self.word_ids,self.word_sq,self.name_embeds =self._get_desc_input(word_emb_path)

    def _get_desc_input(self,word_emb_path):
        name_triples = self.name
        #print(name_triples)
        names = pd.DataFrame(name_triples)
        names.iloc[:, 1] = names.iloc[:, 1].str.split(' ')

        word_sq=names.iloc[:, 1].copy()
        # load word embedding
        with open(word_emb_path ,'r') as f:
            w = f.readlines()
            w = pd.Series(w[1:])

        we = w.str.split(' ')
        word = we.apply(lambda x: x[0])
        w_em = we.apply(lambda x: x[1:])
        print('concat word embeddings')
        word_em = np.stack(w_em.values, axis=0).astype(np.float)
        word_em = np.append(word_em, np.zeros([1, 300]), axis=0)
        print('convert words to ids')
        w_in_desc = []
        for l in names.iloc[:, 1].values:
            w_in_desc += l
        w_in_desc = pd.Series(list(set(w_in_desc)))
        un_logged_words = w_in_desc[~w_in_desc.isin(word)]
        un_logged_id = len(word)

        all_word = pd.concat(
            [pd.Series(word.index, word.values),
             pd.Series([un_logged_id, ] * len(un_logged_words), index=un_logged_words)])
        def lookup_and_padding(x):
            default_length = 4
            vlist=list(all_word.loc[x].values)
            ids = vlist+ [all_word.iloc[-1], ] * default_length
            return ids[:default_length]+[len(vlist)]

        print('look up desc embeddings')
        names.iloc[:, 1] = names.iloc[:, 1].apply(lookup_and_padding)

        # entity-desc-embedding dataframe
        e_desc_input = pd.DataFrame(np.repeat([[un_logged_id, ] * 5], self.entities_num, axis=0),
                                    range(self.entities_num))

        e_desc_input.iloc[names.iloc[:, 0].values] = np.stack(names.iloc[:, 1].values)
        #print(word_em[999994])
        name_embeds1 = word_em[e_desc_input.values[:,:3]]
        name_embeds = np.sum(name_embeds1, axis=1)/e_desc_input.values[:,4].reshape(-1,1)
        name_embeds=F.normalize(torch.FloatTensor(name_embeds), p=2, dim=1).numpy()

        return e_desc_input.values,name_embeds1,name_embeds
