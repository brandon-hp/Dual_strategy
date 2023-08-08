import pickle

import openai
import json
import os
import numpy as np
import string

import torch

openai.api_key = "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"
def askChatGPT():
    #prompt = question
    model_engine = "text-embedding-ada-002"
    #text = "Yves Saint Laurent YSL  is a French luxury fashion house founded by Yves Saint Laurent and his partner, Pierre Bergé. Saint Laurent Paris revived its haute couture collection in 2015 under creative director Hedi Slimane. The new collection, \"Yves Saint Laurent Couture\" or \"Saint Laurent Paris 24, Rue de L’Université\" is the French house's first couture collection ever since the retirement of its legendary founder in 2002."
    #text1 = "伊夫·聖羅蘭是一个奢侈的时装品牌，由设计师伊夫·圣洛朗及其伴侣貝爾傑所创立。风格精致高雅，之前的主要设计师为Hedi Slimane，目前已離開。最初的女装风格曾从男装中大量借鉴物件，如西服裤、运动夹克等等，使得品牌知名度提升。YSL也经营香水等化妆品。"
    #texts=[text,text1]
    # completions = openai.Completion.create(
    #     engine=model_engine,
    #     prompt=prompt,
    #     max_tokens=1024,
    #     n=1,
    #     stop=None,
    #     temperature=0.5,
    # )
    #
    # message = completions.choices[0].text
    texts=['apple: I have an Apple phone.','apple: I bought my Apple phone']
    completions = openai.Embedding.create(input=texts, model=model_engine)
    print(completions['data'][0]['embedding'])
    print(completions['data'][1]['embedding'])
    '''
    os.chdir('.')
    vecList = pickle.load(open('zh_en_name_chatgpt_vectorList.pl','rb'))
    start=len(vecList)
    tap=len(texts)//80
    end=len(texts)
    while start<end:
        start1=start+tap if start+tap<end else end
        print(start, start1)
        completions = openai.Embedding.create(input=texts[start:start1], model=model_engine)
        print(start, start1)
        for i in range(len(completions['data'])):
            vecList.append(completions['data'][i]['embedding'])

        pickle.dump(vecList,open('zh_en_name_chatgpt_vectorList.pl', 'wb'))
        print(start, start1)
        start=start1
    '''
    # with open(file='zh_chatgpt_vectorList.json', mode='r', encoding='utf-8') as f:
    #     embedding_list = json.load(f)
    # print(embedding_list[0])
    # print(embedding_list[1])
def askChatGPT1(texts):
    #prompt = question
    model_engine = "text-embedding-ada-002"
    text = "Yves Saint Laurent YSL  is a French luxury fashion house founded by Yves Saint Laurent and his partner, Pierre Bergé. Saint Laurent Paris revived its haute couture collection in 2015 under creative director Hedi Slimane. The new collection, \"Yves Saint Laurent Couture\" or \"Saint Laurent Paris 24, Rue de L’Université\" is the French house's first couture collection ever since the retirement of its legendary founder in 2002."
    text1 = "伊夫·聖羅蘭是一个奢侈的时装品牌，由设计师伊夫·圣洛朗及其伴侣貝爾傑所创立。风格精致高雅，之前的主要设计师为Hedi Slimane，目前已離開。最初的女装风格曾从男装中大量借鉴物件，如西服裤、运动夹克等等，使得品牌知名度提升。YSL也经营香水等化妆品。"
    completions = openai.Embedding.create(input=[text], model=model_engine)
    print(completions['data'][0]['embedding'])
    #texts=[text,text1]
    # completions = openai.Completion.create(
    #     engine=model_engine,
    #     prompt=prompt,
    #     max_tokens=1024,
    #     n=1,
    #     stop=None,
    #     temperature=0.5,
    # )
    #
    # message = completions.choices[0].text
def load_file(fn, num=1):
    print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret
def get_hits(test_pair, output_layer,stop_mrr,top_k=[1]):
    # self.ws =torch.nn.functional.softmax(self.w.detach(),0)
    ########### 双曲空间 #############
    L = np.array([e1 for e1, e2 in test_pair])
    R = np.array([e2 for e1, e2 in test_pair])
    Lvec =torch.index_select(output_layer,0,torch.tensor(L)) #output_layer[torch.tensor(L)]#list_index_select(output_layer,torch.tensor(L).to(device))
    Rvec = torch.index_select(output_layer,0,torch.tensor(R)) #list_index_select(output_layer,torch.tensor(R).to(device))
    sim=torch.cdist(Lvec, Rvec, 2).pow(2)
    sim = sim.numpy()
    # 计算MRR数值情况
    mrr_l = []  # KG1
    # top_k=10
    top_lr = [0] * len(top_k)
    # shape[0]输出矩阵的行数，这里为10500
    for i in range(L.shape[0]):  #
        # for i in range(Lvec.size(0)):
        rank = sim[i, :].argsort()
        # 找出是在第几个位置匹配上的
        rank_index = np.where(rank == i)[0][0]
        mrr_l.append(1.0 / (rank_index + 1))
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100),end='')
    print('MRR: %.4f' % (np.mean(mrr_l)))

    if stop_mrr:
        return np.mean(mrr_l)
    else:
        return top_lr[0] /  len(test_pair) * 100


def load_id2object(file_paths,cn=''):
    id2object = {}
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            print('loading a (id2object)file...  ' + file_path)
            for line in f:
                th = line.strip('\n').split('\t')
                id2object[int(th[0])] = th[1]
                if cn=='entity' and 'http://' not in th[1]:
                    id2object[int(th[0])] ='http://'+'e'+file_path[-1]+'/'+th[1]
    return id2object
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
# id2comment=[]#load_id2object(['DBP15k/fr_en/comment_1','DBP15k/fr_en/comment_2'])
# id2name=load_id2object(['DBP15k/zh_en/ent_ids_1','DBP15k/zh_en/ent_ids_2'],"entity")
# texts=[]
# for i in range(len(id2name)):
#     comment=id2comment[i] if i in id2comment else get_name(id2name[i])
#     texts.append(comment)
#pickle.dump([], open('zh_en_name_chatgpt_vectorList.pl', 'wb'))
askChatGPT()
# pickle.dump([],open('fr_en_chatgpt_vectorList.pl', 'wb'))
#test()