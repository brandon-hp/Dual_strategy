
import networkx as nx

from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

from utils.data_utils import loadfile
import numpy as np
import scipy.sparse as sp


dataset='DBP15k'
language="ja_en"
data_path="/home/huangpeng/hgcn-master/data/"
prefix = data_path + dataset + '/' + language
kg1 = prefix + '/triples_1'
kg2 = prefix + '/triples_2'
e1 = prefix + '/ent_ids_1'
e2 = prefix + '/ent_ids_2'
ill = prefix + '/ref_ent_ids'
ill_r = prefix + '/ref_r_ids'
e1 = set(loadfile(e1, 1)) # ent1 ids
e2 = set(loadfile(e2, 1)) # ent2 ids
e = len(e1 | e2) # all ent num
kg1 = loadfile(kg1, 3)  # KG1
kg2 = loadfile(kg2, 3)  # KG2
ind, val = [], []
maxind=0
for tri in kg1+kg2:
    if tri[0] == tri[2]:
        continue
    ind.append((tri[0], tri[2]))
    val.append(1)

ind = np.array(ind, dtype=np.int32)
val = np.array(val, dtype=np.float32)
adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(e, e), dtype=np.float32)
import math

def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig
G=nx.Graph(adj.A)

orc =OllivierRicci(G, alpha=0.5, verbose="INFO")
orc.compute_ricci_curvature()
nodeorc={}
noded={}
for triple in G.edges():
    orcn=nodeorc.get(triple[0],0)
    nodeorc[triple[0]]=orcn+round(orc.G[triple[0]][triple[1]]["ricciCurvature"],4)
    d = noded.get(triple[0], 0)
    noded[triple[0]]=d+1

    orcn = nodeorc.get(triple[1], 0)
    nodeorc[triple[1]] = orcn + round(orc.G[triple[0]][triple[1]]["ricciCurvature"], 4)
    d = noded.get(triple[1], 0)
    noded[triple[1]]=d+1
for i in range(e):
    if i in noded:
        print(str(i)+'\t'+str(nodeorc[i]/noded[i]))
    else:
        print(str(i) + '\t0')
'''
name='zh_en'
other = name + '/721_5fold/0'
ent1 = {}
ent2 = {}
with open(name + '/ent_ids_1') as fin1, open(name + '/ent_ids_2') as fin2:
    for line in fin1:
        th = line[:-1].split('\t')
        ent1[th[1]]=th[0]
    for line in fin2:
        th = line[:-1].split('\t')
        ent2[th[1]] = th[0]
train=[]
test=[]
valid=[]
with open(other + '/train_links1') as fin1, open(other + '/valid_links1') as fin2,open(other + '/test_links1') as fin3:
    for line in fin1:
        th = line[:-1].split('\t')
        train.append(ent1[th[0]]+'\t'+ent2[th[1]]+'\n')
    for line in fin2:
        th = line[:-1].split('\t')
        valid.append(ent1[th[0]]+'\t'+ent2[th[1]]+'\n')
    for line in fin3:
        th = line[:-1].split('\t')
        test.append(ent1[th[0]]+'\t'+ent2[th[1]]+'\n')
with open(other + '/train_links',"w") as fin1, open(other + '/valid_links',"w") as fin2,open(other + '/test_links',"w") as fin3:
    for line in train:
        fin1.write(line)
    for line in valid:
        fin2.write(line)
    for line in test:
        fin3.write(line)


'''
'''
with open(name + '/ent_ids_1') as fin1, open(name + '/ent_ids_2') as fin2:
    for line in fin1:
        th = line[:-1].split('\t')
        ent1[int(th[0])]=th[1]
    for line in fin2:
        th = line[:-1].split('\t')
        ent2[int(th[0])] = th[1]
trainlink=[]
testlink=[]
for tr in train:
    trainlink.append((ent1[tr[0]],ent2[tr[1]]))
for tr in test:
    testlink.append((ent1[tr[0]],ent2[tr[1]]))
illL = len(train)
np.random.shuffle(train)
validlink=trainlink[:illL // 9 * 3]
trainlink=trainlink[illL // 9 * 3:]

with open(name+'/test_links',"w") as fin1:
    for tr in testlink:
        fin1.write(tr[0]+'\t'+tr[1]+'\n')
with open(name+'/train_links',"w") as fin1:
    for tr in trainlink:
        fin1.write(tr[0]+'\t'+tr[1]+'\n')
with open(name+'/valid_links',"w") as fin1:
    for tr in validlink:
        fin1.write(tr[0]+'\t'+tr[1]+'\n')
'''