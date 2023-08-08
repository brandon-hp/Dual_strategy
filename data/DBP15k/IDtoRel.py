import pickle
import random

name='fr_en_no'
align=set()
align_dict={}
id2en={}
id2rel={}
with open(name + '/ent_ids_1') as fin1:
    for line in fin1:
        th = line[:-1].split('\t')
        id2en[int(th[0])]=th[1]
with open(name + '/ent_ids_2') as fin1:
    for line in fin1:
        th = line[:-1].split('\t')
        id2en[int(th[0])]=th[1]
with open(name + '/rel_ids_1') as fin1:
    for line in fin1:
        th = line[:-1].split('\t')
        id2rel[int(th[0])]=th[1]
with open(name + '/rel_ids_2') as fin1:
    for line in fin1:
        th = line[:-1].split('\t')
        id2rel[int(th[0])]=th[1]
with open(name + '/triples_1') as fin1,open(name + '/rel_triples_1','w') as fin2:
    for line in fin1:
        th = line[:-1].split('\t')
        fin2.write(id2en[int(th[0])]+'\t'+id2rel[int(th[1])]+'\t'+id2en[int(th[2])]+'\n')

with open(name + '/triples_2') as fin1,open(name + '/rel_triples_2','w') as fin2:
    for line in fin1:
        th = line[:-1].split('\t')
        fin2.write(id2en[int(th[0])]+'\t'+id2rel[int(th[1])]+'\t'+id2en[int(th[2])]+'\n')

