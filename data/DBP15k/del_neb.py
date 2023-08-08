import pickle
import random

name='fr_en_50'
namef='fr_en'
align=set()
align_dict={}
with open(name + '/ref_ent_ids') as fin1:
    for line in fin1:
        th = line[:-1].split('\t')
        align.add((int(th[0]), int(th[1])))
with open(name + '/ref_ent_ids_1') as fin1:
    for line in fin1:
        th = line[:-1].split('\t')
        align_dict[int(th[1])]=int(th[0])
        align_dict[int(th[0])] = int(th[1])
id2nebs={}
with open(namef + '/triples_1') as fin1:
    for line in fin1:
        th = line[:-1].split('\t')
        nebs=id2nebs.get(int(th[0]),set())
        nebs.add(int(th[2]))
        id2nebs[int(th[0])]=nebs
        nebs=id2nebs.get(int(th[2]),set())
        nebs.add(int(th[0]))
        id2nebs[int(th[2])]=nebs
with open(namef + '/triples_2') as fin1:
    for line in fin1:
        th = line[:-1].split('\t')
        if th[0]==th[2]:
            continue
        nebs=id2nebs.get(int(th[0]),set())
        nebs.add(int(th[2]))
        id2nebs[int(th[0])]=nebs
        nebs=id2nebs.get(int(th[2]),set())
        nebs.add(int(th[0]))
        id2nebs[int(th[2])]=nebs
with open(namef + '/triples_1') as fin1,open(name + '/triples_1','w') as fin2:
    for line in fin1:
        th = line[:-1].split('\t')

        if int(th[0]) not in align_dict or int(th[2]) not in align_dict:
            fin2.write(line)
        elif align_dict[int(th[0])] not in id2nebs or align_dict[int(th[2])] not in id2nebs:
            fin2.write(line)
        elif align_dict[int(th[2])] not in id2nebs[align_dict[int(th[0])]] or align_dict[int(th[0])] not in id2nebs[align_dict[int(th[2])]]:
            fin2.write(line)
        elif random.random()<=0.5:
            fin2.write(line)
        '''
        if random.random() <= 0.7:
            fin2.write(line)
        '''

with open(namef + '/triples_2') as fin1,open(name + '/triples_2','w') as fin2:
    for line in fin1:
        th = line[:-1].split('\t')

        if int(th[0]) not in align_dict or int(th[2]) not in align_dict:
            fin2.write(line)
        elif align_dict[int(th[0])] not in id2nebs or align_dict[int(th[2])] not in id2nebs:
            fin2.write(line)
        elif align_dict[int(th[2])] not in id2nebs[align_dict[int(th[0])]] or align_dict[int(th[0])] not in id2nebs[align_dict[int(th[2])]]:
            fin2.write(line)
        elif random.random()<=0.5:
            fin2.write(line)
        '''
        if random.random() <= 0.7:
            fin2.write(line)
        '''

