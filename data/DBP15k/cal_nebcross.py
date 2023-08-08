import pickle

name='ja_en_sp'
align=set()
align_dict={}

# with open(name + '/ref_pairs') as fin1:
#     for line in fin1:
#         th = line[:-1].split('\t')
#         align.add((int(th[0]),int(th[1])))
#         align_dict[int(th[1])] = int(th[0])
# with open(name + '/sup_pairs') as fin1:
#     for line in fin1:
#         th = line[:-1].split('\t')
#         align.add((int(th[0]),int(th[1])))
#         align_dict[int(th[1])] = int(th[0])

with open(name + '/ref_ent_ids') as fin1:
    for line in fin1:
        th = line[:-1].split('\t')
        align.add((int(th[0]), int(th[1])))
with open(name + '/ref_ent_ids') as fin1:
    for line in fin1:
        th = line[:-1].split('\t')
        align_dict[int(th[1])]=int(th[0])
id2nebs={}
with open(name + '/triples_1') as fin1:
    for line in fin1:
        th = line[:-1].split('\t')
        nebs=id2nebs.get(int(th[0]),set())
        nebs.add(int(th[2]))
        id2nebs[int(th[0])]=nebs
        nebs=id2nebs.get(int(th[2]),set())
        nebs.add(int(th[0]))
        id2nebs[int(th[2])]=nebs
with open(name + '/triples_2') as fin1:
    for line in fin1:
        th = line[:-1].split('\t')
        if th[0]==th[2]:
            continue
        nebs=id2nebs.get(int(th[0]),set())
        nebs.add(align_dict[int(th[2])] if int(th[2]) in align_dict else int(th[2]))
        id2nebs[int(th[0])]=nebs
        nebs=id2nebs.get(int(th[2]),set())
        nebs.add(align_dict[int(th[0])] if int(th[0]) in align_dict else int(th[0]))
        id2nebs[int(th[2])]=nebs
cross_n=[0,0,0,0,0,0,0]
no1=0
for tr in align:
    if tr[0] not in  id2nebs or tr[1] not in id2nebs:
        print('----------')
        continue
    cn=id2nebs[tr[0]]&id2nebs[tr[1]]
    no1+=len(cn)
    cindex=len(cn) if len(cn)<=5 else 5
    cross_n[cindex]+=1
    cross_n[-1]+=(len(cn)/len(id2nebs[tr[0]])+len(cn)/len(id2nebs[tr[1]]))
print(len(align),no1)
sum=0
for i in range(6):
    sum+=cross_n[5-i]
    print(sum/len(align))
print(cross_n[-1]/(2*len(align)))
#print(cross_n/len(align),len(align))

'''
ent1 = {}
ent2 = {}
with open(name + '/ent_ids_1') as fin1, open(name + '/ent_ids_2') as fin2:
    for line in fin1:
        th = line[:-1].split('\t')
        ent1[th[1]]=th[0]
    for line in fin2:
        th = line[:-1].split('\t')
        ent2[th[1]] = th[0]
rel1n={}
dup=set()
with open(name + '/rel_triples_1') as fin1, open(name + '/rel_triples_2') as fin2:
    for line in fin1:
        th = line[:-1].split('\t')
        s=rel1n.get(th[1],set())
        s.add(ent1[th[0]]+'|'+ent1[th[2]])
        rel1n[th[1]]=s
        if ent1[th[0]]+'|'+ent1[th[2]] in dup:
            print(ent1[th[0]]+'|'+ent1[th[2]],th[1])
        else:
            dup.add(ent1[th[0]]+'|'+ent1[th[2]])
    for line in fin2:
        th = line[:-1].split('\t')
        s = rel1n.get(th[1], set())
        s.add(ent2[th[0]] + '|' + ent2[th[2]])
        rel1n[th[1]] = s
rel2n={}
rel={}
with open(name + '/triples_1') as fin1, open(name + '/triples_2') as fin2:
    for line in fin1:
        th = line[:-1].split('\t')
        s=rel2n.get(th[1],set())
        s.add(th[0]+'|'+th[2])
        rel2n[th[1]]=s
    for line in fin2:
        th = line[:-1].split('\t')
        s = rel2n.get(th[1], set())
        s.add(th[0] + '|' + th[2])
        rel2n[th[1]] = s
for r1 in rel1n:
    for r2 in rel2n:
        if rel1n[r1]==rel2n[r2]:
            rel[int(r2)]=r1

fw = open("rel_ids", "wb")
pickle.dump(rel, fw)
'''