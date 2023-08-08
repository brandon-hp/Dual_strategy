import pickle

name='DBP15k/fr_en'
def generate_ent_link(name):
    name2index={}
    with open(name + '/ent_ids_1') as fin1:
        for line in fin1:
            th = line[:-1].split('\t')
            name2index[th[1]]=th[0]
    with open(name + '/ent_ids_2') as fin1:
        for line in fin1:
            th = line[:-1].split('\t')
            name2index[th[1]]=th[0]
    with open(name + '/ent_links') as fin1,open(name + '/ent_links_id','w') as fin2:
        for line in fin1:
            th = line[:-1].split('\t')
            fin2.write(name2index[th[0]]+'\t'+name2index[th[1]]+'\n')
def generate_rel_ids(name):
    trp=[]
    index2name={}
    with open(name + '/ent_ids_1') as fin1:
        for line in fin1:
            th = line[:-1].split('\t')
            index2name[th[0]]=th[1]
    with open(name + '/ent_ids_2') as fin1:
        for line in fin1:
            th = line[:-1].split('\t')
            index2name[th[0]]=th[1]
    trp = []
    rel_ids_1={}
    with open(name + '/triples_1') as fin1,open(name + '/rel_triples_1') as fin2,open(name + '/rel_ids_1','w') as fin3:
        for line in fin1:
            th = line[:-1].split('\t')
            trp.append((index2name[th[0]],th[1],index2name[th[2]]))
        for i,line in enumerate(fin2):
            th = line[:-1].split('\t')
            if th[0]==trp[i][0] and th[2]==trp[i][2]:
                rel_ids_1[trp[i][1]]=th[1]
        for id,txt in rel_ids_1.items():
            fin3.write(id+'\t'+txt+'\n')
    trp = []
    rel_ids_2={}
    with open(name + '/triples_2') as fin1,open(name + '/rel_triples_2') as fin2,open(name + '/rel_ids_2','w') as fin3:
        for line in fin1:
            th = line[:-1].split('\t')
            trp.append((index2name[th[0]],th[1],index2name[th[2]]))
        for i,line in enumerate(fin2):
            th = line[:-1].split('\t')
            if th[0]==trp[i][0] and th[2]==trp[i][2]:
                rel_ids_2[trp[i][1]]=th[1]
        for id,txt in rel_ids_2.items():
            fin3.write(id+'\t'+txt+'\n')
generate_rel_ids(name)








