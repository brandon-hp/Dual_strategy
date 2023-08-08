def read_att_data(data_path,cn='1'):
    """
    load attribute triples file.
    """
    print("loading attribute triples file from: ",data_path)
    att_data = []
    with open(data_path,"r",encoding="utf-8") as f:
        for line in f:
            e,a,l = line.rstrip('\n').split('\t',2)
            e = e.strip('<>')
            a = a.strip('<>')
            if "/property/" in a:
                a = a.split(r'/property/')[-1]
            else:
                a = a.split(r'/')[-1]
            l = l.rstrip('@zhenjadefr .')
            if len(l.rsplit('^^',1)) == 2:
                l,l_type = l.rsplit("^^")
            else:
                l_type = 'string'
            l = l.strip("\"")
            if 'http://' not in e:
                e='http://'+'e'+cn+'/'+e
            att_data.append((e,a,l,l_type)) #(entity,attribute,value,value_type)
    return att_data
prefix='zh_en'
attr_1_path=prefix+'/attr_triples_1/2attr'
attr_2_path = prefix + '/attr_triples_2/2attr'
keep_data_1 = read_att_data(attr_1_path, '1')
keep_data_2 = read_att_data(attr_2_path, '2')
str_set=['0','1','2','3','4','5','6','7','8','9','.','#','(',')','*','+','-']
number_cn=0
number_set=set()
number_set1=set()
for e,a,l,l_type in keep_data_1:
    total=len(l)
    if total==0:
        continue
    cn=0
    for sub in str_set:
        if sub in l:
            cn+=l.count(sub)
    if cn/total>0.6:
        number_cn+=1
        number_set.add((a,l))

    else:
        number_set1.add((a,l))
print('attr 1 number:',number_cn/len(keep_data_1))
with open('number_set11','w') as fin:
    fin.write(str(number_set))
with open('number_set12','w') as fin:
    fin.write(str(number_set1))

number_set=set()
number_set1=set()
number_cn=0
for e,a,l,l_type in keep_data_2:
    total=len(l)
    if total==0:
        continue
    cn=0
    for sub in str_set:
        if sub in l:
            cn+=l.count(sub)
    if cn/total>0.6:
        number_cn+=1
        number_set.add((a,l))
    else:
        number_set1.add((a,l))
print('attr 2 number:',number_cn/len(keep_data_2))
with open('number_set21','w') as fin:
    fin.write(str(number_set))
with open('number_set22','w') as fin:
    fin.write(str(number_set1))