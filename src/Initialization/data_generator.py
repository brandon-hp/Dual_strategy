from torch.utils.data import Dataset
import numpy as np

class Batch_TrainData_Generator(Dataset):
    def __init__(self,train_ill,ent_ids1,ent_ids2,index2data,batch_size,neg_num):
        self.ent_ill = train_ill
        self.ent_ids1 = ent_ids1
        self.ent_ids2 = ent_ids2
        self.batch_size = batch_size
        self.neg_num = neg_num
        self.iter_count = 0
        self.index2entity = index2data
        print("In Batch_TrainData_Generator, train ill num: {}".format(len(self.ent_ill)))
        print("In Batch_TrainData_Generator, ent_ids1 num: {}".format(len(self.ent_ids1)))
        print("In Batch_TrainData_Generator, ent_ids2 num: {}".format(len(self.ent_ids2)))


    def __len__(self):
        return len(self.train_index)

    def __getitem__(self, idx):
        """
        item 为数据索引，迭代取第item条数据
        """
        pe1s,pe2s,ne1s,ne2s = self.train_index[idx]
        return pe1s,pe2s,ne1s,ne2s
    def train_index_gene(self,candidate_dict):
        """
        generate training data (entity_index).
        """
        train_index = [] #training data
        candid_num = 999999
        for ent in candidate_dict:
            candid_num = min(candid_num,len(candidate_dict[ent]))
            candidate_dict[ent] = np.array(candidate_dict[ent])
        for pe1,pe2 in self.ent_ill:
            for _ in range(self.neg_num):
                if np.random.rand() <= 0.5:
                    #e1
                    ne1 = candidate_dict[pe2][np.random.randint(candid_num)]
                    ne2 = pe2
                else:
                    ne1 = pe1
                    ne2 = candidate_dict[pe1][np.random.randint(candid_num)]
                #same check
                if pe1!=ne1 or pe2!=ne2:
                    train_index.append([pe1,pe2,ne1,ne2])
        np.random.shuffle(train_index)
        self.train_index = train_index
        self.batch_num = int( np.ceil( len(self.train_index) * 1.0 / self.batch_size ) )
