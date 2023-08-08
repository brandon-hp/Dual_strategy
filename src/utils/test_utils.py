import numpy as np

def hit_res(index_mat,stop_mrr=False,top_k=[1,10,50]):
    ent_num=index_mat.shape[0]
    mrr_l=[]
    top_lr = [0] * len(top_k)
    # shape[0]输出矩阵的行数，这里为10500
    for i in range(ent_num):  #np.where(rank == i)[0][0]
        rank_index = np.where(index_mat[i] == i)[0][0]
        mrr_l.append(1.0 / (rank_index + 1))
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    print('Entity Alignment:',end="")
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] /ent_num * 100),end="")
    print('MRR: %.4f' % (np.mean(mrr_l)))
    if stop_mrr:
        return np.mean(mrr_l)
    else:
        return top_lr[0] / ent_num * 100

