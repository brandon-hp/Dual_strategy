"""Graph decoders."""
import random

import manifolds
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('..')
from utils.data_utils import list_index_select
class CrossDecoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self,args):
        super(CrossDecoder, self).__init__()
        self.k=args.k
        self.gamma=args.gamma_margin
        self.device=args.device

    def decode(self,res,manifolds,metrics_input):
        metrics_input=self.process_data(res,manifolds,metrics_input)
        ILL, neg_left, neg_right, neg2_left, neg2_right=metrics_input['ILL'],metrics_input["neg_left"], metrics_input["neg_right"],metrics_input["neg2_left"], metrics_input["neg2_right"]
        outlayer, _, c = res['M0']

        left = torch.tensor(ILL[:, 0]).to(self.device)  # KG1中的预对齐的实体id 训练集 4500
        right = torch.tensor(ILL[:, 1]).to(self.device)  # KG2中的预对齐的实体id 训练集 4500
        left_x = list_index_select(outlayer,left)#torch.index_select(outlayer, 0, left)  # KG1中的实体emb (4500,300)
        right_x = list_index_select(outlayer,right)#torch.index_select(outlayer, 0, right)  # KG2中的实体emb (4500,300)

        A=0
        for i in range(len(manifolds)):
            A += manifolds[i].sqdist(left_x[i], right_x[i], c[i])

        # gamma表示边缘，即γ
        D = A + self.gamma
        neg_l_x = list_index_select(outlayer,neg_left)#torch.index_select(outlayer, 0, neg_left)
        neg_r_x = list_index_select(outlayer,neg_right)#torch.index_select(outlayer, 0, neg_right)

        B=0
        for i in range(len(manifolds)):
            B += manifolds[i].sqdist(neg_l_x[i], neg_r_x[i], c[i])
        C = - B.view(-1, self.k)
        L1 = torch.nn.functional.relu(torch.add(C, D.view(-1, 1)))

        neg_l_x = list_index_select(outlayer,neg2_left)#torch.index_select(outlayer, 0, neg2_left)
        neg_r_x = list_index_select(outlayer,neg2_right)#torch.index_select(outlayer, 0, neg2_right)


        B=0
        for i in range(len(manifolds)):
            B += manifolds[i].sqdist(neg_l_x[i], neg_r_x[i], c[i])

        C = - B.view(-1, self.k)
        L2 = torch.nn.functional.relu(torch.add(C, D.view(-1, 1)))
        return (torch.mean(L1) + torch.mean(L2)) / 2.0
    def process_data(self,res,manifolds,metrics_input):
        beg, end=metrics_input['index']
        ILL=metrics_input['train_ill'][beg:end]
        neg2_left = self.get_negM(ILL[:,1], res,manifolds, self.k)
        neg_right = self.get_negM(ILL[:,0], res,manifolds, self.k)
        # print(neg_left.device)
        metrics_input['ILL'] = ILL
        t = len(ILL)  # 4500
        L = np.ones((t, self.k)) * (ILL[:,0].reshape((t, 1)))
        neg_left = L.reshape((t *self.k,))
        neg_left = torch.LongTensor(neg_left)
        L = np.ones((t, self.k)) * (ILL[:, 1].reshape((t, 1)))
        neg2_right = L.reshape((t * self.k,))
        neg2_right = torch.LongTensor(neg2_right)
        metrics_input['neg_left'] = neg_left.reshape((t, self.k)).reshape((-1,))
        metrics_input['neg_right'] = neg_right.reshape(((end - beg), self.k)).reshape((-1,))
        metrics_input['neg2_left'] = neg2_left.reshape(((end - beg), self.k)).reshape((-1,))
        metrics_input['neg2_right'] = neg2_right.reshape((t, self.k)).reshape((-1,))
        for x, val in metrics_input.items():
            if torch.is_tensor(metrics_input[x]):
                metrics_input[x] = metrics_input[x].to(self.device)
        return metrics_input
    def get_negM(self, ILL, res, manifolds, k):

        t = len(ILL)  # t=4500
        ILL = torch.LongTensor(ILL).to(self.device)
        output_layer, _, c = res['M0']

        ILL_vec = list_index_select(output_layer, ILL)  # torch.index_select(output_layer, 0, ILL)
        sim = 0
        for i in range(len(manifolds)):
            sim += manifolds[i].sqdist(ILL_vec[i], output_layer[i], c[i], True)
        neg = torch.argsort(sim, 1)[:, 1:k + 1]
        neg = neg.reshape(t * k, )
        return neg
class CrossDecoder1(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self,args):
        super(CrossDecoder1, self).__init__()
        self.k=args.k
        self.gamma=args.gamma_margin
        self.device=args.device
        self.dense1 = nn.Linear(4, 1, True)

    def decode(self,res,manifolds,metrics_input):
        metrics_input=self.process_data(res,manifolds,metrics_input)
        ILL, neg_left, neg_right, neg2_left, neg2_right=metrics_input['ILL'],metrics_input["neg_left"], metrics_input["neg_right"],metrics_input["neg2_left"], metrics_input["neg2_right"]
        outlayer, _, c = res['M0']

        left = torch.tensor(ILL[:, 0]).to(self.device)  # KG1中的预对齐的实体id 训练集 4500
        right = torch.tensor(ILL[:, 1]).to(self.device)  # KG2中的预对齐的实体id 训练集 4500
        left_x = list_index_select(outlayer,left)#torch.index_select(outlayer, 0, left)  # KG1中的实体emb (4500,300)
        right_x = list_index_select(outlayer,right)#torch.index_select(outlayer, 0, right)  # KG2中的实体emb (4500,300)

        A=[]
        for i in range(len(manifolds)):
            A.append(manifolds[i].sqdist(left_x[i], right_x[i], c[i]))
        A=self.dense1(torch)

        # gamma表示边缘，即γ
        D = A + self.gamma
        neg_l_x = list_index_select(outlayer,neg_left)#torch.index_select(outlayer, 0, neg_left)
        neg_r_x = list_index_select(outlayer,neg_right)#torch.index_select(outlayer, 0, neg_right)

        B=0
        for i in range(len(manifolds)):
            B += manifolds[i].sqdist(neg_l_x[i], neg_r_x[i], c[i])
        C = - B.view(-1, self.k)
        L1 = torch.nn.functional.relu(torch.add(C, D.view(-1, 1)))

        neg_l_x = list_index_select(outlayer,neg2_left)#torch.index_select(outlayer, 0, neg2_left)
        neg_r_x = list_index_select(outlayer,neg2_right)#torch.index_select(outlayer, 0, neg2_right)


        B=0
        for i in range(len(manifolds)):
            B += manifolds[i].sqdist(neg_l_x[i], neg_r_x[i], c[i])

        C = - B.view(-1, self.k)
        L2 = torch.nn.functional.relu(torch.add(C, D.view(-1, 1)))
        return (torch.mean(L1) + torch.mean(L2)) / 2.0
    def process_data(self,res,manifolds,metrics_input):
        beg, end=metrics_input['index']
        ILL=metrics_input['train_ill'][beg:end]
        neg2_left = self.get_negM(ILL[:,1], res,manifolds, self.k)
        neg_right = self.get_negM(ILL[:,0], res,manifolds, self.k)
        # print(neg_left.device)
        metrics_input['ILL'] = ILL
        t = len(ILL)  # 4500
        L = np.ones((t, self.k)) * (ILL[:,0].reshape((t, 1)))
        neg_left = L.reshape((t *self.k,))
        neg_left = torch.LongTensor(neg_left)
        L = np.ones((t, self.k)) * (ILL[:, 1].reshape((t, 1)))
        neg2_right = L.reshape((t * self.k,))
        neg2_right = torch.LongTensor(neg2_right)
        metrics_input['neg_left'] = neg_left.reshape((t, self.k)).reshape((-1,))
        metrics_input['neg_right'] = neg_right.reshape(((end - beg), self.k)).reshape((-1,))
        metrics_input['neg2_left'] = neg2_left.reshape(((end - beg), self.k)).reshape((-1,))
        metrics_input['neg2_right'] = neg2_right.reshape((t, self.k)).reshape((-1,))
        for x, val in metrics_input.items():
            if torch.is_tensor(metrics_input[x]):
                metrics_input[x] = metrics_input[x].to(self.device)
        return metrics_input
    def get_negM(self, ILL, res, manifolds, k):

        t = len(ILL)  # t=4500
        ILL = torch.LongTensor(ILL).to(self.device)
        output_layer, _, c = res['M0']

        ILL_vec = list_index_select(output_layer, ILL)  # torch.index_select(output_layer, 0, ILL)
        sim = 0
        for i in range(len(manifolds)):
            sim += manifolds[i].sqdist(ILL_vec[i], output_layer[i], c[i], True)
        neg = torch.argsort(sim, 1)[:, 1:k + 1]
        neg = neg.reshape(t * k, )
        return neg
class CrossDecoderT(CrossDecoder):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self,args):
        super(CrossDecoderT, self).__init__(args)
        self.k=args.k
        self.gamma=args.gamma_margin
        self.device=args.device
        self.type=args.loss_t

    def decode(self,res,manifolds,metrics_input):
        beg, end=metrics_input['index']
        ILL=metrics_input['train_ill'][beg:end]
        outlayer, _, c = res['M0']

        left = torch.tensor(ILL[:, 0]).to(self.device)  # KG1中的预对齐的实体id 训练集 4500
        right = torch.tensor(ILL[:, 1]).to(self.device)  # KG2中的预对齐的实体id 训练集 4500
        left_x = list_index_select(outlayer,left)#torch.index_select(outlayer, 0, left)  # KG1中的实体emb (4500,300)
        right_x = list_index_select(outlayer,right)#torch.index_select(outlayer, 0, right)  # KG2中的实体emb (4500,300)

        A=0
        for i in range(len(manifolds)):
            A += manifolds[i].sqdist(left_x[i], right_x[i], c[i])

        # gamma表示边缘，即γ
        D = A
        L1 = torch.nn.functional.relu( D.view(-1, 1))
        if self.type=='sum':
            return torch.sum(L1)
        else:
            return torch.mean(L1)


class TransEDecoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self,args):
        super(TransEDecoder, self).__init__()

        self.k=args.k
        self.pos_margin, self.neg_margin= args.pos_margin, args.neg_margin
        self.device=args.device
        self.neg_param=args.neg_param
        self.type=args.loss_t

    def decode1(self,res,manifolds,metrics_input):
        metrics_input=self.process_data(res,manifolds,metrics_input)
        pos=metrics_input['pos_triples']
        neg=metrics_input['neg_triples']
        outlayer, output_r, c = res['M0']
        pos_head=list_index_select(outlayer,pos[:,0])#outlayer[pos[:,0]]
        pos_rel=list_index_select(output_r,pos[:,1])#output_r[pos[:,1]]
        pos_tail=list_index_select(outlayer,pos[:,2])

        neg_head=list_index_select(outlayer,neg[:,0])
        neg_rel=list_index_select(output_r,neg[:,1])
        neg_tail=list_index_select(outlayer,neg[:,2])
        pos_score=0
        neg_score=0
        for i in range(len(manifolds)):
            pos_score+=manifolds[i].sqdist(manifolds[i].proj(manifolds[i].mobius_add(pos_head[i], pos_rel[i], c[i]), c[i]), pos_tail[i], c[i]).squeeze(1)
        for i in range(len(manifolds)):
            neg_score += manifolds[i].sqdist(
                manifolds[i].proj(manifolds[i].mobius_add(neg_head[i], neg_rel[i], c[i]), c[i]), neg_tail[i],
                c[i]).squeeze(1)
        if self.type=='sum':
            pos_loss=F.relu(pos_score-self.pos_margin).sum()
            neg_loss=F.relu(self.neg_margin-neg_score).sum()
        else:
            pos_loss=F.relu(pos_score-self.pos_margin).mean()
            neg_loss=F.relu(self.neg_margin-neg_score).mean()
        return pos_loss+self.neg_param*neg_loss
    def process_data(self,res,manifolds,metrics_input):
        beg1, end1=metrics_input['index1']
        beg2, end2 = metrics_input['index2']
        pos_triples1=metrics_input['kg1'][beg1:end1]
        pos_triples2=metrics_input['kg2'][beg2:end2]
        ent_ids_1=metrics_input['ent_ids_1']
        ent_ids_2=metrics_input['ent_ids_2']
        if metrics_input['epoch']>=0 and beg1==0:
            metrics_input['neighbours_dic1']=self.get_negM(ent_ids_1, res, manifolds, self.k*40)
            metrics_input['neighbours_dic2'] = self.get_negM(ent_ids_2, res, manifolds, self.k*40)
        neg_triples1=self.trunc_sampling_multi(pos_triples1,metrics_input['kg1'],  metrics_input['neighbours_dic1'], metrics_input['neighbours_dic2'],ent_ids_1, self.k)
        neg_triples2=self.trunc_sampling_multi(pos_triples2,metrics_input['kg2'],  metrics_input['neighbours_dic1'],metrics_input['neighbours_dic2'], ent_ids_2, self.k)
        pos_triples=pos_triples1+pos_triples2
        neg_triples=neg_triples1+neg_triples2
        metrics_input['pos_triples'] = torch.tensor(np.array(pos_triples))
        metrics_input['neg_triples'] = torch.tensor(np.array(neg_triples))
        for x, val in metrics_input.items():
            if torch.is_tensor(metrics_input[x]):
                metrics_input[x] = metrics_input[x].to(self.device)
        return metrics_input

    def trunc_sampling_multi(self,pos_triples, all_triples, dic1,dic2, ent_list, multi):
        neg_triples = list()
        ent_list = np.array(ent_list)
        for (h, r, t) in pos_triples:
            choice = random.randint(0, 999)
            if choice < 500:
                if h in dic1:
                    candidates=dic1[h]
                elif h in dic2:
                    candidates=dic2[h]
                else:
                    candidates=ent_list
                h2s = random.sample(candidates, multi)
                negs = [(h2, r, t) for h2 in h2s]
                neg_triples.extend(negs)
            elif choice >= 500:
                if t in dic1:
                    candidates = dic1[t]
                elif t in dic2:
                    candidates = dic2[t]
                else:
                    candidates = ent_list
                t2s = random.sample(candidates, multi)
                negs = [(h, r, t2) for t2 in t2s]
                neg_triples.extend(negs)
        neg_triples = list(set(neg_triples) - set(all_triples))
        return neg_triples
    def get_negM(self, ILL, res, manifolds, k):

        t = len(ILL)  # t=4500
        ILL = torch.LongTensor(ILL).to(self.device)
        output_layer, _, c = res['M0']

        ILL_vec = list_index_select(output_layer, ILL)  # torch.index_select(output_layer, 0, ILL)
        sim = 0
        for i in range(len(manifolds)):
            sim += manifolds[i].sqdist(ILL_vec[i].cpu(), ILL_vec[i].cpu(), c[i].cpu(), True)
        neg = torch.argsort(sim, 1)[:, 1:k + 1]
        neighbours_dic={}
        for i in range(neg.shape[0]):
            neighbours_dic[int(ILL[i])]=ILL[neg[i]].tolist()
        return neighbours_dic



class MRAEA(CrossDecoder):
    def __init__(self,args):
        super(MRAEA, self).__init__(args)
class GAT(CrossDecoder):
    def __init__(self,args):
        super(GAT, self).__init__(args)
class MRAEA_adapt(CrossDecoder):
    def __init__(self,args):
        super(MRAEA_adapt, self).__init__(args)
class LightEA_literal(CrossDecoder):
    def __init__(self,args):
        super(LightEA_literal, self).__init__(args)
class DUAL(CrossDecoder):
    def __init__(self,args):
        super(DUAL, self).__init__(args)
class DUAL_adapt(CrossDecoder):
    def __init__(self,args):
        super(DUAL_adapt, self).__init__(args)
class RREA_adapt(CrossDecoder):
    def __init__(self,args):
        super(RREA_adapt, self).__init__(args)
class RREA_adapt1(CrossDecoder):
    def __init__(self,args):
        super(RREA_adapt1, self).__init__(args)
class TestM(CrossDecoder):
    def __init__(self,args):
        super(TestM, self).__init__(args)
class roadEA(CrossDecoder):
    def __init__(self,args):
        super(roadEA, self).__init__(args)
class RREAN_literal(CrossDecoder):
    def __init__(self,args):
        super(RREAN_literal, self).__init__(args)
class ICLEA_literal(CrossDecoder):
    def __init__(self,args):
        super(ICLEA_literal, self).__init__(args)
class ICLEA(CrossDecoder):
    def __init__(self,args):
        super(ICLEA, self).__init__(args)
class RREA(CrossDecoder):
    def __init__(self,args):
        super(RREA, self).__init__(args)
class RAGA(CrossDecoder):
    def __init__(self,args):
        super(RAGA, self).__init__(args)
class RAGA_literal(CrossDecoder):
    def __init__(self,args):
        super(RAGA_literal, self).__init__(args)
class ERMC(CrossDecoder):
    def __init__(self,args):
        super(ERMC, self).__init__(args)
class ERMC_adapt(CrossDecoder):
    def __init__(self,args):
        super(ERMC_adapt, self).__init__(args)
class TransEdgeDecoder(TransEDecoder):
    """
    Decoder abstract class for node classification tasks.
    """
    def __init__(self,args,input_dim,output_dim):
        super(TransEdgeDecoder, self).__init__(args)
        self.act = getattr(F, args.act)
        self.mlp=nn.Linear(2*input_dim,output_dim)
        self.bp_param=args.bp_param
        self.type=args.loss_t

    def decode1(self, res, manifolds, metrics_input):
        metrics_input=self.process_data(res,manifolds,metrics_input)
        pos=metrics_input['pos_triples']
        neg=metrics_input['neg_triples']
        outlayer, output_r, c = res['M0']
        outlayer1, output_r1, c1 = res['M1']
        pos_head=list_index_select(outlayer,pos[:,0])
        pos_rel=list_index_select(output_r,pos[:,1])
        pos_tail=list_index_select(outlayer,pos[:,2])

        neg_head=list_index_select(outlayer,neg[:,0])
        neg_rel=list_index_select(output_r,neg[:,1])
        neg_tail=list_index_select(outlayer,neg[:,2])

        pos_head1=list_index_select(outlayer1,pos[:,0])
        pos_tail1=list_index_select(outlayer1,pos[:,2])

        neg_head1=list_index_select(outlayer1,neg[:,0])
        neg_tail1=list_index_select(outlayer1,neg[:,2])
        pos_rel=self.context_projection(pos_head1,pos_rel,pos_tail1,c,c1,manifolds)
        neg_rel=self.context_projection(neg_head1,neg_rel,neg_tail1,c,c1,manifolds)

        pos_score=0
        neg_score=0
        for i in range(len(manifolds)):
            pos_score+=manifolds[i].sqdist(manifolds[i].proj(manifolds[i].mobius_add(pos_head[i], pos_rel[i], c[i]), c[i]), pos_tail[i], c[i]).squeeze(1)
        for i in range(len(manifolds)):
            neg_score += manifolds[i].sqdist(
                manifolds[i].proj(manifolds[i].mobius_add(neg_head[i], neg_rel[i], c[i]), c[i]), neg_tail[i],
                c[i]).squeeze(1)
        if self.type=='sum':
            pos_loss=F.relu(pos_score-self.pos_margin).sum()
            neg_loss=F.relu(self.neg_margin-neg_score).sum()
        else:
            pos_loss=F.relu(pos_score-self.pos_margin).mean()
            neg_loss=F.relu(self.neg_margin-neg_score).mean()

        return pos_loss+self.neg_param*neg_loss

    def context_projection(self,hs, rss, ts,c,c1,manifolds):
        r=[]
        for i in range(len(manifolds)):
            rs=manifolds[i].proj_tan0(manifolds[i].logmap0(rss[i], c=c[i]), c=c[i])
            head =manifolds[i].proj_tan0(manifolds[i].logmap0(hs[i], c=c1[i]), c=c1[i])
            tail =manifolds[i].proj_tan0(manifolds[i].logmap0(ts[i], c=c1[i]), c=c1[i])
            hts = torch.cat((head, tail), dim=-1)
            hts = F.normalize(hts, p=2, dim=1)
            hts = self.act(self.mlp(hts))
            norm_vec = F.normalize(hts, p=2, dim=1)
            bias=rs - (rs * norm_vec).sum(dim=1).unsqueeze(1) * norm_vec
            bias = F.normalize(bias, p=2, dim=1)
            r.append(manifolds[i].proj(manifolds[i].expmap0(manifolds[i].proj_tan0(bias, c=c[i]), c=c[i]),
                                 c=c[i]))
        return r
    def context_compression(self,hs, rss, ts,c,c1,manifolds):
        r=[]
        for i in range(len(manifolds)):
            rs=manifolds[i].proj_tan0(manifolds[i].logmap0(rss[i], c=c[i]), c=c[i])
            head =manifolds[i].proj_tan0(manifolds[i].logmap0(hs[i], c=c1[i]), c=c1[i])
            tail =manifolds[i].proj_tan0(manifolds[i].logmap0(ts[i], c=c1[i]), c=c1[i])
            hts = F.normalize(torch.cat((head, rs), dim=-1), p=2, dim=1)
            hts = self.act(self.mlp(hts))
            hts1 = F.normalize(torch.cat((rs,tail), dim=-1), p=2, dim=1)
            hts1 = self.act(self.mlp1(hts1))
            crs=F.normalize(torch.cat((hts, hts1), dim=-1), p=2, dim=1)
            crs = self.act(self.mlp2(crs))
            crs = F.normalize(crs, p=2, dim=1)
            r.append(manifolds[i].proj(manifolds[i].expmap0(manifolds[i].proj_tan0(crs, c=c[i]), c=c[i]),
                                 c=c[i]))
        return r
class TransEdge(TransEdgeDecoder):
    """
    Decoder abstract class for node classification tasks.
    """
    def __init__(self,args,input_dim,output_dim):
        super(TransEdge, self).__init__(args,input_dim,output_dim)
        self.crossdecoder=CrossDecoderT(args)
        self.bp_param=args.bp_param
    def decode(self, res, manifolds, metrics_input):
        return self.decode1(res, manifolds, metrics_input)+self.bp_param * self.crossdecoder.decode(res, manifolds, metrics_input)

class KECG(TransEDecoder):
    """
    Decoder abstract class for node classification tasks.
    """
    def __init__(self,args):
        super(KECG, self).__init__(args)
        self.crossdecoder=CrossDecoder(args)
        self.bp_param=args.bp_param
    def decode(self, res, manifolds, metrics_input):
        if metrics_input['epoch']%2==1:
            return self.bp_param*self.crossdecoder.decode(res, manifolds, metrics_input)
        else:
            return self.decode1(res,manifolds,metrics_input)
class KECG1(TransEDecoder):
    """
    Decoder abstract class for node classification tasks.
    """
    def __init__(self,args):
        super(KECG1, self).__init__(args)
        self.crossdecoder=CrossDecoderT(args)
        self.bp_param=args.bp_param
    def decode(self, res, manifolds, metrics_input):
            return self.bp_param*self.crossdecoder.decode(res, manifolds, metrics_input)+self.decode1(res,manifolds,metrics_input)
class SSP(TransEdge):
    def __init__(self,args,input_dim,output_dim):
        super(SSP, self).__init__(args,input_dim,output_dim)
class SSP_adapt(TransEdge):
    def __init__(self,args,input_dim,output_dim):
        super(SSP_adapt, self).__init__(args,input_dim,output_dim)
class SSP1(TransEDecoder):
    """
    Decoder abstract class for node classification tasks.
    """
    def __init__(self,c_k,gamma_margin,t_k, pos_margin, neg_margin,neg_param,bp_param,device,act,input_dim,output_dim):
        super(SSP1, self).__init__(t_k, pos_margin, neg_margin,neg_param,device)
        self.crossdecoder=CrossDecoderT(c_k,gamma_margin,device)
        self.bp_param=bp_param
    def decode(self, res, manifolds, metrics_input):
        return self.decode1(res, manifolds, metrics_input)+self.bp_param * self.crossdecoder.decode(res, manifolds, metrics_input)

class BootEA(TransEDecoder):
    """
    Decoder abstract class for node classification tasks.
    """
    def __init__(self,args):
        super(BootEA, self).__init__(args)
        self.crossdecoder = CrossDecoderT(args)
        self.bp_param=args.bp_param
    def decode(self, res, manifolds, metrics_input):
        return self.decode1(res, manifolds, metrics_input)+self.bp_param * self.crossdecoder.decode(res, manifolds, metrics_input)

class MlP(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(MlP, self).__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim, True)
        self.dense2 = nn.Linear(hidden_dim, 1, True)
        torch.nn.init.xavier_normal_(self.dense1.weight)
        torch.nn.init.xavier_normal_(self.dense2.weight)
    def forward(self,features):
        x = self.dense1(features)#[B,h]
        x = F.relu(x)
        x = self.dense2(x)#[B,1]
        x = F.tanh(x)
        x = torch.squeeze(x,1)#[B]
        return x
class BERT_INT_N(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self,input_dim,output_dim,MARGIN,device):
        super(BERT_INT_N, self).__init__()
        self.mlp=MlP(input_dim,output_dim)
        self.device=device
        self.Criterion=nn.MarginRankingLoss(margin=MARGIN, size_average=True)
    def decode(self,res,manifolds,metrics_input):
        pos_f_ids,neg_f_ids=metrics_input['pos_f_ids'],metrics_input["neg_f_ids"]
        outlayer, _, _ = res['M0']
        f_emb=torch.cat(outlayer,dim=-1)
        pos_feature = f_emb[torch.LongTensor(pos_f_ids)]
        neg_feature = f_emb[torch.LongTensor(neg_f_ids)]
        p_score = self.mlp(pos_feature)
        n_score = self.mlp(neg_feature)
        p_score = p_score.unsqueeze(-1)#[B,1]
        n_score = n_score.unsqueeze(-1)#[B,1]
        label_y = torch.ones(p_score.shape).to(self.device)
        batch_loss = self.Criterion(p_score, n_score, label_y)
        return batch_loss
class GRU(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self,MARGIN,device):
        super(GRU, self).__init__()
        self.device=device
        self.Criterion=nn.MarginRankingLoss(margin=MARGIN, size_average=True)
    def decode(self,res,manifolds,metrics_input):
        p_score,n_score=metrics_input['p_score'],metrics_input["n_score"]
        p_score = p_score.unsqueeze(-1)#[B,1]
        n_score = n_score.unsqueeze(-1)#[B,1]
        label_y = -torch.ones(p_score.shape).to(self.device)
        batch_loss = self.Criterion(p_score, n_score, label_y)
        return batch_loss
class BERT_INT_A(BERT_INT_N):
    def __init__(self,input_dim,output_dim,MARGIN,device):
        super(BERT_INT_A, self).__init__(input_dim,output_dim,MARGIN,device)
class BERT_INT_ALL(BERT_INT_N):
    def __init__(self,input_dim,output_dim,MARGIN,device):
        super(BERT_INT_ALL, self).__init__(input_dim,output_dim,MARGIN,device)

