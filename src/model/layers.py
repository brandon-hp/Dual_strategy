import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from torch.nn.modules.module import Module
import math
def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers)
    dims = [args.feat_dim] + ([args.dim] * args.num_layers)
    n_curvatures = args.num_layers+1
    if args.c is None:
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures
class GRUAttnNet(torch.nn.Module):
    def __init__(self, embed_dim, hidden_dim, hidden_layers, dropout, device):
        super(GRUAttnNet, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.device = device

        self.build_model(dropout)
        self.out_dim = hidden_dim

    def build_model(self, dropout):
        self.attn_layer = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(inplace=True)
        ).to(self.device)

        self.gru_layer = torch.nn.GRU(self.embed_dim, self.hidden_dim, self.hidden_layers, bidirectional=True, batch_first=True, dropout=dropout).to(self.device)
        self.gru_layer_attn_w = torch.nn.GRU(self.embed_dim, self.hidden_dim, self.hidden_layers, bidirectional=True, batch_first=True, dropout=dropout).to(self.device)
        # self.gru_layer = torch.nn.LSTM(self.embed_dim, self.hidden_dim, self.hidden_layers, bidirectional=True, batch_first=True, dropout=dropout).to(self.device)

    def attn_net_with_w(self, rnn_out, rnn_hn, neighbor_mask: torch.Tensor, x):
        """
        :param rnn_out: [batch_size, seq_len, n_hidden * 2]
        :param rnn_hn: [batch_size, num_layers * num_directions, n_hidden]
        :return:
        """
        neighbor_mask = neighbor_mask.unsqueeze(1).to(self.device)
        lstm_tmp_out = torch.chunk(rnn_out, 2, -1)  # 把最后一维度分成两份
        # h [batch_size, time_step(seq_len), hidden_dims] 把两层的结果叠加？
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        #h = torch.where(neighbor_mask_dim == 1, h, neighbor_mask_dim)
        # 计算权重
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(rnn_hn, dim=1, keepdim=True) # 按维度求和
        # atten_w [batch_size, 1, hidden_dims] 算出各个隐藏状态的权重？
        atten_w = self.attn_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = torch.nn.Tanh()(h)
        # atten_w       [batch_size, 1, hidden_dims]
        # m.t(1,2)      [batch_size, hidden_dims, time_step(seq_len)]
        # atten_context [batch_size, 1, time_step(seq_len)]
        atten_context = torch.bmm(m[:,0,:].unsqueeze(1), m.transpose(1, 2)) # bmm批次中每一个step的矩阵乘法， transpose交换两个维度
        # softmax_w [batch_size, 1, time_step]
        neighbor_mask[neighbor_mask==0]=-numpy.inf
        atten_context=torch.where(neighbor_mask == 1, atten_context, neighbor_mask)
        softmax_w = torch.nn.functional.softmax(atten_context, dim=-1)  # 把最后一维度映射到[0,1]
        # 序列结果加权
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        # context = t.bmm(softmax_w, x)
        # context = t.bmm(softmax_w, t.cat((h, x), dim=-1))
        # result [batch_size, hidden_dims]
        result = context.squeeze_(1)     #squeeze(arg)表示若第arg维的维度值为1，则去掉该维度。否则tensor不变。
        return result, softmax_w

    def forward(self, x, neighbor_mask):
        rnn_out, _ = self.gru_layer(x)
        # attention
        _, hn = self.gru_layer_attn_w(x)
        # rnn_out, (hn, _) = self.gru_layer(x)
        hn: torch.Tensor
        hn = hn.permute(1, 0, 2)
        out, weights = self.attn_net_with_w(rnn_out, hn, neighbor_mask, x)

        # gru
        # lstm_tmp_out = t.chunk(rnn_out, 2, -1)
        # h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # out = h[:, -1, :].squeeze()
        # # out = rnn_out[:, -1, :].squeeze()
        #
        # # weights = t.zeros((rnn_out.shape[0], 1, rnn_out.shape[1]))
        # weights = None
        return out, weights

class AdjustableModule(nn.Module):

    def __init__(self, curvature):
        super(AdjustableModule, self).__init__()
        self.curvature = curvature

    def update_curvature(self, curvature):
        self.curvature = curvature
class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, dropout, use_bias,use_att, local_agg,use_w):
        super(HyperbolicGraphConvolution, self).__init__()
        self.manifold=manifold
        self.c=c_in
        self.dropout=dropout
        self.use_w=use_w
        if self.use_w:
            self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
    def forward(self, input):
        x, adj = input
        if self.use_w:
            h1 = self.linear.forward(x)
        else:
            h1=x
        h2 = self.agg.forward(h1, adj)
        return h2
class HighwayGate(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, out_features, c_in, c_out, dropout, use_bias):
        super(HighwayGate, self).__init__()
        self.linear_gate=HypLinear(manifold,  out_features, out_features, c_in, dropout, use_bias)
        self.c_out=c_out
        self.c=c_in
        self.manifold=manifold
    def forward(self, input1,input2):
        transform_gate = self.linear_gate.forward(input1)
        transform_gate = F.sigmoid(transform_gate)
        carry_gate = 1.0 - transform_gate

        res = self.manifold.proj(self.manifold.mobius_add(transform_gate * input2, carry_gate * input1, c=self.c),
                                 self.c)
        return self.manifold.proj(self.manifold.expmap0(self.manifold.logmap0(res, c=self.c), c=self.c_out),
                                  c=self.c_out)
class HighwayGate1(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, out_features, c_in, c_out, dropout, use_bias):
        super(HighwayGate1, self).__init__()
        self.linear_gate=HypLinear(manifold,  out_features, 1, c_in, dropout, use_bias)
        self.c_out=c_out
        self.c=c_in
        self.manifold=manifold
    def forward(self, input1,input2):
        transform_gate = self.linear_gate.forward(input1).mean()
        transform_gate = F.sigmoid(transform_gate)
        carry_gate = 1.0 - transform_gate
        print(transform_gate,carry_gate)
        res = self.manifold.proj(self.manifold.mobius_add(transform_gate * input2, carry_gate * input1, c=self.c),
                                 self.c)
        return self.manifold.proj(self.manifold.expmap0(self.manifold.logmap0(res, c=self.c), c=self.c_out),
                                  c=self.c_out)
class Mlp(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features,out_features, c_in, c_out, act,dropout, use_bias):
        super(Mlp, self).__init__()
        self.linear_gate=HypLinear(manifold,  in_features, out_features, c_out, dropout, use_bias)
        self.c_out=c_out
        self.c_in=c_in
        self.manifold=manifold
        self.act=HypAct(manifold, c_out, c_out, act)
    def forward(self, input1,input2):
        x=self.manifold.proj_tan0(self.manifold.logmap0(input1, c=self.c_in), c=self.c_in)
        for input in input2:
            x1=self.manifold.proj_tan0(self.manifold.logmap0(input, c=self.c_out), c=self.c_out)
            x=torch.cat((x,x1),dim=-1)
        out=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c_out), c=self.c_out),
                                 c=self.c_out)

        return self.linear_gate(out)
class RalationAwareGraphAttentionLayer(AdjustableModule):
    def __init__(self, manifold, input_dim, output_dim, rdim,dropout, alpha, curvature,use_bias,n_heads,heads_concat,use_w):
        """Sparse version of GAT."""
        super(RalationAwareGraphAttentionLayer, self).__init__(curvature)
        self.dropout = dropout
        self.output_dim = output_dim
        self.manifold = manifold
        self.concat=heads_concat
        self.curvature=curvature
        self.attentions = [SpRalationAwareGraphAttentionLayer(
            manifold,
            input_dim,
            output_dim,
            rdim,
            dropout=dropout,
            alpha=alpha,
            curvature=curvature,
            use_bias=use_bias) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def update_curvature(self, c):
        super(RalationAwareGraphAttentionLayer, self).update_curvature(c)
        for layer in self.attentions:
            layer.update_curvature(c)

    def forward(self, input):
        x,r,adjs= input
        if torch.any(torch.isnan(x)):
            raise ValueError('input tensor has NaaN values')
        x = F.dropout(x, self.dropout, training=self.training)
        r = F.dropout(r, self.dropout, training=self.training)
        input=x,r,adjs
        if self.concat:
            if self.manifold.name == 'Euclidean':
                h = torch.cat([att(input) for att in self.attentions], dim=1)
            else:  # No concat
                h = self.attentions[0](input)
        else:
            h=0
            for att in self.attentions:
                h+=self.manifold.proj_tan0(self.manifold.logmap0(att(input), c=self.curvature), c=self.curvature)
            h=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(F.normalize(h/len(self.attentions),p=2,dim=1), self.curvature), c=self.curvature),
                                 c=self.curvature)



        h = F.dropout(h, self.dropout, training=self.training)
        return h

class SpRalationAwareGraphAttentionLayer(AdjustableModule):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, manifold, in_features, out_features,rdim, dropout, alpha , curvature=1, use_bias=False):
        super(SpRalationAwareGraphAttentionLayer, self).__init__(curvature)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.use_bias = use_bias
        self.manifold = manifold
        self.heada = nn.Parameter(torch.zeros(size=(1, in_features)))
        nn.init.xavier_normal_(self.heada.data, gain=1.414)
        self.taila = nn.Parameter(torch.zeros(size=(1, in_features)))
        nn.init.xavier_normal_(self.taila.data, gain=1.414)
        self.ra = nn.Parameter(torch.zeros(size=(1, rdim)))
        nn.init.xavier_normal_(self.ra.data, gain=1.414)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
    def update_curvature(self, c):
        super(SpRalationAwareGraphAttentionLayer, self).update_curvature(c)
        self.linear.update_curvature(c)
    def forward(self, input):
        x,r,adjs=input
        N = x.size()[0]
        edge = adjs[0]._indices()
        h=x #self.linear(x)
        r = self.manifold.proj_tan0(self.manifold.logmap0(r, c=self.curvature), c=self.curvature)
        h = self.manifold.proj_tan0(self.manifold.logmap0(h, c=self.curvature), c=self.curvature)

        edge_e =torch.exp(self.leakyrelu((self.heada.mm(h[edge[0, :], :].t())+self.taila.mm(h[edge[1, :], :].t()+torch.matmul(adjs[1],self.ra.mm(r.t()).t()).t())).squeeze()))
        ########################Euclidean Block (START)########################
        # convert h to Euclidean space.
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        ones = torch.ones(size=(N, 1))
        if h.is_cuda:
            ones = ones.cuda()
        e_rowsum = self.special_spmm(indices=edge, values=edge_e, shape=torch.Size([N, N]), b=ones)
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(indices=edge, values=edge_e, shape=torch.Size([N, N]), b=h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        e_rowsum[e_rowsum==0]=1
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = h_prime
        ########################Euclidean Block (END)##########################
        # convert h back to Hyperbolic space (from Euclidean space).
        out = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(out, self.curvature), c=self.curvature),
                                 c=self.curvature)
        #print(out.shape)
        return out
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
class  SimplifiedRelationalGraphAttentionLayer(AdjustableModule):
    def __init__(self, manifold, input_dim, output_dim, rdim,dropout, alpha, curvature,use_bias,n_heads,heads_concat,use_w):
        """Sparse version of GAT."""
        super(SimplifiedRelationalGraphAttentionLayer, self).__init__(curvature)
        self.dropout = dropout
        self.output_dim = output_dim
        self.manifold = manifold
        self.concat=heads_concat
        self.curvature=curvature
        self.attentions = [SpSimplifiedRelationalGraphAttentionLayer(
            manifold,
            input_dim,
            output_dim,
            rdim,
            use_w,
            dropout=dropout,
            alpha=alpha,
            curvature=curvature,
            use_bias=use_bias) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def update_curvature(self, c):
        super(SimplifiedRelationalGraphAttentionLayer, self).update_curvature(c)
        for layer in self.attentions:
            layer.update_curvature(c)

    def forward(self, input):
        x,r,adjs= input
        if torch.any(torch.isnan(x)):
            raise ValueError('input tensor has NaaN values')
        x = F.dropout(x, self.dropout, training=self.training)
        r = F.dropout(r, self.dropout, training=self.training)
        input=x,r,adjs
        if self.concat:
            if self.manifold.name == 'Euclidean':
                h = torch.cat([att(input) for att in self.attentions], dim=1)
            else:  # No concat
                h = self.attentions[0](input)
        else:
            h=0
            r=0
            for att in self.attentions:
                h1,r1=att(input)
                h+=self.manifold.proj_tan0(self.manifold.logmap0(h1, c=self.curvature), c=self.curvature)
                r+=self.manifold.proj_tan0(self.manifold.logmap0(r1, c=self.curvature), c=self.curvature)
            h=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(F.normalize(h/len(self.attentions),p=2,dim=1), self.curvature), c=self.curvature),
                                 c=self.curvature)
            r=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(F.normalize(r/len(self.attentions),p=2,dim=1), self.curvature), c=self.curvature),
                                 c=self.curvature)



        h = F.dropout(h, self.dropout, training=self.training)
        return h,r

class SpSimplifiedRelationalGraphAttentionLayer(AdjustableModule):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, manifold, in_features, out_features,rdim,use_w, dropout, alpha , curvature=1, use_bias=False):
        super(SpSimplifiedRelationalGraphAttentionLayer, self).__init__(curvature)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.use_bias = use_bias
        self.manifold = manifold
        self.ra = nn.Parameter(torch.zeros(size=(1, rdim)))
        nn.init.xavier_normal_(self.ra.data, gain=1.414)
        self.use_w=use_w
        if use_w:
            self.linear=HypLinear(manifold,  out_features, out_features, curvature, dropout, use_bias)
            self.linear_r = HypLinear(manifold, rdim, rdim, curvature, dropout, use_bias)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
    def update_curvature(self, c):
        super(SpSimplifiedRelationalGraphAttentionLayer, self).update_curvature(c)
        self.linear.update_curvature(c)
    def forward(self, input):
        x,r,adjs=input
        N = x.size()[0]
        edge = adjs[0]._indices()
        edge1 = adjs[1]._indices()
        edge2 = adjs[2]._indices()
        r1=r
        h=self.linear(x) if self.use_w else x
        r=self.linear_r(r) if self.use_w else r
        r = self.manifold.proj_tan0(self.manifold.logmap0(r, c=self.curvature), c=self.curvature)
        h = self.manifold.proj_tan0(self.manifold.logmap0(h, c=self.curvature), c=self.curvature)
        if self.use_w:
            h=F.normalize(h,p=2,dim=-1)
            r=F.normalize(r,p=2,dim=-1)
        rel_sum=r[edge[0, :], :]
        neighs=h[edge2[0, :], :]
        neighs = neighs - (neighs * rel_sum).sum(dim=1).unsqueeze(1) * rel_sum
        edge_e =torch.exp(self.leakyrelu((self.ra.mm(rel_sum.t())).squeeze())).unsqueeze(1)
        e_rowsum=torch.matmul(adjs[1],edge_e)
        e_rowsum[e_rowsum == 0] = 1
        h_prime=torch.matmul(adjs[1],neighs*edge_e).div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = h_prime
        ########################Euclidean Block (END)##########################
        # convert h back to Hyperbolic space (from Euclidean space).
        out = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(out, self.curvature), c=self.curvature),
                                 c=self.curvature)
        #print(out.shape)
        return out,r1
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
class  RelationalReflectionGraphAttentionLayer(AdjustableModule):
    def __init__(self, manifold, input_dim, output_dim, rdim,dropout, alpha, curvature,use_bias,n_heads,heads_concat,use_w):
        """Sparse version of GAT."""
        super(RelationalReflectionGraphAttentionLayer, self).__init__(curvature)
        self.dropout = dropout
        self.output_dim = output_dim
        self.manifold = manifold
        self.concat=heads_concat
        self.curvature=curvature
        self.attentions = [SpRelationalReflectionGraphAttentionLayer(
            manifold,
            input_dim,
            output_dim,
            rdim,
            dropout=dropout,
            alpha=alpha,
            curvature=curvature,
            use_w=use_w,
            use_bias=use_bias) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def update_curvature(self, c):
        super(RelationalReflectionGraphAttentionLayer, self).update_curvature(c)
        for layer in self.attentions:
            layer.update_curvature(c)

    def forward(self, input):
        x,r,adjs= input
        if torch.any(torch.isnan(x)):
            raise ValueError('input tensor has NaaN values')
        if self.manifold.name == 'Euclidean':
            x = F.dropout(x, self.dropout, training=self.training)
            r = F.dropout(r, self.dropout, training=self.training)
        elif self.manifold.name =='PoincareBall':
            x=self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.curvature), c=self.curvature)
            x = F.dropout(x, self.dropout, training=self.training)
            x=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x,self.curvature),
                c=self.curvature),c=self.curvature)

            r=self.manifold.proj_tan0(self.manifold.logmap0(r, c=self.curvature), c=self.curvature)
            r = F.dropout(r, self.dropout, training=self.training)
            r=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(r,self.curvature),
                c=self.curvature),c=self.curvature)
        else:
            pass

        input=x,r,adjs
        if self.concat:
            if self.manifold.name == 'Euclidean':
                h = torch.cat([att(input) for att in self.attentions], dim=1)
            else:  # No concat
                h = self.attentions[0](input)
        else:
            h=0
            r=0
            for att in self.attentions:
                h1,r1=att(input)
                h+=self.manifold.proj_tan0(self.manifold.logmap0(h1, c=self.curvature), c=self.curvature)
                r+=self.manifold.proj_tan0(self.manifold.logmap0(r1, c=self.curvature), c=self.curvature)
            h=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(F.normalize(h/len(self.attentions),p=2,dim=1), self.curvature), c=self.curvature),
                                 c=self.curvature)
            r=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(F.normalize(r/len(self.attentions),p=2,dim=1), self.curvature), c=self.curvature),
                                 c=self.curvature)


        if self.manifold.name == 'Euclidean':
            h = F.dropout(h, self.dropout, training=self.training)
        elif self.manifold.name =='PoincareBall':
            h=self.manifold.proj_tan0(self.manifold.logmap0(h, c=self.curvature), c=self.curvature)
            h = F.dropout(h, self.dropout, training=self.training)
            h=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(h,self.curvature),
                c=self.curvature),c=self.curvature)
        else:
            pass
        return h,r

class SpRelationalReflectionGraphAttentionLayer(AdjustableModule):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, manifold, in_features, out_features,rdim, dropout, alpha , curvature=1,use_w=False, use_bias=False):
        super(SpRelationalReflectionGraphAttentionLayer, self).__init__(curvature)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.use_bias = use_bias
        self.manifold = manifold
        self.a = nn.Parameter(torch.zeros(size=(1, 2*in_features+rdim)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.use_w=use_w
        if use_w:
            # self.linear = HypLinear(manifold, out_features, out_features, curvature, dropout, use_bias)
            # self.linear_r = HypLinear(manifold, rdim, rdim, curvature, dropout, use_bias)
            self.lx = nn.Parameter(torch.zeros(size=(1, in_features)))
            nn.init.xavier_normal_(self.lx.data, gain=1.414)
            self.lr = nn.Parameter(torch.zeros(size=(1, rdim)))
            nn.init.xavier_normal_(self.lr.data, gain=1.414)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
    def update_curvature(self, c):
        super(SpRelationalReflectionGraphAttentionLayer, self).update_curvature(c)
        self.linear.update_curvature(c)
    def linear(self,x):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.curvature), c=self.curvature)
        h=h-torch.mul(torch.mul(h,self.lx).sum(1).unsqueeze(1),self.lx)
        return self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(h, self.curvature), c=self.curvature),
                                 c=self.curvature)

    def linear_r(self,x):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.curvature), c=self.curvature)
        h=h-torch.mul(torch.mul(h,self.lr).sum(1).unsqueeze(1),self.lr)
        return self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(h, self.curvature), c=self.curvature),
                                 c=self.curvature)
    def forward(self, input):
        x,r,adjs=input
        N = x.size()[0]
        edge = adjs[0]._indices()
        edge1 = adjs[1]._indices()
        edge2 = adjs[2]._indices()
        r1=r
        h=self.linear(x) if self.use_w else x
        r=self.linear_r(r) if self.use_w else r
        r = self.manifold.proj_tan0(self.manifold.logmap0(r, c=self.curvature), c=self.curvature)
        h = self.manifold.proj_tan0(self.manifold.logmap0(h, c=self.curvature), c=self.curvature)
        if self.use_w:
            h=F.normalize(h,p=2,dim=-1)
            r=F.normalize(r,p=2,dim=-1)
        rel_sum=r[edge[0, :], :]
        neighs1=h[edge2[0, :], :]
        neighs = neighs1 - (neighs1 * rel_sum).sum(dim=1).unsqueeze(1) * rel_sum
        #rel_sum1=rel_sum - (rel_sum * neighs).sum(dim=1).unsqueeze(1) * neighs
        selfs=h[edge1[0, :], :]
        edge_e =torch.exp(self.leakyrelu((self.a.mm(torch.cat([selfs,neighs,rel_sum],dim=-1).t())).squeeze())).unsqueeze(1)
        e_rowsum=torch.matmul(adjs[1],edge_e)
        e_rowsum[e_rowsum == 0] = 1
        h_prime=torch.matmul(adjs[1],neighs*edge_e).div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = h_prime
        ########################Euclidean Block (END)##########################
        # convert h back to Hyperbolic space (from Euclidean space).
        out = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(out, self.curvature), c=self.curvature),
                                 c=self.curvature)
        #print(out.shape)
        return out,r1
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class  EntityAwareRelationGraphAttentionLayer(AdjustableModule):
    def __init__(self, manifold, input_dim, output_dim, rdim,dropout, alpha, curvature,use_bias,n_heads,heads_concat):
        """Sparse version of GAT."""
        super(EntityAwareRelationGraphAttentionLayer, self).__init__(curvature)
        self.dropout = dropout
        self.output_dim = output_dim
        self.manifold = manifold
        self.concat=heads_concat
        self.curvature=curvature
        self.attentions = [SpEntityAwareRelationGraphAttentionLayer(
            manifold,
            input_dim,
            output_dim,
            rdim,
            dropout=dropout,
            alpha=alpha,
            curvature=curvature,
            use_bias=use_bias) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def update_curvature(self, c):
        super(EntityAwareRelationGraphAttentionLayer, self).update_curvature(c)
        for layer in self.attentions:
            layer.update_curvature(c)

    def forward(self, input):
        x,r,adjs= input
        if torch.any(torch.isnan(x)):
            raise ValueError('input tensor has NaaN values')
        x = F.dropout(x, self.dropout, training=self.training)
        r = F.dropout(r, self.dropout, training=self.training)
        input=x,r,adjs
        if self.concat:
            if self.manifold.name == 'Euclidean':
                h = torch.cat([att(input) for att in self.attentions], dim=1)
            else:  # No concat
                h = self.attentions[0](input)
        else:
            h=0
            for att in self.attentions:
                h+=self.manifold.proj_tan0(self.manifold.logmap0(att(input), c=self.curvature), c=self.curvature)
            h=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(h/len(self.attentions), self.curvature), c=self.curvature),
                                 c=self.curvature)



        h = F.dropout(h, self.dropout, training=self.training)
        return h

class SpEntityAwareRelationGraphAttentionLayer(AdjustableModule):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, manifold, in_features, out_features,rdim, dropout, alpha , curvature=1, use_bias=False):
        super(SpEntityAwareRelationGraphAttentionLayer, self).__init__(curvature)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.use_bias = use_bias
        self.manifold = manifold
        self.linear_h=HypLinear(manifold,  in_features, rdim, curvature, dropout, use_bias)
        self.linear_t = HypLinear(manifold, in_features, rdim, curvature, dropout, use_bias)
        self.ah1 = nn.Parameter(torch.zeros(size=(1,rdim)))
        nn.init.xavier_normal_(self.ah1.data, gain=1.414)
        self.ah2 = nn.Parameter(torch.zeros(size=(1,rdim)))
        nn.init.xavier_normal_(self.ah2.data, gain=1.414)
        self.at1 = nn.Parameter(torch.zeros(size=(1,rdim)))
        nn.init.xavier_normal_(self.at1.data, gain=1.414)
        self.at2 = nn.Parameter(torch.zeros(size=(1,rdim)))
        nn.init.xavier_normal_(self.at2.data, gain=1.414)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
    def update_curvature(self, c):
        super(SpEntityAwareRelationGraphAttentionLayer, self).update_curvature(c)
        self.linear.update_curvature(c)
    def forward(self, input):
        x,r,adjs=input
        N = x.size()[0]
        edge1 = adjs[1]._indices()
        edge2 = adjs[2]._indices()
        x_r_h=self.linear_h(x)
        x_r_t=self.linear_t(x)
        x_r_h = self.manifold.proj_tan0(self.manifold.logmap0(x_r_h, c=self.curvature), c=self.curvature)
        x_r_t = self.manifold.proj_tan0(self.manifold.logmap0(x_r_t, c=self.curvature), c=self.curvature)
        edge_e1 = torch.exp( self.leakyrelu((self.ah1.mm(x_r_h[edge1[0, :], :].t())+self.ah2.mm(x_r_t[edge2[0, :], :].t())).squeeze())).unsqueeze(1)
        edge_e2 = torch.exp( self.leakyrelu((self.at1.mm(x_r_h[edge1[0, :], :].t())+self.at2.mm(x_r_t[edge2[0, :], :].t())).squeeze())).unsqueeze(1)
        neighs1=edge_e1*x_r_h[edge1[0, :], :]
        e_rowsum=torch.matmul(adjs[0],edge_e1)
        e_rowsum[e_rowsum == 0] = 1
        r_h=torch.matmul(adjs[0],neighs1).div(e_rowsum)

        neighs2 = edge_e2 * x_r_t[edge2[0, :], :]
        e_rowsum = torch.matmul(adjs[0], edge_e2)
        e_rowsum[e_rowsum == 0] = 1
        r_t = torch.matmul(adjs[0], neighs2).div(e_rowsum)
        h_prime=r_h+r_t

        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = h_prime
        ########################Euclidean Block (END)##########################
        # convert h back to Hyperbolic space (from Euclidean space).
        out = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(out, self.curvature), c=self.curvature),
                                 c=self.curvature)
        #print(out.shape)
        return out
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
class  EntityAwareRelationGraphAttentionLayer1(AdjustableModule):
    def __init__(self, manifold, input_dim, output_dim, rdim,dropout, alpha, curvature,use_bias,n_heads,heads_concat):
        """Sparse version of GAT."""
        super(EntityAwareRelationGraphAttentionLayer1, self).__init__(curvature)
        self.dropout = dropout
        self.output_dim = output_dim
        self.manifold = manifold
        self.concat=heads_concat
        self.curvature=curvature
        self.attentions = [SpEntityAwareRelationGraphAttentionLayer1(
            manifold,
            input_dim,
            output_dim,
            rdim,
            dropout=dropout,
            alpha=alpha,
            curvature=curvature,
            use_bias=use_bias) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def update_curvature(self, c):
        super(EntityAwareRelationGraphAttentionLayer1, self).update_curvature(c)
        for layer in self.attentions:
            layer.update_curvature(c)

    def forward(self, input):
        x,r,adjs= input
        if torch.any(torch.isnan(x)):
            raise ValueError('input tensor has NaaN values')
        x = F.dropout(x, self.dropout, training=self.training)
        r = F.dropout(r, self.dropout, training=self.training)
        input=x,r,adjs
        if self.concat:
            if self.manifold.name == 'Euclidean':
                h = torch.cat([att(input) for att in self.attentions], dim=1)
            else:  # No concat
                h = self.attentions[0](input)
        else:
            h=0
            for att in self.attentions:
                h+=self.manifold.proj_tan0(self.manifold.logmap0(att(input), c=self.curvature), c=self.curvature)
            h=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(h/len(self.attentions), self.curvature), c=self.curvature),
                                 c=self.curvature)



        h = F.dropout(h, self.dropout, training=self.training)
        return h

class SpEntityAwareRelationGraphAttentionLayer1(AdjustableModule):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, manifold, in_features, out_features,rdim, dropout, alpha , curvature=1, use_bias=False):
        super(SpEntityAwareRelationGraphAttentionLayer1, self).__init__(curvature)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.use_bias = use_bias
        self.manifold = manifold
        self.linear_h=HypLinear(manifold,  in_features, rdim, curvature, dropout, use_bias)
        self.linear_t = HypLinear(manifold, in_features, rdim, curvature, dropout, use_bias)
        self.ah1 = nn.Parameter(torch.zeros(size=(1,rdim)))
        nn.init.xavier_normal_(self.ah1.data, gain=1.414)
        self.ah2 = nn.Parameter(torch.zeros(size=(1,rdim)))
        nn.init.xavier_normal_(self.ah2.data, gain=1.414)
        self.at1 = nn.Parameter(torch.zeros(size=(1,rdim)))
        nn.init.xavier_normal_(self.at1.data, gain=1.414)
        self.at2 = nn.Parameter(torch.zeros(size=(1,rdim)))
        nn.init.xavier_normal_(self.at2.data, gain=1.414)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
    def update_curvature(self, c):
        super(SpEntityAwareRelationGraphAttentionLayer1, self).update_curvature(c)
        self.linear.update_curvature(c)
    def forward(self, input):
        x,r,adjs=input
        N = x.size()[0]
        edge1 = adjs[1]._indices()
        edge2 = adjs[2]._indices()
        x_r_h=self.linear_h(x)
        x_r_t=self.linear_t(x)
        x_r_h = self.manifold.proj_tan0(self.manifold.logmap0(x_r_h, c=self.curvature), c=self.curvature)
        x_r_t = self.manifold.proj_tan0(self.manifold.logmap0(x_r_t, c=self.curvature), c=self.curvature)
        edge_e1 = torch.exp( self.leakyrelu((self.ah2.mm(x_r_t[edge2[0, :], :].t())).squeeze())).unsqueeze(1)
        edge_e2 = torch.exp( self.leakyrelu((self.at2.mm(x_r_t[edge2[0, :], :].t())).squeeze())).unsqueeze(1)
        neighs1=edge_e1*x_r_h[edge1[0, :], :]
        e_rowsum=torch.matmul(adjs[0],edge_e1)
        e_rowsum[e_rowsum == 0] = 1
        r_h=torch.matmul(adjs[0],neighs1).div(e_rowsum)

        neighs2 = edge_e2 * x_r_t[edge2[0, :], :]
        e_rowsum = torch.matmul(adjs[0], edge_e2)
        e_rowsum[e_rowsum == 0] = 1
        r_t = torch.matmul(adjs[0], neighs2).div(e_rowsum)
        h_prime=torch.cat((r_h,r_t),dim=1)#r_h+r_t

        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = F.normalize(h_prime,p=2,dim=1)
        ########################Euclidean Block (END)##########################
        # convert h back to Hyperbolic space (from Euclidean space).
        out = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(out, self.curvature), c=self.curvature),
                                 c=self.curvature)
        #print(out.shape)
        return out
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class  RelationAwareEntityGraphAttentionLayer(AdjustableModule):
    def __init__(self, manifold, input_dim, output_dim, rdim,dropout, alpha, curvature,use_bias,n_heads,heads_concat):
        """Sparse version of GAT."""
        super(RelationAwareEntityGraphAttentionLayer, self).__init__(curvature)
        self.dropout = dropout
        self.output_dim = output_dim
        self.manifold = manifold
        self.concat=heads_concat
        self.curvature=curvature
        self.attentions = [SpRelationAwareEntityGraphAttentionLayer(
            manifold,
            input_dim,
            output_dim,
            rdim,
            dropout=dropout,
            alpha=alpha,
            curvature=curvature,
            use_bias=use_bias) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def update_curvature(self, c):
        super(RelationAwareEntityGraphAttentionLayer, self).update_curvature(c)
        for layer in self.attentions:
            layer.update_curvature(c)

    def forward(self, input):
        x,r,adjs= input
        if torch.any(torch.isnan(x)):
            raise ValueError('input tensor has NaaN values')
        x = F.dropout(x, self.dropout, training=self.training)
        r = F.dropout(r, self.dropout, training=self.training)
        input=x,r,adjs
        if self.concat:
            if self.manifold.name == 'Euclidean':
                h = torch.cat([att(input) for att in self.attentions], dim=1)
            else:  # No concat
                h = self.attentions[0](input)
        else:
            h=0
            for att in self.attentions:
                h+=self.manifold.proj_tan0(self.manifold.logmap0(att(input), c=self.curvature), c=self.curvature)
            h=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(h/len(self.attentions), self.curvature), c=self.curvature),
                                 c=self.curvature)



        h = F.dropout(h, self.dropout, training=self.training)
        return h

class SpRelationAwareEntityGraphAttentionLayer(AdjustableModule):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, manifold, in_features, out_features,rdim, dropout, alpha , curvature=1, use_bias=False):
        super(SpRelationAwareEntityGraphAttentionLayer, self).__init__(curvature)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.use_bias = use_bias
        self.manifold = manifold
        self.linear_h=HypLinear(manifold,  in_features, 1, curvature, dropout, use_bias)
        self.linear_t = HypLinear(manifold, in_features, 1, curvature, dropout, use_bias)
        self.linear_r = HypLinear(manifold, rdim, 1, curvature, dropout, use_bias)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
    def update_curvature(self, c):
        super(SpRelationAwareEntityGraphAttentionLayer, self).update_curvature(c)
        self.linear.update_curvature(c)
    def forward(self, input):
        x,r,adjs=input
        N = x.size()[0]
        edge=adjs[0]._indices()
        edge1 = adjs[1]._indices()
        edge2 = adjs[2]._indices()
        e_h=self.linear_h(x)
        e_t=self.linear_t(x)
        e_r=self.linear_r(r)
        e_h = self.manifold.proj_tan0(self.manifold.logmap0(e_h, c=self.curvature), c=self.curvature)
        e_t = self.manifold.proj_tan0(self.manifold.logmap0(e_t, c=self.curvature), c=self.curvature)
        e_r = self.manifold.proj_tan0(self.manifold.logmap0(e_r, c=self.curvature), c=self.curvature)
        x = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.curvature), c=self.curvature)
        r = self.manifold.proj_tan0(self.manifold.logmap0(r, c=self.curvature), c=self.curvature)
        edge_e1 = torch.exp(self.leakyrelu(e_h[edge1[0, :], :]+e_r[edge[0, :], :]))
        edge_e2 = torch.exp(self.leakyrelu(e_t[edge2[0, :], :]+e_r[edge[0, :], :]))
        neighs1 = edge_e1 * r[edge[0, :], :]
        e_rowsum=torch.matmul(adjs[1],edge_e1)
        e_rowsum[e_rowsum == 0] = 1
        x_h=torch.matmul(adjs[1],neighs1).div(e_rowsum)

        neighs2 = edge_e2 * r[edge[0, :], :]
        e_rowsum = torch.matmul(adjs[2], edge_e2)
        e_rowsum[e_rowsum == 0] = 1
        x_t = torch.matmul(adjs[2], neighs2).div(e_rowsum)
        h_prime=torch.cat((x,x_h,x_t),dim=-1)

        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = h_prime
        ########################Euclidean Block (END)##########################
        # convert h back to Hyperbolic space (from Euclidean space).
        out = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(out, self.curvature), c=self.curvature),
                                 c=self.curvature)
        #print(out.shape)
        return out
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
class  ValGraphAttentionLayer(AdjustableModule):
    def __init__(self, manifold, input_dim, output_dim,dropout, alpha, curvature,use_bias,n_heads,heads_concat,use_w=True):
        """Sparse version of GAT."""
        super(ValGraphAttentionLayer, self).__init__(curvature)
        self.dropout = dropout
        self.output_dim = output_dim
        self.manifold = manifold
        self.concat=heads_concat
        self.curvature=curvature
        self.attentions = [SpValGraphAttentionLayer(
            manifold,
            input_dim,
            output_dim,
            dropout=dropout,
            alpha=alpha,
            curvature=curvature,
            use_w=use_w,
            use_bias=use_bias) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def update_curvature(self, c):
        super(ValGraphAttentionLayer, self).update_curvature(c)
        for layer in self.attentions:
            layer.update_curvature(c)

    def forward(self, input):
        x,r,a,adjs= input
        if torch.any(torch.isnan(x)):
            raise ValueError('input tensor has NaaN values')
        x = F.dropout(x, self.dropout, training=self.training)
        r = F.dropout(r, self.dropout, training=self.training)
        a = F.dropout(a, self.dropout, training=self.training)
        input=x,r,a,adjs
        if self.concat:
            if self.manifold.name == 'Euclidean':
                h = torch.cat([att(input) for att in self.attentions], dim=1)
            else:  # No concat
                h = self.attentions[0](input)
        else:
            h=0
            for att in self.attentions:
                h+=self.manifold.proj_tan0(self.manifold.logmap0(att(input), c=self.curvature), c=self.curvature)
            h=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(F.relu(F.normalize(h/len(self.attentions),p=2,dim=1)), self.curvature), c=self.curvature),
                                 c=self.curvature)



        h = F.dropout(h, self.dropout, training=self.training)
        return h

class SpValGraphAttentionLayer(AdjustableModule):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, manifold, in_features, out_features, dropout, alpha , curvature=1, use_w=True,use_bias=False):
        super(SpValGraphAttentionLayer, self).__init__(curvature)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.use_bias = use_bias
        self.use_w=use_w
        self.manifold = manifold
        self.a = nn.Parameter(torch.zeros(size=(1, in_features+2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        if self.use_w:
            self.linear = HypLinear(manifold, out_features, out_features, curvature, dropout, use_bias)
            #self.linear_z = HypLinear(manifold, in_features,in_features, curvature, dropout, use_bias)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
    def update_curvature(self, c):
        super(SpValGraphAttentionLayer, self).update_curvature(c)
        self.linear.update_curvature(c)
    def forward(self, input):
        x,r,a,adjs=input
        N = x.size()[0]
        M=r.size()[0]
        edge = adjs[0]._indices()
        edge1 = adjs[1]._indices()
        edge2 = adjs[2]._indices()
        r = self.linear(r) if self.use_w else r
        #x = self.linear_z(x) if self.use_w else x
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.curvature), c=self.curvature)
        r = self.manifold.proj_tan0(self.manifold.logmap0(r, c=self.curvature), c=self.curvature)
        a = self.manifold.proj_tan0(self.manifold.logmap0(a, c=self.curvature), c=self.curvature)
        edge_e = torch.exp(self.leakyrelu(
            self.a.mm(torch.cat((h[edge[0, :], :],a[edge1[0, :], :],r[edge2[0, :], :]), dim=-1).t()).squeeze())).unsqueeze(1)
        neighs=r[edge2[0, :], :]
        e_rowsum = torch.matmul(adjs[0], edge_e)
        e_rowsum[e_rowsum == 0] = 1
        h_prime = torch.matmul(adjs[0], neighs * edge_e).div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = h_prime
        '''
        assert not torch.isnan(edge_e).any()
        # edge_e: E
        ones = torch.ones(size=(M, 1))
        if h.is_cuda:
            ones = ones.cuda()
        e_rowsum = self.special_spmm(indices=edge, values=edge_e, shape=torch.Size([N, M]), b=ones)
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E
        h_prime = self.special_spmm(indices=edge, values=edge_e, shape=torch.Size([N, M]), b=r)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        e_rowsum[e_rowsum==0]=1
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = h_prime
        '''

        ########################Euclidean Block (END)##########################
        # convert h back to Hyperbolic space (from Euclidean space).
        out = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(out, self.curvature), c=self.curvature),
                                 c=self.curvature)

        #print(out.shape)
        return out
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
class  relGraphAttentionLayer(AdjustableModule):
    def __init__(self, manifold, input_dim, output_dim,dropout, alpha, curvature,use_bias,n_heads,heads_concat,use_w=True):
        """Sparse version of GAT."""
        super(relGraphAttentionLayer, self).__init__(curvature)
        self.dropout = dropout
        self.output_dim = output_dim
        self.manifold = manifold
        self.concat=heads_concat
        self.curvature=curvature
        self.attentions = [SprelGraphAttentionLayer(
            manifold,
            input_dim,
            output_dim,
            dropout=dropout,
            alpha=alpha,
            curvature=curvature,
            use_w=use_w,
            use_bias=use_bias) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def update_curvature(self, c):
        super(relGraphAttentionLayer, self).update_curvature(c)
        for layer in self.attentions:
            layer.update_curvature(c)

    def forward(self, input):
        x,r,adjs= input
        if torch.any(torch.isnan(x)):
            raise ValueError('input tensor has NaaN values')
        x = F.dropout(x, self.dropout, training=self.training)
        r = F.dropout(r, self.dropout, training=self.training)
        input=x,r,adjs
        if self.concat:
            if self.manifold.name == 'Euclidean':
                h = torch.cat([att(input) for att in self.attentions], dim=1)
            else:  # No concat
                h = self.attentions[0](input)
        else:
            h=0
            for att in self.attentions:
                h+=self.manifold.proj_tan0(self.manifold.logmap0(att(input), c=self.curvature), c=self.curvature)
            h=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(h/len(self.attentions), self.curvature), c=self.curvature),
                                 c=self.curvature)



        h = F.dropout(h, self.dropout, training=self.training)
        return h

class SprelGraphAttentionLayer(AdjustableModule):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, manifold, in_features, out_features, dropout, alpha , curvature=1, use_w=True,use_bias=False):
        super(SprelGraphAttentionLayer, self).__init__(curvature)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.use_bias = use_bias
        self.use_w=use_w
        self.manifold = manifold
        self.a = nn.Parameter(torch.zeros(size=(1, in_features+out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        if self.use_w:
            self.linear = HypLinear(manifold, out_features, out_features, curvature, dropout, use_bias)
            #self.linear_z = HypLinear(manifold, in_features,in_features, curvature, dropout, use_bias)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
    def update_curvature(self, c):
        super(SprelGraphAttentionLayer, self).update_curvature(c)
        self.linear.update_curvature(c)
    def forward(self, input):
        x,r,adjs=input
        N = x.size()[0]
        M = r.size()[0]
        edge = adjs[0]._indices()
        edge1 = adjs[1]._indices()
        r = self.linear(r) if self.use_w else r
        # x = self.linear_z(x) if self.use_w else x
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.curvature), c=self.curvature)
        r = self.manifold.proj_tan0(self.manifold.logmap0(r, c=self.curvature), c=self.curvature)
        edge_e = torch.exp(self.leakyrelu(
            self.a.mm(
                torch.cat((h[edge1[0, :], :],r[edge[0, :], :]), dim=-1).t()).squeeze())).unsqueeze(
            1)
        neighs = r[edge[0, :], :]
        e_rowsum = torch.matmul(adjs[1], edge_e)
        e_rowsum[e_rowsum == 0] = 1
        h_prime = torch.matmul(adjs[1], neighs * edge_e).div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = h_prime

        out = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(out, self.curvature), c=self.curvature),
                                 c=self.curvature)

        #print(out.shape)
        return out
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class  GraphAttentionLayer(AdjustableModule):
    def __init__(self, manifold, input_dim, output_dim,dropout, alpha, curvature,use_bias,n_heads,heads_concat,use_w=True):
        """Sparse version of GAT."""
        super(GraphAttentionLayer, self).__init__(curvature)
        self.dropout = dropout
        self.output_dim = output_dim
        self.manifold = manifold
        self.concat=heads_concat
        self.curvature=curvature
        self.attentions = [SpGraphAttentionLayer(
            manifold,
            input_dim,
            output_dim,
            dropout=dropout,
            alpha=alpha,
            curvature=curvature,
            use_w=use_w,
            use_bias=use_bias) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def update_curvature(self, c):
        super(GraphAttentionLayer, self).update_curvature(c)
        for layer in self.attentions:
            layer.update_curvature(c)

    def forward(self, input):
        x,r,adjs= input
        if torch.any(torch.isnan(x)):
            raise ValueError('input tensor has NaaN values')
        x = F.dropout(x, self.dropout, training=self.training)
        r = F.dropout(r, self.dropout, training=self.training)
        input=x,r,adjs
        if self.concat:
            if self.manifold.name == 'Euclidean':
                h = torch.cat([att(input) for att in self.attentions], dim=1)
            else:  # No concat
                h = self.attentions[0](input)
        else:
            h=0
            for att in self.attentions:
                h+=self.manifold.proj_tan0(self.manifold.logmap0(att(input), c=self.curvature), c=self.curvature)
            h=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(F.normalize(h/len(self.attentions),p=2,dim=1), self.curvature), c=self.curvature),
                                 c=self.curvature)



        h = F.dropout(h, self.dropout, training=self.training)
        return h
class  GraphAttentionLayerTest(AdjustableModule):
    def __init__(self, manifold, input_dim, output_dim,rdim,dropout, alpha, curvature,use_bias,n_heads,heads_concat,use_w=True):
        """Sparse version of GAT."""
        super(GraphAttentionLayerTest, self).__init__(curvature)
        self.dropout = dropout
        self.output_dim = output_dim
        self.manifold = manifold
        self.concat=heads_concat
        self.curvature=curvature
        self.attentions = [SpGraphAttentionLayerTest(
            manifold,
            input_dim,
            output_dim,
            rdim,
            dropout=dropout,
            alpha=alpha,
            curvature=curvature,
            use_w=use_w,
            use_bias=use_bias) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def update_curvature(self, c):
        super(GraphAttentionLayerTest, self).update_curvature(c)
        for layer in self.attentions:
            layer.update_curvature(c)

    def forward(self, input):
        x,r,adjs= input
        if torch.any(torch.isnan(x)):
            raise ValueError('input tensor has NaaN values')
        x = F.dropout(x, self.dropout, training=self.training)
        r = F.dropout(r, self.dropout, training=self.training)
        input=x,r,adjs
        if self.concat:
            if self.manifold.name == 'Euclidean':
                h = torch.cat([att(input) for att in self.attentions], dim=1)
            else:  # No concat
                h = self.attentions[0](input)
        else:
            h=0
            for att in self.attentions:
                h+=self.manifold.proj_tan0(self.manifold.logmap0(att(input), c=self.curvature), c=self.curvature)
            h=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(F.normalize(h/len(self.attentions),p=2,dim=1), self.curvature), c=self.curvature),
                                 c=self.curvature)



        h = F.dropout(h, self.dropout, training=self.training)
        return h
class  GraphAttentionLayerKecg(AdjustableModule):
    def __init__(self, manifold, input_dim, output_dim,dropout, alpha, curvature,use_bias,n_heads,heads_concat,use_w=True):
        """Sparse version of GAT."""
        super(GraphAttentionLayerKecg, self).__init__(curvature)
        self.dropout = dropout
        self.output_dim = output_dim
        self.manifold = manifold
        self.concat=heads_concat
        self.curvature=curvature
        self.attentions = [SpGraphAttentionLayer(
            manifold,
            input_dim,
            output_dim,
            dropout=dropout,
            alpha=alpha,
            curvature=curvature,
            use_w=use_w,
            use_bias=use_bias) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def update_curvature(self, c):
        super(GraphAttentionLayerKecg, self).update_curvature(c)
        for layer in self.attentions:
            layer.update_curvature(c)

    def forward(self, input):
        x,r,adjs= input
        if torch.any(torch.isnan(x)):
            raise ValueError('input tensor has NaaN values')
        x = F.dropout(x, self.dropout, training=self.training)
        r = F.dropout(r, self.dropout, training=self.training)
        input=x,r,adjs
        if self.concat:
            if self.manifold.name == 'Euclidean':
                h = torch.cat([att(input) for att in self.attentions], dim=1)
            else:  # No concat
                h = self.attentions[0](input)
        else:
            h=0
            for att in self.attentions:
                h+=self.manifold.proj_tan0(self.manifold.logmap0(att(input), c=self.curvature), c=self.curvature)
            h=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(h/len(self.attentions), self.curvature), c=self.curvature),
                                 c=self.curvature)



        h = F.dropout(h, self.dropout, training=self.training)
        return h

class SpGraphAttentionLayer(AdjustableModule):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, manifold, in_features, out_features, dropout, alpha , curvature=1, use_w=True,use_bias=False):
        super(SpGraphAttentionLayer, self).__init__(curvature)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.use_bias = use_bias
        self.use_w=use_w
        self.manifold = manifold
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        if self.use_w:
            self.linear = HypLinear(manifold, in_features, out_features, curvature, dropout, use_bias)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
    def update_curvature(self, c):
        super(SpGraphAttentionLayer, self).update_curvature(c)
        self.linear.update_curvature(c)
    def forward(self, input):
        x,r,adjs=input
        N = x.size()[0]
        edge1 = adjs[1]._indices()
        edge2 = adjs[2]._indices()
        r1 = r
        x = self.linear(x) if self.use_w else x
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.curvature), c=self.curvature)
        edge_e = torch.exp(self.leakyrelu(
            self.a.mm(torch.cat((h[edge1[0, :], :], h[edge2[0, :], :]), dim=-1).t()).squeeze())).unsqueeze(1)
        neighs = edge_e * h[edge2[0, :], :]
        e_rowsum = torch.matmul(adjs[1], edge_e)
        e_rowsum[e_rowsum == 0] = 1
        h_prime = torch.matmul(adjs[1], neighs).div(e_rowsum)

        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = h_prime
        ########################Euclidean Block (END)##########################
        # convert h back to Hyperbolic space (from Euclidean space).
        out = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(out, self.curvature), c=self.curvature),
                                 c=self.curvature)

        '''
        edge = adjs[0]._indices()
        x=self.linear(x) if self.use_w else x
        h= self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.curvature), c=self.curvature)
        edge_e = torch.exp(self.leakyrelu(self.a.mm(torch.cat((h[edge[0, :], :],h[edge[1, :], :]),dim=-1).t()).squeeze()))
        ########################Euclidean Block (START)########################
        # convert h to Euclidean space.
        assert not torch.isnan(edge_e).any()
        # edge_e: E
        ones = torch.ones(size=(N, 1))
        if h.is_cuda:
            ones = ones.cuda()
        e_rowsum = self.special_spmm(indices=edge, values=edge_e, shape=torch.Size([N, N]), b=ones)
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E
        h_prime = self.special_spmm(indices=edge, values=edge_e, shape=torch.Size([N, N]), b=h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        e_rowsum[e_rowsum==0]=1
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = h_prime

        ########################Euclidean Block (END)##########################
        # convert h back to Hyperbolic space (from Euclidean space).
        out = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(out, self.curvature), c=self.curvature),
                                 c=self.curvature)
        '''
        #print(out.shape)
        return out
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
class  HyperbolicGraphAttentionLayer(AdjustableModule):
    def __init__(self, manifold, input_dim, output_dim,dropout, alpha, curvature,use_bias,n_heads,heads_concat,use_w=True):
        """Sparse version of GAT."""
        super(HyperbolicGraphAttentionLayer, self).__init__(curvature)
        self.dropout = dropout
        self.output_dim = output_dim
        self.manifold = manifold
        self.concat=heads_concat
        self.curvature=curvature
        self.attentions = [SpHyperbolicGraphAttentionLayer(
            manifold,
            input_dim,
            output_dim,
            dropout=dropout,
            alpha=alpha,
            curvature=curvature,
            use_w=use_w,
            use_bias=use_bias) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def update_curvature(self, c):
        super(HyperbolicGraphAttentionLayer, self).update_curvature(c)
        for layer in self.attentions:
            layer.update_curvature(c)

    def forward(self, input):
        x,r,adjs= input
        if torch.any(torch.isnan(x)):
            raise ValueError('input tensor has NaaN values')
        input=x,r,adjs
        if self.concat:
            if self.manifold.name == 'Euclidean':
                h = torch.cat([att(input) for att in self.attentions], dim=1)
            else:  # No concat
                h = self.attentions[0](input)
        else:
            h=0
            for att in self.attentions:
                h+=self.manifold.proj_tan0(self.manifold.logmap0(att(input), c=self.curvature), c=self.curvature)
            h=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(h/len(self.attentions), self.curvature), c=self.curvature),
                                 c=self.curvature)
        return h

class SpHyperbolicGraphAttentionLayer(AdjustableModule):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, manifold, in_features, out_features, dropout, alpha , curvature=1, use_w=True,use_bias=False):
        super(SpHyperbolicGraphAttentionLayer, self).__init__(curvature)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.use_bias = use_bias
        self.use_w=use_w
        self.manifold = manifold
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        if self.use_w:
            self.linear = HypLinear(manifold, in_features, out_features, curvature, dropout, use_bias)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
    def update_curvature(self, c):
        super(SpHyperbolicGraphAttentionLayer, self).update_curvature(c)
        self.linear.update_curvature(c)
    def forward(self, input):
        x,r,adjs=input
        N = x.size()[0]
        edge1 = adjs[1]._indices()
        edge2 = adjs[2]._indices()
        r1 = r
        x = self.linear(x) if self.use_w else x
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.curvature), c=self.curvature)
        edge_e = torch.exp(self.leakyrelu(
            self.a.mm(torch.cat((h[edge1[0, :], :], h[edge2[0, :], :]), dim=-1).t()).squeeze())).unsqueeze(1)
        e_rowsum = torch.matmul(adjs[1], edge_e)
        e_rowsum[e_rowsum == 0] = 1
        neighs = edge_e * h[edge2[0, :], :]
        h_prime = torch.matmul(adjs[1], neighs).div(e_rowsum)

        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = h_prime
        ########################Euclidean Block (END)##########################
        # convert h back to Hyperbolic space (from Euclidean space).
        out = self.dropout(out)
        out = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(out, self.curvature), c=self.curvature),
                                 c=self.curvature)

        '''
        edge = adjs[0]._indices()
        x=self.linear(x) if self.use_w else x
        h= self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.curvature), c=self.curvature)
        edge_e = torch.exp(self.leakyrelu(self.a.mm(torch.cat((h[edge[0, :], :],h[edge[1, :], :]),dim=-1).t()).squeeze()))
        ########################Euclidean Block (START)########################
        # convert h to Euclidean space.
        assert not torch.isnan(edge_e).any()
        # edge_e: E
        ones = torch.ones(size=(N, 1))
        if h.is_cuda:
            ones = ones.cuda()
        e_rowsum = self.special_spmm(indices=edge, values=edge_e, shape=torch.Size([N, N]), b=ones)
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E
        h_prime = self.special_spmm(indices=edge, values=edge_e, shape=torch.Size([N, N]), b=h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        e_rowsum[e_rowsum==0]=1
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = h_prime

        ########################Euclidean Block (END)##########################
        # convert h back to Hyperbolic space (from Euclidean space).
        out = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(out, self.curvature), c=self.curvature),
                                 c=self.curvature)
        '''
        #print(out.shape)
        return out
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
class SpGraphAttentionLayerTest(AdjustableModule):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, manifold, in_features, out_features, rdim, dropout, alpha, curvature=1, use_bias=False,
                 use_w=False):
        super(SpGraphAttentionLayerTest, self).__init__(curvature)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.use_bias = use_bias
        self.manifold = manifold
        self.use_w = use_w
        self.a = nn.Parameter(torch.zeros(size=(1,  rdim)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        if self.use_w:
            #self.linear = HypLinear(manifold, in_features, out_features, curvature, dropout, use_bias)
            self.linear_r = HypLinear(manifold, rdim, rdim, curvature, dropout, use_bias)
        self.a1 = nn.Parameter(torch.zeros(size=(1, in_features)))
        nn.init.xavier_normal_(self.a1.data, gain=1.414)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def update_curvature(self, c):
        super(SpGraphAttentionLayerTest, self).update_curvature(c)
        self.linear.update_curvature(c)


    def forward(self, input):
        x, r, adjs = input
        N = x.size()[0]
        edge = adjs[0]._indices()
        edge1 = adjs[1]._indices()
        edge2 = adjs[2]._indices()
        r1 = r
        x = torch.mul(x,self.a1)
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.curvature), c=self.curvature)
        r = self.manifold.proj_tan0(self.manifold.logmap0(r, c=self.curvature), c=self.curvature)
        edge_e = torch.exp(self.leakyrelu(self.a.mm(r[edge[0, :], :].t()).squeeze())).unsqueeze(1)
        neighs = edge_e * h[edge2[0, :], :]
        e_rowsum = torch.matmul(adjs[1], edge_e)
        e_rowsum[e_rowsum == 0] = 1
        h_prime = torch.matmul(adjs[1], neighs).div(e_rowsum)

        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = h_prime
        ########################Euclidean Block (END)##########################
        # convert h back to Hyperbolic space (from Euclidean space).
        out = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(out, self.curvature), c=self.curvature),
            c=self.curvature)
        # print(out.shape)
        return out
class  Test(AdjustableModule):
    def __init__(self, manifold, input_dim, output_dim, rdim, dropout, alpha, curvature, use_bias, n_heads,
                 heads_concat, use_w=False):
        """Sparse version of GAT."""
        super(Test, self).__init__(curvature)
        self.dropout = dropout
        self.output_dim = output_dim
        self.manifold = manifold
        self.concat = heads_concat
        self.curvature = curvature
        self.attentions = [SpTest(
            manifold,
            input_dim,
            output_dim,
            rdim,
            dropout=dropout,
            alpha=alpha,
            curvature=curvature,
            use_bias=use_bias,
            use_w=use_w
        ) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def update_curvature(self, c):
        super(Test, self).update_curvature(c)
        for layer in self.attentions:
            layer.update_curvature(c)

    def forward(self, input):
        x, r, adjs = input
        if torch.any(torch.isnan(x)):
            raise ValueError('input tensor has NaaN values')
        x = F.dropout(x, self.dropout, training=self.training)
        r = F.dropout(r, self.dropout, training=self.training)
        input = x, r, adjs
        if self.concat:
            if self.manifold.name == 'Euclidean':
                h = torch.cat([att(input) for att in self.attentions], dim=1)
            else:  # No concat
                h = self.attentions[0](input)
        else:
            h = 0
            r = 0
            for att in self.attentions:
                h1, r1 = att(input)
                h += self.manifold.proj_tan0(self.manifold.logmap0(h1, c=self.curvature), c=self.curvature)
                r += self.manifold.proj_tan0(self.manifold.logmap0(r1, c=self.curvature), c=self.curvature)
            h = self.manifold.proj(self.manifold.expmap0(
                self.manifold.proj_tan0(F.normalize(h / len(self.attentions), p=2, dim=1), self.curvature),
                c=self.curvature),
                                   c=self.curvature)
            r = self.manifold.proj(self.manifold.expmap0(
                self.manifold.proj_tan0(F.normalize(r / len(self.attentions), p=2, dim=1), self.curvature),
                c=self.curvature),
                                   c=self.curvature)

        h = F.dropout(h, self.dropout, training=self.training)
        return h, r

class SpTest(AdjustableModule):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, manifold, in_features, out_features, rdim, dropout, alpha, curvature=1, use_bias=False,
                 use_w=False):
        super(SpTest, self).__init__(curvature)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.use_bias = use_bias
        self.manifold = manifold
        self.use_w = use_w
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * in_features + rdim)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        if self.use_w:
            #self.linear = HypLinear(manifold, in_features, out_features, curvature, dropout, use_bias)
            self.linear_r = HypLinear(manifold, rdim, rdim, curvature, dropout, use_bias)
        self.a1 = nn.Parameter(torch.zeros(size=(1, in_features)))
        nn.init.xavier_normal_(self.a1.data, gain=1.414)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def update_curvature(self, c):
        super(SpTest, self).update_curvature(c)
        self.linear.update_curvature(c)


    def forward(self, input):
        x, r, adjs = input
        N = x.size()[0]
        edge = adjs[0]._indices()
        edge1 = adjs[1]._indices()
        edge2 = adjs[2]._indices()
        r1 = r
        x = torch.mul(x,self.a1)
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.curvature), c=self.curvature)
        r = self.manifold.proj_tan0(self.manifold.logmap0(r, c=self.curvature), c=self.curvature)
        edge_e = torch.exp(self.leakyrelu(self.a.mm((torch.cat(
            (h[edge1[0, :], :], r[edge[0, :], :], h[edge2[0, :], :]), dim=-1)).t()).squeeze())).unsqueeze(1)
        neighs = edge_e * h[edge2[0, :], :]
        e_rowsum = torch.matmul(adjs[1], edge_e)
        e_rowsum[e_rowsum == 0] = 1
        h_prime = torch.matmul(adjs[1], neighs).div(e_rowsum)

        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = h_prime
        ########################Euclidean Block (END)##########################
        # convert h back to Hyperbolic space (from Euclidean space).
        out = self.manifold.proj(
            self.manifold.expmap0(self.manifold.proj_tan0(out, self.curvature), c=self.curvature),
            c=self.curvature)
        # print(out.shape)
        return out, r1

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class  GraphAttentionLayer1(AdjustableModule):
    def __init__(self, manifold, input_dim, output_dim,dropout, alpha, curvature,use_bias,n_heads,heads_concat,use_w=True):
        """Sparse version of GAT."""
        super(GraphAttentionLayer1, self).__init__(curvature)
        self.dropout = dropout
        self.output_dim = output_dim
        self.manifold = manifold
        self.concat=heads_concat
        self.curvature=curvature
        self.attentions = [SpGraphAttentionLayer1(
            manifold,
            input_dim,
            output_dim,
            dropout=dropout,
            alpha=alpha,
            curvature=curvature,
            use_w=use_w,
            use_bias=use_bias) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def update_curvature(self, c):
        super(GraphAttentionLayer1, self).update_curvature(c)
        for layer in self.attentions:
            layer.update_curvature(c)

    def forward(self, input):
        x,r,adjs= input
        if torch.any(torch.isnan(x)):
            raise ValueError('input tensor has NaaN values')
        x = F.dropout(x, self.dropout, training=self.training)
        r = F.dropout(r, self.dropout, training=self.training)
        input=x,r,adjs
        if self.concat:
            if self.manifold.name == 'Euclidean':
                h = torch.cat([att(input) for att in self.attentions], dim=1)
            else:  # No concat
                h = self.attentions[0](input)
        else:
            h=0
            for att in self.attentions:
                h+=self.manifold.proj_tan0(self.manifold.logmap0(att(input), c=self.curvature), c=self.curvature)
            h=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(F.normalize(h/len(self.attentions),p=2,dim=1), self.curvature), c=self.curvature),
                                 c=self.curvature)



        h = F.dropout(h, self.dropout, training=self.training)
        return h

class SpGraphAttentionLayer1(AdjustableModule):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, manifold, in_features, out_features, dropout, alpha , curvature=1, use_w=True,use_bias=False):
        super(SpGraphAttentionLayer1, self).__init__(curvature)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.use_bias = use_bias
        self.use_w=use_w
        self.manifold = manifold
        self.a = nn.Parameter(torch.zeros(size=(1, 2*in_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        if self.use_w:
            self.xa = nn.Parameter(torch.zeros(size=(1, in_features)))
            nn.init.xavier_normal_(self.xa.data, gain=1.414)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
    def update_curvature(self, c):
        super(SpGraphAttentionLayer1, self).update_curvature(c)
    def linear(self,x):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.curvature), c=self.curvature)
        h=torch.mul(h,self.xa)
        return self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(h, self.curvature), c=self.curvature),
                                 c=self.curvature)
    def forward(self, input):
        x,r,adjs=input
        N = x.size()[0]
        edge = adjs[0]._indices()
        r1 = r
        x = self.linear(x) if self.use_w else x
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.curvature), c=self.curvature)
        edge_e = torch.exp(self.leakyrelu(
            self.a.mm(torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=-1).t()).squeeze())).unsqueeze(1)
        ########################Euclidean Block (START)########################
        # convert h to Euclidean space.
        assert not torch.isnan(edge_e).any()
        # edge_e: E
        ones = torch.ones(size=(N, 1))
        if h.is_cuda:
            ones = ones.cuda()
        e_rowsum = self.special_spmm(indices=edge, values=edge_e, shape=torch.Size([N, N]), b=ones)
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E
        h_prime = self.special_spmm(indices=edge, values=edge_e, shape=torch.Size([N, N]), b=h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        e_rowsum[e_rowsum == 0] = 1
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = h_prime

        ########################Euclidean Block (END)##########################
        # convert h back to Hyperbolic space (from Euclidean space).
        out = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(out, self.curvature), c=self.curvature),
                                 c=self.curvature)
        '''
        edge = adjs[0]._indices()
        x=self.linear(x) if self.use_w else x
        h= self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.curvature), c=self.curvature)
        edge_e = torch.exp(self.leakyrelu(self.a.mm(torch.cat((h[edge[0, :], :],h[edge[1, :], :]),dim=-1).t()).squeeze()))
        ########################Euclidean Block (START)########################
        # convert h to Euclidean space.
        assert not torch.isnan(edge_e).any()
        # edge_e: E
        ones = torch.ones(size=(N, 1))
        if h.is_cuda:
            ones = ones.cuda()
        e_rowsum = self.special_spmm(indices=edge, values=edge_e, shape=torch.Size([N, N]), b=ones)
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E
        h_prime = self.special_spmm(indices=edge, values=edge_e, shape=torch.Size([N, N]), b=h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        e_rowsum[e_rowsum==0]=1
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = h_prime

        ########################Euclidean Block (END)##########################
        # convert h back to Hyperbolic space (from Euclidean space).
        out = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(out, self.curvature), c=self.curvature),
                                 c=self.curvature)
        '''
        #print(out.shape)
        return out
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
class  GraphAttentionLayer2(AdjustableModule):
    def __init__(self, manifold, input_dim, output_dim,dropout, alpha, curvature,use_bias,n_heads,heads_concat,use_w=True):
        """Sparse version of GAT."""
        super(GraphAttentionLayer2, self).__init__(curvature)
        self.dropout = dropout
        self.output_dim = output_dim
        self.manifold = manifold
        self.concat=heads_concat
        self.curvature=curvature
        self.attentions = [SpGraphAttentionLayer2(
            manifold,
            input_dim,
            output_dim,
            dropout=dropout,
            alpha=alpha,
            curvature=curvature,
            use_w=use_w,
            use_bias=use_bias) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def update_curvature(self, c):
        super(GraphAttentionLayer2, self).update_curvature(c)
        for layer in self.attentions:
            layer.update_curvature(c)

    def forward(self, input):
        x,r,adjs= input
        if torch.any(torch.isnan(x)):
            raise ValueError('input tensor has NaaN values')
        x = F.dropout(x, self.dropout, training=self.training)
        r = F.dropout(r, self.dropout, training=self.training)
        input=x,r,adjs
        if self.concat:
            if self.manifold.name == 'Euclidean':
                h = torch.cat([att(input) for att in self.attentions], dim=1)
            else:  # No concat
                h = self.attentions[0](input)
        else:
            h=0
            for att in self.attentions:
                h+=self.manifold.proj_tan0(self.manifold.logmap0(att(input), c=self.curvature), c=self.curvature)
            h=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(F.normalize(h/len(self.attentions),p=2,dim=1), self.curvature), c=self.curvature),
                                 c=self.curvature)



        h = F.dropout(h, self.dropout, training=self.training)
        return h

class SpGraphAttentionLayer2(AdjustableModule):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, manifold, in_features, out_features, dropout, alpha , curvature=1, use_w=True,use_bias=False):
        super(SpGraphAttentionLayer2, self).__init__(curvature)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.use_bias = use_bias
        self.use_w=use_w
        self.manifold = manifold
        self.a = nn.Parameter(torch.zeros(size=(1, 2*in_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        if self.use_w:
            self.linear=HypLinear(manifold, out_features, out_features, curvature, dropout, use_bias)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
    def update_curvature(self, c):
        super(SpGraphAttentionLayer2, self).update_curvature(c)
    def forward(self, input):
        x,r,adjs=input
        N = x.size()[0]
        M=r.size()[0]
        edge = adjs[0]._indices()
        edge1 = adjs[1]._indices()
        r1 = r
        x = self.linear(x) if self.use_w else x
        r = self.linear(r) if self.use_w else r
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.curvature), c=self.curvature)
        r = self.manifold.proj_tan0(self.manifold.logmap0(r, c=self.curvature), c=self.curvature)
        edge_e = torch.exp(self.leakyrelu(
            self.a.mm(torch.cat((h[edge[0, :], :], r[edge1[0, :], :]), dim=-1).t()).squeeze())).unsqueeze(1)
        neighs = edge_e * r[edge1[0, :], :]
        e_rowsum = torch.matmul(adjs[0], edge_e)
        e_rowsum[e_rowsum == 0] = 1
        h_prime = torch.matmul(adjs[0], neighs).div(e_rowsum)

        out = h_prime
        ########################Euclidean Block (END)##########################
        # convert h back to Hyperbolic space (from Euclidean space).
        out = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(out, self.curvature), c=self.curvature),
                                 c=self.curvature)
        '''
        edge = adjs[0]._indices()
        x=self.linear(x) if self.use_w else x
        h= self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.curvature), c=self.curvature)
        edge_e = torch.exp(self.leakyrelu(self.a.mm(torch.cat((h[edge[0, :], :],h[edge[1, :], :]),dim=-1).t()).squeeze()))
        ########################Euclidean Block (START)########################
        # convert h to Euclidean space.
        assert not torch.isnan(edge_e).any()
        # edge_e: E
        ones = torch.ones(size=(N, 1))
        if h.is_cuda:
            ones = ones.cuda()
        e_rowsum = self.special_spmm(indices=edge, values=edge_e, shape=torch.Size([N, N]), b=ones)
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E
        h_prime = self.special_spmm(indices=edge, values=edge_e, shape=torch.Size([N, N]), b=h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        e_rowsum[e_rowsum==0]=1
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = h_prime

        ########################Euclidean Block (END)##########################
        # convert h back to Hyperbolic space (from Euclidean space).
        out = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(out, self.curvature), c=self.curvature),
                                 c=self.curvature)
        '''
        #print(out.shape)
        return out
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class  RelationGraphAttentionLayer(AdjustableModule):
    def __init__(self, manifold, input_dim,output_dim,rdim,dropout, alpha, curvature,use_bias,n_heads,heads_concat,use_w=False):
        """Sparse version of GAT."""
        super(RelationGraphAttentionLayer, self).__init__(curvature)
        self.dropout = dropout
        self.output_dim = output_dim
        self.manifold = manifold
        self.concat=heads_concat
        self.curvature=curvature
        self.attentions = [SpRelationGraphAttentionLayer(
            manifold,
            input_dim,
            output_dim,
            rdim,
            dropout=dropout,
            alpha=alpha,
            curvature=curvature,
            use_bias=use_bias,
            use_w=use_w
        ) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def update_curvature(self, c):
        super(RelationGraphAttentionLayer, self).update_curvature(c)
        for layer in self.attentions:
            layer.update_curvature(c)

    def forward(self, input):
        x,r,adjs= input
        if torch.any(torch.isnan(x)):
            raise ValueError('input tensor has NaaN values')
        x = F.dropout(x, self.dropout, training=self.training)
        r = F.dropout(r, self.dropout, training=self.training)
        input=x,r,adjs
        if self.concat:
            if self.manifold.name == 'Euclidean':
                h = torch.cat([att(input) for att in self.attentions], dim=1)
            else:  # No concat
                h = self.attentions[0](input)
        else:
            h=0
            r=0
            for att in self.attentions:
                h1,r1=att(input)
                h+=self.manifold.proj_tan0(self.manifold.logmap0(h1, c=self.curvature), c=self.curvature)
                r+=self.manifold.proj_tan0(self.manifold.logmap0(r1, c=self.curvature), c=self.curvature)
            h=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(F.normalize(h/len(self.attentions),p=2,dim=1), self.curvature), c=self.curvature),
                                 c=self.curvature)
            r=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(F.normalize(r/len(self.attentions),p=2,dim=1), self.curvature), c=self.curvature),
                                 c=self.curvature)



        h = F.dropout(h, self.dropout, training=self.training)
        return h,r

class SpRelationGraphAttentionLayer(AdjustableModule):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, manifold, in_features, out_features,rdim, dropout, alpha , curvature=1, use_bias=False,use_w=False):
        super(SpRelationGraphAttentionLayer, self).__init__(curvature)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.use_bias = use_bias
        self.manifold = manifold
        self.use_w=use_w
        self.a = nn.Parameter(torch.zeros(size=(1, 2*in_features+rdim)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        if self.use_w:
            self.linear=HypLinear(manifold, in_features, out_features, curvature, dropout, use_bias)
            self.linear_r = HypLinear(manifold, rdim, rdim, curvature, dropout, use_bias)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
    def update_curvature(self, c):
        super(SpRelationGraphAttentionLayer, self).update_curvature(c)
        self.linear.update_curvature(c)
    def forward(self, input):
        x,r,adjs=input
        N = x.size()[0]
        edge = adjs[0]._indices()
        edge1 = adjs[1]._indices()
        edge2 = adjs[2]._indices()
        r1=r
        x=self.linear(x) if self.use_w else x
        r = self.linear_r(r) if self.use_w else r
        h= self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.curvature), c=self.curvature)
        r = self.manifold.proj_tan0(self.manifold.logmap0(r, c=self.curvature), c=self.curvature)
        edge_e = torch.exp(self.leakyrelu(self.a.mm((torch.cat((h[edge1[0, :], :],r[edge[0, :], :],h[edge2[0, :], :]),dim=-1)).t()).squeeze())).unsqueeze(1)
        neighs = edge_e * h[edge2[0, :], :]
        e_rowsum=torch.matmul(adjs[1],edge_e)
        e_rowsum[e_rowsum == 0] = 1
        h_prime=torch.matmul(adjs[1],neighs).div(e_rowsum)

        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = h_prime
        ########################Euclidean Block (END)##########################
        # convert h back to Hyperbolic space (from Euclidean space).
        out = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(out, self.curvature), c=self.curvature),
                                 c=self.curvature)
        #print(out.shape)
        return out,r1
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class  RelationGraphAttentionLayer1(AdjustableModule):
    def __init__(self, manifold, input_dim,output_dim,rdim,dropout, alpha, curvature,use_bias,n_heads,heads_concat,use_w=False):
        """Sparse version of GAT."""
        super(RelationGraphAttentionLayer1, self).__init__(curvature)
        self.dropout = dropout
        self.output_dim = output_dim
        self.manifold = manifold
        self.concat=heads_concat
        self.curvature=curvature
        self.attentions = [SpRelationGraphAttentionLayer1(
            manifold,
            input_dim,
            output_dim,
            rdim,
            dropout=dropout,
            alpha=alpha,
            curvature=curvature,
            use_bias=use_bias,
            use_w=use_w
        ) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def update_curvature(self, c):
        super(RelationGraphAttentionLayer1, self).update_curvature(c)
        for layer in self.attentions:
            layer.update_curvature(c)

    def forward(self, input):
        x,r,adjs= input
        if torch.any(torch.isnan(x)):
            raise ValueError('input tensor has NaaN values')
        x = F.dropout(x, self.dropout, training=self.training)
        r = F.dropout(r, self.dropout, training=self.training)
        input=x,r,adjs
        if self.concat:
            if self.manifold.name == 'Euclidean':
                h = torch.cat([att(input) for att in self.attentions], dim=1)
            else:  # No concat
                h = self.attentions[0](input)
        else:
            h=0
            for att in self.attentions:
                h+=self.manifold.proj_tan0(self.manifold.logmap0(att(input), c=self.curvature), c=self.curvature)
            h=self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(h/len(self.attentions), self.curvature), c=self.curvature),
                                 c=self.curvature)



        h = F.dropout(h, self.dropout, training=self.training)
        return h

class SpRelationGraphAttentionLayer1(AdjustableModule):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, manifold, in_features, out_features,rdim, dropout, alpha , curvature=1, use_bias=False,use_w=False):
        super(SpRelationGraphAttentionLayer1, self).__init__(curvature)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.use_bias = use_bias
        self.manifold = manifold
        self.use_w=use_w
        self.a = nn.Parameter(torch.zeros(size=(1, rdim)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        if self.use_w:
            self.linear=HypLinear(manifold, in_features, out_features, curvature, dropout, use_bias)
        self.linear_r = HypLinear(manifold, rdim, rdim//2, curvature, dropout, use_bias)
        self.linear_r1 = HypLinear(manifold, rdim//2, 1, curvature, dropout, use_bias)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
    def update_curvature(self, c):
        super(SpRelationGraphAttentionLayer1, self).update_curvature(c)
        self.linear.update_curvature(c)
    def forward(self, input):
        x,r,adjs=input
        N = x.size()[0]
        edge = adjs[0]._indices()
        edge1 = adjs[1]._indices()
        edge2 = adjs[2]._indices()
        x=self.linear(x) if self.use_w else x
        r = F.relu(self.linear_r1(F.relu(self.linear_r(r))))
        h= self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.curvature), c=self.curvature)
        r = self.manifold.proj_tan0(self.manifold.logmap0(r, c=self.curvature), c=self.curvature)
        edge_e = torch.exp(self.leakyrelu(r[edge[1,:],:].squeeze())).unsqueeze(1)
        neighs = edge_e * h[edge2[0, :], :]
        e_rowsum=torch.matmul(adjs[1],edge_e)
        e_rowsum[e_rowsum == 0] = 1
        h_prime=torch.matmul(adjs[1],neighs).div(e_rowsum)

        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        out = h_prime
        ########################Euclidean Block (END)##########################
        # convert h back to Hyperbolic space (from Euclidean space).
        out = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(out, self.curvature), c=self.curvature),
                                 c=self.curvature)
        #print(out.shape)
        return out
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )
class HypLinear1(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear1, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        self.weight.data.copy_(torch.tensor(np.diag(np.random.rand(self.out_features))))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)
        else:
            support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        assert not torch.isnan(self.manifold.logmap0(x, c=self.c_in)).any()
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        assert not torch.isnan(xt).any()
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        assert not torch.isnan(xt).any()
        assert not torch.isnan(self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)).any()
        #print(-1,torch.mean(xt))
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
class HypNorm(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out):
        super(HypNorm, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out

    def forward(self, x):
        xt = F.normalize(self.manifold.logmap0(x, c=self.c_in),p=2,dim=-1)
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        #print(-1,torch.mean(xt))
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )

class DenseAtt(nn.Module):
    def __init__(self, in_features, dropout):
        super(DenseAtt, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(2 * in_features, 1, bias=True)
        self.in_features = in_features

    def forward (self, x, adj):
        n = x.size(0)
        # n x 1 x d
        x_left = torch.unsqueeze(x, 1)
        x_left = x_left.expand(-1, n, -1)
        # 1 x n x d
        x_right = torch.unsqueeze(x, 0)
        x_right = x_right.expand(n, -1, -1)
        #x_cat = torch.cat((x_left.reshape(-1,self.in_features), x_right.reshape(-1,self.in_features)), dim=0)
        x_cat = torch.cat((x_left, x_right), dim=2)
        att_adj = self.linear(x_cat).squeeze()
        att_adj = F.sigmoid(att_adj)
        att_adj = torch.mul(adj.to_dense(), att_adj)
        return att_adj

class BERT_INT_MlP(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(BERT_INT_MlP, self).__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim, True)
        self.dense2 = nn.Linear(hidden_dim, 1, True)
        init.xavier_normal_(self.dense1.weight)
        init.xavier_normal_(self.dense2.weight)
    def forward(self,features):
        x = self.dense1(features)#[B,h]
        x = F.relu(x)
        x = self.dense2(x)#[B,1]
        x = F.tanh(x)
        x = torch.squeeze(x,1)#[B]
        return x
class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)
