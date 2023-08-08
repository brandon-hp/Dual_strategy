"""Graph encoders."""
import math

import torch
import torch.nn as nn
import model.layers as layers
import torch.nn.functional as F
import numpy as np
class Encoder(nn.Module):
    """
    Encoder abstract class.
    """
    def __init__(self):
        super(Encoder, self).__init__()

    def encode(self, input):
        output= self.layers.forward(input)
        return output
class MRAEA(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, rdim,e2eself_adj,e2r_adj,r2t_adj,h2t_adj,t2t_adj,manifold,args):
        super(MRAEA, self).__init__()

        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs=[r2t_adj,h2t_adj,t2t_adj,e2eself_adj,e2r_adj]
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        self.inact=layers.HypAct(manifold, self.curvatures[0], self.curvatures[0], acts[0])
        gc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            ragat=layers.RelationGraphAttentionLayer(
                self.manifold, in_dim+rdim, out_dim+rdim,rdim, args.dropout, 0.2,c_in,args.bias,args.n_heads,args.heads_concat,args.use_w
            )
            setattr(self, 'ragat{}'.format(i), ragat)
            hact=layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact{}'.format(i), hact)
            if self.multi_hop=='hgate':
                hgate=layers.HighwayGate(self.manifold, out_dim+rdim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate{}'.format(i), hgate)
        self.layers = nn.ModuleList(gc_layers)#nn.Sequential(*hgc_layers)
        self.encode_graph = True
        self.len=len(dims) - 1


    def encode(self,input):
        x,r=input

        x=torch.cat((x,torch.matmul(self.adjs[4],r)),dim=-1)#F.normalize(torch.cat((torch.matmul(self.adjs[3],x),torch.matmul(self.adjs[4],r)),dim=-1),p=2,dim=-1)

        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(r, self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])

        outputs=[]
        c=[]
        if self.multi_hop=='concat':
            outputs.append(x_hyp)
            c.append(self.curvatures[0])
        h=x_hyp
        for i in range(self.len):
            ragat=getattr(self, 'ragat{}'.format(i))
            h1,r_hyp=ragat((h,r_hyp,self.adjs[:3]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                h2=hgate(h,h2)
                h = h2
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
            else:
                h = h2
                c.append(self.curvatures[i + 1])
                outputs.append(h)

        return outputs,r_hyp,c
class GAT(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, rdim,e2eself_adj,e2r_adj,r2t_adj,h2t_adj,t2t_adj,manifold,args):
        super(GAT, self).__init__()

        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs=[r2t_adj,h2t_adj,t2t_adj,e2eself_adj,e2r_adj]
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        self.inact=layers.HypAct(manifold, self.curvatures[0], self.curvatures[0], acts[0])
        gc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            ragat=layers.GraphAttentionLayer(
                self.manifold, in_dim, out_dim, args.dropout, 0.2,c_in,args.bias,args.n_heads,args.heads_concat,args.use_w
            )
            setattr(self, 'ragat{}'.format(i), ragat)
            hact=layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact{}'.format(i), hact)
            if self.multi_hop=='hgate':
                hgate=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate{}'.format(i), hgate)
        self.layers = nn.ModuleList(gc_layers)#nn.Sequential(*hgc_layers)
        self.encode_graph = True
        self.len=len(dims) - 1
        self.mlp=nn.Linear(len(dims)*dims[0],dims[0])
        self.dropout=args.dropout


    def encode(self,input):
        x,r=input
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(r, self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])

        outputs=[]
        c=[]
        if self.multi_hop=='concat':
            outputs.append(x_hyp)
            c.append(self.curvatures[0])
        h=x_hyp
        for i in range(self.len):
            ragat=getattr(self, 'ragat{}'.format(i))
            h1=ragat((h,r_hyp,self.adjs[:3]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                h2=hgate(h,h2)
                h = h2
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
            else:
                h = h2
                c.append(self.curvatures[i + 1])
                outputs.append(h)
        return outputs,r_hyp,c
class MRAEA_adapt(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, rdim,e2eself_adj,e2r_adj,r2t_adj,h2t_adj,t2t_adj,manifold,args):
        super(MRAEA_adapt, self).__init__()

        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs=[r2t_adj,h2t_adj,t2t_adj,e2eself_adj,e2r_adj]
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            ragat=layers.RelationGraphAttentionLayer(
                self.manifold, in_dim, out_dim,rdim, args.dropout, 0.2,c_in,args.bias,args.n_heads,args.heads_concat,args.use_w
            )
            setattr(self, 'ragat{}'.format(i), ragat)
            hact=layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact{}'.format(i), hact)
            if self.multi_hop=='hgate':
                hgate=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate{}'.format(i), hgate)
        self.layers = nn.ModuleList(gc_layers)#nn.Sequential(*hgc_layers)
        self.encode_graph = True
        self.len=len(dims) - 1


    def encode(self,input):
        x,r=input
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(r, self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])

        outputs=[]
        c=[]
        if self.multi_hop=='concat':
            outputs.append(x_hyp)
            c.append(self.curvatures[0])
        h=x_hyp
        for i in range(self.len):
            ragat=getattr(self, 'ragat{}'.format(i))
            h1,r_hyp=ragat((h,r_hyp,self.adjs[:3]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                h2=hgate(h,h2)
                h = h2
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
            else:
                h = h2
                c.append(self.curvatures[i + 1])
                outputs.append(h)

        return outputs,r_hyp,c
class DUAL(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, rdim,r2t_adj,h2t_adj,t2t_adj,e2eself_adj,e2r_adj,manifold,args):
        super(DUAL, self).__init__()

        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs=[r2t_adj,h2t_adj,t2t_adj,e2eself_adj,e2r_adj]
        self.inact=getattr(F, args.act)
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            ragat=layers.SimplifiedRelationalGraphAttentionLayer(
                self.manifold, in_dim, out_dim,rdim, args.dropout, 0.2,c_in,args.bias,args.n_heads,args.heads_concat,args.use_w
            )
            setattr(self, 'srgat{}'.format(i), ragat)
            hact=layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact{}'.format(i), hact)
            if self.multi_hop=='hgate':
                hgate=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate{}'.format(i), hgate)
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            ragat=layers.SimplifiedRelationalGraphAttentionLayer(
                self.manifold, in_dim, out_dim,rdim, args.dropout, 0.2,c_in,args.bias,args.n_heads,args.heads_concat,args.use_w
            )
            setattr(self, 'srgat_r{}'.format(i), ragat)
            hact=layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact_r{}'.format(i), hact)
            if self.multi_hop=='hgate':
                hgate=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate_r{}'.format(i), hgate)
        self.proxy = nn.Embedding(64, len(dims)*dims[-1])
        nn.init.kaiming_normal_(self.proxy.weight, mode='fan_out', nonlinearity='relu')
        #self.proxy = nn.Parameter(torch.zeros(size=(64, len(dims)*dims[-1])))
        #nn.init.xavier_normal_(self.proxy.data, gain=1.414)
        self.gate_kernel = nn.Parameter(torch.zeros(size=(len(dims)*dims[-1],1)))
        nn.init.xavier_normal_(self.gate_kernel.data, gain=1.414)

        self.proxy_r = nn.Embedding(64, len(dims)*dims[-1])
        nn.init.kaiming_normal_(self.proxy_r.weight, mode='fan_out', nonlinearity='relu')
        #self.proxy_r = nn.Parameter(torch.zeros(size=(64, len(dims)*dims[-1])))
        #nn.init.xavier_normal_(self.proxy_r.data, gain=1.414)
        self.gate_kernel_r = nn.Parameter(torch.zeros(size=(len(dims)*dims[-1],1)))
        nn.init.xavier_normal_(self.gate_kernel_r.data, gain=1.414)
        self.bias_r=nn.Parameter(torch.zeros(size=(1,len(dims)*dims[-1])))
        nn.init.xavier_normal_(self.bias_r.data, gain=1.414)
        self.bias=nn.Parameter(torch.zeros(size=(1,len(dims)*dims[-1])))
        nn.init.xavier_normal_(self.bias.data, gain=1.414)
        self.encode_graph = True
        self.len=len(dims) - 1


    def encode(self,input):
        x,r=input
        #x=F.normalize(torch.matmul(self.adjs[3],x),p=2,dim=-1)
        x_r=F.normalize(torch.matmul(self.adjs[4],r),p=2,dim=-1)
        # x=self.inact(x)
        # x_r = self.inact(x_r)

        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        x_tan_r = self.manifold.proj_tan0(x_r, self.curvatures[0])
        x_hyp_r = self.manifold.expmap0(x_tan_r, c=self.curvatures[0])
        x_hyp_r = self.manifold.proj(x_hyp_r, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(r, self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])

        outputs=[]
        c=[]
        if self.multi_hop=='concat':
            outputs.append(x_hyp)
            c.append(self.curvatures[0])
        h=x_hyp
        for i in range(self.len):
            ragat=getattr(self, 'srgat{}'.format(i))
            h1,r_hyp=ragat((h,r_hyp,self.adjs[:3]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                h2=hgate(h,h2)
                h = h2
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
            else:
                h = h2
                c.append(self.curvatures[i + 1])
                outputs.append(h)
        x_o=torch.cat(outputs,dim=-1)
        proxy_att = torch.matmul(x_o,self.proxy.weight.transpose(0,1))
        proxy_att=torch.softmax(proxy_att,dim=-1)
        proxy_feature=x_o-torch.matmul(proxy_att,self.proxy.weight)

        gate_rate = F.sigmoid(torch.matmul(proxy_feature, self.gate_kernel))#+self.bias)
        x_o_1 = (gate_rate) * x_o + (1 - gate_rate) * proxy_feature

        # proxy_att = K.softmax(proxy_att,axis = -1)
        # proxy_feature = outputs - K.dot(proxy_att,self.proxy)
        outputs = []
        if self.multi_hop=='concat':
            outputs.append(x_hyp_r)
            c.append(self.curvatures[0])
        h=x_hyp_r
        for i in range(self.len):
            ragat=getattr(self, 'srgat_r{}'.format(i))
            h1,r_hyp=ragat((h,r_hyp,self.adjs[:3]))
            hact=getattr(self, 'hact_r{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate_r{}'.format(i))
                h2=hgate(h,h2)
                h = h2
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
            else:
                h = h2
                c.append(self.curvatures[i + 1])
                outputs.append(h)

        x_o_r=torch.cat(outputs,dim=-1)
        proxy_att_r = torch.matmul(x_o_r,self.proxy_r.weight.transpose(0,1))
        proxy_att_r=torch.softmax(proxy_att_r,dim=-1)
        proxy_feature_r=x_o_r-torch.matmul(proxy_att_r,self.proxy_r.weight)

        gate_rate_r = F.sigmoid(torch.matmul(proxy_feature_r, self.gate_kernel_r))#+self.bias_r)
        x_o_2 = (gate_rate_r) * x_o_r + (1 - gate_rate_r) * proxy_feature_r
        # x_o_1 = F.dropout(x_o_1, 0.3, training=self.training)
        # x_o_2=F.dropout(x_o_2, 0.3, training=self.training)

        outputs=[x_o_1,x_o_2]
        c=[c[0],c[1]]

        return outputs,r_hyp,c
class DUAL_adapt(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, rdim,r2t_adj,h2t_adj,t2t_adj,e2eself_adj,e2r_adj,manifold,args):
        super(DUAL_adapt, self).__init__()

        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs=[r2t_adj,h2t_adj,t2t_adj,e2eself_adj,e2r_adj]
        self.inact=getattr(F, args.act)
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            ragat=layers.SimplifiedRelationalGraphAttentionLayer(
                self.manifold, in_dim, out_dim,rdim, args.dropout, 0.2,c_in,args.bias,args.n_heads,args.heads_concat,args.use_w
            )
            setattr(self, 'srgat{}'.format(i), ragat)
            hact=layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact{}'.format(i), hact)
            if self.multi_hop=='hgate':
                hgate=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate{}'.format(i), hgate)
        '''
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            ragat=layers.SimplifiedRelationalGraphAttentionLayer(
                self.manifold, in_dim, out_dim,rdim, args.dropout, 0.2,c_in,args.bias,args.n_heads,args.heads_concat,args.use_w
            )
            setattr(self, 'srgat_r{}'.format(i), ragat)
            hact=layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact_r{}'.format(i), hact)
            if self.multi_hop=='hgate':
                hgate=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate_r{}'.format(i), hgate)
        '''
        self.encode_graph = True
        self.len=len(dims) - 1


    def encode(self,input):
        x,r=input

        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(r, self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])

        outputs=[]
        c=[]
        if self.multi_hop=='concat':
            outputs.append(x_hyp)
            c.append(self.curvatures[0])
        h=x_hyp
        for i in range(self.len):
            ragat=getattr(self, 'srgat{}'.format(i))
            h1,r_hyp=ragat((h,r_hyp,self.adjs[:3]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                h2=hgate(h,h2)
                h = h2
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
            else:
                h = h2
                c.append(self.curvatures[i + 1])
                outputs.append(h)
        '''
        if self.multi_hop=='concat':
            outputs.append(x_hyp_r)
            c.append(self.curvatures[0])
        h=x_hyp_r
        for i in range(self.len):
            ragat=getattr(self, 'srgat_r{}'.format(i))
            h1,r_hyp=ragat((h,r_hyp,self.adjs[:3]))
            hact=getattr(self, 'hact_r{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate_r{}'.format(i))
                h2=hgate(h,h2)
                h = h2
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
            else:
                h = h2
                c.append(self.curvatures[i + 1])
                outputs.append(h)
        '''

        return outputs,r_hyp,c
class RREA(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, rdim,r2t_adj,h2t_adj,t2t_adj,e2eself_adj,e2r_adj,manifold,args):
        super(RREA, self).__init__()

        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs=[r2t_adj,h2t_adj,t2t_adj,e2eself_adj,e2r_adj]
        self.inact=getattr(F, args.act)
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            ragat=layers.RelationalReflectionGraphAttentionLayer(
                self.manifold, in_dim, out_dim,rdim, args.dropout, 0.2,c_in,args.bias,args.n_heads,args.heads_concat,args.use_w
            )
            setattr(self, 'rrgat{}'.format(i), ragat)
            hact=layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact{}'.format(i), hact)
            if self.multi_hop=='hgate':
                hgate=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate{}'.format(i), hgate)

        self.encode_graph = True
        self.len=len(dims) - 1


    def encode(self,input):
        x,r=input
        #x=F.normalize(torch.matmul(self.adjs[3],x),p=2,dim=-1)
        x_r=F.normalize(torch.matmul(self.adjs[4],r),p=2,dim=-1)
        # x=self.inact(x)
        # x_r = self.inact(x_r)

        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        x_tan_r = self.manifold.proj_tan0(x_r, self.curvatures[0])
        x_hyp_r = self.manifold.expmap0(x_tan_r, c=self.curvatures[0])
        x_hyp_r = self.manifold.proj(x_hyp_r, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(r, self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])

        outputs=[]
        c=[]
        if self.multi_hop=='concat':
            outputs.append(x_hyp)
            c.append(self.curvatures[0])
        h=x_hyp
        for i in range(self.len):
            ragat=getattr(self, 'rrgat{}'.format(i))
            h1,r_hyp=ragat((h,r_hyp,self.adjs[:3]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                h2=hgate(h,h2)
                h = h2
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
            else:
                h = h2
                c.append(self.curvatures[i + 1])
                outputs.append(h)

        if self.multi_hop=='concat':
            outputs.append(x_hyp_r)
            c.append(self.curvatures[0])
        h=x_hyp_r
        for i in range(self.len):
            ragat=getattr(self, 'rrgat{}'.format(i))
            h1,r_hyp=ragat((h,r_hyp,self.adjs[:3]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                h2=hgate(h,h2)
                h = h2
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
            else:
                h = h2
                c.append(self.curvatures[i + 1])
                outputs.append(h)

        return outputs,r_hyp,c
class RREA_adapt1(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, rdim,r2t_adj,h2t_adj,t2t_adj,e2eself_adj,e2r_adj,manifold,args):
        super(RREA_adapt1, self).__init__()

        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs=[r2t_adj,h2t_adj,t2t_adj,e2eself_adj,e2r_adj]
        self.inact=getattr(F, args.act)
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            ragat=layers.RelationalReflectionGraphAttentionLayer(
                self.manifold, in_dim, out_dim,rdim, args.dropout, 0.2,c_in,args.bias,args.n_heads,args.heads_concat,args.use_w
            )
            setattr(self, 'rrgat{}'.format(i), ragat)
            hact=layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact{}'.format(i), hact)
            if self.multi_hop=='hgate':
                hgate=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate{}'.format(i), hgate)

        self.encode_graph = True
        self.len=len(dims) - 1


    def encode(self,input):
        x,r=input
        x_r=F.normalize(torch.matmul(self.adjs[4],r),p=2,dim=-1)
        # x=self.inact(x)
        # x_r = self.inact(x_r)

        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        x_tan_r = self.manifold.proj_tan0(x_r, self.curvatures[0])
        x_hyp_r = self.manifold.expmap0(x_tan_r, c=self.curvatures[0])
        x_hyp_r = self.manifold.proj(x_hyp_r, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(r, self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])

        outputs=[]
        c=[]
        if self.multi_hop=='concat':
            outputs.append(x_hyp)
            c.append(self.curvatures[0])
        h=x_hyp
        for i in range(self.len):
            ragat=getattr(self, 'rrgat{}'.format(i))
            h1,_=ragat((h,r_hyp,self.adjs[:3]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                h2=hgate(h,h2)
                h = h2
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
            else:
                h = h2
                c.append(self.curvatures[i + 1])
                outputs.append(h)

        if self.multi_hop=='concat':
            outputs.append(x_hyp_r)
            c.append(self.curvatures[0])
        h=x_hyp_r
        for i in range(self.len):
            ragat=getattr(self, 'rrgat{}'.format(i))
            h1,_=ragat((h,r_hyp,self.adjs[:3]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                h2=hgate(h,h2)
                h = h2
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
            else:
                h = h2
                c.append(self.curvatures[i + 1])
                outputs.append(h)

        return outputs,r_hyp,c
class RREA_adapt(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, rdim,r2t_adj,h2t_adj,t2t_adj,e2eself_adj,e2r_adj,manifold,args):
        super(RREA_adapt, self).__init__()

        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs=[r2t_adj,h2t_adj,t2t_adj,e2eself_adj,e2r_adj]
        self.inact=getattr(F, args.act)
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            ragat=layers.RelationalReflectionGraphAttentionLayer(
                self.manifold, in_dim, out_dim,rdim, args.dropout, 0.2,c_in,args.bias,args.n_heads,args.heads_concat,args.use_w
            )
            setattr(self, 'rrgat{}'.format(i), ragat)
            hact=layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact{}'.format(i), hact)
            if self.multi_hop=='hgate':
                hgate=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate{}'.format(i), hgate)

        self.encode_graph = True
        self.len=len(dims) - 1
        # self.ins_embeddings = nn.Embedding(38960, 300, max_norm=1)
        # nn.init.kaiming_normal_(self.ins_embeddings.weight, mode='fan_out', nonlinearity='relu')
        # self.rel_embeddings = nn.Embedding(6049, 300, max_norm=1)
        # nn.init.kaiming_normal_(self.rel_embeddings.weight, mode='fan_out', nonlinearity='relu')


    def encode(self,input):
        x,r=input
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(r, self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])

        outputs=[]
        c=[]
        if self.multi_hop=='concat':
            outputs.append(x_hyp)
            c.append(self.curvatures[0])
        h=x_hyp
        for i in range(self.len):
            ragat=getattr(self, 'rrgat{}'.format(i))
            h1,r_hyp=ragat((h,r_hyp,self.adjs[:3]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                h2=hgate(h,h2)
                h = h2
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
            else:
                h = h2
                c.append(self.curvatures[i + 1])
                outputs.append(h)
        '''
        if self.multi_hop=='concat':
            outputs.append(x_hyp_r)
            c.append(self.curvatures[0])
        h=x_hyp_r
        for i in range(self.len):
            ragat=getattr(self, 'rrgat{}'.format(i))
            h1,r_hyp=ragat((h,r_hyp,self.adjs[:3]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                h2=hgate(h,h2)
                h = h2
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
            else:
                h = h2
                c.append(self.curvatures[i + 1])
                outputs.append(h)
        '''
        return outputs,r_hyp,c
class TestM1(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, rdim,r2t_adj,h2t_adj,t2t_adj,e2eself_adj,e2r_adj,e2v_adj,manifold,args):
        super(TestM1, self).__init__()

        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs=[r2t_adj,h2t_adj,t2t_adj,e2eself_adj,e2r_adj,e2v_adj]
        self.inact=getattr(F, args.act)
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        self.attg= layers.ValGraphAttentionLayer(
            self.manifold,  dims[0], dims[0], args.dropout, 0.2, acts[0], args.bias, 2, args.heads_concat,
            args.use_w
        )
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            ragat=layers.RelationalReflectionGraphAttentionLayer(
                self.manifold, in_dim, out_dim,rdim, args.dropout, 0.2,c_in,args.bias,args.n_heads,args.heads_concat,args.use_w
            )
            setattr(self, 'rrgat{}'.format(i), ragat)
            hact=layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact{}'.format(i), hact)
            if self.multi_hop=='hgate':
                hgate=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate{}'.format(i), hgate)

        self.encode_graph = True
        self.len=len(dims) - 1
        # self.ins_embeddings = nn.Embedding(38960, 300, max_norm=1)
        # nn.init.kaiming_normal_(self.ins_embeddings.weight, mode='fan_out', nonlinearity='relu')
        # self.rel_embeddings = nn.Embedding(6049, 300, max_norm=1)
        # nn.init.kaiming_normal_(self.rel_embeddings.weight, mode='fan_out', nonlinearity='relu')


    def encode(self,input):
        x,x1,r,v=input
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        x_tan1 = self.manifold.proj_tan0(x1, self.curvatures[0])
        x_hyp1 = self.manifold.expmap0(x_tan1, c=self.curvatures[0])
        x_hyp1 = self.manifold.proj(x_hyp1, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(r, self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])

        x_v=self.attg((x,v,self.adjs[-1]))#F.normalize(torch.matmul(self.adjs[-1],v),p=2,dim=1)
        x_tan_v = self.manifold.proj_tan0(x_v, self.curvatures[0])
        x_hyp_v = self.manifold.expmap0(x_tan_v, c=self.curvatures[0])
        x_hyp_v = self.manifold.proj(x_hyp_v, c=self.curvatures[0])


        outputs=[]
        c=[]
        if self.multi_hop=='concat':
            outputs.append(x_hyp)
            c.append(self.curvatures[0])
        h=x_hyp
        for i in range(self.len):
            ragat=getattr(self, 'rrgat{}'.format(i))
            h1,r_hyp=ragat((h,r_hyp,self.adjs[:3]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                h2=hgate(h,h2)
                h = h2
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
            else:
                h = h2
                c.append(self.curvatures[i + 1])
                outputs.append(h)

        if self.multi_hop=='concat':
            outputs.append(x_hyp_v)
            c.append(self.curvatures[0])
        h=x_hyp_v
        for i in range(0):
            ragat=getattr(self, 'rrgat{}'.format(i))
            h1,r_hyp=ragat((h,r_hyp,self.adjs[:3]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                h2=hgate(h,h2)
                h = h2
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
            else:
                h = h2
                c.append(self.curvatures[i + 1])
                outputs.append(h)

        if self.multi_hop=='concat':
            outputs.append(x_hyp1)
            c.append(self.curvatures[0])
        h=x_hyp1
        for i in range(self.len):
            ragat=getattr(self, 'rrgat{}'.format(i))
            h1,r_hyp=ragat((h,r_hyp,self.adjs[:3]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                h2=hgate(h,h2)
                h = h2
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
            else:
                h = h2
                c.append(self.curvatures[i + 1])
                outputs.append(h)
        return outputs,r_hyp,c


class TestM(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, rdim,vdim, r2t_adj, h2t_adj, t2t_adj, e2eself_adj, e2r_adj, e2v_adj,e2a_adj,e2t_adj,a2t_adj,v2t_adj,manifold, args):
        super(TestM, self).__init__()

        self.manifold = manifold
        self.multi_hop = args.multi_hop
        self.adjs = [r2t_adj, h2t_adj, t2t_adj, e2eself_adj, e2r_adj, e2v_adj,e2a_adj,e2t_adj,a2t_adj,v2t_adj]
        self.inact = getattr(F, args.act)
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        self.attg = layers.ValGraphAttentionLayer(
            self.manifold, dims[0], vdim, 0, 0.2, self.curvatures[0], args.bias, 2, args.heads_concat,
            True
        )
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            ragat = layers.RelationalReflectionGraphAttentionLayer(
                self.manifold, in_dim, out_dim, rdim, args.dropout, 0.2, c_in, args.bias, args.n_heads,
                args.heads_concat, args.use_w
            )
            setattr(self, 'rrgat{}'.format(i), ragat)
            hact = layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact{}'.format(i), hact)
            if self.multi_hop == 'hgate':
                hgate = layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate{}'.format(i), hgate)

        self.encode_graph = True
        self.len = len(dims) - 1
        # self.ins_embeddings = nn.Embedding(38960, 300, max_norm=1)
        # nn.init.kaiming_normal_(self.ins_embeddings.weight, mode='fan_out', nonlinearity='relu')
        # self.rel_embeddings = nn.Embedding(6049, 300, max_norm=1)
        # nn.init.kaiming_normal_(self.rel_embeddings.weight, mode='fan_out', nonlinearity='relu')

    def update_attribute(self, e2t_adj,a2t_adj,v2t_adj):
        self.adjs[-1]=v2t_adj
        self.adjs[-2] = a2t_adj
        self.adjs[-3] = e2t_adj

    def encode(self, input):
        x, r, v,a = input
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(r, self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])

        x_v = self.attg((x, v,a, self.adjs[-3:]))  # F.normalize(torch.matmul(self.adjs[-1],v),p=2,dim=1)
        x_tan_v = self.manifold.proj_tan0(x_v, self.curvatures[0])
        x_hyp_v = self.manifold.expmap0(x_tan_v, c=self.curvatures[0])
        x_hyp_v = self.manifold.proj(x_hyp_v, c=self.curvatures[0])
        #
        # x_a = self.attg((x, a, self.adjs[-1]))  # F.normalize(torch.matmul(self.adjs[-1],v),p=2,dim=1)
        # x_tan_a = self.manifold.proj_tan0(x_a, self.curvatures[0])
        # x_hyp_a = self.manifold.expmap0(x_tan_a, c=self.curvatures[0])
        # x_hyp_a = self.manifold.proj(x_hyp_a, c=self.curvatures[0])

        outputs = []
        c = []
        if self.multi_hop == 'concat':
            outputs.append(x_hyp)
            c.append(self.curvatures[0])
        h = x_hyp
        for i in range(self.len):
            ragat = getattr(self, 'rrgat{}'.format(i))
            h1, r_hyp = ragat((h, r_hyp, self.adjs[:3]))
            hact = getattr(self, 'hact{}'.format(i))
            h2 = hact(h1)
            if self.multi_hop == 'hgate':
                hgate = getattr(self, 'hgate{}'.format(i))
                h2 = hgate(h, h2)
                h = h2
                if i == self.len - 1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
            else:
                h = h2
                c.append(self.curvatures[i + 1])
                outputs.append(h)

        if self.multi_hop == 'concat':
            outputs.append(x_hyp_v)
            c.append(self.curvatures[0])

        h = x_hyp_v
        for i in range(0):
            ragat = getattr(self, 'rrgat{}'.format(i))
            h1, r_hyp = ragat((h, r_hyp, self.adjs[:3]))
            hact = getattr(self, 'hact{}'.format(i))
            h2 = hact(h1)
            if self.multi_hop == 'hgate':
                hgate = getattr(self, 'hgate{}'.format(i))
                h2 = hgate(h, h2)
                h = h2
                if i == self.len - 1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
            else:
                h = h2
                c.append(self.curvatures[i + 1])
                outputs.append(h)
        return outputs, r_hyp, c
class RREAN_literal(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, rdim,r2t_adj,h2t_adj,t2t_adj,r2t_adj_o,h2t_adj_o,t2t_adj_o,e2r_adj,eself_adj,manifold,args):
        super(RREAN_literal, self).__init__()

        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs=[r2t_adj,h2t_adj,t2t_adj,r2t_adj_o,h2t_adj_o,t2t_adj_o,e2r_adj,eself_adj]
        self.inact=getattr(F, args.act)
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        self.attg= layers.relGraphAttentionLayer(
            self.manifold,  dims[0], dims[0], args.dropout, 0.2, acts[0], args.bias, 2, args.heads_concat,
            True
        )
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            ragat=layers.GraphAttentionLayer(
                self.manifold, in_dim, out_dim, args.dropout, 0.2,c_in,args.bias,args.n_heads,args.heads_concat,args.use_w
            )
            setattr(self, 'ragat{}'.format(i), ragat)
            hact=layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact{}'.format(i), hact)
            #if self.multi_hop=='hgate':
            hgate=layers.HighwayGate1(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
            setattr(self, 'hgate{}'.format(i), hgate)
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * 300)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.a1 = nn.Parameter(torch.zeros(size=(1, 2 * 300)))
        nn.init.xavier_normal_(self.a1.data, gain=1.414)
        self.linear = layers.HypLinear(manifold, dims[-1] * 3, dims[-1], self.curvatures[-1], args.dropout, args.bias)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.encode_graph = True
        self.len=len(dims) - 1
        #self.cpe = nn.Linear(300, 300,bias=False)
    def encode(self,input):
        x,r=input
        x_r=self.attg((x,r,self.adjs[0:3]))##F.normalize(torch.matmul(self.adjs[-2],r),p=2,dim=1)
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        #
        x_tan_r = self.manifold.proj_tan0(x_r, self.curvatures[0])
        x_hyp_r = self.manifold.expmap0(x_tan_r, c=self.curvatures[0])
        x_hyp_r = self.manifold.proj(x_hyp_r, c=self.curvatures[0])


        r_tan = self.manifold.proj_tan0(r, self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp1 = self.manifold.proj(r_hyp, c=self.curvatures[0])
        r_hyp=r_hyp1
        #print(wg/2)
        # r_hyp = self.gat_e_to_r((x_hyp.detach(), r_hyp, self.adjs[3:6]))
        # r_hyp=torch.cat((r_hyp,r_hyp[:-1]),dim=0)
        #
        # x_hyp_r = F.normalize(torch.matmul(self.adjs[-2], r_hyp), p=2, dim=1)

        outputs=[]
        c=[]
        w=x_hyp
        '''
        if self.multi_hop=='concat':
            outputs.append(x_hyp)
            c.append(self.curvatures[0])
        h=x_hyp
        for i in range(self.len):
            ragat=getattr(self, 'ragat{}'.format(i))
            h1=ragat((h,r_hyp,self.adjs[:3]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                h2=hgate(x_hyp,h2)
                h = h2
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
            else:
                c.append(self.curvatures[i + 1])
                outputs.append(h2)
                h=h2
        con_embedding=torch.cat(outputs,dim=1).reshape(-1,(self.len+1),300)
        ent_embedding=con_embedding[:,0,:].unsqueeze(1).expand(-1,(self.len+1),-1)
        w_embedding=torch.cat((ent_embedding,con_embedding),dim=-1).reshape(-1,600)
        weight = torch.exp(self.leakyrelu(self.a.mm((w_embedding.t())).squeeze())).reshape(-1,self.len+1).mean(0)
        weight=(weight*3)/weight.sum()
        print(weight)
        outputs=[outputs[0]*weight[0],outputs[1]*weight[1],outputs[2]*weight[2]]
        '''

        r_hyp = r_hyp1
        if self.multi_hop=='concat':
            outputs.append(x_hyp_r)
            c.append(self.curvatures[0])
        h=x_hyp_r
        for i in range(self.len):
            ragat=getattr(self, 'ragat{}'.format(i))
            h1=ragat((h,r_hyp,self.adjs[:3]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate1{}'.format(i))
                h2=hgate(h,h2)
                h = h2
                if i==0:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
            else:
                h = h2
                c.append(self.curvatures[i + 1])
                outputs.append(h)
        '''
        con_embedding=torch.cat(outputs,dim=1).reshape(-1,(self.len+1),300)
        ent_embedding=con_embedding[:,0,:].unsqueeze(1).expand(-1,(self.len+1),-1)
        w_embedding=torch.cat((ent_embedding,con_embedding),dim=-1).reshape(-1,600)
        weight = torch.exp(self.leakyrelu(self.a.mm((w_embedding.t())).squeeze())).reshape(-1,(self.len+1),1)
        res1=(con_embedding*weight).sum(1).div(weight.sum(1))

        weight = torch.exp(self.leakyrelu(self.a1.mm((w_embedding.t())).squeeze())).reshape(-1,(self.len+1),1)
        res=F.dropout(F.relu(F.normalize(((con_embedding*weight).sum(1).div(weight.sum(1)+res1)/2),p=2,dim=1)),0.2, training=self.training)
        '''



        # e=self.cpe(con_embedding)
        # a=F.softmax(e.reshape(-1,(self.len+1)),dim=1)*(self.len+1)
        # # #U=torch.matmul(a,encoded_layer[-1][:,0].reshape(-1,self.an,self.inputdim))
        # #U = torch.matmul(a,con_embedding).squeeze(1)
        # # outputs+=[F.normalize(U,p=2,dim=1)]
        # # c.append(self.curvatures[-1])
        # outputs1=[]
        # for i in range(self.len+1):
        #     outputs1.append(a[:,i].unsqueeze(1)*outputs[i])

        return outputs,r_hyp,c
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
class LightEA_literal(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, rdim,e2e_adj,e2eself_adj,e2r_adj,r2e_adj,manifold,args):
        super(LightEA_literal, self).__init__()

        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs=[e2eself_adj,e2r_adj,r2e_adj,e2e_adj]
        self.inact=getattr(F, args.act)
        dims, acts, self.curvature = layers.get_dim_act_curv(args)
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvature[i], self.curvature[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            if self.multi_hop=='hgate':
                hgate=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate{}'.format(i), hgate)

        self.encode_graph = True
        self.len=len(dims) - 1
        self.device=args.device
        self.special_spmm = SpecialSpmm()



    def encode(self,input):
        x,r=input

        x_hyp =x
        node_size=x.shape[0]
        r_hyp =r

        x_outputs=[]
        r_outputs=[]
        c=[]
        if self.multi_hop=='concat':
            x_outputs.append(x_hyp)
            r_outputs.append(r_hyp)
            c.append(self.curvature[0])
        hx=x_hyp
        hr=r_hyp
        for i in range(self.len):
            hx1 =hx
            hr1 =hr
            hr2=F.normalize(torch.matmul(self.adjs[2],hx1), p=2, dim=1)
            hx2 = torch.matmul(self.adjs[0], hx1)+torch.matmul(self.adjs[1], hr1)
            hx2= F.normalize(hx2, p=2, dim=1)
            hx3=hx2
            hr3=hr2
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                hx3=hgate(hx,hx3)
                hr3 = hgate(hr, hr3)
                if i==self.len-1:
                    x_outputs.append(hx3)
                    r_outputs.append(hr3)
                    c.append(self.curvature[i + 1])
            else:
                x_outputs.append(hx3)
                r_outputs.append(hr3)
                c.append(self.curvature[i + 1])
            hx = hx3
            hr = hr3
        ent_feature =F.normalize(torch.cat(x_outputs, dim=-1), p=2, dim=1)
        rel_feature = F.normalize(torch.cat(r_outputs, dim=-1), p=2, dim=1)
        rel_dim=x.shape[-1]//2
        rel_feature = self.random_projection(rel_feature,rel_dim)
        adj_value=rel_feature[self.adjs[2]._indices()[0,:],:]
        batch_size = ent_feature.shape[-1] // 16
        features_list = []
        c = []
        for batch in range(rel_dim // batch_size + 1):
            temp_list = []
            for head in range(batch_size):
                if batch * batch_size + head >= rel_dim:
                    break
                # sparse_graph =torch.sparse_coo_tensor(self.adjs[0]._indices(), adj_value[:, batch * batch_size + head], (node_size, node_size))
                # val=torch.exp(-adj_value[:, batch * batch_size + head])
                # N=ent_feature.shape[0]
                # ones = torch.ones(size=(N, 1))
                # if ent_feature.is_cuda:
                #     ones = ones.cuda()
                #
                # feature =  self.special_spmm(indices=self.adjs[3]._indices(), values=val, shape=torch.Size([node_size, node_size]), b=self.random_projection(ent_feature, 16))#torch.matmul(sparse_graph, self.random_projection(ent_feature, 16))
                # enum=self.special_spmm(indices=self.adjs[3]._indices(), values=val, shape=torch.Size([node_size, node_size]), b=ones)
                # enum[enum==0]=1
                # temp_list.append(feature.div(enum))
                val=adj_value[:, batch * batch_size + head]
                h=self.random_projection(ent_feature,16)
                feature = self.special_spmm(indices=self.adjs[3]._indices(), values=val,
                                            shape=torch.Size([node_size, node_size]),
                                            b=h)
                temp_list.append(F.normalize(feature,p=2,dim=1))
            if len(temp_list):
                features_list.append(torch.cat(temp_list, dim=-1))
                c.append(self.curvature[-1])
        vec=torch.cat(features_list, dim=-1).numpy()
        return [torch.FloatTensor(vec / np.linalg.norm(vec, axis=-1, keepdims=True))],r_hyp,[self.curvature[-1]]


    def random_projection(self,x, out_dim):
        random_vec = F.normalize(torch.FloatTensor(np.random.normal(0,1,(x.shape[-1],out_dim))), p=2, dim=1).to(self.device)
        return torch.matmul(x, random_vec)
class RAGA(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, rdim,e2e_adj,e2eself_adj,r2t_adj,h2t_adj,t2t_adj,r2t_adj_o,h2t_adj_o,t2t_adj_o,manifold,args):
        super(RAGA, self).__init__()

        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs=[e2e_adj,r2t_adj_o,h2t_adj_o,t2t_adj_o,e2e_adj,h2t_adj,t2t_adj]
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            ragat=layers.HyperbolicGraphConvolution(
                self.manifold, in_dim, out_dim,c_in, args.dropout, args.bias,args.use_att,args.local_agg,args.use_w
            )
            setattr(self, 'gcn{}'.format(i), ragat)
            if self.multi_hop=='hgate':
                hact = layers.HypAct(manifold, c_in, c_in, act)
                setattr(self, 'hact{}'.format(i), hact)
                hgate=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate{}'.format(i), hgate)
            else:
                hact = layers.HypAct(manifold, c_in, c_out, act)
                setattr(self, 'hact{}'.format(i), hact)
        input_dim=output_dim=dims[-1]
        rdim = output_dim // 3
        self.gat_e_to_r=layers.EntityAwareRelationGraphAttentionLayer(manifold, input_dim, output_dim, rdim,args.dropout, 0.2, self.curvatures[-1],args.bias,args.n_heads,args.heads_concat)
        self.gat_r_to_e = layers.RelationAwareEntityGraphAttentionLayer(manifold, input_dim, output_dim, rdim,
                                                                        args.dropout, 0.2, self.curvatures[-1],
                                                                        args.bias, args.n_heads, args.heads_concat)
        self.gat=layers.GraphAttentionLayer(manifold, input_dim+2*rdim, input_dim+2*rdim,
                                                                        args.dropout, 0.2, self.curvatures[-1],
                                                                        args.bias, args.n_heads, args.heads_concat,args.use_w)

        self.layers = nn.ModuleList(gc_layers)#nn.Sequential(*hgc_layers)

        self.encode_graph = True
        self.len=len(dims) - 1


    def encode(self,input):
        x,r=input
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(r, self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])

        outputs=[]
        c=[]

        if self.multi_hop=='concat':
            outputs.append(x_hyp)
            c.append(self.curvatures[0])
        h=x_hyp
        for i in range(self.len):
            gcn=getattr(self, 'gcn{}'.format(i))
            h1=gcn((h,self.adjs[0]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                h2=hgate(h,h2)
                h=h2
                if i==self.len-1:
                    outputs.append(h)
                    c.append(self.curvatures[i + 1])
            else:
                h = h2
                outputs.append(h)
                c.append(self.curvatures[i + 1])
        if self.multi_hop == 'hgate':
            x_r=self.gat_e_to_r((outputs[0],r_hyp,self.adjs[1:4]))
            x_e=self.gat_r_to_e((outputs[0],x_r,self.adjs[1:4]))
            x_e1=self.gat((x_e,x_r,self.adjs[4:]))
            outputs=[F.normalize(torch.cat((x_e,F.relu(x_e1)),dim=-1),p=2,dim=-1)]
            c=[self.curvatures[-1]]
        else:
            x_r=self.gat_e_to_r((outputs[-1],r_hyp,self.adjs[1:4]))
            x_e=self.gat_r_to_e((outputs[-1],x_r,self.adjs[1:4]))
            x_e1=self.gat((x_e,x_r,self.adjs[4:]))
            outputs.append(x_e)
            c.append(self.curvatures[-1])
            outputs.append(F.relu(x_e1))
            c.append(self.curvatures[-1])

        return outputs,r_hyp,c


class RAGA_literal(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, rdim, e2e_adj, e2eself_adj, r2t_adj, h2t_adj, t2t_adj, r2t_adj_o, h2t_adj_o, t2t_adj_o, manifold,
                 args):
        super(RAGA_literal, self).__init__()
        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs=[e2e_adj,r2t_adj_o,h2t_adj_o,t2t_adj_o,e2e_adj,h2t_adj,t2t_adj]
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            ragat=layers.GraphAttentionLayer(
                self.manifold, in_dim, out_dim, args.dropout, 0.2,c_in,args.bias, args.n_heads, args.heads_concat,args.use_w
            )
            setattr(self, 'gcn{}'.format(i), ragat)
            if self.multi_hop=='hgate':
                hact = layers.HypAct(manifold, c_in, c_in, act)
                setattr(self, 'hact{}'.format(i), hact)
                hgate=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate{}'.format(i), hgate)
            else:
                hact = layers.HypAct(manifold, c_in, c_out, act)
                setattr(self, 'hact{}'.format(i), hact)
        input_dim=output_dim=dims[-1]
        rdim = output_dim // 3
        input_dim=len(dims)*input_dim
        self.input_dim=input_dim
        self.gat_e_to_r=layers.EntityAwareRelationGraphAttentionLayer(manifold, input_dim, output_dim, rdim,args.dropout, 0.2, self.curvatures[-1],args.bias,args.n_heads,args.heads_concat)
        self.gat_r_to_e = layers.RelationAwareEntityGraphAttentionLayer(manifold, input_dim, output_dim, rdim,
                                                                        args.dropout, 0.2, self.curvatures[-1],
                                                                        args.bias, args.n_heads, args.heads_concat)
        self.gat=layers.GraphAttentionLayer(manifold, input_dim+2*rdim, input_dim+2*rdim,
                                                                        args.dropout, 0.2, self.curvatures[-1],
                                                                        args.bias, args.n_heads, args.heads_concat,args.use_w)

        self.layers = nn.ModuleList(gc_layers)#nn.Sequential(*hgc_layers)

        self.encode_graph = True
        self.len=len(dims) - 1


    def encode(self,input):
        x,r=input
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(r, self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])

        outputs=[]
        c=[]

        if self.multi_hop=='concat':
            outputs.append(x_hyp)
            c.append(self.curvatures[0])
        h=x_hyp
        for i in range(self.len):
            gcn=getattr(self, 'gcn{}'.format(i))
            h1=gcn((h,r_hyp,self.adjs[4:]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                h2=hgate(h,h2)
                h=h2
                if i==self.len-1:
                    h = h2
                    outputs.append(h)
                    c.append(self.curvatures[i + 1])
            else:
                h = h2
                outputs.append(h)
                c.append(self.curvatures[i + 1])
        if self.multi_hop == 'hgate':
            x_r=self.gat_e_to_r((outputs[0],r_hyp,self.adjs[1:4]))
            x_e=self.gat_r_to_e((outputs[0],x_r,self.adjs[1:4]))
            x_e1=self.gat((x_e,x_r,self.adjs[4:]))
            outputs=[F.normalize(torch.cat((x_e,F.relu(x_e1)),dim=-1),p=2,dim=-1)]
            c=[self.curvatures[-1]]
        else:
            h=torch.cat(outputs,dim=-1)
            x_r=self.gat_e_to_r((h,r_hyp,self.adjs[1:4]))
            x_e=self.gat_r_to_e((h,x_r,self.adjs[1:4]))
            x_e1=self.gat((x_e,x_r,self.adjs[4:]))
            outputs+=[F.normalize(F.relu(x_e1),p=2,dim=-1),F.normalize(x_e[:,self.input_dim:],p=2,dim=-1)]
            c+=[self.curvatures[-1],self.curvatures[-1]]

        return outputs,r_hyp,c
class Test(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, rdim,e2r_adj, e2e_adj, e2eself_adj, r2t_adj, h2t_adj, t2t_adj, r2t_adj_o, h2t_adj_o, t2t_adj_o, manifold,
                 args):
        super(Test, self).__init__()
        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs=[e2eself_adj,r2t_adj_o,h2t_adj_o,t2t_adj_o,r2t_adj,h2t_adj,t2t_adj,e2r_adj]
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        gc_layers = []

        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gcn=layers.HyperbolicGraphAttentionLayer(
                self.manifold, in_dim, out_dim,args.dropout, 0.2, c_in,
                                                                        args.bias, args.n_heads, args.heads_concat,args.use_w)
            setattr(self, 'gcn{}'.format(i), gcn)
            hact=layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact{}'.format(i), hact)
            if self.multi_hop == 'hgate':
                hgate=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate{}'.format(i), hgate)

        self.layers = nn.ModuleList(gc_layers)#nn.Sequential(*hgc_layers)

        self.encode_graph = True
        self.len=len(dims) - 1

        self.linear_gate = layers.HypLinear(self.manifold, 3*dims[-1], 3,self.curvatures[-1], args.dropout, args.bias)
    def encode(self,input):
        x,r=input
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(r, self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])

        x_r=F.normalize(torch.matmul(self.adjs[-1],r),p=2,dim=-1)

        x_tan_r = self.manifold.proj_tan0(x_r, self.curvatures[0])
        x_hyp_r = self.manifold.expmap0(x_tan_r, c=self.curvatures[0])
        x_hyp_r = self.manifold.proj(x_hyp_r, c=self.curvatures[0])

        outputs=[]
        c=[]
        h=x_hyp
        if self.multi_hop=='concat':
            outputs.append(h)
            c.append(self.curvatures[0])
        for i in range(self.len):
            hact = getattr(self, 'hact{}'.format(i))
            gcn=getattr(self, 'gcn{}'.format(i))
            h2=hact(gcn((h,r_hyp,self.adjs[4:])))
            if self.multi_hop == 'hgate':
                hgate = getattr(self, 'hgate{}'.format(i))
                h2 = hgate(h, h2)
                h = h2
                if i==self.len-1:
                    outputs.append(h)
                    c.append(self.curvatures[i + 1])
            else:
                h = h2
                outputs.append(h)
                c.append(self.curvatures[i + 1])
        h=x_hyp_r
        if self.multi_hop=='concat':
            outputs.append(h)
            c.append(self.curvatures[0])
        for i in range(self.len):
            hact = getattr(self, 'hact{}'.format(i))
            gcn=getattr(self, 'gcn{}'.format(i))
            h2=hact(gcn((h,r_hyp,self.adjs[4:])))
            if self.multi_hop == 'hgate':
                hgate = getattr(self, 'hgate{}'.format(i))
                h2 = hgate(h, h2)
                h = h2
                if i==self.len-1:
                    outputs.append(h)
                    c.append(self.curvatures[i + 1])
            else:
                h = h2
                outputs.append(h)
                c.append(self.curvatures[i + 1])

        return outputs,r_hyp,c

class Test1(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, rdim, e2e_adj, e2eself_adj, r2t_adj, h2t_adj, t2t_adj, r2t_adj_o, h2t_adj_o, t2t_adj_o, manifold,
                 args):
        super(Test1, self).__init__()
        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs=[e2e_adj,r2t_adj_o,h2t_adj_o,t2t_adj_o,r2t_adj,h2t_adj,t2t_adj]
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        gc_layers = []

        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            ragat=layers.Test(
                self.manifold, in_dim, out_dim, rdim,args.dropout, 0.2,c_in,args.bias, args.n_heads, args.heads_concat,args.use_w
            )
            setattr(self, 'gcn{}'.format(i), ragat)
            if self.multi_hop=='hgate':
                hact = layers.HypAct(manifold, c_in, c_in, act)
                setattr(self, 'hact{}'.format(i), hact)
                hgate=layers.HighwayGate1(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate{}'.format(i), hgate)
            else:
                hact = layers.HypAct(manifold, c_in, c_out, act)
                setattr(self, 'hact{}'.format(i), hact)
        input_dim=output_dim=dims[-1]
        rdim = output_dim // 2
        input_dim=len(dims)*input_dim
        self.input_dim=input_dim
        self.gat_e_to_r=layers.EntityAwareRelationGraphAttentionLayer(manifold, input_dim, output_dim, rdim,args.dropout, 0.2, self.curvatures[-1],args.bias,args.n_heads,args.heads_concat)
        self.gat_r_to_e = layers.RelationAwareEntityGraphAttentionLayer(manifold, input_dim, output_dim, rdim,
                                                                        args.dropout, 0.2, self.curvatures[-1],
                                                                        args.bias, args.n_heads, args.heads_concat)
        self.layers = nn.ModuleList(gc_layers)#nn.Sequential(*hgc_layers)

        self.encode_graph = True
        self.len=len(dims) - 1

        self.linear_gate = layers.HypLinear(self.manifold, 3*dims[-1], 3,self.curvatures[-1], args.dropout, args.bias)
    def encode(self,input):
        x,r=input
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(r, self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])

        outputs=[]
        c=[]

        if self.multi_hop=='concat':
            outputs.append(x_hyp)
            c.append(self.curvatures[0])
        h=x_hyp
        for i in range(self.len):
            gcn=getattr(self, 'gcn{}'.format(i))
            h1,_=gcn((h,r_hyp,self.adjs[4:]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                #if i==self.len-1:
                outputs.append(hgate(x_hyp,h2))
                h=h2
                c.append(self.curvatures[i + 1])
            else:
                h = h2
                outputs.append(h)
                c.append(self.curvatures[i + 1])
        h=torch.cat(outputs,dim=-1)
        x_r=self.gat_e_to_r((h,r_hyp,self.adjs[1:4]))
        x_e=self.gat_r_to_e((h,x_r,self.adjs[1:4]))[:,self.input_dim:]

        if self.multi_hop=='concat':
            outputs.append(x_e)
            c.append(self.curvatures[0])
        h=F.normalize(x_e,p=2,dim=-1)
        for i in range(self.len):
            gcn=getattr(self, 'gcn{}'.format(i))
            h1,_=gcn((h,r_hyp,self.adjs[4:]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                #if i==self.len-1:
                outputs.append(hgate(x_hyp,h2))
                h=h2
                c.append(self.curvatures[i + 1])
            else:
                h = h2
                outputs.append(h)
                c.append(self.curvatures[i + 1])


        return outputs,r_hyp,c

class ERMC(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, rdim,e2ei_adj,e2eo_adj,e2ri_adj,e2ro_adj,r2ei_adj,r2eo_adj,manifold,args):
        super(ERMC, self).__init__()

        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs=[e2ei_adj,e2eo_adj,e2ri_adj,e2ro_adj,r2ei_adj,r2eo_adj]
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gcn_e2ei=layers.HyperbolicGraphConvolution(
                self.manifold, in_dim, out_dim,c_in, args.dropout, args.bias,args.use_att,args.local_agg,args.use_w
            )
            setattr(self, 'gcn_e2ei{}'.format(i), gcn_e2ei)

            gcn_e2eo=layers.HyperbolicGraphConvolution(
                self.manifold, in_dim, out_dim,c_in, args.dropout, args.bias,args.use_att,args.local_agg,args.use_w
            )
            setattr(self, 'gcn_e2eo{}'.format(i), gcn_e2eo)

            gcn_e2ri=layers.HyperbolicGraphConvolution(
                self.manifold, rdim, out_dim,c_in, args.dropout, args.bias,args.use_att,args.local_agg,args.use_w
            )
            setattr(self, 'gcn_e2ri{}'.format(i), gcn_e2ri)

            gcn_e2ro=layers.HyperbolicGraphConvolution(
                self.manifold, rdim, out_dim,c_in, args.dropout, args.bias,args.use_att,args.local_agg,args.use_w
            )
            setattr(self, 'gcn_e2ro{}'.format(i), gcn_e2ro)

            gcn_r2ei = layers.HyperbolicGraphConvolution(
                self.manifold, in_dim, rdim, c_in, args.dropout, args.bias, args.use_att, args.local_agg,args.use_w
            )
            setattr(self, 'gcn_r2ei{}'.format(i), gcn_r2ei)

            gcn_r2eo = layers.HyperbolicGraphConvolution(
                self.manifold, in_dim, rdim, c_in, args.dropout, args.bias, args.use_att, args.local_agg,args.use_w
            )
            setattr(self, 'gcn_r2eo{}'.format(i), gcn_r2eo)
            hact=layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact{}'.format(i), hact)
            if self.multi_hop=='hgate':
                hgate=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate{}'.format(i), hgate)
            elif self.multi_hop=='mlp':
                mlpe=layers.Mlp(self.manifold,out_dim*5,out_dim,c_in, c_out, act,args.dropout, args.bias)
                setattr(self, 'mlpe{}'.format(i), mlpe)
                mlpr=layers.Mlp(self.manifold,rdim*3,rdim,c_in, c_out,act ,args.dropout, args.bias)
                setattr(self, 'mlpr{}'.format(i), mlpr)
        self.layers = nn.ModuleList(gc_layers)#nn.Sequential(*hgc_layers)
        self.encode_graph = True
        self.len=len(dims) - 1


    def encode(self,input):
        x,r=input
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(r, self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])

        outputs=[]
        c=[]
        if self.multi_hop=='concat':
            outputs.append(x_hyp)
            c.append(self.curvatures[0])
        h=x_hyp
        r=r_hyp
        for i in range(self.len):
            hact = getattr(self, 'hact{}'.format(i))
            gcn_e2ei=getattr(self, 'gcn_e2ei{}'.format(i))
            e2ei=F.normalize(hact(gcn_e2ei((h,self.adjs[0]))),p=2,dim=-1)
            gcn_e2eo=getattr(self, 'gcn_e2eo{}'.format(i))
            e2eo=F.normalize(hact(gcn_e2eo((h,self.adjs[1]))),p=2,dim=-1)
            gcn_e2ri=getattr(self, 'gcn_e2ri{}'.format(i))
            e2ri=F.normalize(hact(gcn_e2ri((r,self.adjs[2]))),p=2,dim=-1)
            gcn_e2ro=getattr(self, 'gcn_e2ro{}'.format(i))
            e2ro=F.normalize(hact(gcn_e2ro((r,self.adjs[3]))),p=2,dim=-1)

            gcn_r2ei=getattr(self, 'gcn_r2ei{}'.format(i))
            r2ei=F.normalize(hact(gcn_r2ei((h,self.adjs[4]))),p=2,dim=-1)
            gcn_r2eo=getattr(self, 'gcn_r2eo{}'.format(i))
            r2eo=F.normalize(hact(gcn_r2eo((h,self.adjs[5]))),p=2,dim=-1)

            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                h2=hgate(h,h2)
                h=h2
                if i==self.len-1:
                    h = h2
                    outputs.append(h)
                    c.append(self.curvatures[i + 1])
            elif self.multi_hop=='mlp':
                mlpe=getattr(self, 'mlpe{}'.format(i))
                h=mlpe(h,[e2ei,e2eo,e2ri,e2ro])
                mlpe=getattr(self, 'mlpr{}'.format(i))
                r=mlpe(r,[r2ei,r2eo])
                if i==self.len-1:
                    outputs.append(F.normalize(h,p=2,dim=-1))
                    c.append(self.curvatures[i + 1])
            else:
                h = h2
                outputs.append(h)
                c.append(self.curvatures[i + 1])
        return outputs,r_hyp,c
class ERMC_adapt(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, rdim,e2ei_adj,e2eo_adj,e2ri_adj,e2ro_adj,r2ei_adj,r2eo_adj,r2t_adj_o,h2t_adj_o,t2t_adj_o,manifold,args):
        super(ERMC_adapt, self).__init__()

        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs=[e2ei_adj,e2eo_adj,e2ri_adj,e2ro_adj,r2ei_adj,r2eo_adj,r2t_adj_o,h2t_adj_o,t2t_adj_o]
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gcn_e2ei=layers.GraphAttentionLayer2(
                self.manifold, in_dim, out_dim, args.dropout, 0.2,c_in,args.bias,args.n_heads, args.heads_concat,args.use_w
            )
            setattr(self, 'gcn_e2ei{}'.format(i), gcn_e2ei)

            gcn_e2eo=layers.GraphAttentionLayer2(
                self.manifold, in_dim, out_dim, args.dropout, 0.2,c_in,args.bias,args.n_heads, args.heads_concat,args.use_w
            )
            setattr(self, 'gcn_e2eo{}'.format(i), gcn_e2eo)

            gcn_e2ri=layers.GraphAttentionLayer2(
                self.manifold, in_dim, out_dim, args.dropout, 0.2,c_in,args.bias,args.n_heads, args.heads_concat,args.use_w
            )
            setattr(self, 'gcn_e2ri{}'.format(i), gcn_e2ri)

            gcn_e2ro=layers.GraphAttentionLayer2(
                self.manifold, in_dim, out_dim, args.dropout, 0.2,c_in,args.bias,args.n_heads, args.heads_concat,args.use_w
            )
            setattr(self, 'gcn_e2ro{}'.format(i), gcn_e2ro)

            gcn_r2ei = layers.GraphAttentionLayer2(
                self.manifold, in_dim, out_dim, args.dropout, 0.2,c_in,args.bias,args.n_heads, args.heads_concat,args.use_w
            )
            setattr(self, 'gcn_r2ei{}'.format(i), gcn_r2ei)

            gcn_r2eo = layers.GraphAttentionLayer2(
                self.manifold, in_dim, out_dim, args.dropout, 0.2,c_in,args.bias,args.n_heads, args.heads_concat,args.use_w
            )
            setattr(self, 'gcn_r2eo{}'.format(i), gcn_r2eo)
            hact=layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact{}'.format(i), hact)
            if self.multi_hop=='hgate':
                hgate=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate{}'.format(i), hgate)
            else:
                mlpe=layers.Mlp(self.manifold,out_dim*4,out_dim,c_in, c_out, act,args.dropout, args.bias)
                setattr(self, 'mlpe{}'.format(i), mlpe)
                mlpr=layers.Mlp(self.manifold,rdim*3,rdim,c_in, c_out,act ,args.dropout, args.bias)
                setattr(self, 'mlpr{}'.format(i), mlpr)
        self.layers = nn.ModuleList(gc_layers)#nn.Sequential(*hgc_layers)
        self.encode_graph = True
        self.len=len(dims) - 1


    def encode(self,input):
        x,r=input
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(r, self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])

        outputs=[]
        c=[]
        if self.multi_hop=='concat':
            outputs.append(x_hyp)
            c.append(self.curvatures[0])
        h=x_hyp
        r=r_hyp
        for i in range(self.len):
            hact = getattr(self, 'hact{}'.format(i))
            gcn_e2ei=getattr(self, 'gcn_e2ei{}'.format(i))
            e2ei=F.normalize(hact(gcn_e2ei((h,h,[self.adjs[8],self.adjs[7]]))),p=2,dim=-1)
            gcn_e2eo=getattr(self, 'gcn_e2eo{}'.format(i))
            e2eo=F.normalize(hact(gcn_e2eo((h,h,self.adjs[7:]))),p=2,dim=-1)
            gcn_e2ri=getattr(self, 'gcn_e2ri{}'.format(i))
            e2ri=F.normalize(hact(gcn_e2ri((h,r,[self.adjs[8],self.adjs[6]]))),p=2,dim=-1)
            gcn_e2ro=getattr(self, 'gcn_e2ro{}'.format(i))
            e2ro=F.normalize(hact(gcn_e2ro((h,r,[self.adjs[7],self.adjs[6]]))),p=2,dim=-1)

            gcn_r2ei=getattr(self, 'gcn_r2ei{}'.format(i))
            r2ei=F.normalize(hact(gcn_r2ei((r,h,self.adjs[6:]))),p=2,dim=-1)
            gcn_r2eo=getattr(self, 'gcn_r2eo{}'.format(i))
            r2eo=F.normalize(hact(gcn_r2eo((r,h,[self.adjs[6],self.adjs[8],self.adjs[7]]))),p=2,dim=-1)

            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                h2=hgate(h,h2)
                h=h2
                if i==self.len-1:
                    h = h2
                    outputs.append(h)
                    c.append(self.curvatures[i + 1])
            elif self.multi_hop=='mlp':
                mlpe=getattr(self, 'mlpe{}'.format(i))
                h=mlpe(h,[e2ei,e2eo,e2ri,e2ro])
                mlpe=getattr(self, 'mlpr{}'.format(i))
                r=mlpe(r,[r2ei,r2eo])
                if i==self.len-1:
                    outputs.append(F.normalize(h,p=2,dim=-1))
                    c.append(self.curvatures[i + 1])
            else:
                mlpe=getattr(self, 'mlpe{}'.format(i))
                h=mlpe(e2ei,[e2eo,e2ri,e2ro])
                mlpe=getattr(self, 'mlpr{}'.format(i))
                r=mlpe(r,[r2ei,r2eo])
                outputs.append(F.normalize(h,p=2,dim=-1))
                c.append(self.curvatures[i + 1])


        return outputs,r_hyp,c
class BERT_INT_N(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, manifold,ent_pairs,neigh_dict,ent_pad_id,args,batch_size = 1024):
        super(BERT_INT_N, self).__init__()

        self.manifold =manifold
        self.kernel_num=args.kernel_num
        self.device=args.device
        self.ent_pairs=ent_pairs
        self.neigh_dict=neigh_dict
        self.ent_pad_id=ent_pad_id
        self.batch_size=batch_size

    def encode(self,input):
        x,r=input
        e_emb = F.normalize(x, p=2, dim=-1)
        sigmas = self.kernel_sigmas(self.kernel_num).to(self.device)
        mus = self.kernel_mus(self.kernel_num).to(self.device)
        # print("sigmas:",sigmas)
        # print("mus:",mus)
        sigmas = sigmas.view(1, 1, -1)
        mus = mus.view(1, 1, -1)

        all_ent_pairs = []
        all_features = []
        all_features1=[]
        # print("entity_embedding shape:",e_emb.shape)
        for start_pos in range(0, len(self.ent_pairs), self.batch_size):
            batch_ent_pairs = self.ent_pairs[start_pos: start_pos + self.batch_size]
            e1s = [e1 for e1, e2 in batch_ent_pairs]
            e2s = [e2 for e1, e2 in batch_ent_pairs]
            e1_tails = [self.neigh_dict[int(e1)] for e1 in e1s]  # size: [B(Batchsize),ne1(e1_neighbor_max_num)]
            e2_tails = [self.neigh_dict[int(e2)] for e2 in e2s]  # [B,ne2]
            e1_masks = np.ones(np.array(e1_tails).shape)
            e2_masks = np.ones(np.array(e2_tails).shape)
            e1_masks[np.array(e1_tails) == self.ent_pad_id] = 0
            e2_masks[np.array(e2_tails) ==  self.ent_pad_id] = 0
            e1_masks = torch.FloatTensor(e1_masks.tolist()).to(self.device).unsqueeze(-1)  # [B,ne1,1]
            e2_masks = torch.FloatTensor(e2_masks.tolist()).to(self.device).unsqueeze(-1)  # [B,ne2,1]
            e1_tails = torch.LongTensor(e1_tails).to(self.device)  # [B,ne1]
            e2_tails = torch.LongTensor(e2_tails).to(self.device)  # [B,ne2]
            e1_tail_emb = e_emb[e1_tails]  # [B,ne1,embedding_dim]
            e2_tail_emb = e_emb[e2_tails]  # [B,ne2,embedding_dim]
            sim_matrix = torch.bmm(e1_tail_emb, torch.transpose(e2_tail_emb, 1, 2))  # [B,ne1,ne2]
            features = self.batch_dual_aggregation_feature_gene(sim_matrix, mus, sigmas, e1_masks, e2_masks)
            features = features.detach().cpu().tolist()
            all_ent_pairs.extend(batch_ent_pairs)
            all_features.extend(features)

            e1s = torch.LongTensor(e1s).to(self.device)
            e2s = torch.LongTensor(e2s).to(self.device)
            e1_embs = e_emb[e1s]  # [B,embedding_dim]
            e2_embs = e_emb[e2s]
            cos_sim = F.cosine_similarity(e1_embs, e2_embs)
            cos_sim = cos_sim.detach().cpu().unsqueeze(-1).tolist()
            all_ent_pairs.extend(batch_ent_pairs)
            all_features1.extend(cos_sim)

        return [torch.FloatTensor(all_features).to(self.device),torch.FloatTensor(all_features1).to(self.device)],None,None

    def batch_dual_aggregation_feature_gene(self,batch_sim_matrix, mus, sigmas, attn_ne1, attn_ne2):
        """
        Dual Aggregation.
        [similarity matrix -> feature]
        :param batch_sim_matrix: [B,ne1,ne2]
        :param mus: [1,1,k(kernel_num)]
        :param sigmas: [1,1,k]
        :param attn_ne1: [B,ne1,1]
        :param attn_ne2: [B,ne2,1]
        :return feature: [B,kernel_num * 2].
        """
        sim_maxpooing_1, _ = batch_sim_matrix.topk(k=1, dim=-1)  # [B,ne1,1] #get max value.
        pooling_value_1 = torch.exp((- ((sim_maxpooing_1 - mus) ** 2) / (sigmas ** 2) / 2))  # [B,ne1,k]
        log_pooling_sum_1 = torch.log(torch.clamp(pooling_value_1, min=1e-10)) * attn_ne1 * 0.01  # [B,ne1,k]
        log_pooling_sum_1 = torch.sum(log_pooling_sum_1, 1)  # [B,k]

        sim_maxpooing_2, _ = torch.transpose(batch_sim_matrix, 1, 2).topk(k=1, dim=-1)  # [B,ne2,1]
        pooling_value_2 = torch.exp((- ((sim_maxpooing_2 - mus) ** 2) / (sigmas ** 2) / 2))  # [B,ne2,k]
        log_pooling_sum_2 = torch.log(torch.clamp(pooling_value_2, min=1e-10)) * attn_ne2 * 0.01  # [B,ne2,k]
        log_pooling_sum_2 = torch.sum(log_pooling_sum_2, 1)  # [B,k]

        batch_ne2_num = attn_ne2.sum(dim=1)  # [B,1]
        batch_ne2_num = torch.clamp(batch_ne2_num, min=1e-10)
        log_pooling_sum_2 = log_pooling_sum_2 * (1 / batch_ne2_num)  # [B,k]

        batch_ne1_num = attn_ne1.sum(dim=1)  # [B,1]
        batch_ne1_num = torch.clamp(batch_ne1_num, min=1e-10)
        log_pooling_sum_1 = log_pooling_sum_1 * (1 / batch_ne1_num)  # [B,k]
        return torch.cat([log_pooling_sum_1, log_pooling_sum_2], dim=-1)

    def kernel_mus(self,n_kernels):
        l_mu = [1]
        if n_kernels == 1:
            return l_mu
        bin_size = 1.0 / (n_kernels - 1)  # score range from [0, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return torch.FloatTensor(l_mu)

    def kernel_sigmas(self,n_kernels):
        l_sigma = [0.001]  # for exact match.
        # small variance -> exact match
        if n_kernels == 1:
            return l_sigma
        l_sigma += [0.1] * (n_kernels - 1)
        return torch.FloatTensor(l_sigma)
class GRU(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, manifold,pad_idx,args,batch_size = 1024):
        super(GRU, self).__init__()

        self.manifold =manifold
        self.device=args.device
        self.pad_idx=pad_idx
        self.rnn = layers.GRUAttnNet(300, 300, 1, dropout=0.2, device=args.device)
        self.bn = torch.nn.Sequential(
            torch.nn.BatchNorm1d(300),
            # t.nn.BatchNorm1d(bert_output_dim + rnn_hidden_dim),
            # t.nn.BatchNorm1d(bert_output_dim),
            torch.nn.Softsign(),
        )
        # #########2021-11-17
        attr_out_dim = 300
        self.combiner = torch.nn.Sequential(
            torch.nn.Linear(attr_out_dim, attr_out_dim),
            # t.nn.Linear(attr_out_dim, attr_out_dim),
        )

    def encode(self,input):
        batch_neighbors,batch_self,batch_neighbors1,batch_self1,all_embed=input

        pos1=self.get_emb1(batch_neighbors,batch_self,all_embed)
        pos2 = self.get_emb1(batch_neighbors1,batch_self1,all_embed)


        return (pos1-pos2).pow(2).sum(dim=-1)
    def get_emb1(self,batch_neighbors,batch_self,all_embed):
        ones = torch.ones(batch_neighbors.shape).to(self.device)
        zeros = torch.zeros(batch_neighbors.shape).to(self.device)
        neighbor_mask = torch.where(batch_neighbors == self.pad_idx, zeros, ones)

        h = all_embed[batch_neighbors]
        h = F.relu(h)
        h_prime, _ = self.rnn.forward(h, neighbor_mask)
        #print(batch_neighbors[6],h_prime[6])
        # print(torch.isnan(h_prime).sum())
        # h_prime = self.bn(h_prime)
        # print(torch.isnan(h_prime).sum())
        return torch.cat((F.normalize(h_prime,p=2,dim=-1),all_embed[batch_self]),dim=-1)
    def get_emb(self,batch_neighbors,batch_self,all_embed):
        ones = torch.ones(batch_neighbors.shape).to(self.device)
        zeros = torch.zeros(batch_neighbors.shape).to(self.device)
        neighbor_mask = torch.where(batch_neighbors == self.pad_idx, zeros, ones)

        h = all_embed[batch_neighbors]
        h = F.relu(h)
        h_prime, _ = self.rnn.forward(h, neighbor_mask)
        #print(batch_neighbors[6],h_prime[6])
        # print(torch.isnan(h_prime).sum())
        # h_prime = self.bn(h_prime)
        # print(torch.isnan(h_prime).sum())
        return torch.cat((h_prime,all_embed[batch_self]),dim=-1)


class BERT_INT_A(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, manifold,ent_pairs,ent2valueids,cosin_features,value_pad_id,args,batch_size = 1024):
        super(BERT_INT_A, self).__init__()

        self.manifold =manifold
        self.kernel_num=args.kernel_num
        self.device=args.device
        self.ent_pairs=ent_pairs
        self.ent2valueids=ent2valueids
        self.value_pad_id=value_pad_id
        self.batch_size=batch_size
        self.cosin_features=cosin_features
        self.reverse=args.sim_reverse

    def encode(self,input):
        x,r=input

        value_emb = F.normalize(x, p=2, dim=-1)
        sigmas = self.kernel_sigmas(self.kernel_num).to(self.device)
        mus = self.kernel_mus(self.kernel_num).to(self.device)
        # print("sigmas:",sigmas)
        # print("mus:",mus)
        sigmas = sigmas.view(1, 1, -1)
        mus = mus.view(1, 1, -1)

        all_ent_pairs = []
        all_features = []
        all_features1=[]
        # print("attributeValue embedding shape:", value_emb.shape)
        for start_pos in range(0, len(self.ent_pairs), self.batch_size):
            batch_ent_pairs = self.ent_pairs[start_pos: start_pos + self.batch_size]
            e1s = [e1 for e1, e2 in batch_ent_pairs]
            e2s = [e2 for e1, e2 in batch_ent_pairs]
            e1_values = [self.ent2valueids[e1] for e1 in e1s]  # size: [B(Batchsize), ne1(e1_attributeValue_max_num)]
            e2_values = [self.ent2valueids[e2] for e2 in e2s]  # [B,ne2]

            e1_masks = np.ones(np.array(e1_values).shape)
            e2_masks = np.ones(np.array(e2_values).shape)
            e1_masks[np.array(e1_values) == self.value_pad_id] = 0
            e2_masks[np.array(e2_values) == self.value_pad_id] = 0
            e1_masks = torch.FloatTensor(e1_masks.tolist()).to(self.device).unsqueeze(-1)  # [B,ne1,1]
            e2_masks = torch.FloatTensor(e2_masks.tolist()).to(self.device).unsqueeze(-1)  # [B,ne2,1]

            e1_values = torch.LongTensor(e1_values).to(self.device)  # [B,ne1]
            e2_values = torch.LongTensor(e2_values).to(self.device)   # [B,ne2]
            e1_values_emb = value_emb[e1_values]  # [B,ne1,embedding_dim]
            e2_values_emb = value_emb[e2_values]  # [B,ne2,embedding_dim]

            sim_matrix = torch.bmm(e1_values_emb, torch.transpose(e2_values_emb, 1, 2))  # [B,ne1,ne2]
            features =  self.batch_dual_aggregation_feature_gene(sim_matrix, mus, sigmas, e1_masks, e2_masks)
            features = features.detach().cpu().tolist()

            all_ent_pairs.extend(batch_ent_pairs)
            all_features.extend(features)
            all_features1.extend(self.cosin_features[start_pos: start_pos + self.batch_size])
        ft=torch.FloatTensor(all_features1).to(self.device).unsqueeze(1)
        if self.reverse:
            ft=ft.max()-ft
            ft=ft/ft.max()
        return [torch.FloatTensor(all_features).to(self.device),ft],None,None

    def batch_dual_aggregation_feature_gene(self,batch_sim_matrix, mus, sigmas, attn_ne1, attn_ne2):
        """
        Dual Aggregation.
        [similarity matrix -> feature]
        :param batch_sim_matrix: [B,ne1,ne2]
        :param mus: [1,1,k(kernel_num)]
        :param sigmas: [1,1,k]
        :param attn_ne1: [B,ne1,1]
        :param attn_ne2: [B,ne2,1]
        :return feature: [B,kernel_num * 2].
        """
        sim_maxpooing_1, _ = batch_sim_matrix.topk(k=1, dim=-1)  # [B,ne1,1] #get max value.
        pooling_value_1 = torch.exp((- ((sim_maxpooing_1 - mus) ** 2) / (sigmas ** 2) / 2))  # [B,ne1,k]
        log_pooling_sum_1 = torch.log(torch.clamp(pooling_value_1, min=1e-10)) * attn_ne1 * 0.01  # [B,ne1,k]
        log_pooling_sum_1 = torch.sum(log_pooling_sum_1, 1)  # [B,k]

        sim_maxpooing_2, _ = torch.transpose(batch_sim_matrix, 1, 2).topk(k=1, dim=-1)  # [B,ne2,1]
        pooling_value_2 = torch.exp((- ((sim_maxpooing_2 - mus) ** 2) / (sigmas ** 2) / 2))  # [B,ne2,k]
        log_pooling_sum_2 = torch.log(torch.clamp(pooling_value_2, min=1e-10)) * attn_ne2 * 0.01  # [B,ne2,k]
        log_pooling_sum_2 = torch.sum(log_pooling_sum_2, 1)  # [B,k]

        batch_ne2_num = attn_ne2.sum(dim=1)  # [B,1]
        batch_ne2_num = torch.clamp(batch_ne2_num, min=1e-10)
        log_pooling_sum_2 = log_pooling_sum_2 * (1 / batch_ne2_num)  # [B,k]

        batch_ne1_num = attn_ne1.sum(dim=1)  # [B,1]
        batch_ne1_num = torch.clamp(batch_ne1_num, min=1e-10)
        log_pooling_sum_1 = log_pooling_sum_1 * (1 / batch_ne1_num)  # [B,k]
        return torch.cat([log_pooling_sum_1, log_pooling_sum_2], dim=-1)

    def kernel_mus(self,n_kernels):
        l_mu = [1]
        if n_kernels == 1:
            return l_mu
        bin_size = 1.0 / (n_kernels - 1)  # score range from [0, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return torch.FloatTensor(l_mu)

    def kernel_sigmas(self,n_kernels):
        l_sigma = [0.001]  # for exact match.
        # small variance -> exact match
        if n_kernels == 1:
            return l_sigma
        l_sigma += [0.1] * (n_kernels - 1)
        return torch.FloatTensor(l_sigma)
class BERT_INT_ALL(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, manifold,ent_pairs,ent2valueids,neigh_dict,value_pad_id,ent_pad_id,args,batch_size = 1024):
        super(BERT_INT_ALL, self).__init__()

        self.manifold =manifold
        self.kernel_num=args.kernel_num
        self.device=args.device
        self.ent_pad_id=ent_pad_id
        self.ent_pairs=ent_pairs
        self.ent2valueids=ent2valueids
        self.value_pad_id=value_pad_id
        self.batch_size=batch_size
        self.neigh_dict=neigh_dict
        self.reverse=args.sim_reverse

    def encode(self,input):
        x_e,x_v=input

        value_emb = F.normalize(x_v, p=2, dim=-1)
        e_emb = F.normalize(x_e, p=2, dim=-1)
        sigmas = self.kernel_sigmas(self.kernel_num).to(self.device)
        mus = self.kernel_mus(self.kernel_num).to(self.device)
        # print("sigmas:",sigmas)
        # print("mus:",mus)
        sigmas = sigmas.view(1, 1, -1)
        mus = mus.view(1, 1, -1)

        all_ent_pairs = []
        all_features = []
        all_features1=[]
        all_features2 = []
        # print("attributeValue embedding shape:", value_emb.shape)
        for start_pos in range(0, len(self.ent_pairs), self.batch_size):
            batch_ent_pairs = self.ent_pairs[start_pos: start_pos + self.batch_size]
            e1s = [e1 for e1, e2 in batch_ent_pairs]
            e2s = [e2 for e1, e2 in batch_ent_pairs]
            e1_values = [self.ent2valueids[e1] for e1 in e1s]  # size: [B(Batchsize), ne1(e1_attributeValue_max_num)]
            e2_values = [self.ent2valueids[e2] for e2 in e2s]  # [B,ne2]

            e1_masks = np.ones(np.array(e1_values).shape)
            e2_masks = np.ones(np.array(e2_values).shape)
            e1_masks[np.array(e1_values) == self.value_pad_id] = 0
            e2_masks[np.array(e2_values) == self.value_pad_id] = 0
            e1_masks = torch.FloatTensor(e1_masks.tolist()).to(self.device).unsqueeze(-1)  # [B,ne1,1]
            e2_masks = torch.FloatTensor(e2_masks.tolist()).to(self.device).unsqueeze(-1)  # [B,ne2,1]

            e1_values = torch.LongTensor(e1_values).to(self.device)  # [B,ne1]
            e2_values = torch.LongTensor(e2_values).to(self.device)   # [B,ne2]
            e1_values_emb = value_emb[e1_values]  # [B,ne1,embedding_dim]
            e2_values_emb = value_emb[e2_values]  # [B,ne2,embedding_dim]

            sim_matrix = torch.bmm(e1_values_emb, torch.transpose(e2_values_emb, 1, 2))  # [B,ne1,ne2]
            features =  self.batch_dual_aggregation_feature_gene(sim_matrix, mus, sigmas, e1_masks, e2_masks)
            features = features.detach().cpu().tolist()

            all_ent_pairs.extend(batch_ent_pairs)
            all_features.extend(features)
            e1_tails = [self.neigh_dict[int(e1)] for e1 in e1s]  # size: [B(Batchsize),ne1(e1_neighbor_max_num)]
            e2_tails = [self.neigh_dict[int(e2)] for e2 in e2s]  # [B,ne2]
            e1_masks = np.ones(np.array(e1_tails).shape)
            e2_masks = np.ones(np.array(e2_tails).shape)
            e1_masks[np.array(e1_tails) == self.ent_pad_id] = 0
            e2_masks[np.array(e2_tails) ==  self.ent_pad_id] = 0
            e1_masks = torch.FloatTensor(e1_masks.tolist()).to(self.device).unsqueeze(-1)  # [B,ne1,1]
            e2_masks = torch.FloatTensor(e2_masks.tolist()).to(self.device).unsqueeze(-1)  # [B,ne2,1]
            e1_tails = torch.LongTensor(e1_tails).to(self.device)  # [B,ne1]
            e2_tails = torch.LongTensor(e2_tails).to(self.device)  # [B,ne2]
            e1_tail_emb = e_emb[e1_tails]  # [B,ne1,embedding_dim]
            e2_tail_emb = e_emb[e2_tails]  # [B,ne2,embedding_dim]
            sim_matrix = torch.bmm(e1_tail_emb, torch.transpose(e2_tail_emb, 1, 2))  # [B,ne1,ne2]
            features = self.batch_dual_aggregation_feature_gene(sim_matrix, mus, sigmas, e1_masks, e2_masks)
            features = features.detach().cpu().tolist()
            all_features1.extend(features)

            e1s = torch.LongTensor(e1s).to(self.device)
            e2s = torch.LongTensor(e2s).to(self.device)
            e1_embs = e_emb[e1s]  # [B,embedding_dim]
            e2_embs = e_emb[e2s]
            cos_sim = F.cosine_similarity(e1_embs, e2_embs)
            cos_sim = cos_sim.detach().cpu().unsqueeze(-1).tolist()
            all_features2.extend(cos_sim)

        return [torch.FloatTensor(all_features).to(self.device),torch.FloatTensor(all_features1).to(self.device),torch.FloatTensor(all_features2).to(self.device)],None,None

    def batch_dual_aggregation_feature_gene(self,batch_sim_matrix, mus, sigmas, attn_ne1, attn_ne2):
        """
        Dual Aggregation.
        [similarity matrix -> feature]
        :param batch_sim_matrix: [B,ne1,ne2]
        :param mus: [1,1,k(kernel_num)]
        :param sigmas: [1,1,k]
        :param attn_ne1: [B,ne1,1]
        :param attn_ne2: [B,ne2,1]
        :return feature: [B,kernel_num * 2].
        """
        sim_maxpooing_1, _ = batch_sim_matrix.topk(k=1, dim=-1)  # [B,ne1,1] #get max value.
        pooling_value_1 = torch.exp((- ((sim_maxpooing_1 - mus) ** 2) / (sigmas ** 2) / 2))  # [B,ne1,k]
        log_pooling_sum_1 = torch.log(torch.clamp(pooling_value_1, min=1e-10)) * attn_ne1 * 0.01  # [B,ne1,k]
        log_pooling_sum_1 = torch.sum(log_pooling_sum_1, 1)  # [B,k]

        sim_maxpooing_2, _ = torch.transpose(batch_sim_matrix, 1, 2).topk(k=1, dim=-1)  # [B,ne2,1]
        pooling_value_2 = torch.exp((- ((sim_maxpooing_2 - mus) ** 2) / (sigmas ** 2) / 2))  # [B,ne2,k]
        log_pooling_sum_2 = torch.log(torch.clamp(pooling_value_2, min=1e-10)) * attn_ne2 * 0.01  # [B,ne2,k]
        log_pooling_sum_2 = torch.sum(log_pooling_sum_2, 1)  # [B,k]

        batch_ne2_num = attn_ne2.sum(dim=1)  # [B,1]
        batch_ne2_num = torch.clamp(batch_ne2_num, min=1e-10)
        log_pooling_sum_2 = log_pooling_sum_2 * (1 / batch_ne2_num)  # [B,k]

        batch_ne1_num = attn_ne1.sum(dim=1)  # [B,1]
        batch_ne1_num = torch.clamp(batch_ne1_num, min=1e-10)
        log_pooling_sum_1 = log_pooling_sum_1 * (1 / batch_ne1_num)  # [B,k]
        return torch.cat([log_pooling_sum_1, log_pooling_sum_2], dim=-1)

    def kernel_mus(self,n_kernels):
        l_mu = [1]
        if n_kernels == 1:
            return l_mu
        bin_size = 1.0 / (n_kernels - 1)  # score range from [0, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return torch.FloatTensor(l_mu)

    def kernel_sigmas(self,n_kernels):
        l_sigma = [0.001]  # for exact match.
        # small variance -> exact match
        if n_kernels == 1:
            return l_sigma
        l_sigma += [0.1] * (n_kernels - 1)
        return torch.FloatTensor(l_sigma)
class TransEdge(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self,manifold,args):
        super(TransEdge, self).__init__()
        self.manifold=manifold
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        self.mlp=layers.HypLinear(manifold,  args.feat_dim, args.feat_dim, self.curvatures[0], args.dropout, args.bias)
        self.hact = layers.HypAct(manifold, self.curvatures[0], self.curvatures[0], acts[0])
        self.hnorm=layers.HypNorm(manifold, self.curvatures[0], self.curvatures[0])
    def encode(self,input):
        x,r=input
        x_tan = self.manifold.proj_tan0(F.normalize(x,p=2,dim=-1), self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        x_hyp1=self.hnorm(self.hact(self.mlp(x_hyp)))

        r_tan = self.manifold.proj_tan0(F.normalize(r,p=2,dim=-1), self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])

        return [x_hyp],[x_hyp1],[r_hyp],[self.curvatures[0]]
class BootEA(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self,manifold,args):
        super(BootEA, self).__init__()
        self.manifold=manifold
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
    def encode(self,input):
        x,r=input
        x_tan = self.manifold.proj_tan0(F.normalize(x,p=2,dim=-1), self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(F.normalize(r,p=2,dim=-1), self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])
        return [x_hyp],[r_hyp],[self.curvatures[0]]
class SSP(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, e2eself_adj,manifold,args):
        super(SSP, self).__init__()

        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs=[e2eself_adj]
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gcn=layers.HyperbolicGraphConvolution(
                self.manifold, in_dim, out_dim,c_in, args.dropout, args.bias,args.use_att,args.local_agg,args.use_w
            )
            setattr(self, 'gcn{}'.format(i), gcn)
            hact=layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact{}'.format(i), hact)
            if self.multi_hop == 'hgate':
                hgate=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate{}'.format(i), hgate)

        self.layers = nn.ModuleList(gc_layers)#nn.Sequential(*hgc_layers)
        self.encode_graph = True
        self.len=len(dims) - 1


    def encode(self,input):
        x,r=input
        x_tan = self.manifold.proj_tan0(F.normalize(x,p=2,dim=-1), self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(F.normalize(r,p=2,dim=-1), self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])

        outputs=[]
        c=[]
        h=x_hyp
        if self.multi_hop=='concat':
            outputs.append(h)
            c.append(self.curvatures[0])
        for i in range(self.len):
            hact = getattr(self, 'hact{}'.format(i))
            gcn=getattr(self, 'gcn{}'.format(i))
            h2=hact(F.normalize(gcn((h,self.adjs[0])),p=2,dim=-1))
            if self.multi_hop == 'hgate':
                hgate = getattr(self, 'hgate{}'.format(i))
                h2 = hgate(h, h2)
                h = h2
                if i==self.len-1:
                    outputs.append(h)
                    c.append(self.curvatures[i + 1])
            else:
                h = h2
                outputs.append(h)
                c.append(self.curvatures[i + 1])
        r_tan = self.manifold.proj_tan0(r, c[0])
        r_hyp = self.manifold.expmap0(r_tan, c=c[0])
        r_hyp = self.manifold.proj(r_hyp, c=c[0])

        return outputs,[r_hyp]*len(outputs),c
class SSP_adapt(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, r2t_adj_self, h2t_adj_self, t2t_adj_self,manifold,args):
        super(SSP_adapt, self).__init__()

        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs = [r2t_adj_self, h2t_adj_self, t2t_adj_self]
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gcn=layers.GraphAttentionLayer(manifold, in_dim, out_dim, args.dropout, 0.2, c_in,
                                             args.bias, args.n_heads, args.heads_concat, args.use_w)
            setattr(self, 'gcn{}'.format(i), gcn)
            hact=layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact{}'.format(i), hact)
            if self.multi_hop=='hgate':
                hgate=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate{}'.format(i), hgate)

        self.layers = nn.ModuleList(gc_layers)#nn.Sequential(*hgc_layers)
        self.encode_graph = True
        self.len=len(dims) - 1


    def encode(self,input):
        x,r=input
        x_tan = self.manifold.proj_tan0(F.normalize(x,p=2,dim=-1), self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(F.normalize(r,p=2,dim=-1), self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])

        outputs=[]
        c=[]
        h=x_hyp
        if self.multi_hop=='concat':
            outputs.append(h)
            c.append(self.curvatures[0])
        for i in range(self.len):
            hact = getattr(self, 'hact{}'.format(i))
            gcn=getattr(self, 'gcn{}'.format(i))
            h2=hact(F.normalize(gcn((h,r_hyp,self.adjs[0:])),p=2,dim=-1))
            if self.multi_hop == 'hgate':
                hgate = getattr(self, 'hgate{}'.format(i))
                h2 = hgate(h, h2)
                h = h2
                if i==self.len-1:
                    outputs.append(F.normalize(h,p=2,dim=-1))
                    c.append(self.curvatures[i + 1])
            else:
                h = h2
                outputs.append(h)
                c.append(self.curvatures[i + 1])

        r_tan = self.manifold.proj_tan0(r, c[0])
        r_hyp = self.manifold.expmap0(r_tan, c=c[0])
        r_hyp = self.manifold.proj(r_hyp, c=c[0])

        return outputs,[r_hyp]*len(outputs),c
class KECG(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, r2t_adj,h2t_adj,t2t_adj,manifold,args):
        super(KECG, self).__init__()

        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs=[r2t_adj,h2t_adj,t2t_adj]
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gat=layers.GraphAttentionLayerKecg(manifold,  in_dim, out_dim,args.dropout, 0.2, c_in,
                                                                        args.bias, args.n_heads, args.heads_concat)

            setattr(self, 'gat{}'.format(i), gat)
            hact=layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact{}'.format(i), hact)

        self.layers = nn.ModuleList(gc_layers)#nn.Sequential(*hgc_layers)
        self.encode_graph = True
        self.len=len(dims) - 1


    def encode(self,input):
        x,r=input

        x_tan = self.manifold.proj_tan0(F.normalize(x,p=2,dim=-1), self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(F.normalize(r,p=2,dim=-1), self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])
        outputs=[]
        c=[]
        h=x_hyp
        for i in range(self.len):
            hact = getattr(self, 'hact{}'.format(i))
            gat=getattr(self, 'gat{}'.format(i))
            h2=gat((h,r_hyp,self.adjs[0:3]))
            h = h2 if i==self.len-1 else hact(h2)
            if i==self.len-1:
                outputs.append(F.normalize(h,p=2,dim=1))
                c.append(self.curvatures[i + 1])

        return outputs,[r_hyp],c
class KECG_literal(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, e2eself_adj,h2t_adj,t2t_adj,manifold,args):
        super(KECG_literal, self).__init__()

        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs=[e2eself_adj,h2t_adj,t2t_adj]
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gat=layers.GraphAttentionLayer(manifold,  in_dim, out_dim,args.dropout, 0.2, c_in,
                                                                        args.bias, args.n_heads, args.heads_concat,args.use_w)

            setattr(self, 'gat{}'.format(i), gat)
            hact=layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact{}'.format(i), hact)
            hgate=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
            setattr(self, 'hgate{}'.format(i), hgate)
        self.hnorm = layers.HypNorm(manifold, self.curvatures[-1], self.curvatures[-1])
        self.layers = nn.ModuleList(gc_layers)#nn.Sequential(*hgc_layers)
        self.encode_graph = True
        self.len=len(dims) - 1


    def encode(self,input):
        x,r=input
        x_tan = self.manifold.proj_tan0(F.normalize(x,p=2,dim=-1), self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(F.normalize(r,p=2,dim=-1), self.curvatures[-1])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[-1])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[-1])
        outputs=[]
        c=[]
        h=x_hyp
        for i in range(self.len):
            hact = getattr(self, 'hact{}'.format(i))
            gat=getattr(self, 'gat{}'.format(i))
            h2=hact(gat((h,r_hyp,self.adjs)))
            hgate=getattr(self, 'hgate{}'.format(i))
            h2=hgate(h,h2)
            h = h2
            if i==self.len-1:
                outputs.append(self.hnorm(h))
                c.append(self.curvatures[i + 1])

        return outputs,[r_hyp],c
class ICLEA_literal(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, rdim,r2t_adj_self,h2t_adj_self,t2t_adj_self,r2t_adj,h2t_adj,t2t_adj,e2eself_adj,e2r_adj,manifold,args):
        super(ICLEA_literal, self).__init__()

        self.manifold =manifold
        self.multi_hop=args.multi_hop
        self.adjs=[r2t_adj_self,h2t_adj_self,t2t_adj_self,r2t_adj,h2t_adj,t2t_adj,e2eself_adj,e2r_adj]
        self.inact=getattr(F, args.act)
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gat=layers.GraphAttentionLayer(manifold,  in_dim, out_dim,args.dropout, 0.2, c_in,
                                                                        args.bias, args.n_heads, args.heads_concat,args.use_w)

            setattr(self, 'gat{}'.format(i), gat)
            gat_r=layers.GraphAttentionLayer(manifold,  rdim, rdim,args.dropout, 0.2, c_in,
                                                                        args.bias, args.n_heads, args.heads_concat,args.use_w)

            setattr(self, 'gat_r{}'.format(i), gat_r)
            ragat = layers.RelationGraphAttentionLayer1(
                self.manifold, in_dim, out_dim, rdim, args.dropout, 0.2, c_in, args.bias, args.n_heads,
                args.heads_concat, args.use_w
            )
            setattr(self, 'ragat{}'.format(i), ragat)
            hact=layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact{}'.format(i), hact)
            if self.multi_hop=='hgate':
                hgate=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate{}'.format(i), hgate)
                hgate1=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate1{}'.format(i), hgate1)
                hgate2=layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate2{}'.format(i), hgate2)
        self.linear = layers.HypLinear(manifold, dims[-1]*3, dims[-1], self.curvatures[-1], args.dropout, args.bias)
        self.hact = layers.HypAct(manifold, self.curvatures[-1], self.curvatures[-1], acts[-1])

        self.encode_graph = True
        self.len=len(dims) - 1


    def encode(self,input):
        x,r=input
        #x=torch.matmul(self.adjs[3],x)
        x_r=torch.matmul(self.adjs[7],r)#F.normalize(torch.matmul(self.adjs[4],r),p=2,dim=1)
        # x=self.inact(x)
        # x_r = self.inact(x_r)

        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        x_tan_r = self.manifold.proj_tan0(x_r, self.curvatures[0])
        x_hyp_r = self.manifold.expmap0(x_tan_r, c=self.curvatures[0])
        x_hyp_r = self.manifold.proj(x_hyp_r, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(r, self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])

        outputs=[]
        c=[]
        if self.multi_hop=='concat':
            outputs.append(x_hyp)
            c.append(self.curvatures[0])
            outputs.append(x_hyp_r)
            c.append(self.curvatures[0])
            outputs.append(x_hyp)
            c.append(self.curvatures[0])
        h=x_hyp
        h01=x_hyp
        h_r=x_hyp_r
        for i in range(self.len):
            gat=getattr(self, 'gat{}'.format(i))
            h1=gat((h,r_hyp,self.adjs[0:3]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            gat_r=getattr(self, 'gat_r{}'.format(i))
            h1_r=gat_r((h_r,r_hyp,self.adjs[0:3]))
            hact=getattr(self, 'hact{}'.format(i))
            h2_r=hact(h1_r)
            ragat=getattr(self, 'ragat{}'.format(i))
            h11=ragat((h01,r_hyp,self.adjs[3:6]))
            hact=getattr(self, 'hact{}'.format(i))
            h21=hact(h11)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                h2=hgate(h,h2)
                h = h2
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
                hgate1=getattr(self, 'hgate1{}'.format(i))
                h21=hgate1(h01,h21)
                h01 = h21
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h01)
                hgate2=getattr(self, 'hgate2{}'.format(i))
                h2_r=hgate2(h_r,h2_r)
                h_r = h2_r
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h_r)
            elif self.multi_hop == 'concat':
                h = h2
                c.append(self.curvatures[i + 1])
                outputs.append(h)
                h01 = h21
                c.append(self.curvatures[i + 1])
                outputs.append(h01)
                h_r = h2_r
                c.append(self.curvatures[i + 1])
                outputs.append(h_r)
            else:
                if i == self.len - 1:
                    h = h2
                    h01 = h21
                    h_r = h2_r
                    c.append(self.curvatures[i + 1])
                    outputs.append(self.linear(torch.cat((h, h01, h_r), dim=-1)))

        return outputs, [r_hyp], c
class ICLEA(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, rdim, r2t_adj_self, h2t_adj_self, t2t_adj_self, r2t_adj, h2t_adj, t2t_adj, e2eself_adj, e2r_adj,
                 manifold, args):
        super(ICLEA, self).__init__()

        self.manifold = manifold
        self.multi_hop = args.multi_hop
        self.adjs = [r2t_adj_self, h2t_adj_self, t2t_adj_self, r2t_adj, h2t_adj, t2t_adj, e2eself_adj, e2r_adj]
        self.inact = getattr(F, args.act)
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gat = layers.GraphAttentionLayer(manifold, in_dim, out_dim, args.dropout, 0.2, c_in,
                                             args.bias, args.n_heads, args.heads_concat, args.use_w)

            setattr(self, 'gat{}'.format(i), gat)
            ragat = layers.RelationGraphAttentionLayer1(
                self.manifold, in_dim, out_dim, rdim, args.dropout, 0.2, c_in, args.bias, args.n_heads,
                args.heads_concat, args.use_w
            )
            setattr(self, 'ragat{}'.format(i), ragat)
            hact = layers.HypAct(manifold, c_in, c_out, act)
            setattr(self, 'hact{}'.format(i), hact)
            if self.multi_hop == 'hgate':
                hgate = layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate{}'.format(i), hgate)
                hgate1 = layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate1{}'.format(i), hgate1)
                hgate2 = layers.HighwayGate(self.manifold, out_dim, c_in, c_out, args.dropout, args.bias)
                setattr(self, 'hgate2{}'.format(i), hgate2)
        self.linear = layers.HypLinear(manifold, dims[-1] * 3, dims[-1], self.curvatures[-1], args.dropout, args.bias)
        #self.hact = layers.HypAct(manifold, self.curvatures[-1], self.curvatures[-1], acts[-1])
        self.rel_embeddings = nn.Embedding(r2t_adj._indices().shape[1], 300, max_norm=1)
        nn.init.kaiming_normal_(self.rel_embeddings.weight, mode='fan_out', nonlinearity='relu')
        self.encode_graph = True
        self.len = len(dims) - 1

    def encode(self,input):
        x,r=input
        x_r=F.normalize(torch.matmul(self.adjs[7],r),p=2,dim=1)

        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])

        x_tan_r = self.manifold.proj_tan0(x_r, self.curvatures[0])
        x_hyp_r = self.manifold.expmap0(x_tan_r, c=self.curvatures[0])
        x_hyp_r = self.manifold.proj(x_hyp_r, c=self.curvatures[0])

        r_tan = self.manifold.proj_tan0(r, self.curvatures[0])
        r_hyp = self.manifold.expmap0(r_tan, c=self.curvatures[0])
        r_hyp = self.manifold.proj(r_hyp, c=self.curvatures[0])

        outputs=[]
        c=[]
        if self.multi_hop=='concat':
            outputs.append(x_hyp)
            c.append(self.curvatures[0])
            outputs.append(x_hyp_r)
            c.append(self.curvatures[0])
            outputs.append(x_hyp)
            c.append(self.curvatures[0])
        h=x_hyp
        h01=x_hyp
        h_r=x_hyp_r
        for i in range(self.len):
            gat=getattr(self, 'gat{}'.format(i))
            h1=gat((h,r_hyp,self.adjs[0:3]))
            hact=getattr(self, 'hact{}'.format(i))
            h2=hact(h1)
            h1_r=gat((h_r,r_hyp,self.adjs[0:3]))
            hact=getattr(self, 'hact{}'.format(i))
            h2_r=hact(h1_r)
            ragat=getattr(self, 'ragat{}'.format(i))
            h11=ragat((h01,self.rel_embeddings.weight,self.adjs[3:6]))
            hact=getattr(self, 'hact{}'.format(i))
            h21=hact(h11)
            if self.multi_hop == 'hgate':
                hgate=getattr(self, 'hgate{}'.format(i))
                h2=hgate(h,h2)
                h = h2
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h)
                hgate1=getattr(self, 'hgate1{}'.format(i))
                h21=hgate1(h01,h21)
                h01 = h21
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h01)
                hgate2=getattr(self, 'hgate2{}'.format(i))
                h2_r=hgate2(h_r,h2_r)
                h_r = h2_r
                if i==self.len-1:
                    c.append(self.curvatures[i + 1])
                    outputs.append(h_r)
            elif self.multi_hop == 'concat':
                h = h2
                c.append(self.curvatures[i + 1])
                outputs.append(h)
                h01 = h21
                c.append(self.curvatures[i + 1])
                outputs.append(h01)
                h_r = h2_r
                c.append(self.curvatures[i + 1])
                outputs.append(h_r)
            else:
                h = F.normalize(h2,p=2,dim=-1)
                h01 = F.normalize(h21,p=2,dim=-1)
                h_r = F.normalize(h2_r,p=2,dim=-1)
                if i==self.len-1:
                    outputs.append(F.normalize(self.linear(torch.cat((h,h01,h_r),dim=-1)),p=2,dim=-1))
                    c.append(self.curvatures[i + 1])

        return outputs, [r_hyp], c

class roadEA(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, sorted_value_list,ent_init_list,value_attr_concate,attr_value, att_adj,batch_size,manifold,args):
        super(roadEA, self).__init__()
        self.sorted_value_list=sorted_value_list
        self.ent_init_list=ent_init_list
        self.value_attr_concate=value_attr_concate
        self.att_conv=nn.Conv1d(in_channels=300, out_channels=1, kernel_size=1)
        self.conv = nn.Conv1d(in_channels=100, out_channels=10, kernel_size=1)
        self.attr_value=attr_value
        self.batch_size=batch_size
        self.manifold = manifold
        self.multi_hop = args.multi_hop
        dims, acts, self.curvatures = layers.get_dim_act_curv(args)
        self.att_gcn=layers.HyperbolicGraphConvolution(
                self.manifold, 30, 30,self.curvatures[0], args.dropout, args.bias,args.use_att,args.local_agg,args.use_w
            )
        self.linear_v=layers.HypLinear(manifold,  args.feat_dim+30, args.feat_dim, self.curvatures[0], args.dropout, args.bias)
        self.adjs = att_adj
        self.temp_attr_map=nn.Parameter(torch.Tensor(300, 30))
        nn.init.xavier_uniform_(self.temp_attr_map, gain=math.sqrt(2))
        self.device=args.device


    def encode(self, input):
        x,a,v = input
        attr_embeddings=torch.tensor([])
        for start_pos in range(0, len(self.attr_value), self.batch_size):
            id_table = self.attr_value[start_pos:start_pos+self.batch_size]
            embeddings =v[id_table]
            raw_shape = embeddings.shape
            inputs = embeddings.reshape(raw_shape[0], raw_shape[1], raw_shape[2])
            conv_embeddings = self.conv(inputs)
            conv_embeddings = conv_embeddings.reshape(raw_shape[0],raw_shape[1],30)
            conv_embeddings = torch.mean(conv_embeddings, dim=1)
            attr_embeddings=torch.cat([attr_embeddings,conv_embeddings],dim=0)
        attr_embeddings=attr_embeddings+torch.matmul(a[:-1,:],self.temp_attr_map)
        a_tan = self.manifold.proj_tan0(attr_embeddings, self.curvatures[0])
        a_hyp = self.manifold.expmap0(a_tan, c=self.curvatures[0])
        a_hyp = self.manifold.proj(a_hyp, c=self.curvatures[0])
        attr_embeddings=self.att_gcn((a_hyp,self.adjs))
        concate_embeddings=attr_embeddings[self.value_attr_concate]
        concate_embeddings = torch.cat((concate_embeddings,torch.zeros(1,30).to(self.device)),dim=0)
        concate_embeddings =torch.cat((v[:len(self.value_attr_concate)+1],concate_embeddings),dim=1)
        v = F.normalize(self.linear_v(concate_embeddings),p=2,dim=1)

        value_embeddings = v[self.sorted_value_list]
        attr_len=value_embeddings.shape[1]
        ent_embeddings = v[self.ent_init_list]

        value_embeddings =value_embeddings.reshape(-1,v.shape[-1])
        tile_ent_embeddings = ent_embeddings.repeat([1, attr_len])
        repeat_ent_embeddings = tile_ent_embeddings.reshape(-1,v.shape[-1])

        add_embedding = value_embeddings+repeat_ent_embeddings
        add_embedding = add_embedding.reshape(1,-1,v.shape[-1])
        con_embedding = self.att_conv(add_embedding.permute(0,2,1)).permute(0,2,1)
        con_embedding = con_embedding.reshape(-1,attr_len)
        con = torch.softmax(con_embedding, dim=1)
        con =con.reshape(-1,attr_len,1)
        combine_embedding = con * (value_embeddings.reshape(-1,attr_len,v.shape[-1]))

        combine_embedding = torch.sum(combine_embedding, dim=1)
        ent_embeddings=F.normalize((combine_embedding+ent_embeddings),p=2,dim=1)

        return ent_embeddings