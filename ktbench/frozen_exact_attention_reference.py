"""CPU inference-only exact causal reference; not a trainable model variant."""
import torch
from torch import nn


def exact_causal_reference(q,k,v,qm,km):
    if any(x.device.type!='cpu' for x in (q,k,v,qm,km)):
        raise ValueError('frozen CPU reference only')
    q=torch.where(qm[:,None,:,None],q,0.).double()
    k=torch.where(km[:,None,:,None],k,0.).double()
    v=torch.where(km[:,None,:,None],v,0.).double()
    n=q.shape[2];allowed=torch.ones(n,n,dtype=torch.bool).tril()[None,None]&km[:,None,None,:]
    score=(q@k.transpose(-1,-2))/q.shape[-1]**.5;has=allowed.any(-1,keepdim=True)
    masked=torch.where(allowed,score,-torch.inf);masked=torch.where(has,masked,0.)
    weights=masked.softmax(-1)*allowed
    return (weights@v)*qm[:,None,:,None]


class FrozenExactAttention(nn.Module):
    def __init__(self,original):
        super().__init__();self.original=original

    def forward(self,query,key,value,*,query_mask,key_mask,causal=True):
        if self.training or torch.is_grad_enabled() or not causal:
            raise RuntimeError('eval/no-grad causal sensitivity reference only')
        m=self.original
        q,k,v=[m._heads(layer(torch.where(mask[...,None],x,0.))) for layer,x,mask in
               ((m.query,query,query_mask),(m.key,key,key_mask),(m.value,value,key_mask))]
        out=exact_causal_reference(q,k,v,query_mask,key_mask).to(query.dtype)
        out=out.transpose(1,2).contiguous().reshape(query.shape)
        return m.dropout(m.output(out))*query_mask[...,None]
