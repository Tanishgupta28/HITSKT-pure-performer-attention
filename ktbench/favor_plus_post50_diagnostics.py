"""CPU-only, frozen-state research diagnostics; no optimizer or CUDA entry point."""
from __future__ import annotations
import math
import os
from collections import Counter
import numpy as np
import torch


def require_cpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1') or torch.cuda.is_initialized():
        raise RuntimeError('diagnostics require CUDA hidden and uninitialized')


def describe(x):
    a=np.asarray(x,dtype=np.float64).reshape(-1)
    if not len(a) or not np.isfinite(a).all():raise ValueError('finite nonempty observations required')
    return dict(minimum=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),
        p95=float(np.percentile(a,95)),maximum=float(a.max()),count=len(a))


def attention_weights(q,k,projection,qm,km):
    """Independent explicit FP64 feature/exact causal matrices [H,L,L].

    Per-feature key gauges are compensated in queries before a row gauge.
    They cancel algebraically; the causal mask is applied before normalization.
    This offline oracle does not call the production feature map or prefix scan.
    """
    q,k,w=[np.asarray(a,dtype=np.float64) for a in (q,k,projection)]
    qm,km=np.asarray(qm,bool),np.asarray(km,bool)
    if q.ndim!=3 or k.shape!=q.shape or w.ndim!=3 or w.shape[0]!=q.shape[0] or w.shape[-1]!=q.shape[-1]:raise ValueError('Q/K/projection dimensions')
    h,n,d=q.shape
    if qm.shape!=(n,) or km.shape!=(n,):raise ValueError('aligned boolean masks required')
    q=np.where(qm[None,:,None],q,0.);k=np.where(km[None,:,None],k,0.)
    if not all(np.isfinite(a).all() for a in (q,k,w)):raise FloatingPointError('nonfinite active inputs')
    valid=np.tri(n,dtype=bool)&km[None,:]&qm[:,None]
    feature=np.zeros((h,n,n));exact=np.zeros_like(feature);pair_ess=[];iid_logvar=[]
    if not km.any():return feature,exact,dict(pair_feature_ess=None,iid_log10_relative_variance_proxy=None)
    for head in range(h):
        x,y=q[head]/d**.25,k[head]/d**.25
        lq=x@w[head].T-.5*np.sum(x*x,axis=-1,keepdims=True)-.5*np.log(w.shape[1])
        lk=y@w[head].T-.5*np.sum(y*y,axis=-1,keepdims=True)-.5*np.log(w.shape[1])
        gauge=lk[km].max(axis=0);query=lq+gauge;query-=query.max(axis=-1,keepdims=True)
        kernel=np.exp(query)@np.exp(np.where(km[:,None],lk-gauge,-np.inf)).T
        kernel=np.where(valid,kernel,0.);total=kernel.sum(axis=-1,keepdims=True)
        if np.any((total[:,0]<=0)&valid.any(-1)):raise FloatingPointError('offline gauge underflow; no silent floor')
        feature[head]=np.divide(kernel,total,out=np.zeros_like(kernel),where=total>0)
        score=q[head]@k[head].T/np.sqrt(d);score=np.where(valid,score,-np.inf)
        maximum=np.max(score,axis=-1,keepdims=True);maximum=np.where(np.isfinite(maximum),maximum,0.)
        weight=np.where(valid,np.exp(score-maximum),0.);total=weight.sum(axis=-1,keepdims=True)
        exact[head]=np.divide(weight,total,out=np.zeros_like(weight),where=total>0)
        # Fixed sparse pair grid, not every O(L²*m) feature contribution tensor.
        for i in np.unique(np.linspace(0,n-1,min(n,8)).astype(int)):
            js=np.flatnonzero(valid[i])
            for j in js[np.unique(np.linspace(0,len(js)-1,min(len(js),12)).astype(int))] if len(js) else []:
                terms=lq[i]+lk[j];mass=np.exp(terms-terms.max());mass/=mass.sum()
                pair_ess.append(float(1/(mass@mass)))
                t=float(np.sum((x[i]+y[j])**2))
                # IID Gaussian PRF relative variance, NOT an ORF bound.
                iid_logvar.append(float((t+np.log(-np.expm1(-t)) if t>0 else -690.)/np.log(10)-np.log10(w.shape[1])))
    return feature,exact,dict(pair_feature_ess=describe(pair_ess) if pair_ess else None,
        iid_log10_relative_variance_proxy=describe(iid_logvar) if iid_logvar else None)


def direct_log_feature_weights(q,k,w,qm,km):
    """Small independent pairwise logsumexp oracle for the compensated gauge."""
    h,n,d=q.shape;out=np.zeros((h,n,n))
    for head in range(h):
        for i in range(n):
            if not qm[i]:continue
            js=np.flatnonzero(km[:i+1]);logs=[]
            for j in js:
                x=q[head,i]/d**.25;y=k[head,j]/d**.25
                logs.append(np.logaddexp.reduce(w[head]@(x+y)-(x@x+y@y)/2)-np.log(w.shape[1]))
            if logs:
                values=np.exp(np.asarray(logs)-max(logs));out[head,i,js]=values/values.sum()
    return out


def attention_diagnostics(q,k,v,w,qm,km,actual=None):
    q,k,v,w=[np.asarray(a,dtype=np.float64) for a in (q,k,v,w)]
    v=np.where(np.asarray(km)[None,:,None],v,0.)
    if not np.isfinite(v).all():raise FloatingPointError('nonfinite active values')
    feature,exact,extra=attention_weights(q,k,w,qm,km)
    af=feature@v;ae=exact@v;eligible=np.broadcast_to(np.asarray(qm)[None,:,None],ae.shape)
    error=(af-ae)[eligible];target=ae[eligible]
    rows=np.broadcast_to(np.asarray(qm)[None,:],ae.shape[:2])& (exact.sum(-1)>0)
    tv=.5*np.abs(feature-exact).sum(-1)
    entropy=-np.sum(np.where(exact>0,exact*np.log(np.maximum(exact,1e-300)),0.),axis=-1)
    rowerror=np.linalg.norm(af-ae,axis=-1)/np.maximum(np.linalg.norm(ae,axis=-1),1e-30)
    value_rms=float(np.sqrt(np.mean(v[:,np.asarray(km)]**2)))
    result=dict(relative_output_l2=float(np.linalg.norm(error)/max(np.linalg.norm(target),1e-30)),
        output_rmse=float(np.sqrt(np.mean(error**2))),error_over_value_rms=float(np.sqrt(np.mean(error**2))/max(value_rms,1e-30)),
        exact_output_rms=float(np.sqrt(np.mean(target**2))),value_rms=value_rms,
        attention_total_variation=describe(tv[rows]),exact_attention_max_weight=describe(exact.max(-1)[rows]),
        exact_attention_effective_keys=describe(np.exp(entropy[rows])),head_query_relative_l2=describe(rowerror[rows]),
        query_norm=describe(np.linalg.norm(q,axis=-1)[:,np.asarray(qm)]),key_norm=describe(np.linalg.norm(k,axis=-1)[:,np.asarray(km)]),
        **extra)
    if actual is not None:
        delta=(np.asarray(actual,dtype=np.float64)-af)[eligible]
        result['production_vs_feature_max_abs']=float(np.max(np.abs(delta)))
        result['feature_kernel_agreement']='PASS' if result['production_vs_feature_max_abs']<=5e-5 else 'FAIL'
    return result


def binary_summary(probability,label,threshold=.5):
    p=np.asarray(probability,dtype=np.float64);y=np.asarray(label,dtype=np.int64)
    if p.ndim!=1 or p.shape!=y.shape or not len(y) or not np.isfinite(p).all() or np.any((p<=0)|(p>=1)) or not set(np.unique(y)).issubset({0,1}):raise ValueError('finite aligned probabilities strictly inside(0,1) and binary labels')
    pred=p>=threshold;positive=y==1;tp=int(np.sum(pred&positive));fp=int(np.sum(pred&~positive));fn=int(np.sum(~pred&positive));tn=int(np.sum(~pred&~positive))
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
    order=np.argsort(p,kind='stable');rank=np.empty(len(y));i=0
    while i<len(y):
        j=i+1
        while j<len(y) and p[order[j]]==p[order[i]]:j+=1
        rank[order[i:j]]=(i+1+j)/2;i=j
    npos=int(positive.sum());nneg=len(y)-npos
    losses=-(y*np.log(p)+(1-y)*np.log1p(-p))
    return dict(targets=len(y),positive=npos,negative=nneg,threshold=threshold,tp=tp,fp=fp,tn=tn,fn=fn,
        accuracy=(tp+tn)/len(y),precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
        roc_auc=float((rank[positive].sum()-npos*(npos+1)/2)/(npos*nneg)) if npos and nneg else None,
        rounded_probability_bce=float(losses.mean()),brier=float(np.mean((p-y)**2)),predicted_positive_rate=float(pred.mean()),
        mean_probability=float(p.mean()),mean_positive_probability=float(p[positive].mean()) if npos else None,
        mean_negative_probability=float(p[~positive].mean()) if nneg else None,
        positive_bce=float(losses[positive].mean()) if npos else None,negative_bce=float(losses[~positive].mean()) if nneg else None,
        within_1e6_of_half=int(np.sum(np.abs(p-.5)<=1e-6)),within_1e3_of_half=int(np.sum(np.abs(p-.5)<=1e-3)))


def shape_memory(shapes,parameter_bytes,buffer_bytes,*,width=256,heads=4,ff=2048,m=128,chunk=32):
    """Component accounting, NOT a GPU peak forecast. No tensor allocation."""
    b=len(shapes);hist=sum(len(s.history_lengths) for s in shapes)
    lh=max(max(s.history_lengths)+1 for s in shapes);lt=max(s.target_length+1 for s in shapes);ls=max(len(s.history_lengths)+1 for s in shapes)
    stages=[('action',hist,lh),('session',b,ls),('correct',b,lt),('decoder',b,lt)]
    out={};d=width//heads
    for name,rows,length in stages:
        boundaries=math.ceil(length/chunk)
        out[name]=dict(rows=rows,length=length,padded_tokens=rows*length,
            saved_scan_boundaries=rows*heads*boundaries*m*(d+2)*4+rows*boundaries,
            saved_scan_log_features_and_v=rows*heads*length*(2*m+d)*4+rows*length,
            two_ff_hidden_activations=2*rows*length*ff*4,
            single_weighted_state_per_recomputed_token=rows*heads*min(length,chunk)*m*d*4)
    persistent=4*parameter_bytes+buffer_bytes  # weights+grads+Adam m/v, FP32
    retained=sum(x['saved_scan_boundaries']+x['saved_scan_log_features_and_v']+x['two_ff_hidden_activations'] for x in out.values())
    return dict(batch_size=b,history_rows=hist,max_history_actions_including_eos=lh,max_target_actions_including_eos=lt,
        token_cost=hist*lh+b*lt+b*ls,persistent_weights_grad_adam_buffers_bytes=persistent,
        retained_component_accounting_bytes=retained,component_sum_bytes=persistent+retained,
        max_one_weighted_state_per_recomputed_token_bytes=max(x['single_weighted_state_per_recomputed_token'] for x in out.values()),
        stages=out,limitations='Component accounting excludes other saved tensors, recomputation graph multiplicity, allocator/workspaces/driver. Not a complete GPU peak; workspace component not a measured peak.')


class SavedStorageTracker:
    """Measure unique live autograd-saved storage, excluding parameter buffers.

    Detached wrappers preserve saved values and share storage. They do not retain
    the forward grad_fn and are released with autograd's saved tensors.
    """
    class Packed:
        def __init__(self,tensor,tracker,key):self.value=tensor.detach();self.tracker=tracker;self.key=key
        def __del__(self):self.tracker.release(self.key)
    def __init__(self,excluded=()):
        self.excluded=set(excluded);self.refs=Counter();self.sizes={};self.live=0;self.peak=0;self.phase='forward';self.phase_peak={};self.pack_calls=0
    @staticmethod
    def key(t):return (t.device.type,t.untyped_storage().data_ptr(),t.untyped_storage().nbytes())
    def pack(self,tensor):
        key=self.key(tensor);self.pack_calls+=1
        if key not in self.excluded:
            if self.refs[key]==0:self.sizes[key]=key[-1];self.live+=key[-1]
            self.refs[key]+=1;self.peak=max(self.peak,self.live);self.phase_peak[self.phase]=max(self.phase_peak.get(self.phase,0),self.live)
        return self.Packed(tensor,self,key)
    def release(self,key):
        if key in self.excluded:return
        self.refs[key]-=1
        if not self.refs[key]:self.live-=self.sizes.pop(key);del self.refs[key]
    @staticmethod
    def unpack(packed):return packed.value
