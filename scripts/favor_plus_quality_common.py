"""Independent bounded numerical oracles; never used by production attention."""
from __future__ import annotations

import numpy as np
import torch
from scipy.special import logsumexp


def describe(values) -> dict:
    a = np.asarray(values,dtype=np.float64).reshape(-1)
    return dict(count=int(a.size),minimum=float(a.min()),median=float(np.median(a)),
                p90=float(np.quantile(a,.90)),p95=float(np.quantile(a,.95)),
                maximum=float(a.max()),mean=float(a.mean()))


def independent_attention(q,k,v,omega,qm,km):
    """NumPy/SciPy L² oracle, independent of all production feature helpers.

    Directly build log(phi(q) dot phi(k)) in bounded query tiles, then normalize
    over eligible keys. Also return exact softmax and descriptive feature ESS.
    """
    q,k,v,omega = [np.asarray(t,dtype=np.float64) for t in (q,k,v,omega)]
    d,m = q.shape[-1],omega.shape[1]
    x,y = q/d**.25,k/d**.25
    a = np.einsum('bhld,hmd->bhlm',x,omega)-np.sum(x*x,-1,keepdims=True)/2-np.log(m)/2
    b = np.einsum('bhld,hmd->bhlm',y,omega)-np.sum(y*y,-1,keepdims=True)/2-np.log(m)/2
    n = q.shape[2]
    log_kernel = np.empty((*q.shape[:3],n),dtype=np.float64)
    ess = []
    allowed = np.tri(n,dtype=bool)[None,None,:,:] & np.asarray(km)[:,None,None,:]
    allowed_q = allowed & np.asarray(qm)[:,None,:,None]
    for start in range(0,n,8):
        z = a[:,:,start:start+8,None,:]+b[:,:,None,:,:]
        lk = logsumexp(z,axis=-1)
        log_kernel[:,:,start:start+8] = lk
        effective = np.exp(2*lk-logsumexp(2*z,axis=-1))
        eligible = np.broadcast_to(allowed_q[:,:,start:start+8],lk.shape)
        ess.extend(effective[eligible].tolist())
    exact_scores = np.einsum('bhid,bhjd->bhij',q,k)/d**.5
    def normalize(score):
        valid = np.broadcast_to(allowed,score.shape)
        has = valid.any(-1,keepdims=True)
        masked = np.where(valid,score,-np.inf)
        masked = np.where(has,masked,0.)
        logw = masked-logsumexp(masked,axis=-1,keepdims=True)
        return np.where(valid,np.exp(logw),0.),logw
    approx_weights,approx_logw = normalize(log_kernel)
    exact_weights,exact_logw = normalize(exact_scores)
    query_valid = np.asarray(qm)[:,None,:,None]
    feature_output = np.where(query_valid,approx_weights@v,0.)
    exact_output = np.where(query_valid,exact_weights@v,0.)
    eligible = np.broadcast_to(allowed_q,log_kernel.shape)
    norm_sum_squared = np.maximum((np.sum(q*q,-1)[...,None]+np.sum(k*k,-1)[:,:,None,:]
                                  +2*np.einsum('bhid,bhjd->bhij',q,k))/d**.5,0.)
    s = norm_sum_squared[eligible]
    # iid PRF coefficient of variation squared; an explanatory reference,
    # not a variance formula for the correlated ORF rows.
    log_cv2 = np.where(s>50,s,np.log(np.expm1(np.minimum(s,50))+np.finfo(float).tiny))-np.log(m)
    row_valid = np.broadcast_to(np.asarray(qm)[:,None,:],log_kernel.shape[:3])
    kl_terms = np.where(eligible,exact_weights*(np.where(eligible,exact_logw,0.)-
                                             np.where(eligible,approx_logw,0.)),0.)
    diagnostic = dict(feature_ess=describe(ess),
        absolute_log_kernel_error=describe(np.abs((log_kernel-exact_scores)[eligible])),
        signed_log_kernel_error=describe((log_kernel-exact_scores)[eligible]),
        iid_log10_cv_squared=describe(log_cv2/np.log(10)),
        mean_attention_total_variation=float((np.abs(approx_weights-exact_weights).sum(-1)/2)[row_valid].mean()),
        mean_exact_to_feature_kl=float(kl_terms.sum(-1)[row_valid].mean()))
    return feature_output,exact_output,diagnostic


def output_errors(actual,exact,mask) -> dict:
    actual,exact = np.asarray(actual),np.asarray(exact)
    eligible = np.broadcast_to(np.asarray(mask)[:,None,:,None],actual.shape)
    delta = (actual-exact)[eligible]
    target = exact[eligible]
    return dict(mse=float(np.mean(delta**2)),rmse=float(np.sqrt(np.mean(delta**2))),
                max_abs=float(np.max(np.abs(delta))),
                relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(target),1e-30)))


def torch_feature_reference(q,k,v,omega,qm,km):
    """Autograd oracle using explicit pairwise log features, no prefix states."""
    d,m = q.shape[-1],omega.shape[1]
    x,y = q/d**.25,k/d**.25
    phi_q = torch.einsum('bhid,hmd->bhim',x,omega)-x.square().sum(-1,keepdim=True)/2
    phi_k = torch.einsum('bhjd,hmd->bhjm',y,omega)-y.square().sum(-1,keepdim=True)/2
    score = torch.logsumexp(phi_q[:,:,:,None,:]+phi_k[:,:,None,:,:],-1)-np.log(m)
    n = q.shape[2]
    valid = torch.ones(n,n,device=q.device,dtype=torch.bool).tril()[None,None] & km[:,None,None,:]
    has = valid.any(-1,keepdim=True)
    masked = torch.where(valid,score,-torch.inf)
    masked = torch.where(has,masked,0.)
    weights = masked.softmax(-1)*valid
    return (weights@v)*qm[:,None,:,None]
