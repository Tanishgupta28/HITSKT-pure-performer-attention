"""Independent small exhaustive oracles for diagnostic counts, not model training."""
from itertools import product
import numpy as np
import pytest
from scripts.diagnose_hitskt_metrics import confusion_metrics,infer_confusion,disagreement_bounds
from scripts.diagnose_hitskt_timing import sweep


def confusion(y,p):
    return dict(tp=int(np.sum((y==1)&p)),fp=int(np.sum((y==0)&p)),
                tn=int(np.sum((y==0)&~p)),fn=int(np.sum((y==1)&~p)))


def test_confusion_bounds_attained_exhaustively():
    y=np.array([0,0,0,1,1,1]);groups={}
    for bits in product([False,True],repeat=6):
        p=np.array(bits);c=confusion(y,p);key=tuple(c.values())
        groups.setdefault(key,[]).append(p)
    for first,aa in groups.items():
        for second,bb in groups.items():
            counts=[int(np.count_nonzero(a!=b)) for a in aa for b in bb]
            bounds=disagreement_bounds(dict(zip(['tp','fp','tn','fn'],first)),dict(zip(['tp','fp','tn','fn'],second)))
            assert min(counts)==bounds['minimum'] and max(counts)==bounds['maximum']
            assert all(n%2==bounds['parity'] for n in counts)


@pytest.mark.parametrize('c',[dict(tp=23689,fp=16694,tn=52016,fn=19853),dict(tp=173,fp=89,tn=218,fn=121)])
def test_integer_confusion_reconstruction(c):
    assert infer_confusion(dict(targets=sum(c.values()),**confusion_metrics(c)))==c


def brute(starts,ends,bundles,offsets,students,splits,history):
    out=np.zeros((3,9),np.int64)
    for a,b in zip(students[:-1],students[1:]):
        for s in range(a+1,b):
            lo=offsets[max(a,s-history)];ss=offsets[s]
            for i in range(ss,offsets[s+1]):
                prior=range(lo,i);over=[j for j in prior if ends[j]>starts[i]]
                ties=[j for j in prior if starts[j]==starts[i]]
                same=[j for j in prior if bundles[i]>=0 and bundles[j]==bundles[i]]
                out[splits[s]]+=np.array([1,i>lo,len(over)>0,len(over),any(j<ss for j in over),len(ties)>0,len(ties),len(same)>0,len(same)])
    return out


@pytest.mark.parametrize('seed',range(6))
def test_timing_context_sweep_against_enumerated_history(seed):
    rng=np.random.default_rng(seed);lengths=rng.integers(1,6,size=43)
    offsets=np.r_[0,np.cumsum(lengths)].astype('int64');students=np.array([0,22,43],dtype='int64')
    starts=np.empty(offsets[-1],dtype='int64')
    for a,b in zip(students[:-1],students[1:]):starts[offsets[a]:offsets[b]]=np.cumsum(rng.integers(0,5,size=offsets[b]-offsets[a]))
    ends=starts+rng.integers(0,20,size=len(starts));bundles=rng.integers(-1,5,size=len(starts));splits=np.arange(43,dtype='int64')%3
    assert np.array_equal(sweep(starts,ends,bundles,offsets,students,splits,15),brute(starts,ends,bundles,offsets,students,splits,15))


def test_older_response_overlap_missed_by_adjacent_check():
    starts=np.array([0,1,2,3]);ends=np.array([100,1,2,3]);bundles=np.full(4,-1,dtype='int64')
    offsets=np.array([0,1,4]);students=np.array([0,2]);splits=np.array([0,2])
    out=sweep(starts,ends,bundles,offsets,students,splits)
    assert out[2,2]==3 and out[2,3]==3 and out[2,4]==3
