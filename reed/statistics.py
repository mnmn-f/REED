import numpy as np
EPS=1e-12
def correlation_redundancy(x):
    x = x - x.mean(0)
    norms = np.linalg.norm(x, axis=0)
    x = x[:, norms > EPS] / norms[norms > EPS]
    d = x.shape[1]
    if d < 2:
        return 0.
    # Gram identity avoids a d*d matrix when B << d.
    return float(max(0., ((x @ x.T)**2).sum() - d) / (d*(d-1)))
def rank_transform(raw, reference, two_sided=False, extrapolate=False):
    result=np.empty_like(raw,dtype=float)
    for m in range(2):
        for j in range(raw.shape[-1]):
            r=np.sort(reference[:,m,j]); n=len(r)
            left=np.searchsorted(r,raw[:,m,j],side='left')
            right=np.searchsorted(r,raw[:,m,j],side='right')
            result[:,m,j]=(left+right+1)/(2*(n+1))
            if extrapolate:
                # Preserve empirical ranks in range, continue tails smoothly.
                # This is a score calibration heuristic, not a tail p guarantee.
                scale=max(float(np.std(r,ddof=1)),1e-10)
                high=raw[:,m,j]>r[-1];low=raw[:,m,j]<r[0]
                result[high,m,j]=1-(.5/(n+1))*np.exp(-np.minimum((raw[high,m,j]-r[-1])/scale,700))
                result[low,m,j]=(.5/(n+1))*np.exp(-np.minimum((r[0]-raw[low,m,j])/scale,700))
    return np.abs(2*result-1) if two_sided else result

def fuse(x, method, normal):
    if method != "max":
        raise ValueError("Expected max fusion for the released calibration")
    return np.clip(x, 1e-6, 1-1e-6).max(axis=-1)
