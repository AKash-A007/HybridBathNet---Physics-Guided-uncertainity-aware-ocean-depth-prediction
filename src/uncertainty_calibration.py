"""Post-hoc GEBCO uncertainty calibration; never modifies depth predictions.

Scaling follows regression variance calibration (Levi et al.); empirical signed
residual quantiles adapt CRUDE with equal geographic-group weighting. Hierarchical
conformal uses Lee/Barber/Willett equation6, including the test-group infinity
atom. Hierarchical exchangeability is an assumption, not ensured by a grid.
"""
import numpy as np
from scipy.special import ndtr


def group_weights(groups):
    groups=np.asarray(groups)
    _,inverse,counts=np.unique(groups,return_inverse=True,return_counts=True)
    return 1.0/(len(counts)*counts[inverse])


def weighted_quantile(values,weights,probability):
    values=np.asarray(values,dtype='float64');weights=np.asarray(weights,dtype='float64')
    if not 0<=probability<=1 or np.any(weights<0) or weights.sum()<=0:
        raise ValueError('Invalid quantile probability/weights')
    order=np.argsort(values,kind='stable');cdf=np.cumsum(weights[order])/weights.sum()
    i=min(int(np.searchsorted(cdf,probability,side='left')),len(values)-1)
    return float(values[order[i]])


class ResidualCalibrator:
    def __init__(self,method):
        if method not in ('raw','gaussian_scale','empirical_group_crude','hierarchical_conformal','group_max_conformal'):
            raise ValueError(method)
        self.method=method

    def fit(self,target,mean,sigma,groups):
        target,mean,sigma=(np.asarray(a,dtype='float64') for a in (target,mean,sigma))
        groups=np.asarray(groups)
        if not (target.shape==mean.shape==sigma.shape==groups.shape):raise ValueError('Shape mismatch')
        if not np.isfinite([target,mean,sigma]).all() or np.any(sigma<=0):raise ValueError('Nonfinite data or nonpositive sigma')
        self.z=(target-mean)/sigma;self.weights=group_weights(groups)
        self.n_groups=len(np.unique(groups))
        self.scale=float(np.sqrt(np.sum(self.weights*self.z**2)))
        self.scale=max(self.scale,1e-6)
        self.maxima=np.array([abs(self.z[groups==g]).max() for g in np.unique(groups)])
        order=np.argsort(self.z);z=self.z[order];w=self.weights[order]
        self.empirical_half_pair_distance=float(np.sum(w*z*(2*np.cumsum(w)-w-1)))
        return self

    def interval(self,mean,sigma,coverage=.9):
        from scipy.special import ndtri
        if not 0<coverage<1:raise ValueError('Coverage must be between0 and1')
        mean=np.asarray(mean);sigma=np.asarray(sigma);alpha=1-coverage
        if not np.isfinite(mean).all() or not np.isfinite(sigma).all() or np.any(sigma<=0):
            raise ValueError('Inference requires finite means and strictly positive finite sigma')
        if self.method in ('raw','gaussian_scale'):
            scale=self.scale if self.method=='gaussian_scale' else 1.
            radius=float(ndtri((1+coverage)/2))*sigma*scale
            return mean-radius,mean+radius
        if self.method=='empirical_group_crude':
            low=weighted_quantile(self.z,self.weights,alpha/2)
            high=weighted_quantile(self.z,self.weights,1-alpha/2)
            return mean+low*sigma,mean+high*sigma
        if self.method=='hierarchical_conformal':
            weights=np.r_[self.weights*self.n_groups/(self.n_groups+1),1/(self.n_groups+1)]
            values=np.r_[abs(self.z),np.inf]
        else:
            weights=np.ones(self.n_groups+1)/(self.n_groups+1)
            values=np.r_[self.maxima,np.inf]
        radius=weighted_quantile(values,weights,coverage)*sigma
        return mean-radius,mean+radius

    def crps(self,target,mean,sigma):
        """Proper distribution score; no finite density assigned to conformal sets."""
        target,mean,sigma=(np.asarray(a,dtype='float64') for a in (target,mean,sigma))
        if self.method=='empirical_group_crude':
            residual=(target-mean)/sigma
            return sigma*(np.sum(self.weights[:,None]*abs(self.z[:,None]-residual[None]),axis=0)-self.empirical_half_pair_distance)
        if self.method not in ('raw','gaussian_scale'):return None
        sd=sigma*(self.scale if self.method=='gaussian_scale' else 1.)
        z=(target-mean)/sd;phi=np.exp(-.5*z*z)/np.sqrt(2*np.pi)
        return sd*(z*(2*ndtr(z)-1)+2*phi-1/np.sqrt(np.pi))

    def state(self):
        return {'method':self.method,'n_groups':self.n_groups,'gaussian_scale':self.scale,
            'signed_normalized_residuals':self.z.tolist(),'equal_group_point_weights':self.weights.tolist(),
            'group_maxima':self.maxima.tolist(),
            'interpretation':'Calibrated total uncertainty only. Raw head/ensemble decomposition remains unchanged.'}

    @classmethod
    def from_state(cls,state):
        """Inference-only reload; no labels, fitting, or validation access."""
        self=cls(state['method'])
        self.z=np.asarray(state['signed_normalized_residuals'],dtype='float64')
        self.weights=np.asarray(state['equal_group_point_weights'],dtype='float64')
        self.n_groups=int(state['n_groups']);self.scale=float(state['gaussian_scale'])
        self.maxima=np.asarray(state['group_maxima'],dtype='float64')
        if self.z.shape!=self.weights.shape or not np.isfinite(self.z).all() or np.any(self.weights<=0):
            raise ValueError('Invalid saved residual calibration')
        if not np.isclose(self.weights.sum(),1) or self.scale<=0 or not np.isfinite(self.scale):
            raise ValueError('Invalid saved calibration scale/weights')
        order=np.argsort(self.z);z=self.z[order];w=self.weights[order]
        self.empirical_half_pair_distance=float(np.sum(w*z*(2*np.cumsum(w)-w-1)))
        return self


def interval_score(target,lower,upper,coverage=.9):
    target,lower,upper=map(np.asarray,(target,lower,upper));alpha=1-coverage
    return upper-lower+(2/alpha)*np.maximum(lower-target,0)+(2/alpha)*np.maximum(target-upper,0)
