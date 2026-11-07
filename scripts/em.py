"""Bernoulli, Naive Bayes categórico e mistura hurdle EM para volumes defasados."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

def bernoulli_mle(x):
    x=np.asarray(x,float);return float(x.mean())
def bernoulli_map(x,a=2,b=2):
    """Moda posterior Beta(s+a, n-s+b); não a média posterior."""
    x=np.asarray(x,float);s=x.sum();n=len(x)
    if s+a<=1 or n-s+b<=1:
        return float(np.clip((s+a-1)/max(n+a+b-2,1),0,1))
    return float((s+a-1)/(n+a+b-2))

def bernoulli_posterior_mean(x,a=2,b=2):
    x=np.asarray(x,float);return float((x.sum()+a)/(len(x)+a+b))

class CategoricalNB:
    def __init__(self,alpha=1):self.alpha=alpha
    def fit(self,X,y):
        self.classes=np.unique(y);self.features=list(X)
        self.priors={};self.cond={};self.K={};self.N={}
        for col in self.features:self.K[col]=len(set(X[col].astype(str)))+1 # <UNK>
        y=np.asarray(y)
        for c in self.classes:
            sub=X[y==c];self.N[c]=len(sub)
            self.priors[c]=(len(sub)+self.alpha)/(len(X)+len(self.classes)*self.alpha)
            for col in self.features:
                counts=sub[col].astype(str).value_counts()
                den=len(sub)+self.alpha*self.K[col]
                self.cond[c,col]={v:(n+self.alpha)/den for v,n in counts.items()}
                self.cond[c,col]['<UNK>']=self.alpha/den
        return self
    def predict_proba(self,X):
        res=[]
        for _,row in X.iterrows():
            logs=[]
            for c in self.classes:
                v=np.log(self.priors[c])
                for col in self.features:
                    probs=self.cond[c,col]
                    v+=np.log(probs.get(str(row[col]),probs['<UNK>']))
                logs.append(v)
            logs=np.array(logs,float);res.append(np.exp(logs-logsumexp(logs)))
        return np.asarray(res)
    def predict(self,X):return self.classes[self.predict_proba(X).argmax(axis=1)]

def hurdle_em(lags,K=2,variance_floor=.05,max_iter=300,tol=1e-5):
    """Zeros explícitos e mistura gaussiana em log1p(lag), condicional a x>0.
    A curva corresponde à log-verossimilhança do subconjunto positivo.
    """
    x=np.asarray(lags,float);zero=x==0;positive=np.log1p(x[~zero]);n=len(positive)
    if n<K*10:raise ValueError('Amostra positiva insuficiente')
    means=np.quantile(positive,np.linspace(.25,.75,K))
    variances=np.full(K,max(np.var(positive),variance_floor))
    weights=np.full(K,1/K);hist=[]
    for _ in range(max_iter):
        log_joint=np.column_stack([np.log(weights[k])+norm.logpdf(positive,means[k],np.sqrt(variances[k])) for k in range(K)])
        r=np.exp(log_joint-logsumexp(log_joint,axis=1)[:,None]);nk=np.maximum(r.sum(axis=0),1e-10)
        weights=nk/nk.sum();means=(r*positive[:,None]).sum(axis=0)/nk
        variances=np.maximum((r*(positive[:,None]-means)**2).sum(axis=0)/nk,variance_floor)
        updated=np.column_stack([np.log(weights[k])+norm.logpdf(positive,means[k],np.sqrt(variances[k])) for k in range(K)])
        ll=float(logsumexp(updated,axis=1).sum());hist.append(ll)
        if len(hist)>1 and abs(hist[-1]-hist[-2])<tol:break
    order=np.argsort(means)
    if np.any(np.diff(hist)<-1e-6):raise RuntimeError('EM: log-verossimilhança decresceu de forma relevante')
    return {'zero_fraction':float(zero.mean()),'means':means[order],'variances':variances[order],
            'weights':weights[order],'ll':hist,'variance_floor':variance_floor}

def hurdle_posterior(lags,model):
    x=np.asarray(lags,float);out=np.zeros((len(x),3));zero=(x==0);out[zero,0]=1
    if (~zero).any():
        t=np.log1p(x[~zero]);lp=np.column_stack([np.log(model['weights'][k])+norm.logpdf(t,model['means'][k],np.sqrt(model['variances'][k])) for k in range(2)])
        out[~zero,1:]=np.exp(lp-logsumexp(lp,axis=1)[:,None])
    if not np.allclose(out.sum(axis=1),1):raise AssertionError('posterior não normalizada')
    return out
