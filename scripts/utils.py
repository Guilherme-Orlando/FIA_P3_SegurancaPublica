"""Preparação transparente do painel SINESP VDE 2026 (UF × indicador × mês).

Nunca se usa a variável-alvo de uma amostra de teste para definir limiares.
O CSV é extração do XLSX anexado; rebuild_csv.py documenta o procedimento.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
SEED=42
EVENTOS={
 'Feminicídio','Homicídio doloso','Lesão corporal seguida de morte',
 'Roubo seguido de morte (latrocínio)','Tentativa de feminicídio',
 'Tentativa de homicídio','Estupro','Estupro de vulnerável',
 'Furto de veículo','Roubo de veículo','Roubo de carga',
 'Roubo a instituição financeira'
}
CATEGORICAS=['uf','evento','semestre2','lag_alto','alta_previa']

def read_source():
    src=DATA/'vde_2026_extraido.csv'
    if not src.exists():
        raise FileNotFoundError('CSV ausente. Execute python scripts/rebuild_csv.py')
    df=pd.read_csv(src,low_memory=False,dtype={'uf':str,'municipio':str,'evento':str,'abrangencia':str})
    expected={'uf','municipio','evento','data_referencia','total_vitima','total','abrangencia'}
    if not expected.issubset(df):raise ValueError('Colunas não reconhecidas: '+str(sorted(expected-set(df))))
    return df

def panel(rebuild=False):
    cache=DATA/'painel_2026.csv'
    src=DATA/'vde_2026_extraido.csv'
    popfile=DATA/'ibge_pop_uf_2026.csv'
    if not rebuild and cache.exists() and cache.stat().st_mtime >= max(src.stat().st_mtime,popfile.stat().st_mtime):
        p=pd.read_csv(cache,parse_dates=['data'],dtype={'uf':str,'evento':str})
        if 'taxa_por100mil' in p.columns:return p
    df=read_source()
    df=df[(df['abrangencia']=='Estadual') & df.evento.isin(EVENTOS)].copy()
    # Não somar total_vitima e total: representam unidades distintas.
    vitima=df.total_vitima.notna()
    df['contagem']=pd.to_numeric(df.total_vitima.where(vitima,df.total),errors='coerce')
    df['unidade']=np.where(vitima,'vitimas','ocorrencias')
    df['data']=pd.to_datetime(pd.to_numeric(df.data_referencia,errors='coerce'),unit='D',origin='1899-12-30',errors='coerce')
    df=df[df.contagem.ge(0)&df.data.notna()&df.uf.notna()].copy()
    if (df.data.dt.year!=2026).any():raise ValueError('Existem datas fora de 2026. Verificar fonte.')
    # Filtra duplicações de escopo; existem linhas municipais e linhas estaduais, mas
    # não se deve computar outros órgãos federais para o mesmo indicador.
    p=(df.groupby(['uf','evento','unidade','data'],as_index=False)
       .agg(contagem=('contagem','sum'),municipios=('municipio','nunique'))
       .sort_values(['uf','evento','data']).reset_index(drop=True))
    if p.duplicated(['uf','evento','data']).any():raise ValueError('Indicador com mistura de unidades.')
    p['mes']=p.data.dt.month
    g=p.groupby(['uf','evento'],sort=False)
    p['lag1']=g.contagem.shift(1)
    p['media_previa']=g.contagem.transform(lambda s:s.shift(1).expanding().mean())
    p['delta_previo']=g.contagem.shift(1)-g.contagem.shift(2)
    prev_date=g.data.shift(1)
    # Garantia: lag1 é sempre do mês imediatamente anterior (não do último mês disponível).
    p=p[(p.data.dt.to_period('M')-prev_date.dt.to_period('M')).apply(lambda z: getattr(z,'n',None)==1)].copy()
    # delta_previo não existe na primeira observação após lag1; codifica-se 0, explicitamente.
    p['delta_previo']=p['delta_previo'].fillna(0)
    p['alta_previa']=(p['delta_previo']>0).astype(int)
    p['semestre2']=(p.mes>=7).astype(int)
    pop=pd.read_csv(DATA/'ibge_pop_uf_2026.csv',usecols=['uf','populacao_2026'])
    assert len(pop)==27 and pop.uf.is_unique and (pop.populacao_2026>0).all()
    p=p.merge(pop,on='uf',how='left',validate='many_to_one')
    if p.populacao_2026.isna().any():raise ValueError('UF sem denominador IBGE 2026')
    # Por indicador e mês: NÃO somar taxas de unidades distintas.
    p['taxa_por100mil']=p.contagem/p.populacao_2026*100000.
    if len(p)<200: raise ValueError('Painel anormalmente pequeno; revisar fonte.')
    p.to_csv(cache,index=False)
    return p

def indices_random(p):
    from sklearn.model_selection import train_test_split
    return tuple(train_test_split(np.arange(len(p)),test_size=.2,random_state=SEED))

def prepare_splits(p,tr,te):
    """Fita parâmetros da rotulagem SOMENTE em tr e aplica sem reajuste em te."""
    train=p.iloc[np.asarray(tr)].copy()
    test=p.iloc[np.asarray(te)].copy()
    cut=train.groupby('evento').contagem.quantile(.75)
    mlag=train.groupby('evento').lag1.median()
    def make(x):
        x=x.copy()
        x['alto']=(x.contagem>x.evento.map(cut)).astype(int)
        x['lag_alto']=(x.lag1>x.evento.map(mlag)).astype(int)
        return x
    return make(train),make(test),{'p75_train_por_evento':cut.to_dict(), 'mediana_lag_train_por_evento':mlag.to_dict()}

def training_split():
    p=panel()
    tr,te=indices_random(p)
    a,b,meta=prepare_splits(p,tr,te)
    return p,a,b,meta

def temporal_split(p):
    tr=np.where(p.mes<=7)[0]
    te=np.where(p.mes>=8)[0]
    return prepare_splits(p,tr,te)

def class_features(d):return d[CATEGORICAS].astype(str).copy()

def ent(y):
    n=pd.Series(y).value_counts(normalize=True).to_numpy(dtype=float)
    return float(-(n*np.log2(n)).sum()) if len(n) else 0.

def gain(d,feature,target='alto'):
    return ent(d[target])-sum((len(g)/len(d))*ent(g[target]) for _,g in d.groupby(feature))

class ID3:
    def __init__(self,max_depth=4,min_leaf=8):self.max_depth=max_depth;self.min_leaf=min_leaf
    def fit(self,X,y):
        d=X.copy();d['_alvo']=np.asarray(y,dtype=int)
        def build(sub,avail,level):
            mode=int(sub._alvo.mode().iloc[0])
            if level>=self.max_depth or len(sub)<self.min_leaf or sub._alvo.nunique()==1 or not avail:return mode
            best=max(avail,key=lambda a:gain(sub,a,'_alvo'))
            if gain(sub,best,'_alvo')<1e-10:return mode
            return {'attribute':best,'default':mode,'branches':{str(k):build(s,[a for a in avail if a!=best],level+1) for k,s in sub.groupby(best)}}
        self.tree=build(d,list(X),0);return self
    def predict(self,X):
        def walk(row,t):
            if not isinstance(t,dict):return t
            return walk(row,t['branches'].get(str(row[t['attribute']]),t['default']))
        return np.asarray([walk(row,self.tree) for _,row in X.iterrows()],dtype=int)

def logistic_gd(X,y,lr=.07,steps=2500,l2=.0):
    z=np.c_[np.ones(len(X)),X];w=np.zeros(z.shape[1]);y=np.asarray(y,dtype=float)
    for _ in range(steps):
        h=1/(1+np.exp(-np.clip(z@w,-35,35)))
        g=z.T@(h-y)/len(y);g[1:]+=l2*w[1:];w-=lr*g
    return w

def linear_gd(X,y,lr=.04,steps=3500):
    z=np.c_[np.ones(len(X)),X];w=np.zeros(z.shape[1]);y=np.asarray(y,float);history=[]
    for i in range(steps):
        err=z@w-y;w-=lr*(z.T@err/len(y))
        if i%10==0:history.append(float(np.mean(err**2)))
    return w,history

def regression_features(d):
    signed=lambda v:np.sign(v)*np.log1p(np.abs(v))
    return np.column_stack([np.log1p(d.lag1.to_numpy(float)),np.log1p(d.media_previa.to_numpy(float)),signed(d.delta_previo.to_numpy(float))])

def write_metrics(name,obj):
    OUT.mkdir(exist_ok=True)
    def conv(v):
        if isinstance(v,(np.integer,)):return int(v)
        if isinstance(v,(np.floating,)):return float(v)
        if isinstance(v,np.ndarray):return v.tolist()
        if hasattr(v,'isoformat'):return v.isoformat()
        raise TypeError(type(v).__name__)
    path=OUT/name
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=conv),encoding='utf-8')
    return path
