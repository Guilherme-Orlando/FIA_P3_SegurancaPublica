"""Rede Bayesiana multinomial discreta de seis nós: CPT e enumeração próprias."""
from itertools import product
import numpy as np

PARENTS={
 'uf':[],
 'semestre2':[],
 'lag_alto':['uf'],
 'regime_em':['uf'],
 'alto':['uf','lag_alto','regime_em'],
 'crescimento':['semestre2','alto']
}
NODES=list(PARENTS)

def fit_cpt(train,alpha=1.0):
    states={n:sorted(train[n].unique().tolist()) for n in NODES}
    combos={n:list(product(*(states[p] for p in parents))) for n,parents in PARENTS.items()}
    cpt={}
    for n,parents in PARENTS.items():
        cpt[n]={}
        for parvals in combos[n]:
            s=train
            for p,v in zip(parents,parvals):s=s[s[p]==v]
            counts=s[n].value_counts()
            den=len(s)+alpha*len(states[n])
            cpt[n][parvals]={v:(counts.get(v,0)+alpha)/den for v in states[n]}
            if abs(sum(cpt[n][parvals].values())-1)>1e-12:raise AssertionError('CPT não normalizada')
    return {'states':states,'tables':cpt}

def joint(assignment,model):
    q=1.
    for n,pars in PARENTS.items():
        q*=model['tables'][n][tuple(assignment[p] for p in pars)][assignment[n]]
    return q

def query(target,state,evidence,model):
    if target in evidence:
        return float(evidence[target]==state)
    free=[n for n in NODES if n not in evidence]
    # Enumera configurações consistentes com a evidência (limite 6 nós).
    num=den=0.
    for values in product(*(model['states'][n] for n in free)):
        a=dict(evidence,**dict(zip(free,values)))
        p=joint(a,model);den+=p
        if a[target]==state:num+=p
    return float(num/den) if den else float('nan')

def param_counts(model):
    from math import prod
    total=prod(len(v) for v in model['states'].values())-1
    bn=sum((len(model['states'][n])-1)*prod(len(model['states'][p]) for p in pars) for n,pars in PARENTS.items())
    return int(total),int(bn)

def to_pgmpy(model):
    """Converte as tabelas na mesma ordenação usada na enumeração.
    Executar se pgmpy>=1 estiver instalado (o módulo não é necessário ao modelo próprio).
    """
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    edges=[(p,n) for n,pars in PARENTS.items() for p in pars]
    graph=DiscreteBayesianNetwork(edges)
    cpds=[]
    for n,pars in PARENTS.items():
        ps=list(product(*(model['states'][p] for p in pars)))
        vals=[[model['tables'][n][cfg][v] for cfg in ps] for v in model['states'][n]]
        cpd=TabularCPD(n,len(model['states'][n]),vals,
                       evidence=pars if pars else None,
                       evidence_card=[len(model['states'][p]) for p in pars] if pars else None,
                       state_names={q:model['states'][q] for q in [n]+pars})
        cpds.append(cpd)
    graph.add_cpds(*cpds)
    assert graph.check_model()
    return graph,VariableElimination(graph)
