"""Verificação do pacote: dados, modelos, resultados das etapas e execução opcional real dos notebooks."""
import sys, json, importlib.util, argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--run-notebooks',action='store_true',help='Reexecuta os seis notebooks com nbclient antes da auditoria.')
args=parser.parse_args()
if args.run_notebooks:
    from nbclient import NotebookClient
    import nbformat
    for i in range(1,7):
        path=ROOT/'notebooks'/f'etapa{i}.ipynb'
        nb=nbformat.read(path,as_version=4)
        print(f'EXECUTANDO {path.name}', flush=True)
        NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT/'notebooks')}}).execute()
        nbformat.write(nb,path)
        if any(o.get('output_type')=='error' for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[])):
            raise RuntimeError(f'{path.name} possui uma célula com erro')
    print('REEXECUÇÃO REAL DOS SEIS NOTEBOOKS: OK',flush=True)
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import pandas as pd
from utils import panel,indices_random,prepare_splits,ent,gain,EVENTOS
from em import bernoulli_mle,bernoulli_map,bernoulli_posterior_mean,hurdle_em,hurdle_posterior
from bayes import fit_cpt,query,param_counts
from fuzzy import tri,mamdani
import nbformat

def check(condition,msg):
    if not condition:raise AssertionError(msg)
    print('OK',msg)

P=panel()
pop=pd.read_csv(ROOT/'data'/'ibge_pop_uf_2026.csv')
check(len(pop)==27 and pop.populacao_2026.sum()==214211951,'IBGE 2026: 27 UFs e total de 214.211.951 habitantes')
check(len(P)==2590 and P.uf.nunique()==27 and P.evento.nunique()==12,'Painel: 2.590 linhas, 27 UFs, 12 eventos')
check((ROOT/'SHA256SUMS.txt').exists(),'Manifesto SHA256 das fontes incluído')
check(P['taxa_por100mil'].notna().all() and (P.taxa_por100mil>=0).all(),'Taxas por 100 mil calculadas sem valores ausentes')
check(P.mes.ge(2).all() and P.mes.le(9).all(),'Séries defasadas usam fevereiro a setembro')
tr,te=indices_random(P)
a,b,limits=prepare_splits(P,tr,te)
p_mod=P.copy();p_mod.loc[te,'contagem']=999999.0
a2,b2,limits2=prepare_splits(p_mod,tr,te)
check(limits['p75_train_por_evento']==limits2['p75_train_por_evento'],'Limiar P75 não depende dos valores do teste')
check(np.array_equal(a.alto.to_numpy(),a2.alto.to_numpy()),'Rótulos de treino imunes a alterações artificiais do teste')
check(ent([0,0,1,1])==1 and np.isfinite(gain(a,'uf')),'Entropia e ganho são calculados corretamente')
check(abs(bernoulli_mle([0,0,1])-(1/3))<1e-12,'MLE de Bernoulli calculado corretamente')
check(abs(bernoulli_map([0,0,1])-(2/5))<1e-12,'MAP é a MODA posterior, não a média')
check(abs(bernoulli_posterior_mean([0,0,1])-(3/7))<1e-12,'Média posterior distinta do MAP')
res=hurdle_em(a.lag1,variance_floor=.05)
post=hurdle_posterior(a.lag1,res)
check(np.allclose(post.sum(axis=1),1) and np.all(res['variances']>=.05),'EM: responsabilidades normalizadas e variâncias válidas')
check(np.all(np.diff(res['ll'])>=-1e-6),'EM: log-verossimilhança não decrescente')
assert np.isfinite(mamdani(.1,.2))
check(abs(float(tri(0,0,0,.5))-1)<1e-12 and abs(float(tri(1,.5,1,1))-1)<1e-12,'Pertinência triangular correta nos ombros')
check(mamdani(.1,.1)<mamdani(.9,.9) and mamdani(.9,.1)!=mamdani(.9,.9),'Fuzzy depende do risco e da tendência')
trbn=a.copy();trbn['regime_em']=np.argmax(hurdle_posterior(a.lag1,res),axis=1).astype(int)
trbn['crescimento']=(trbn.contagem>trbn.lag1).astype(int)
model=fit_cpt(trbn);j,k=param_counts(model)
check((j,k)==(1295,274),'BN: economia de parâmetros consistente')
check(abs(query('semestre2',1,{},model)-query('semestre2',1,{'alto':1},model))<1e-9,'BN: d-separação sem observar colisor')
if not (ROOT/'outputs'/'etapa1.json').exists():
    sys.exit('Resultados das etapas ausentes: execute primeiro "python scripts/run_all.py".')
for i in range(1,7):
    f=ROOT/'notebooks'/f'etapa{i}.ipynb'; n=nbformat.read(f,as_version=4)
    codes=[c for c in n.cells if c.cell_type=='code']
    check(len(codes)>2 and all(c.get('execution_count') is not None for c in codes) and all(not any(o.output_type=='error' for o in c.get('outputs',[])) for c in codes),f'Notebook etapa{i}: resultados salvos, sem erro nas células')
    out=ROOT/'outputs'/f'etapa{i}.json';check(out.exists() and len(json.loads(out.read_text()))>0,f'Métricas etapa{i} salvas')
# Verificação explícita de integração simbólica da Etapa 1 com a Etapa 6
integration=pd.read_csv(ROOT/'outputs'/'integracao_teste.csv')
check(len(integration)==518 and {'id3_pred','alto_real','p_alto','prioridade'}.issubset(integration.columns),
      'Etapa 6 reutiliza a decisão ID3 e as probabilidades Bayes/fuzzy')
metrics6=json.loads((ROOT/'outputs'/'etapa6.json').read_text())
check(int((integration.id3_pred!=integration.hard).sum())==metrics6['id3_bn_divergences'],
      'Discordâncias entre decisão ID3 e Bayes armazenadas corretamente')
from pypdf import PdfReader
pdf=ROOT/'artigo'/'relatorio_p3.pdf'
pages=len(PdfReader(str(pdf)).pages)
reader=PdfReader(str(pdf))
body=next((i for i,pg in enumerate(reader.pages) if any(l.strip()=='Referências' for l in (pg.extract_text() or '').splitlines())),pages)
check(6<=body<=8,f'Corpo do artigo dentro do limite: {body} páginas (além de referências e apêndices; total {pages})')
print('TODAS AS VERIFICAÇÕES AUTOMÁTICAS LOCAIS PASSARAM.')
if not args.run_notebooks: print('Para EXECUTAR de fato os notebooks: python scripts/verify_project.py --run-notebooks')
print('pgmpy instalado?',bool(importlib.util.find_spec('pgmpy')),'| scikit-fuzzy instalado?',bool(importlib.util.find_spec('skfuzzy')))

# As comparações com pgmpy e scikit-fuzzy são executadas nos notebooks 5 e 6.
m5=json.loads((ROOT/'outputs'/'etapa5.json').read_text())
m6=json.loads((ROOT/'outputs'/'etapa6.json').read_text())
check(m5['pgmpy_status']=='executado_e_verificado_localmente' and m5['pgmpy_queries']==10 and m5['pgmpy_max_abs_diff']<1e-8,'Enumeração própria x pgmpy: 10 consultas executadas')
check(m6['skfuzzy_status']=='executado_e_verificado_localmente' and m6['skfuzzy_cases']==25 and m6['skfuzzy_max_abs_diff']<=.3,'Mamdani próprio x scikit-fuzzy: 25 casos executados')
