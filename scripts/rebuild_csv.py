"""Extrai o XLSX enviado para um CSV uniforme sem alterar valores.

Usado apenas quando a equipe deseja reconstituir vde_2026_extraido.csv.
"""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
x=ROOT/'data'/'BancoVDE 2026.xlsx'
if not x.exists():raise FileNotFoundError(str(x))
d=pd.read_excel(x,sheet_name='2026')
rename={
 'UF':'uf','Município':'municipio','Municipio':'municipio',
 'Evento':'evento','Data de Referência':'data_referencia',
 'Agente':'agente','Arma':'arma','Faixa Etária':'faixa_etaria',
 'Feminino':'feminino','Masculino':'masculino','Não Informado':'nao_informado',
 'Total de Vítimas':'total_vitima','Total':'total','Total Peso':'total_peso',
 'Abrangência':'abrangencia'}
d=d.rename(columns=rename)
print('Cabeçalhos lidos:',d.columns.to_list())
needed={'uf','municipio','evento','data_referencia','total_vitima','total','abrangencia'}
if not needed.issubset(d):
 raise ValueError('Adapte mapeamento aos cabeçalhos do XLSX; faltam: '+repr(needed-set(d)))
if pd.api.types.is_datetime64_any_dtype(d['data_referencia']):
 d['data_referencia']=(d['data_referencia']-pd.Timestamp('1899-12-30')).dt.days
out=ROOT/'data'/'vde_2026_extraido.csv';d.to_csv(out,index=False)
print('CSV reconstruído:',out,'linhas:',len(d))
