"""Executa as seis etapas efetivamente; grava notebooks e métricas resultantes."""
import sys, subprocess
from pathlib import Path
from nbclient import NotebookClient
import nbformat
ROOT=Path(__file__).resolve().parents[1]
for i in range(1,7):
    path=ROOT/'notebooks'/f'etapa{i}.ipynb'
    nb=nbformat.read(path,as_version=4)
    print(f'Executando etapa {i} ...',flush=True)
    NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT/'notebooks')}}).execute()
    nbformat.write(nb,path)
    assert all(not any(o.output_type=='error' for o in cell.get('outputs',[])) for cell in nb.cells if cell.cell_type=='code')
print('Seis notebooks efetivamente executados sem erros.',flush=True)
if '--no-verify' not in sys.argv:
    subprocess.run([sys.executable,str(ROOT/'scripts'/'verify_project.py')],check=True,cwd=ROOT)
