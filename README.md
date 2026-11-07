# Do Simbólico à Incerteza — P3 Segurança Pública (2026)

**Fundamentos de Inteligência Artificial** • IComp/UFAM • Prof. Edjard Mota • **Equipe 5**
Fernando Naum Santos Batista · Gabriel Lucas Araújo Matos · Guilherme Paiva Orlando · Luis Felipe Ferreira Cavalcante · Yuri Garcia e Silva

Pipeline de seis etapas (árvore de decisão → regressão → probabilidade → Naive Bayes/EM → rede Bayesiana → fuzzy) sobre a base SINESP VDE 2026 (MJSP) com denominadores populacionais do IBGE. A unidade de análise é **UF × indicador × mês**; não há dado individual nem inferência sobre pessoas.

---

## 1. Estrutura

```text
FIA_P3_SegurancaPublica_Equipe5/
├── README.md
├── requirements.txt          # dependências com versões limitadas
├── SHA256SUMS.txt            # checksums das fontes e dos dados derivados
├── data/                     # BancoVDE 2026.xlsx, CSVs derivados, população IBGE
├── notebooks/                # etapa1.ipynb … etapa6.ipynb (já executados)
├── scripts/                      # código reutilizável
│   ├── utils.py              # painel, split sem vazamento, ID3, entropia/ganho, GD linear e logístico
│   ├── em.py                 # MLE, MAP, Naive Bayes categórico, EM hurdle
│   ├── bayes.py              # CPTs, enumeração, contagem de parâmetros, conversão p/ pgmpy
│   ├── fuzzy.py              # Mamdani próprio e versão scikit-fuzzy
│   ├── rebuild_csv.py        # reconstrói o CSV a partir do XLSX
│   ├── run_all.py            # executa os 6 notebooks e audita
│   └── verify_project.py     # autotestes
└── artigo/                   # relatorio_p3.tex, relatorio_p3.pdf e figuras PNG
```

## 2. Dados

| Fonte | Arquivo | Observação |
|---|---|---|
| MJSP / SINESP VDE 2026 | `data/BancoVDE 2026.xlsx` | 618.206 linhas, 31 eventos, jan–set/2026. Portal: <https://www.gov.br/mj/pt-br/assuntos/sua-seguranca/seguranca-publica/estatistica> (acesso em 03/11/2026). |
| IBGE | `data/ibge_pop_uf_2026.csv` | 27 estimativas, referência 01/07/2026 (total 214.211.951). <https://ftp.ibge.gov.br/Estimativas_de_Populacao/Estimativas_2026/POP2026_20261029.pdf> (acesso em 03/11/2026). |
| Derivados | `data/vde_2026_extraido.csv`, `data/painel_2026.csv` | extração do XLSX e painel de modelagem (cache regenerável). |

**Preparação (`scripts/utils.py::panel`)**: somente abrangência *Estadual* e 12 indicadores criminais; soma por `uf × evento × mês`. `total_vitima` e `total` têm unidades diferentes e **nunca são somados**. Janeiro só fornece `lag1`; a base de modelagem tem **2.590 linhas** (fev–set, 27 UFs, 12 eventos).

**Alvo (`alto`)**: 1 se a contagem supera o P75 do indicador **calculado apenas no treino**. A mediana usada em `lag_alto` também sai só do treino. Divisão aleatória 80/20 com `seed=42` (2.072 treino / 518 teste); há também um corte temporal (fev–jul treino, ago–set teste).

## 3. Instalação

Requer Python 3.11+. Na raiz do projeto:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`pgmpy` e `scikit-fuzzy` são obrigatórios: os notebooks 5 e 6 comparam as implementações próprias com essas bibliotecas e falham com mensagem clara se elas faltarem.

## 4. Como executar

```bash
python scripts/run_all.py                 # executa as 6 etapas na ordem e roda os autotestes
python scripts/verify_project.py          # apenas autotestes (requer run_all.py executado antes)
sha256sum -c SHA256SUMS.txt           # Linux/macOS/WSL: integridade das fontes
```

Uso interativo: `jupyter lab` e execute `notebooks/etapa1.ipynb` … `etapa6.ipynb` **em ordem**, como no enunciado (cada etapa grava resultados que os autotestes consultam depois). Os notebooks localizam a raiz tanto a partir da raiz do projeto quanto de `notebooks/`.

Para reconstruir o CSV a partir do XLSX (opcional, mais lento): `python scripts/rebuild_csv.py`. Confira o resultado contra `SHA256SUMS.txt`.

Tempo típico de `run_all.py`: alguns minutos (a Etapa 6 faz inferência por enumeração em 518 linhas de teste).

### 4.1 Google Colab

O projeto também roda no Colab. Suba o zip para `/content`, descompacte e instale as dependências numa célula **antes** de abrir qualquer notebook:

```python
!unzip -q FIA_P3_SegurancaPublica_Equipe5.zip -d /content
%cd /content/FIA_P3_SegurancaPublica_Equipe5
!pip install -q -r requirements.txt
```

Se o `pip` pedir, use *Runtime → Restart session* e rode apenas a célula `%cd` de novo. Depois execute os notebooks em ordem. Sem o `pip install`, as etapas 5 e 6 param com `ImportError` (pgmpy e scikit-fuzzy não vêm no Colab).

## 5. Como conferir cada etapa

Cada item diz **o que o notebook implementa**, **o que verificar** e **o valor de referência** obtido com `seed=42`.

### Etapa 1 — Árvore de decisão (`etapa1.ipynb`, `scripts/utils.py`)
- Implementa entropia, ganho e `ID3` do zero (multiway, profundidade 4, mínimo de 8 por folha); compara com `DecisionTreeClassifier(criterion='entropy')`.
- **Verificar:** o ganho do atributo-raiz é calculado no treino; confira que `gain = H(pai) − Σ p_v·H(filho_v)` (`ent` e `gain` em `utils.py`).
- **Referência:** ID3 teste **93,44%**; scikit-learn teste **89,38%**; 5-fold (scikit-learn) **86,37%**; baseline por UF **88,03%**; raiz = `uf` (ganho 0,3658).
- **Leitura correta:** o alvo em contagens reflete porte do estado (SP/RJ ≈ 90% “alto”; várias UFs ≈ 0%). A versão com taxa por 100 mil dá 83,59%.

### Etapa 2 — Regressão linear e logística (`etapa2.ipynb`)
- Fórmula fechada univariada, GD multivariado, Ridge com λ escolhido por CV **interna ao treino**, logística própria (sigmoide + gradiente).
- **Verificar:** alvo `log1p(contagem)` e atributos `log1p(lag1)`, `log1p(media_previa)`, `sign(Δ)·log1p(|Δ|)` — alvo e atributos na mesma escala; padronização ajustada no treino.
- **Referência:** R² teste univariada **0,9578**, GD **0,9661**, scikit-learn **0,9659**, Ridge **0,9659** (λ=0,117); logística própria **84,94%** vs scikit-learn **86,10%**.
- **Leitura correta:** R² alto mede persistência da ordem de grandeza de cada série UF–evento.

### Etapa 3 — Probabilidade (`etapa3.ipynb`)
- Conjunta 2×2 de A (`alto`) e B (aumento no mês anterior), marginais, condicionais, regra da cadeia, Bayes e independência condicional de B e C dado A.
- **Verificar:** a conjunta soma 1; P(A|B) por contagem direta é igual ao de Bayes.
- **Referência:** P(A)=0,2244; P(B)=0,3296; P(A∩B)=0,0883; P(A|B)=0,2679.

### Etapa 4 — MLE, MAP, Naive Bayes e EM (`etapa4.ipynb`, `scripts/em.py`)
- MLE e MAP (Beta(2,2): **moda** posterior, distinta da média posterior); Naive Bayes categórico com Laplace; EM **hurdle** — zeros de `lag1` tratados como massa discreta e mistura de 2 gaussianas em `log1p(lag1)` para os positivos, com variância mínima 0,05.
- **Verificar:** `res['ll']` é não decrescente; variâncias finais acima do piso (se uma variância colar no piso, a gaussiana está colapsando num ponto).
- **Referência:** MLE 0,2244; MAP 0,2247; NB **87,07%**; NB+EM **87,64%** (11 ganhos × 8 perdas, McNemar exato p=0,648 → diferença **não** significativa); 132 iterações; variâncias 0,141 e 2,143.

### Etapa 5 — Rede Bayesiana (`etapa5.ipynb`, `scripts/bayes.py`)
- DAG de 6 nós: `uf → lag_alto`, `uf → regime_em`, `{uf, lag_alto, regime_em} → alto`, `{semestre2, alto} → crescimento`. CPTs com Laplace estimadas só no treino; enumeração própria.
- **Verificar:** (i) cada CPT normaliza (assert em `fit_cpt`); (ii) enumeração própria = `pgmpy.VariableElimination` em 10 consultas (o notebook faz `assert` com tolerância 1e-8); (iii) d-separação: `semestre2 ⟂ alto` sem observar `crescimento`; ao observar o colisor `crescimento` a probabilidade muda.
- **Referência:** 1.295 parâmetros na conjunta × 274 na rede (economia 1.021); P(S=1)=0,3698 = P(S=1|alto=1); com `crescimento` observado: 0,3036 (c=0) e 0,4217 (c=1).

### Etapa 6 — Mamdani e integração (`etapa6.ipynb`, `scripts/fuzzy.py`)
- Entradas: P(alto) e P(crescimento) vindas da rede da Etapa 5; 5 regras; centroide; comparação com limiar rígido 0,5 e com a decisão do ID3 da Etapa 1.
- **Verificar:** Mamdani próprio × `scikit-fuzzy` em 25 pontos (o notebook faz `assert` com tolerância de 0,3 na escala 0–100; a diferença observada é muito menor).
- **Referência:** 169 pares de entrada distintos; 12 divergências hard × fuzzy em 518; 39 divergências ID3 × rede; grade 11×11 com 34/121 divergências.

## 6. Artigo (LaTeX)

`artigo/relatorio_p3.tex` (formato SBC; corpo de 8 páginas, seguido de referências e apêndices; figuras PNG na própria pasta `artigo/`). Para recompilar:

```bash
cd artigo
pdflatex -interaction=nonstopmode -halt-on-error relatorio_p3.tex
pdflatex -interaction=nonstopmode -halt-on-error relatorio_p3.tex
```

Pacotes LaTeX usados: `babel` (português), `newtx`, `microtype`, `titlesec`, `tikz`, `booktabs`, `caption`, `hyperref`. Em Ubuntu/Debian: `texlive-latex-extra texlive-lang-portuguese texlive-fonts-extra texlive-plain-generic`; em MiKTeX/TeX Live completo já estão incluídos.

## 7. Solução de problemas

| Sintoma | Causa e correção |
|---|---|
| `ModuleNotFoundError: utils` ou `Raiz do projeto não encontrada` | O Jupyter foi aberto fora da pasta do projeto. Os notebooks procuram `scripts/utils.py` no diretório atual, nos pais e até duas pastas abaixo; abra o Jupyter dentro de `FIA_P3_SegurancaPublica_Equipe5/` (ou, no Colab, siga a seção 4.1). |
| `ImportError: pgmpy / scikit-fuzzy não encontrado` | `python -m pip install -r requirements.txt` no ambiente ativo. |
| `AttributeError: numpy has no attribute ...` | NumPy fora da faixa de `requirements.txt` (`>=2.0,<3`). |
| `CSV ausente` | `python scripts/rebuild_csv.py` (gera `data/vde_2026_extraido.csv` a partir do XLSX). |
| Painel com número de linhas diferente de 2.590 | `data/painel_2026.csv` é cache; apague-o e rode de novo para regenerar. |
| `babel Error: Unknown option 'brazil'` ou `binhex.tex not found` | Faltam pacotes TeX (seção 6). |
