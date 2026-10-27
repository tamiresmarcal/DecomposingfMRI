# Piloto: brain states × mapas de referência

Como ler um brain state em termos neuroanatômicos: a ativação média do estado,
no córtex e no subcórtex, e ao lado a correlação desse padrão com quatro
famílias de mapas. Há uma figura para cada atlas do pipeline.

![harvardoxford](state_maps_results/simulated/harvardoxford/states_figure.png)

| família | o que é | fonte | status |
|---|---|---|---|
| **Molecular** | 19 mapas PET de receptores/transportadores (5-HT, DA, NE, ACh, GABA, Glu, CB1, MOR, H3) | conjunto de Hansen et al. 2022, que é o que o `neuromaps` serve em MNI152; baixado do repositório do artigo | **real** |
| **Comportamental** | perfil de cada sub-região do Brainnetome na taxonomia do BrainMap, em todas as subcategorias (ex.: Perception → Vision → Motion), agrupadas por domínio | atlas Brainnetome real (246 regiões, **com subcórtex**); **perfis simulados** | **atlas real, perfis SIMULADOS** |
| **Canônico: Yeo 7** | redes de Yeo 2011 | volume liberal MNI152, o mesmo que o `yeo7` do repo usa | **real** |
| **Canônico: nossas 14 redes** | o atlas `networks` deste repo (254 esferas de 5 mm) | `fmri_decomposition.get_atlas("networks")` | **real** |
| **Estados** | médias por estado no atlas em que foram ajustados | simulados, ou `fmri-decomp state-means --csv` | **simulados** neste piloto |

**Pelo notebook:** `notebooks/08_state_maps.ipynb`. Os parâmetros ficam numa
célula só: a definição de cluster (`STATES`, `WINDOW_S`, `COHORTS`), os atlas
mostrados (`ATLASES`), as famílias (`FAMILIES`), a profundidade da taxonomia
comportamental (`BN_LEVELS`: `"domain"`, `"subcategory"` ou os dois) e os
nulos (`N_NULLS`, `NULL`).

O notebook chama dois scripts, que também rodam pela linha de comando:

```bash
pip install -e ".[atlases]" matplotlib               # + brainsmash para --null burt
notebooks/fetch_reference_maps.sh             # uma vez, ~285 MB em notebooks/state_maps_data/

# 1. tabelas -> state_maps_results/<estados>/<atlas>/{state_means,correlations}.csv + provenance.json
python notebooks/state_correlations.py --states simulate
python notebooks/state_correlations.py --states HMM2_pca14_10 --window-s -1 \
    --atlases harvardoxford --output-root /caminho/outputs

# 2. figuras, a partir das tabelas (nenhum número é recalculado aqui)
python notebooks/state_figures.py notebooks/state_maps_results/simulated \
    --atlases harvardoxford --bn-levels domain subcategory
```

`--states` aceita `simulate`, uma coluna de estados (lida por
`fmri_decomposition.state_means`, a mesma tabela que o `fmri-decomp state-means`
imprime) ou um CSV do `state-means --csv`.

Saídas em `state_maps_results/<estados>/<atlas>/`:

- `state_maps_overview.png`: uma linha por estado. Mostra o córtex (lateral e medial, esquerda e depois direita), dois cortes subcorticais (axial z=2 e coronal y=−10) e uma faixa de correlação por família.
- `state_card_S<k>.png`: um estado em tamanho de leitura, com barras ordenadas.
- `correlations.csv`: r, p e q (FDR) para cada estado × mapa.
- `state_means.csv`: os estados, no formato do `state-means --csv`.
- `provenance.json`: de onde vieram o atlas, os estados e cada família, e a verdade da simulação.

## Os nossos atlas

O piloto roda nos quatro atlas que o pipeline extrai, com o código do próprio
repo:

| atlas | parcelas | de onde veio aqui |
|---|---|---|
| `harvardoxford` | 111 (96 corticais + 15 subcorticais) | `get_atlas` primeiro. Como o nilearn baixa do NITRC, que estava bloqueado, o atlas foi **reconstruído pelo mesmo caminho de código** (`plan_harvard_oxford_merge`, `AtlasSpec.from_label_image`) a partir das mesmas imagens FSL vindas do TemplateFlow: os mesmos 111 nomes, na mesma ordem. |
| `networks_nodes` | 254 esferas | `get_atlas`, sem download (o CSV está no repo) |
| `networks` | 14 redes | `get_atlas` |
| `yeo7` | 7 redes | `get_atlas` primeiro. Como o nilearn baixa do servidor do FreeSurfer, que estava bloqueado, o atlas foi reconstruído a partir do mesmo volume (`Yeo2011_7Networks…LiberalMask`, cópia do neuroparc) |

No cluster, onde o cache do nilearn existe, o `get_atlas` funciona direto. O
`provenance.json` registra qual caminho foi usado.

**Tudo é parcelado no espaço do pipeline** (MNI152NLin6Asym, 2 mm) com
`AtlasSpec.membership`. É a mesma matriz que transformou voxels nas séries
temporais em que os estados foram ajustados, então o valor de um mapa na
parcela k é calculado exatamente como o valor do estado na parcela k,
subcórtex incluído. A superfície só entra no desenho.

## Estatística

Mapas cerebrais suaves se correlacionam muito mais do que vetores
aleatórios, porque parcelas vizinhas não são independentes. Por isso o p vem de
**mapas substitutos que preservam a autocorrelação espacial** de cada mapa de
referência, medida pelas distâncias entre os centróides das parcelas. Isso vale
para subcórtex e para atlas de esferas, onde o spin test não se aplica.

- `--null moran` (padrão): Moran spectral randomization (Wagner & Dray 2015, como no BrainSpace), com 10.000 substitutos por mapa (`--n-nulls`). Roda em segundos.
- `--null burt`: variogram matching (Burt et al. 2020, brainsmash). Leva cerca de 45 s por mapa com 111 parcelas e cerca de 2,5 min com 254; com 85 mapas, isso dá horas. É um job de cluster, para a análise final.

O FDR (BH) é aplicado dentro de cada família, sobre todos os pares estado ×
mapa. O número de substitutos importa: o menor p possível é 1/(n+1), e na
família molecular (6 estados × 19 mapas = 114 testes) 1000 substitutos não
deixam nenhum q chegar a .05. Por isso o padrão é 10.000. Na figura, `•` marca p < .05 sem correção, `✱` marca q < .05 e `★` marca
q < .01.

**`yeo7` (7) e `networks` (14) não têm p-valor.** Uma correlação sobre 7 ou 14
pontos é descrição, e nenhum modelo nulo transforma isso em inferência. Para
esses atlas a figura mostra o padrão e as correlações sem marcas.

## Os estados simulados têm resposta conhecida

Cada estado é uma mistura, com sinal, de redes de Yeo e das quatro estruturas
profundas do Brainnetome (amígdala, hipocampo, núcleos da base e tálamo), mais
ruído espacialmente suave. O piloto sorteia 6000 frames ruidosos numa cadeia
de Markov "pegajosa" e tira a média por rótulo, o mesmo groupby do
`state-means`.

| estado | mistura (verdade) |
|---|---|
| Default | +Default −DorsAttn −SomMot +hipocampo |
| Visual | +Visual −Default +tálamo |
| Sensorimotor | +SomMot +VentAttn −FPN +núcleos da base +tálamo |
| Task-positive | +DorsAttn +FPN −Default +núcleos da base |
| Limbic | +Limbic +Default −Visual +amígdala +hipocampo |
| Salience | +VentAttn +SomMot −Default +tálamo +amígdala |

Se o painel de Yeo não recupera a mistura, o problema está no pipeline, não nos
estados. As correlações moleculares são **reais**, mas são entre mapas de
receptores e padrões sintéticos construídos a partir das redes. Descrevem as
redes, não um achado.

## A parte comportamental: taxonomia completa, perfis simulados

O Brainnetome caracteriza cada sub-região pelos rótulos do BrainMap (Fan et al.
2016), que são hierárquicos: domínio → subdomínio → sub-subdomínio. Com `BN_LEVELS` o painel mostra os **5 domínios** (média das subcategorias de
cada um), **todas as folhas**, ou os dois lado a lado. O painel de folhas
mostra, por exemplo `Vision · Motion`, `Language ·
Semantics`, `Somesthesis · Pain`, `Memory · Working`, agrupadas sob os cinco
domínios (Action, Cognition, Emotion, Interoception e Perception).

**Os perfis estão simulados.** Os perfis reais por sub-região estão em
atlas.brainnetome.org, e nem o DNS desse site resolveu daqui; brainmap.org
também não. O atlas em si (volume com as 246 regiões e a tabela de
sub-regiões) veio da cópia que o DPABI distribui. A tabela simulada usa um
prior de livro-texto por giro, refinado para as sub-regiões cujo nome diz mais
(V5/MT+ → Vision·Motion, A41/42 → Audition, A44/45 → Language, A4tl →
Execution·Speech, etc.), mais ruído. A figura marca esse painel com hachura e
"SIMULATED".

A lista de categorias em `refmaps.BRAINMAP_TAXONOMY` foi escrita à mão a partir
da taxonomia publicada, sem consulta à fonte. Confira contra a tabela real; quando ela entrar,
as colunas passam a vir dela.

**Formato da tabela real** (`--behavioural arquivo.csv`): uma coluna
`bn_index` (1..246, numeração do BNA: ímpar = esquerdo, par = direito) e uma
coluna por categoria com o caminho completo (`Perception.Vision.Motion`), com
qualquer escore monotônico (z, razão de verossimilhança ou 0/1). Os grupos de
domínio da figura saem do primeiro nível do nome.

## Sobre o `neuromaps`

`neuromaps.datasets.fetch_annotation` baixa do OSF, que estava bloqueado
neste ambiente. Os PET vêm do repositório do artigo
(`netneurolab/hansen_receptors`), que tem os mesmos volumes. Num ambiente com
OSF dá para trocar por `fetch_annotation` e usar o `neuromaps.nulls`.

## Decisões a revisitar

- **Estados de DFC** (janelas de conectividade) não são mapas de ativação. Para entrarem aqui, é preciso derivar um mapa por parcela, por exemplo a força nodal.
- **`networks_nodes`**: as esferas de 5 mm amostram pouco tecido. Nos mapas PET de baixa resolução (3 mm), uma esfera tem poucos voxels independentes.
- **`harvardoxford` reconstruído vs. nilearn**: o `symmetric_split` do nilearn e a versão lateralizada do FSL (HOCPAL) podem diferir em alguns voxels na linha média. Com os estados reais, use o atlas do cluster.

## Perfis comportamentais a partir do Neurosynth (padrão)

`BEHAVIOURAL = "neurosynth"` (o padrão) mantém a **taxonomia do BrainMap** e
as **246 regiões do Brainnetome**, mas calcula os perfis a partir do Neurosynth
v7 (14.371 estudos), sem simulação. O cálculo:

1. Cada categoria do BrainMap recebe os termos do vocabulário do Neurosynth que a representam (`refmaps.NEUROSYNTH_TERMS`). Por exemplo, `Perception.Vision.Motion` usa "visual motion" e "motion". Um estudo pertence à categoria se algum desses termos passa do limiar usual de tf-idf do Neurosynth (0,001).
2. Um estudo "ativa" uma sub-região se algum pico dele cai dentro dela ou a até 6 mm. Os picos em Talairach vão para MNI pela transformação de Lancaster 2007.
3. Para cada região e categoria, calcula-se a **inferência direta**, a mesma pergunta que Fan et al. 2016 fizeram ao BrainMap: os estudos da categoria ativam a região mais que os estudos em geral? O escore é z = (k − n·p0) / √(n·p0·(1−p0)), a aproximação normal do teste binomial.

Categorias sem termo no vocabulário ficam de fora: Air Hunger, Bladder, Thermoregulation e Thirst. O resultado é salvo em `state_maps_data/neurosynth/brainnetome_neurosynth.csv`, junto com o número de estudos por categoria (`..._n_studies.csv`).

Conferência: as regiões no topo saem como esperado. V5/MT+ em Vision·Motion, A41/42 em Audition, a amígdala em Fear, o hipocampo em Explicit memory, SMA/M1 em Execution e a ínsula dorsal/SII em Pain.

Os exemplos em `state_maps_results/simulated/` foram gerados antes, com a tabela simulada. Rode o notebook de novo para vê-los com os perfis do Neurosynth.

## Taxonomia: BrainMap ou Beam et al. 2021

`TAXONOMY` escolhe as categorias usadas nos perfis do Neurosynth:

- `"brainmap"` (padrão): a taxonomia do BrainMap (Fox et al. 2005), a que o Brainnetome usa. São 5 domínios e 41 subcategorias.
- `"beam2021"`: a ontologia data-driven de Beam, Potts, Poldrack & Etkin 2021 (*Nature Neuroscience* 24:1733). São 6 domínios: Memory, Reward, Cognition, Vision, Manipulation e Language. Ela foi derivada de cerca de 18.000 artigos de fMRI e, em artigos fora da amostra, replica melhor que o RDoC. As subcategorias são os termos que os autores listaram para cada domínio (`ontology/lists/lists_data-driven_lr.csv` do repositório `ehbeam/neuro-knowledge-engine`). Cada termo vira uma categoria do Neurosynth.

No Beam, 81 dos 112 termos existem no vocabulário do Neurosynth, por domínio: Cognition 23, Memory 16, Vision 15, Manipulation 14, Language 10 e Reward 3. Os que faltam são, na maioria, expressões compostas, como "speech processing", "syntax" e "monetary incentive delay task".

Alguns termos aparecem em mais de um domínio, como "emotion" em Memory e em Cognition. Isso segue a lista original, e cada domínio recebe a sua cópia. No Beam, emoção e medo ficam dentro de Memory, porque não existe domínio de emoção.

Os resultados do Beam vão para `state_maps_results/<estados>_beam2021/`, para não sobrescrever os do BrainMap.


## A figura principal (`states_figure.png`)

Cada estado ocupa uma linha:

- **Cérebro em 2 × 3.** Em cima: lateral esquerda, lateral direita e o corte axial z=2 (núcleos da base e tálamo). Embaixo: medial esquerda, medial direita e o corte coronal y=−10 (amígdala e hipocampo). Cada vista é desenhada sozinha e recortada, para nada ser cortado nas bordas.
- **Um gráfico de barras por família em `PANELS`.** O padrão é Yeo 7 e as subcategorias do BrainMap. Cada gráfico mostra as `TOP` maiores |r|; Yeo 7 mostra sempre as 7 redes. As categorias "… · Other" do BrainMap ficam de fora, porque são restos e não funções.
- **Barras = média de r entre sujeitos ± desvio padrão entre sujeitos.** Isso vale quando os estados vêm de uma coluna de estados (cada sujeito tem o seu mapa por estado; o arquivo é `subject_correlations.parquet`) ou da simulação (30 sujeitos). Com um CSV do `state-means` só existe o mapa do grupo, e as barras saem sem erro.

O painel antigo de faixas de calor e os cards por estado continuam disponíveis com `draw(..., overview=True, cards=True)`, ou `--overview --cards` na linha de comando.

## De onde vem o mapa de cada estado

O mapa de cada estado é sempre **ativação**, lida de `outputs/activation/`. O caminho é o mesmo para PCA, raw e UMAP, e não usa transformação inversa:

1. Dos latents saem o rótulo de cada linha e a chave `(task, sub, window_id)`, mais `stimulus_start_s` nas janelas.
2. São lidos apenas os sujeitos presentes nos latents, ou seja, o conjunto em que o `decompose` ajustou o modelo depois da política de censura. Sujeitos excluídos pelo pipeline nunca são abertos.
3. As ativações por TR são reconstruídas por `frames.read_cohort`:
   - **`window_s = −1`:** com o band-pass e o z-score do próprio ajuste. Cada frame rotulado entra na média do seu estado.
   - **Janelas de DFC (`window_s > 0`):** com os padrões de `frames.py` (band-pass 0,01–0,1 Hz e z-score por run). Uma janela cobre os frames com `stimulus_time_s` em `[stimulus_start_s, stimulus_start_s + window_s)`, a mesma regra de `windows.py`. O mapa do estado é a média de ativação sobre os frames cobertos pelas suas janelas. Com janelas sobrepostas, cada frame conta uma vez por janela.
4. As médias são calculadas por estado (grupo) e por sujeito × estado (para o desvio padrão).

Testado nos arquivos sintéticos do pipeline (`fixtures`), nos dois casos: a diferença em relação a um cálculo manual foi de cerca de 5×10⁻⁸.

O resultado para os estados de DFC mudou. Antes, cada estado virava a força nodal da conectividade; agora é a ativação média durante as janelas do estado. A força nodal continua sendo usada só quando a entrada é um CSV do `state-means` em unidades de aresta.

## Matriz entre mapas de referência

`state_figures.reference_matrix(atlas, rows, cols)` calcula a correlação espacial entre cada mapa de uma família e cada mapa de outra, entre as parcelas do atlas escolhido. O padrão é Yeo 7 × as 38 subcategorias do BrainMap (sem as "Other"). Os asteriscos vêm do mesmo teste espacial das figuras de estados, com FDR sobre a matriz inteira. A função grava `state_maps_results/reference_matrix/reference_matrix_<atlas>_<rows>_x_<cols>.{png,csv}`. No notebook, a seção 3 tem os parâmetros `MATRIX_ATLAS`, `MATRIX_ROWS` e `MATRIX_COLS`.
