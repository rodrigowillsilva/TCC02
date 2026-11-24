# Especificação da Suíte Oficial de Testes — Artigo TCC02

Este documento descreve exatamente o que precisamos que seja implementado e executado na **Suíte Oficial de Experimentos** para gerar todos os dados quantitativos, tabelas em LaTeX e análises da **Seção IV (Resultados e Discussão)** do artigo de TCC02 (`TCC02/ECO_artigo_TCC02/ECO_artigo_TCC02/access.tex`).

---

## 1. Contexto e Código Existente
- O código do pipeline fica em `TCC02/experiment/`.
- Já existe um script funcional de referência em `TCC02/experiment/gerar_tudo_apresentacao.py` que roda métodos sobre vídeos, calcula telemetria e gera `relatorio_quantitativo.md` e `relatorio_quantitativo.json`.
- **Diferença para a Suíte Oficial do Artigo:** Na apresentação nós gravamos vídeos `.mp4` (`cv2.VideoWriter`), o que consome I/O. Para o **benchmark oficial do artigo**, **NÃO devemos gravar vídeos `.mp4` nem abrir janelas `cv2.imshow`** durante a medição de tempo, garantindo que a telemetria (latência em `ms`, `FPS` e `% CPU`) reflita puramente o custo algorítmico dos 4 estágios do pipeline:
  1. `01_Fatiamento (Tiler)`
  2. `02_Filtro (DecisionBlock)`
  3. `03_IA_Inferência (YOLO)` (e atualização de estado/cache quando aplicável)
  4. `04_Limpeza (Consolidator NMS)`

---


## 3. Configuração Fixa do Pipeline e Referência (*Ground Truth*)
- **Tiler:** `tile_size=(320, 320)`, `grid_size=(5, 3)` $\rightarrow$ 15 *tiles* por quadro (sobreposição de 80 px em $X$ e 120 px em $Y$).
- **Detector (Caixa-Preta):** `YOLODetector(model_name="yolov8n.pt", conf_thresh=0.25, imgsz=640)`, classe `0` (*person*).
- **Consolidador:** `Consolidator(iou_threshold=0.45, iom_threshold=0.50)`.
- **Tracker / Orquestrador (Scheduler):** **Desligados (`None`)**. Não fazem parte dos experimentos do artigo.
- **Referência (*Ground Truth*):** É a saída consolidada quadro a quadro do próprio **Baseline** (`decision_block=None`, onde todos os 15 *tiles* são submetidos ao YOLOv8n em todos os quadros).

---

## 4. Baterias de Testes Necessárias

### Bateria A — Comparação Principal entre as 5 Estratégias (para a Tabela II do Artigo)
Executar sobre todos os vídeos selecionados os 5 métodos abaixo com seus parâmetros padrão calibrados:
1. **`BASELINE`**: `decision_block = None` (15 *tiles*/quadro).
2. **`FRAMEDIFF`**: `FrameDiffDecision(debug=False, min_pixels=150, threshold=25, scale_factor=0.75, frame_step=3)`
3. **`KNN`**: `KNNDecision(debug=False, history=500, dist2Threshold=200.0, min_pixels=150, scale_factor=0.75)`
4. **`MOG2`**: `BGSDecision(debug=False, history=500, var_threshold=50.0, min_pixels=150)`
5. **`HASH` (Wavelet)**: `HashDecision(levels=(5, 10, 20, 40, 80), hash_type="WAVELET", tolerance_ratio=0.015, ttl_positive=6, ttl_empty=30, debug=False)`

### Bateria B — Estudo de Ablação / Variação de Parâmetros (Opcional / Recomendado se viável no tempo de execução)
Para enriquecer a discussão e justificar as escolhas da Tabela I do artigo, rodar sobre 1 ou 2 vídeos representativos (ex.: `VIRAT_S_010004_01_000163_000188.mp4` e `VIRAT_S_010203_08_000895_000975.mp4`):
1. **Comparação das Modalidades de Hash (`hash_type`):**
   - Testar `"VALUE"`, `"EDGE"`, `"HYBRID"` (com `levels=(8, 10, 13, 16, 20, 24, 32)`) vs. `"WAVELET"` (com `levels=(5, 10, 20, 40, 80)`), mantendo `tolerance_ratio=0.015, ttl_positive=6, ttl_empty=30`.
2. **Sensibilidade de Tolerância e TTL no Hash Wavelet:**
   - Variar `tolerance_ratio` (ex.: `0.005`, `0.015`, `0.030`) e `ttl_positive` / `ttl_empty` (ex.: `(3, 15)`, `(6, 30)`, `(24, 90)`, e sem expiração `(0, 0)`).
3. **Sensibilidade de `min_pixels` nos métodos de movimento (KNN / FrameDiff):**
   - Variar `min_pixels` (ex.: `75`, `150`, `300`).

---

## 5. Métricas que Devem ser Calculadas e Salvas

Para cada execução (por vídeo e na **Média/Agregado Global** de todos os vídeos):

### 5.1. Métricas de Eficácia (comparando contra as caixas consolidadas do Baseline no mesmo quadro):
- Calcular tanto para **$\mathrm{IoU} \ge 0{,}40$** (padrão principal do artigo) quanto para **$\mathrm{IoU} \ge 0{,}50$**:
  - Total de caixas de referência ($\mathrm{GT}$), Verdadeiros Positivos ($\mathrm{TP}$), Falsos Negativos ($\mathrm{FN}$) e Falsos Positivos ($\mathrm{FP}$).
  - **$\mathrm{Precision}$ (%)**, **$\mathrm{Recall}$ (%)** e **$F_1\text{-Score}$ (%)**.
  - **$\mathrm{IoU}$ Médio** dos pares casados ($\mathrm{TP}$).

### 5.2. Métricas de Eficiência Computacional (descartando os 10 frames de warmup):
- **Latência média por quadro (ms)** de cada estágio:
  - `tiler_ms` (Divisão em tiles)
  - `filter_ms` (Bloco de decisão)
  - `yolo_ms` (Inferência YOLO amortizada por quadro do vídeo, isto é: `tempo_total_yolo / total_de_quadros_medidos`, e também o tempo por chamada quando o YOLO é acionado)
  - `consolidator_ms` (Consolidação IoU/IoM)
  - `total_ms` (Tempo total do pipeline por quadro, sem renderização visual)
- **FPS equivalente do pipeline:** `1000.0 / total_ms`
- **Redução de tempo total (%):** `(base_total_ms - method_total_ms) / base_total_ms * 100`
- **Fator de Aceleração (*Speedup*):** `base_total_ms / method_total_ms`
- **Proporção de *Tiles* Enviados ao YOLO (%):** `(total_tiles_enviados / (15 * total_frames)) * 100`
- **Taxa de Redução de Inferência / *Tiles* Poupados (%):** `100.0 - tiles_enviados_percent`
- **Para o modo HASH:** registrar também `hit_rate_percent`, `cache_hits`, `cache_misses`, `expired_revalidations` e `unique_paths_stored`.
- **Hardware / Sistema:** registrar automaticamente no JSON o modelo da CPU, número de núcleos físicos/lógicos, quantidade de RAM total da máquina e uso médio de CPU (%).

---

## 6. Formato de Saída Esperado (Arquivos para o Assistente de Escrita)

Salvar todos os resultados consolidados na pasta `TCC02/results/OFFICIAL_SUITE/`:
1. **`official_results.json`**: JSON completo com metadados da máquina (CPU/RAM), parâmetros exatos de cada método, métricas vídeo a vídeo, **tabela agregada (média global entre todos os vídeos)** e resultados da Bateria B (ablação).
2. **`official_report.md`**: Relatório legível em Markdown contendo:
   - Especificações do ambiente de hardware detectado;
   - Tabelas Markdown por vídeo e a **Tabela Global Consolidada**;
   - **Blocos de código LaTeX (`tabular`) prontos para colar na Tabela II do `access.tex`**.
