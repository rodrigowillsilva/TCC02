# Plano de Implementação — Pipeline Experimental TCC02

## Objetivo desta fase
Construir o **esqueleto completo do pipeline experimental** rodando no PC local, lendo vídeos VIRAT do disco. O bloco de decisão fica como interface plugável — implementamos os métodos depois.

---

## Estrutura de Arquivos Atualizada

Optamos por uma estrutura mais simples e plana (sem pacotes cheios de `__init__.py` ou configs prematuras) para manter o foco no desenvolvimento ágil.

```
TCC02/
├── videos/                   # Vídeos de teste (VIRAT) copiados
├── notas_metodos/            # Anotações e mapa metodológico
│
├── experiment/
│   ├── main.py               # Ponto de entrada — roda o experimento
│   │
│   ├── pipeline/
│   │   ├── video_reader.py   # Lê o vídeo, retorna FrameData
│   │   ├── tiler.py          # Fatia frame em grade fixa devolvendo objetos Tile
│   │   ├── annotations.py    # Estruturas de dados (BBox, FrameAnnotations)
│   │   ├── frame_annotator.py# Desenha os metadados (BBoxes, labels) no frame
│   │   ├── display.py        # Janela OpenCV (play/pause/frame-a-frame)
│   │   ├── pipeline_runner.py# Orquestra os componentes acima
│   │   │
│   │   ├── detector.py       # (FUTURO) Wrapper caixa-preta do YOLO
│   │   └── consolidator.py   # (FUTURO) NMS para juntar detecções
│   │
│   ├── decision/
│   │   ├── base.py           # Classe abstrata DecisionBlock (interface)
│   │   ├── baseline_all.py   # Baseline: processa TODOS os tiles
│   │   └── (métodos futuros aqui)
│   │
│   └── metrics/
│       └── collector.py      # (FUTURO) Medição de FPS, latência, CPU, RAM
```

---

## Checklist de Implementação

### 1. Infraestrutura Base (✅ CONCLUÍDO)
- [x] **`video_reader.py`** — Iterador que retorna frames com timestamp e índice.
- [x] **`display.py`** — Mostra imagem com controles nativos de Play/Pause (Espaço) e Frame-Step (N/D).
- [x] **`annotations.py` / `frame_annotator.py`** — Desenha as BBoxes no frame, lidando com posicionamento inteligente de rótulos.
- [x] **`tiler.py`** — Divide a imagem em objetos `Tile`. Usa `np.linspace` para garantir tamanho exato dos tiles e overlap calculado dinamicamente baseado na grade (ex: 6x4).

### 2. Orquestrador (✅ CONCLUÍDO)
- [x] **`pipeline_runner.py`** — Loop principal amarrando o leitor, tiler, annotator e display.
- [x] **`main.py`** — Script de entrada com função de debug isolado de tiles (`teste_visual_tiles`).

### 3. Bloco de Decisão (⏳ PRÓXIMO PASSO)
- [ ] **`base.py`** — Classe abstrata `DecisionBlock` (define a interface `select`).
- [ ] **`baseline_all.py`** — Classe `BaselineAll` que simplesmente retorna todos os tiles.

### 4. Inferência e Consolidação (⏸️ PAUSADO)
- [ ] **`detector.py`** — Wrapper do Ultralytics YOLO.
- [ ] **`consolidator.py`** — Mapeamento de coordenadas locais para globais + NMS.

### 5. Métricas (⏸️ PAUSADO)
- [ ] **`collector.py`** — Mede tempo por estágio, uso de CPU e RAM (`psutil`).

---

## Decisões Arquiteturais Tomadas
- **Configurações:** Sem `config.py` por enquanto. Parâmetros passados direto na instanciação dentro do `main.py` para manter simplicidade no desenvolvimento.
- **Tiler:** Usa tamanho fixo de tile (ex: 320x320) e tamanho de grade fixa (ex: 6x4). O *overlap* não é uma porcentagem fixa, ele é resolvido matematicamente pelo algoritmo `numpy.linspace`, garantindo que todos os tiles tenham exatamente o mesmo tamanho e caibam perfeitamente na imagem.
- **Visualização:** Separada em Dados (`annotations.py`) e Renderização (`frame_annotator.py`), permitindo que a visualização seja desligada no futuro sem quebrar o processamento de dados.
