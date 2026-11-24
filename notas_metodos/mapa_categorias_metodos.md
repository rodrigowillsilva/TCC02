# Mapa de Categorias de Métodos — Seleção de Tiles para Inferência Seletiva

Organizado por **família/paradigma**, não por artigo individual. Cada categoria é uma "gaveta" de técnicas que compartilham o mesmo princípio de funcionamento.

> [!NOTE]
> ✅ = Você já tem notas sobre isso  
> 🔍 = Categoria nova para explorar

---

## Categoria 1 ✅ — Modelagem e Subtração de Fundo
**Princípio:** Manter um modelo estatístico do fundo da cena. Pixels que divergem do fundo → primeiro plano → tile ativo.

| Aspecto | Detalhe |
|---------|---------|
| **Algoritmos** | MOG2 (Zivkovic, 2004), KNN (Zivkovic & van der Heijden, 2006), ViBe (Barnich & Van Droogenbroeck, 2011) |
| **Custo CPU** | Baixo-Médio (operação por pixel, mas otimizada em OpenCV) |
| **Forças** | Sem treino, adaptativo, bem suportado em OpenCV, funciona em qualquer cena |
| **Fraquezas** | Sombras → falsos positivos; objetos parados são absorvidos pelo fundo; mudanças de iluminação global |
| **Artigos-âncora** | Reducto (Li et al., 2020), Glimpse (Chen et al., 2015) |

**Artigos para pesquisar:**
- Zivkovic (2004) — *"Improved Adaptive Gaussian Mixture Model for Background Subtraction"* — ICPR 2004
- Zivkovic & van der Heijden (2006) — *"Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction"* — Pattern Recognition Letters
- Barnich & Van Droogenbroeck (2011) — *"ViBe: A Universal Background Subtraction Algorithm for Video Sequences"* — IEEE TIP
- Li et al. (2020) — *"Reducto: On-Camera Filtering for Resource-Efficient Real-Time Video Analytics"* — ACM SIGCOMM
- Chen et al. (2015) — *"Glimpse: Continuous, Real-Time Object Recognition on Mobile Devices"* — ACM SenSys

---

## Categoria 2 🔍 — Comparação Direta de Tiles (Métricas de Similaridade)
**Princípio:** Comparar cada tile do quadro atual com o tile correspondente de um quadro anterior (ou referência). Se são "parecidos demais" → pular.

### Sub-família 2a: Diferença de Pixels / MSE
- Calcula a diferença média de intensidade entre tiles
- Ultra-rápido (`np.mean(np.abs(tile_atual - tile_ref))`)
- Usado no **NoScope** como "difference detector" (até 100k FPS em CPU)
- **Fraqueza:** sensível a qualquer variação de iluminação

### Sub-família 2b: SSIM (Structural Similarity Index)
- Compara **luminância, contraste e estrutura** entre tiles
- Mais robusto que MSE contra variações de iluminação
- Score de 0 a 1 → threshold para decidir "mudou" vs "não mudou"
- Custo um pouco maior que MSE, mas ainda leve
- **Artigo base:** Wang et al. (2004) — *Image Quality Assessment: From Error Visibility to Structural Similarity* (IEEE TIP)

### Sub-família 2c: Hashing Perceptual (pHash / dHash)
- Gera uma "impressão digital" compacta (hash) de cada tile (reduz tile a 8x8 ou 16x16 → calcula hash)
- Compara hashes por **distância de Hamming** → decisão binária ultra-rápida
- Custo computacional **extremamente baixo**
- **Fraqueza:** menos granular — bom para "mudou muito" vs "não mudou nada", ruim para mudanças sutis
- Variantes: aHash (average), dHash (difference/gradiente), pHash (DCT), wHash (wavelet)

> [!TIP]
> Esta categoria é um **upgrade direto** da ideia de "diferença de quadros" que aparece em vários papers. A diferença é o nível de sofisticação: MSE < SSIM < pHash em termos de robustez a ruído, com custo similar.

**Artigos para pesquisar:**
- Wang et al. (2004) — *"Image Quality Assessment: From Error Visibility to Structural Similarity"* — IEEE Transactions on Image Processing (artigo original do SSIM)
- Kang et al. (2017) — *"NoScope: Optimizing Neural Network Queries over Video at Scale"* — VLDB (usa MSE/difference detector como primeiro estágio)
- Zauner (2010) — *"Implementation and Benchmarking of Perceptual Image Hash Functions"* — Thesis, Upper Austria University (referência para pHash/dHash)

---

## Categoria 3 🔍 — Fluxo Óptico Leve
**Princípio:** Estimar o deslocamento de pixels entre quadros consecutivos. Regiões com movimento significativo → tiles ativos.

| Aspecto | Detalhe |
|---------|---------|
| **Algoritmos leves** | Lucas-Kanade (esparso, pontos-chave), Farnebäck (denso mas leve no OpenCV) |
| **Custo CPU** | Médio (mais caro que BGS, mas calculável em CPU) |
| **Forças** | Detecta **direção e velocidade** do movimento, não só "mudou/não mudou" |
| **Fraquezas** | Mais pesado que subtração de fundo; sensível a texturas uniformes; não funciona bem em áreas de baixo contraste |
| **Artigos-âncora** | Zhu et al. (2017) — Deep Feature Flow usa fluxo óptico internamente; Glimpse (Chen et al., 2015) usa trigger-and-track |

> [!IMPORTANT]
> Para o seu cenário (CPU limitada, câmera estática), o fluxo óptico **esparso** (Lucas-Kanade em pontos-chave) seria mais viável que o denso. Mas é provável que MOG2/KNN já cubra o mesmo objetivo com custo menor.

**Artigos para pesquisar:**
- Lucas & Kanade (1981) — *"An Iterative Image Registration Technique with an Application to Stereo Vision"* — IJCAI (artigo original do fluxo óptico esparso)
- Farnebäck (2003) — *"Two-Frame Motion Estimation Based on Polynomial Expansion"* — SCIA (fluxo óptico denso leve, implementado no OpenCV)
- Zhu et al. (2017) — *"Deep Feature Flow for Video Recognition"* — CVPR (usa fluxo óptico para propagar features entre keyframes)
- Hui et al. (2018) — *"LiteFlowNet: A Lightweight Convolutional Neural Network for Optical Flow Estimation"* — CVPR (fluxo óptico leve por CNN)

---

## Categoria 4 ✅ — Domínio Comprimido (Metadados do Codec)
**Princípio:** Extrair informação de movimento diretamente do bitstream de vídeo comprimido (H.264/HEVC), sem decodificar pixels.

### Sub-família 4a: Vetores de Movimento (seus MV notes)
- Lê deslocamento de macroblocos dos P-frames/B-frames
- Custo de decisão ≈ zero (já calculado pelo encoder)
- **Problema:** precisa PyAV/FFmpeg, cegueira em I-frames, ruído de compressão

### Sub-família 4b 🔍: Coeficientes DCT
- Analisa os coeficientes de frequência (DCT) dos blocos comprimidos
- Permite detectar bordas, textura e mudanças diretamente no domínio de frequência
- Pode funcionar como um pipeline **coarse-to-fine**: DCT rápido decide se vale decodificar
- **Problema:** mesma limitação de engenharia (precisa acessar o stream comprimido)

> [!WARNING]
> As duas sub-famílias do domínio comprimido compartilham o mesmo gargalo de engenharia: OpenCV não expõe esses dados. Ambas exigem ferramentas externas (PyAV, FFmpeg, mpegflow). Se for implementar uma, o custo marginal de implementar a outra é baixo.

**Artigos para pesquisar:**
- Kantorov & Laptev (2014) — *"Efficient Feature Extraction, Encoding and Classification for Action Recognition"* — CVPR (extrai features diretamente do domínio comprimido)
- Chadha et al. (2017) — *"Video Classification With CNNs: Using The Codec As A Spatio-Temporal Activity Sensor"* — IEEE TCSVT (usa MVs e coeficientes DCT do codec para classificação)
- Wu et al. (2018) — *"Compressed Video Action Recognition"* — CVPR (combina I-frames, MVs e residuais do codec para reconhecimento de ação)
- Shen et al. (2019) — *"Fast Video Shot Transition Localization with Deep Structured Models"* — ACCV (detecção de mudanças no domínio comprimido)

---

## Categoria 5 🔍 — Rastreamento Leve entre Keyframes
**Princípio:** Rodar YOLO apenas em "keyframes" espaçados. Entre keyframes, usar um rastreador leve para prever onde os objetos estão → só reprocessar tiles onde o tracker perdeu confiança.

| Aspecto | Detalhe |
|---------|---------|
| **Trackers leves** | Kalman Filter, SORT, ByteTrack, Norfair |
| **Custo CPU** | Baixo (Kalman é uma multiplicação de matrizes 4x4) |
| **Forças** | Predição de posição é quase gratuita; IDs mantidos entre quadros; naturalmente complementar à fila de prioridade |
| **Fraquezas** | Precisa de detecção inicial para começar; drift se o intervalo entre keyframes for muito longo; objetos novos entrando na cena podem ser perdidos até o próximo keyframe |
| **Artigos-âncora** | Glimpse (Chen et al., 2015) — trigger + track; Plastiras et al. (2018/2019) — memória temporal de tiles |

> [!TIP]
> Este método se combina muito bem com a sua **Fila de Prioridade**: tiles com objetos rastreados recebem score alto para serem priorizados, e o Kalman filter preenche os gaps entre inferências.

**Artigos para pesquisar:**
- Bewley et al. (2016) — *"Simple Online and Realtime Tracking"* — IEEE ICIP (SORT — tracker leve com Kalman + Hungarian algorithm)
- Zhang et al. (2022) — *"ByteTrack: Multi-Object Tracking by Associating Every Detection Box"* — ECCV (ByteTrack — estado da arte em tracking leve)
- Chen et al. (2015) — *"Glimpse: Continuous, Real-Time Object Recognition on Mobile Devices"* — ACM SenSys (trigger-and-track: roda detector em keyframes, rastreia entre eles)
- Plastiras et al. (2018) — *"Efficient Embedded Object Detection with Selective Tile Processing"* — ACM/IEEE ICDSC (memória temporal de tiles + rastreamento em UAVs)
- Zhu et al. (2017) — *"Deep Feature Flow for Video Recognition"* — CVPR (propagação de features entre keyframes via fluxo óptico)

---

## Categoria 6 🔍 — Mapas de Saliência / Atenção Visual
**Princípio:** Usar modelos computacionais de atenção visual (imitando o olho humano) para identificar regiões "interessantes" de uma imagem.

| Aspecto | Detalhe |
|---------|---------|
| **Métodos clássicos (sem NN)** | Baseados em contraste de cor, textura, bordas — ex: Itti & Koch (1998), GBVS |
| **Custo CPU** | Médio (mais caro que BGS, depende do método) |
| **Forças** | Captura "o que chama atenção" sem depender de movimento — funciona em frames individuais |
| **Fraquezas** | Saliência visual ≠ relevância semântica (uma placa colorida pode ser mais "saliente" que uma pessoa); métodos clássicos podem ser caros demais para borda |

> [!WARNING]
> Para o seu caso específico (câmera estática, foco em movimento), esta categoria pode ser **redundante**. Saliência brilha quando o quadro é estático e você precisa encontrar "o que é interessante" sem informação temporal. Se você tem informação temporal (frames anteriores), BGS ou comparação direta são mais eficientes.

**Artigos para pesquisar:**
- Itti et al. (1998) — *"A Model of Saliency-Based Visual Attention for Rapid Scene Analysis"* — IEEE TPAMI (modelo clássico de saliência)
- Harel et al. (2006) — *"Graph-Based Visual Saliency"* — NIPS (GBVS — saliência por grafo, mais rápido)
- Borji et al. (2015) — *"Salient Object Detection: A Benchmark"* — IEEE TIP (survey comparativo de métodos de saliência)

---

## Categoria 7 ✅ — Escalonamento por Orçamento / Scheduling
**Princípio:** Definir um budget fixo de tiles por quadro e usar uma função de scoring para priorizar quais tiles processar.

| Aspecto | Detalhe |
|---------|---------|
| **Seu método** | Score linear com "Fome Temporal" ($A_i$) e "Presença Histórica" ($D_i$) |
| **Variantes** | Round-robin, aging + decay, reinforcement learning adaptativo |
| **Forças** | Garante throughput constante; sem dependência de sinais visuais (pode combinar com qualquer outra categoria) |
| **Fraquezas** | Sozinho é "cego" — não sabe o que mudou, só garante cobertura |

> [!NOTE]
> Esta categoria é mais um **orquestrador** do que um detector. Funciona melhor **combinado** com outra categoria que forneça o sinal de "importância" de cada tile.

**Artigos para pesquisar:**
- Jiang et al. (2018) — *"Chameleon: Scalable Adaptation of Video Analytics"* — ACM SIGCOMM (adaptação dinâmica de configuração de pipeline de vídeo)
- Zhang et al. (2017) — *"Live Video Analytics at Scale with Approximation and Delay-Tolerance"* — NSDI (VideoStorm — scheduling de queries de vídeo com budget)

---

## Categoria 8 — Classificadores Especializados (Cascata)
**Princípio:** Treinar um modelo auxiliar pequeno para classificar tiles como "interessante" vs "vazio".

| Aspecto | Detalhe |
|---------|---------|
| **Artigos** | NoScope (Kang et al., 2017), FilterForward (Canel et al., 2019) |
| **Custo CPU** | Baixo (modelo muito pequeno) |
| **Forças** | Mais robusto semanticamente que BGS ou métricas de pixel |
| **Fraquezas** | ⚠️ **Exige treino por câmera/cena** — não é genérico. Muda a câmera → retreina |

> [!IMPORTANT]
> Como discutimos: este é poderoso mas **não se encaixa bem** no seu cenário de TCC (abordagem caixa-preta, generalidade). Vale como referência na revisão bibliográfica, não como candidato principal para implementação.

**Artigos para pesquisar:**
- Kang et al. (2017) — *"NoScope: Optimizing Neural Network Queries over Video at Scale"* — VLDB (cascata de difference detector + modelo especializado por câmera)
- Canel et al. (2019) — *"Scaling Video Analytics on Constrained Edge Nodes"* — USENIX ATC (FilterForward — microclassificadores por cena na borda)
- Hsieh et al. (2018) — *"Focus: Querying Large Video Datasets with Low Latency and Low Cost"* — OSDI (ingest-time indexing com CNNs especializadas)

---

## Resumo Visual

```
┌─────────────────────────────────────────────────────────┐
│           CATEGORIAS DE SELEÇÃO DE TILES                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  BASEADAS EM MUDANÇA TEMPORAL (precisam de histórico)   │
│  ├── Cat.1  Subtração de Fundo (MOG2/KNN/ViBe)     ✅  │
│  ├── Cat.2  Comparação Direta (MSE/SSIM/pHash)     🔍  │
│  ├── Cat.3  Fluxo Óptico Leve (L-K / Farnebäck)   🔍  │
│  └── Cat.4  Domínio Comprimido (MVs / DCT)         ✅  │
│                                                         │
│  BASEADAS EM PREDIÇÃO (usam resultado do YOLO)          │
│  ├── Cat.5  Rastreamento entre Keyframes            🔍  │
│  └── Cat.7  Escalonamento por Orçamento             ✅  │
│                                                         │
│  BASEADAS EM CONTEÚDO VISUAL (frame individual)         │
│  └── Cat.6  Saliência / Atenção Visual              🔍  │
│                                                         │
│  BASEADAS EM APRENDIZADO (precisam de treino)           │
│  └── Cat.8  Classificadores Especializados          ❌  │
│             (NoScope — exige treino por câmera)         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Categorias Mais Promissoras para Explorar

Considerando seu cenário (câmera estática, borda sem GPU, caixa-preta, sem treino):

1. **Cat.2 — Comparação Direta (SSIM/pHash):** Complemento natural ou alternativa ao MOG2. Mais simples, menos estado interno. Pesquise: Wang et al. (2004) para SSIM.

2. **Cat.5 — Rastreamento entre Keyframes:** Combina perfeitamente com sua Fila de Prioridade. Pesquise: Kalman Filter + SORT/ByteTrack como complemento ao YOLO espaçado.

3. **Cat.4b — Coeficientes DCT:** Se for investir na engenharia do domínio comprimido (PyAV), vale analisar DCT junto com MVs.
