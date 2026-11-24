---
# yaml-language-server: $schema=schemas\page.schema.json
Object type:
    - Page
Backlinks:
    - estudo-de-metodos.md
Creation date: "2026-08-24T23:46:06Z"
Created by:
    - Rodrigo
id: bafyreifs7xeb44apocz4f6r5vyxhtaazykvmti23gjv5pzcgonkordzoam
---
# Seleção por Vetores de Movimento do Codec (Compressed Motion Vectors)   
**Conceito Central:**
Exploração direta dos metadados gerados pela compressão de vídeo (codecs como H.264 ou HEVC) na etapa de codificação, um processo inerente a pipelines de vídeo na borda. O método descarta a análise de pixels decodificados na CPU, lendo as matrizes matemáticas (vetores de movimento) que indicam o deslocamento dos macroblocos entre os quadros.   
**Mecânica de Execução:**   
1. O fluxo de vídeo é interceptado no nível do *stream* comprimido, antes da conversão para matrizes RGB.   
2. Os vetores de movimento são extraídos dos quadros preditivos (P-frames e B-frames).   
3. A malha de macroblocos (ex.: $16\times16$ pixels) do codec é geometricamente mapeada para a grade de recortes (*tiles*) estabelecida na arquitetura.   
4. *Tiles* contendo vetores de movimento com magnitude superior ao limite de ruído de compressão são marcados como ativos e submetidos ao YOLO.   
   
**Vantagem Arquitetural:**   
- **Custo Computacional Virtualmente Nulo para Decisão:** O cálculo de movimento já foi realizado pelo hardware da câmera para viabilizar a transmissão. O bloco de decisão atua sem processamento matricial adicional de imagens, eliminando o gargalo de CPU para a etapa de filtragem.   
   
**Fraqueza Analítica e Gargalos de Engenharia:**   
- **Quebra de Paradigma de Implementação:** A biblioteca OpenCV, estipulada para a implementação do pipeline base, não preserva metadados de compressão na função `VideoCapture`, entregando apenas os quadros decodificados. É obrigatória a integração de *wrappers* do FFmpeg (ex.: `PyAV` em Python ou `mpegflow` em C++) para extrair os vetores, o que eleva a complexidade de software.   
- **Cegueira Periódica (I-Frames):** O fluxo de vídeo insere quadros-chave independentes (I-frames) periodicamente para redefinir o estado do vídeo. Como os I-frames não contêm vetores de movimento, o bloco de decisão fica cego durante sua passagem, exigindo um mecanismo de *fallback* (como processamento integral ou retenção forçada do estado anterior).   
- **Ruído de Compressão vs. Realidade Física:** Codecs calculam vetores para maximizar a compressão de dados, não para representar o fluxo óptico real. Variações mínimas de iluminação em superfícies texturizadas podem gerar vetores grandes (falsos positivos), forçando inferências inúteis no detector pesado.   
