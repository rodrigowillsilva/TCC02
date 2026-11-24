# Pendências da escrita do artigo TCC02

Arquivo vivo: tudo que falta para terminar o artigo. Atualizar a cada conversa.

## Decisões já tomadas
- Título atualizado (sem "sem GPU"): "Inferência seletiva por tiles para detecção de objetos em tempo real em dispositivos de borda". Revisar menções a "sem GPU" ao longo do artigo (Introdução, Revisão, Síntese e Conclusão) para deixar claro que a redução de inferências do YOLO beneficia qualquer hardware de borda (com ou sem GPU), e que câmeras de segurança reais hoje usam algoritmos simples não-IA, sendo a viabilização de IA (2025/2026) a motivação central.
- Dados: VIRAT, vídeos em 1280x720. Grade 5x3 de tiles 320x320 (overlap por `linspace`: 80 px em x, 120 px em y). Grade escolhida para 720p; outras resoluções exigiriam outra grade.
- Métodos na metodologia (lista): Baseline, FrameDiff, KNN, MOG2, Hash (Wavelet). ViBe só citado (lento em Python, equivalente ao MOG2).
- Baseline: YOLOv8n em todos os tiles (`decision=None`).
- Scheduler: fora do artigo; Rodrigo vai retirá-lo do código.
- Fora do artigo (talvez "trabalhos futuros"): tracker de velocidade, orquestrador de orçamento, modo realtime.
- Hash (árvore hierárquica, tolerância, TTL): ideia própria, sem citação por ora.
- Pseudocódigo inline (ambiente `algorithm`); código real só se curto. Decidir depois de ver como ficam.
- Introdução por último; resumo/abstract no fim (gerar com IA, conforme orientador).
- Tom: impessoal. Estrutura das seções definida pelo Prof. Enzo (I Introdução, II Revisão, III Metodologia, IV Resultados, V Conclusão).

## Estado das seções
- [x] II Revisão Bibliográfica (adaptada do TCC01 e limpa das menções restritivas a "sem GPU")
- [x] III Metodologia (completa e alinhada aos parâmetros finais: III-A Visão geral + Fig. 1 + Alg. 1; III-B Tiles Eq. 1; III-C Estratégias + Alg. 2; III-D Inferência e consolidação Eq. 4)
- [x] IV Resultados e Discussão (completa com dados reais da suíte oficial de 20 vídeos VIRAT / 36.355 quadros / 149.719 instâncias: IV-A Base de dados; IV-B Métricas Eq. 5; IV-C Hardware, calibração/ablação + Tabela 1 de ablação + Tabela 2 de parâmetros; IV-D Resultados globais + Tabela 3 consolidada e discussão detalhada)
- [x] V Conclusão (completa: retomada explícita dos 3 objetivos, síntese dos resultados KNN vs. Hash Wavelet, papel dos mecanismos auxiliares, ponderação equilibrada sobre escopo experimental e trabalhos futuros)
- [x] I Introdução (completa: contextualização + motivação realista sobre viabilização recente de IA em câmeras + os 3 objetivos específicos espelhados na conclusão)
- [x] Resumo / Palavras-chave (Resumo escrito seguindo estritamente as diretrizes do template IEEE Access/UNIFEI: parágrafo único, ~245 palavras, autossuficiente, sem siglas não definidas, equações ou citações; versão em inglês disponível caso o orientador solicite)
- [ ] Agradecimentos (deixado para o final)

## Código / modificações
- [ ] Remover Scheduler do código.
- [ ] Remover tracker e orquestrador do pipeline de experimentos (manter só se usado em demo/realtime).
- [ ] Rodada de testes variando parâmetros de cada método (limiares, `min_pixels`, `scale_factor`, `frame_step`, etc.) e tipo de hash (VALUE/EDGE/HYBRID/WAVELET), tolerância e TTL. Depois: Claude relê o código e atualiza a III-C com os valores finais.
- [ ] Conferir se o `compare.py` calcula as métricas finais desejadas (hoje: precision/recall por IoU 0.4, latência e CPU; faltam mAP, F1, FPS, RAM se forem usados).
- [ ] Consolidador: confirmar que a descrição (IoU + IoM + costura em borda) é a versão final.

## Experimentos / resultados (Seção IV)
- [ ] Definir o conjunto de vídeos VIRAT usados e como o ground truth é gerado (qual modelo YOLO, em qual configuração).
- [ ] Rodar o test suite completo (todos os vídeos x todos os métodos) com `WARMUP_FRAMES` definido.
- [ ] Definir métricas finais: eficiência (FPS, tempo de inferência, latência, CPU, RAM, % de tiles inferidos/reaproveitados) e acurácia (precision, recall, F1, mAP, IoU) vs baseline.
- [ ] Observação do Rodrigo (a confirmar com os dados): KNN com acurácia próxima de 100% do baseline em alguns vídeos; hash o mais rápido, com cerca de 80% de acurácia relativa.
- [ ] Tabelas e gráficos: formatar para coluna ou largura de página.
- [ ] Demonstração em hardware limitado/realtime (objetivo 6 do TCC01), se for incluída.
- [ ] Análise de falhas/limitações (falsos negativos por tiles pulados, objetos parados, etc.).

## Referências
- [ ] Adicionar ao `referencias.bib`: Zauner (hash perceptual, 2010) e Haar wavelet (se citados). Já adicionados na III-C, mas CONFERIR dados: Zivkovic 2004 (MOG2), Zivkovic & van der Heijden 2006 (KNN), OpenCV (Bradski 2000).
- [ ] Revisar chaves/preprints no `.bib` (alguns são arXiv/Preprints, sinalizar).

## Formatação / template
- [ ] Diagrama da arquitetura (TikZ, largura de página) na III-A (há `% TODO` no código).
- [ ] Regras de figuras no template de duas colunas (`\Figure` escolhe coluna x página pela largura).
- [ ] Legendas do template estão em inglês (`FIGURE`/`TABLE`); perguntar ao Prof. Enzo se podem ficar em português.
- [ ] Limite de páginas do artigo (verificar com o orientador).
- [ ] Data da defesa no cabeçalho (`\dataDaDefesa`), e-mail do autor correspondente (comentado), `\corresp`.
- [ ] Remover biografias/EOD do template se não forem exigidas (já removidas; confirmar).
- [ ] Revisão final: siglas definidas na primeira ocorrência, "Fig." no texto, ortografia, consistência de tempo verbal.

## Para conversar com o orientador (Prof. Enzo)
- [ ] Idioma dos rótulos/legendas.
- [ ] Scheduler fora e tracker/orquestrador em trabalhos futuros: ok?
- [ ] Tamanho esperado e nível de detalhe da metodologia (código/pseudocódigo).
