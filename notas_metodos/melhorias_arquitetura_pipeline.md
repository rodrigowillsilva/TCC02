# Melhorias na Arquitetura Global do Pipeline

**Ponto Atual do Projeto:** 
A base dos métodos de seleção de recortes (Tile Selectors: Baseline, KNN, Frame Differencing) foi estabelecida e validada. O TCC agora avança para uma nova fase: estruturar melhorias e ferramentas **em torno** desses métodos. O objetivo é desenvolver módulos agnósticos no pipeline que estabilizem o sistema, garantam a economia de CPU e melhorem a precisão visual, *independentemente* de qual método seletor esteja sendo utilizado.

---

## 1. Módulo Intermediário: Orquestrador de Orçamento (Limitador)
**Posição no Pipeline:** Entre o *Bloco de Decisão* e a *Inferência (YOLO)*.

**O Conceito:**
Nenhum método seletor é à prova de falhas em cenários ruidosos (ex: ventania forte ativando o Frame Diff na tela inteira). Para que o sistema seja realmente viável em hardware Edge, o pipeline deve possuir um funil limitador.
- **Como funciona:** O Seletor (qualquer um deles) analisa o frame e "sugere" uma lista de tiles com movimento. O Orquestrador intercepta essa lista e aplica um Orçamento Máximo (ex: max 2 tiles por frame).
- **Inteligência da Fila:** Se o Seletor sugerir mais tiles do que o orçamento permite, o Orquestrador utiliza mecânicas de Fila de Prioridade (Avaliando Tempo de Fome e Histórico de Ocupação) para decidir quais tiles daquela lista são os mais críticos.

## 2. Módulo de Pós-Processamento: Estabilizador de Saída (Persistência e Interpolação)
**Posição no Pipeline:** Entre a *Inferência (YOLO)* e a *Geração de Resultados/Vídeo*.

**O Conceito:**
Técnicas de supressão de frames (pular tiles para economizar CPU) naturalmente geram "falsos negativos" momentâneos. Se uma pessoa caminhar para um tile que o Orquestrador cortou naquele frame, o YOLO não a verá. Além disso, a própria rede neural sofre de "flicker" (falhas de detecção intermitentes). O pipeline precisa de um estabilizador agnóstico para resolver isso.
- **Persistência Temporária:** Ao perder a detecção de uma pessoa no frame atual, o pipeline mantém a caixa delimitadora (bounding box) ativa na memória por uma pequena janela de tolerância (ex: 3 a 5 frames).
- **Interpolação de Movimento:** Utilizando as detecções dos frames anteriores, o pipeline calcula o vetor de movimento (velocidade horizontal e vertical) do alvo. Nos frames em que o tile do alvo for ignorado pela IA, o pipeline projeta matematicamente a nova posição da caixa na tela ($Posição_{nova} = Posição_{anterior} + Velocidade$).
- **Benefício:** A carga de processamento da IA cai drasticamente, mas o resultado final (JSON / Vídeo) continua fluido e sem perdas de rastreio visíveis, mascarando completamente o fato de que a IA está operando "cega" em várias partes da tela.

---

## 3. Anotações Adicionais e Ideias Futuras
*(Espaço reservado para documentar pesquisas em andamento, novos conceitos de estabilização do pipeline e links de referência)*

- [ ] 
- [ ] 
- [ ] 
