---
# yaml-language-server: $schema=schemas\page.schema.json
Object type:
    - Page
Backlinks:
    - estudo-de-metodos.md
Creation date: "2026-08-24T22:54:38Z"
Created by:
    - Rodrigo
id: bafyreid634m6qa4v7q4rxqup64qw4w5eqynxhsfdokof263pnrkqs6gx44
---
# Inferência Condicional por Subtração de Fundo   
**Conceito Central:**
Utilização de algoritmos clássicos de modelagem e subtração de fundo (como MOG2, ViBe, KNN, etc.) operando no nível do recorte (*tile*) para atuar como um filtro rápido de decisão. O objetivo principal do método é identificar e separar os *tiles* que apresentam mudanças ativas em relação ao fundo estático da cena, submetendo ao detector pesado (YOLO) exclusivamente as regiões onde a máscara de primeiro plano for ativada.   
**Melhorias e Heurísticas Complementares:**
As técnicas abaixo são módulos adicionais que podem ser acoplados ao extrator de fundo básico para mitigar suas falhas e otimizar o desempenho em cenários reais de hardware de borda:   
- **Máquina de Estados por "Esfriamento" da Máscara:** Para resolver o problema de objetos parados que são absorvidos pelo fundo ao longo do tempo. O sistema monitora a queda gradativa da quantidade de pixels ativos (esfriamento); se a máscara esfria após um pico de ativação, assume-se que o alvo está estático. O YOLO é desativado no *tile* e a última predição é reaproveitada.   
- **Veto de Iluminação Global:** Prevenção contra saturação de processamento. Se a máscara de fundo ativar uma quantidade massiva de *tiles* simultaneamente (ex.: nuvens, ajuste automático da câmera), o evento é interpretado como ruído ambiental. A inferência no YOLO é vetada no quadro atual para dar tempo ao algoritmo de fundo para se adaptar à nova luz.   
- **Ativação de Vizinhança Espacial:** Prevenção contra perda de continuidade de alvos velozes. *Tiles* adjacentes àqueles que detectaram movimento recebem um alerta (redução de limiar ou ativação preventiva) para garantir que um objeto cruzando as fronteiras dos recortes seja detectado a tempo.   
- **Sincronização Periódica (Refresh):** Medida de segurança contra caixas delimitadoras "fantasmas". Execução forçada e esparsa do YOLO em *tiles* marcados como inativos (seja por *Round-Robin* ou quadros-chave) para corrigir derivas, confirmar se o alvo estático ainda está lá e limpar falsos positivos residuais.   
