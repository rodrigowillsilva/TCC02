---
# yaml-language-server: $schema=schemas\page.schema.json
Object type:
    - Page
Backlinks:
    - estudo-de-metodos.md
Creation date: "2026-08-16T23:44:49Z"
Created by:
    - Rodrigo
id: bafyreieh3ahaon2ymlk545aoy4mosoegyeaphfqae6flw64uy7hchnll3e
---
# Escalonamento Dinâmico por Fila de Prioridade   
   
**Conceito Central:**   
Alocação de inferência baseada estritamente em um orçamento de tempo ou limite fixo de execuções por quadro (ex.: processar no máximo $X$*tiles* por quadro). A seleção ocorre por uma fila reordenada matematicamente a cada iteração, garantindo distribuição de recursos sem dependência de rastreadores externos.   
**Heurística de Pontuação:**   
A prioridade de um *tile*$i$ no quadro atual é definida por uma equação linear que avalia o histórico recente da própria rede neural:   
$\text{Score}\_i = (W\_1 \times A\_i) + (W\_2 \times D\_i)$   
1. **$A\_i$ (Fome Temporal):** O número de quadros desde a última inferência do *tile*$i$. Cresce $+1$ a cada quadro em que o *tile* é ignorado. Este fator empurra *tiles* inativos para o topo da fila gradativamente, garantindo atualização e prevenindo pontos cegos prolongados.   
2. **$D\_i$ (Presença Histórica):** Um peso ativado caso a última inferência do YOLO naquele *tile* tenha detectado objetos. Pode ser tratado como um booleano (1 ou 0) ou como a contagem bruta de objetos encontrados. Retém a atenção do pipeline nas áreas ativas da cena.   
   
**Mecânica de Execução:**   
1. **Inicialização:** Todos os *tiles* começam com score equivalente, resultando em uma varredura sequencial nos primeiros quadros.   
2. **Atualização:** A cada novo quadro, o score de todos os *tiles* é calculado e ordenado.   
3. **Inferência:** Os *tiles* no topo da fila são enviados ao YOLO até o limite predefinido.   
4. **Feedback:** Para os *tiles* processados, $A\_i$ é zerado e $D\_i$ recebe o resultado atual da rede. Para os *tiles* que ficaram na fila, $A\_i$ sofre incremento e $D\_i$ mantém o valor anterior.   
