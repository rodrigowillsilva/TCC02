import numpy as np
from typing import List, Dict
from .base import DecisionBlock, Tile


class SchedulerDecision(DecisionBlock):
    def __init__(self, budget: int = 8, hunger_weight: float = 1.0, heat_weight: float = 5.0, decay: float = 0.9):
        """
        Orquestrador de Seleção usando Fila de Prioridade (Temporal + Histórico).
        
        Args:
            budget: Cota máxima de tiles que podem ser enviados para o YOLO por frame.
            hunger_weight: Peso da "Fome" (tiles esquecidos por muito tempo ganham pontos).
            heat_weight: Peso do "Calor" (tiles onde pessoas foram achadas ganham muitos pontos).
            decay: Taxa de esquecimento do calor (0.9 = perde 10% do calor por frame).
        """
        self.budget = budget
        self.hunger_weight = hunger_weight
        self.heat_weight = heat_weight
        self.decay = decay
        
        # Metadados para o JSON
        self.config = {
            "method": "SCHEDULER",
            "budget": budget,
            "hunger_weight": hunger_weight,
            "heat_weight": heat_weight,
            "decay": decay
        }
        
        # Estados
        self.hunger: Dict[int, int] = {}      # Índice do Tile -> Frames sem visitar
        self.heat_map: Dict[int, float] = {}  # Índice do Tile -> Pontuação de detecções recentes
        
    def select(self, frame: np.ndarray, tiles: List[Tile], frame_index: int) -> List[int]:
        # Inicializa o estado para tiles novos (primeiro frame)
        for t in tiles:
            if t.index not in self.hunger:
                self.hunger[t.index] = 0
                self.heat_map[t.index] = 0.0

        scores = []
        # Calcula a nota (Score) de cada tile
        for t in tiles:
            fome = self.hunger[t.index]
            calor = self.heat_map[t.index]
            
            # FÓRMULA MATEMÁTICA DA TESE: Score = (Fome * alpha) + (Calor * beta)
            score = (fome * self.hunger_weight) + (calor * self.heat_weight)
            scores.append((score, t.index))

        # Ordena do maior Score para o menor
        scores.sort(reverse=True, key=lambda x: x[0])
        
        # O Orquestrador aprova apenas os Top N tiles baseados no orçamento (budget)
        selected_indices = [idx for score, idx in scores[:self.budget]]
        
        # Atualiza a fome para o próximo frame
        for t in tiles:
            if t.index in selected_indices:
                self.hunger[t.index] = 0 # Foi visitado, a fome zera
            else:
                self.hunger[t.index] += 1 # Não foi visitado, fica com mais fome
                
        return selected_indices

    def update_state(self, selected_tiles: List[Tile], tile_detections: Dict[int, List]):
        """
        Feedback Loop: O PipelineRunner chama isso depois do YOLO processar os escolhidos.
        É aqui que o Orquestrador aprende onde tem gente.
        """
        # 1. Resfria todos os tiles da tela (esquecimento natural)
        for idx in self.heat_map:
            self.heat_map[idx] *= self.decay 
            
        # 2. Esquenta brutalmente os tiles onde o YOLO encontrou pessoas neste frame
        for tile in selected_tiles:
            # Pega as detecções do dicionário para este tile específico
            detections = tile_detections.get(tile.index, [])
            
            # Soma 1 ponto de calor para cada pessoa achada no tile
            self.heat_map[tile.index] += len(detections)
