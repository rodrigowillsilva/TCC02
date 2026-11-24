from abc import ABC, abstractmethod
import numpy as np

from pipeline.tiler import Tile


class DecisionBlock(ABC):
    """
    Interface base para todos os métodos de decisão de tiles.
    Qualquer método novo (Fila, MVs, BGS) deve herdar desta classe.
    """
    
    @abstractmethod
    def select(self, frame: np.ndarray, tiles: list[Tile], frame_index: int) -> list[int]:
        """
        Analisa o frame atual e decide quais tiles devem ser processados pelo YOLO.
        
        Args:
            frame: A imagem completa original (útil para métodos globais como MOG2).
            tiles: Lista de objetos Tile gerados pelo Tiler.
            frame_index: Número do frame atual no vídeo (útil para lógicas temporais).
            
        Returns:
            Lista com os índices (int) dos tiles selecionados.
        """
        pass

