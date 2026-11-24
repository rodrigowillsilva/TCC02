import cv2
import numpy as np

from pipeline.tiler import Tile
from .base import DecisionBlock

PIXEL_THRESHOLD = 1000  # Quantos pixels em movimento para considerar o tile "ativo"

class BGSDecision(DecisionBlock):
    def __init__(self, debug: bool = False, history: int = 500, var_threshold: float = 50.0, min_pixels: int = 250):
        """
        Método de decisão baseado em Subtração de Fundo (MOG2).
        
        Args:
            debug: Se True, abre uma janela própria mostrando a máscara preta e branca.
            history: Tamanho do histórico do MOG2.
            var_threshold: Aumentado para 50.0 (ignora variações leves como ruído de compressão).
            min_pixels: Aumentado para 250 (evita que pequenas folhas ativem o tile).
        """
        self.debug = debug
        self.min_pixels = min_pixels
        self.scale_factor = 1.0
        self.window_name = "Debug BGS Mask"
        self.is_initialized = False
        self.last_mask = None
        self.config = {
            "method": "MOG2",
            "history": history,
            "var_threshold": var_threshold,
            "min_pixels": min_pixels,
        }
        
        # MOG2 lida com sombras (detectShadows=True por padrão)
        self.bgs = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold)

    def select(self, frame: np.ndarray, tiles: list[Tile], frame_index: int) -> list[int]:
        # --- PRÉ-PROCESSAMENTO ---
        # 1. Converte para tons de cinza (remove ruído de cor/chroma do sensor)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # (CLAHE removido pois estava amplificando os artefatos de compressão do MP4)
        
        # 2. Aplica um Blur mais agressivo (7x7)
        blurred_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)
        
        # --- AQUECIMENTO (WARM-UP) NO PRIMEIRO FRAME ---
        if not self.is_initialized:
            for _ in range(10):
                self.bgs.apply(blurred_frame)
            self.is_initialized = True

        # --- SUBTRAÇÃO DE FUNDO ---
        # 3. Aplica o MOG2 na imagem tratada
        fg_mask = self.bgs.apply(blurred_frame)
        
        # 4. Limpa sombras residuais e tons de cinza. Queremos apenas movimento puro (255)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # 5. Operação Morfológica mais forte (7x7): Elimina ruídos ainda maiores
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        self.last_mask = fg_mask

        selected_indices = []
        
        # Cria uma cópia APENAS para desenhar o debug. 
        # Desenhar na fg_mask original contamina os tiles seguintes por causa do overlap!
        debug_mask = fg_mask.copy() if self.debug else None

        # 5. Avalia cada tile
        for t in tiles:
            # Recorta a máscara na mesma posição do tile
            tile_mask = fg_mask[t.y : t.y + t.h, t.x : t.x + t.w]
            
            # Conta os pixels brancos
            moving_pixels = cv2.countNonZero(tile_mask)
            
            # Só ativa se o movimento for considerável (maior que min_pixels)
            if moving_pixels > self.min_pixels:
                selected_indices.append(t.index)
                
            # Apenas para debug: desenha as informações NA CÓPIA
            if debug_mask is not None and moving_pixels > self.min_pixels:
                cv2.rectangle(debug_mask, (t.x, t.y), (t.x + t.w, t.y + t.h), (255, 255, 255), 2)
                cv2.putText(debug_mask, f"{moving_pixels} px", (t.x + 5, t.y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 6. Exibe o debug de forma isolada, sem sujar o pipeline principal
        if debug_mask is not None:
            h, w = debug_mask.shape[:2]
            cv2.imshow(self.window_name, cv2.resize(debug_mask, (w // 2, h // 2)))

        return selected_indices

