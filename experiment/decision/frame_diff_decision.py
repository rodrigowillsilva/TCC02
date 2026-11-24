import cv2
import numpy as np
from typing import List
from .base import DecisionBlock, Tile


class FrameDiffDecision(DecisionBlock):
    def __init__(self, debug: bool = False, min_pixels: int = 150, threshold: int = 25, scale_factor: float = 0.5, frame_step: int = 3):
        """
        Selecionador usando Diferença de Quadros com Buffer Rolante (N-Frame Differencing).
        
        Args:
            debug: Se True, mostra a máscara.
            min_pixels: Quantidade mínima de pixels brancos.
            threshold: Diferença mínima de cor.
            scale_factor: Fator de redução.
            frame_step: Distância (em frames) para a comparação. Ex: 3 = compara o frame atual com o de 3 frames atrás.
        """
        self.debug = debug
        self.min_pixels_original = min_pixels
        self.threshold = threshold
        self.scale_factor = scale_factor
        self.frame_step = frame_step
        
        self.scaled_min_pixels = int(min_pixels * (scale_factor ** 2))
        
        # Buffer de memória rolante para evitar o efeito "piscando"
        self.frame_buffer = []
        self.window_name = "Debug Frame Diff Mask"
        
        self.config = {
            "method": "FRAMEDIFF",
            "threshold": threshold,
            "min_pixels_original": min_pixels,
            "scale_factor": scale_factor,
            "frame_step": frame_step
        }

    def select(self, frame: np.ndarray, tiles: List[Tile], frame_index: int) -> List[int]:
        if self.scale_factor != 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            small_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            small_frame = frame
            
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
        
        # Coloca o frame atual no final da fila de memória
        self.frame_buffer.append(blurred)
        
        # Se ainda não temos frames suficientes no histórico, aguardamos
        if len(self.frame_buffer) <= self.frame_step:
            self.last_mask = np.zeros_like(blurred)
            return []
            
        # Pega o quadro mais antigo da fila para comparar (ex: 3 frames atrás)
        past_frame = self.frame_buffer.pop(0)
            
        # A Matemática: Subtração Absoluta entre hoje e 3 frames atrás
        diff = cv2.absdiff(past_frame, blurred)
        
        _, fg_mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
        self.last_mask = fg_mask
        
        # Exibe a máscara se estiver no modo debug
        if self.debug:
            cv2.imshow(self.window_name, fg_mask)
            
        # 7. Avalia qual tile deve ir pro YOLO
        selected_indices = []
        for tile in tiles:
            # Converte as coordenadas do tile original para a escala reduzida
            tx = int(tile.x * self.scale_factor)
            ty = int(tile.y * self.scale_factor)
            tw = int(tile.w * self.scale_factor)
            th = int(tile.h * self.scale_factor)
            
            # Recorta a máscara na região do tile
            tile_mask = fg_mask[ty:ty+th, tx:tx+tw]
            
            # Conta os pixels brancos
            white_pixels = cv2.countNonZero(tile_mask)
            
            if white_pixels > self.scaled_min_pixels:
                selected_indices.append(tile.index)
                
        return selected_indices
