import cv2
import numpy as np

from pipeline.tiler import Tile
from .base import DecisionBlock


class KNNDecision(DecisionBlock):
    def __init__(self, debug: bool = False, history: int = 500, dist2Threshold: float = 200.0, 
                 min_pixels: int = 250, scale_factor: float = 0.5):
        """
        Método de decisão baseado em KNN (K-Nearest Neighbors).
        
        Args:
            debug: Mostra a máscara de debug.
            history: Tamanho do histórico.
            dist2Threshold: Limiar de distância (abaixamos para 200 para ficar MAIS sensível a movimentos sutis).
            min_pixels: Mínimo de pixels (na escala original) para ativar o tile.
            scale_factor: Fator de redução (0.5 = metade do tamanho).
        """
        self.debug = debug
        self.scale_factor = scale_factor
        
        # Metadados para o JSON
        self.config = {
            "method": "KNN",
            "history": history,
            "dist2Threshold": dist2Threshold,
            "min_pixels_original": min_pixels,
            "scale_factor": scale_factor
        }
        
        self.scaled_min_pixels = int(min_pixels * (scale_factor ** 2))
        self.window_name = "Debug KNN Mask"
        self.is_initialized = False
        
        self.bgs = cv2.createBackgroundSubtractorKNN(history=history, dist2Threshold=dist2Threshold)

    def select(self, frame: np.ndarray, tiles: list[Tile], frame_index: int) -> list[int]:
        if self.scale_factor != 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            small_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            small_frame = frame
            
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # --- AQUECIMENTO (WARM-UP) ---
        # Se for o primeiro frame, alimenta o algoritmo 10x para ele decorar o fundo instantaneamente.
        if not self.is_initialized:
            for _ in range(10):
                self.bgs.apply(blurred)
            self.is_initialized = True
        
        fg_mask = self.bgs.apply(blurred)
        
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Em vez de apagar ruídos (MORPH_OPEN), nós "engordamos" os pixels pra não perder ninguém de longe
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
        self.last_mask = fg_mask

        selected_indices = []
        debug_mask = fg_mask.copy() if self.debug else None

        # --- 4. AVALIAÇÃO DOS TILES NA MÁSCARA REDUZIDA ---
        for t in tiles:
            # Transforma as coordenadas do recorte para a escala pequena
            sx = int(t.x * self.scale_factor)
            sy = int(t.y * self.scale_factor)
            sw = int(t.w * self.scale_factor)
            sh = int(t.h * self.scale_factor)
            
            # Recorta a máscara na escala pequena
            tile_mask = fg_mask[sy : sy + sh, sx : sx + sw]
            moving_pixels = cv2.countNonZero(tile_mask)
            
            if moving_pixels > self.scaled_min_pixels:
                selected_indices.append(t.index)
                
            if debug_mask is not None and moving_pixels > self.scaled_min_pixels:
                cv2.rectangle(debug_mask, (sx, sy), (sx + sw, sy + sh), (255, 255, 255), 2)
                # Texto menor já que a imagem encolheu
                cv2.putText(debug_mask, f"{moving_pixels}", (sx + 2, sy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if debug_mask is not None:
            # Mostra a máscara na tela (ela vai parecer pequenininha, mas é de propósito!)
            cv2.imshow(self.window_name, debug_mask)

        return selected_indices

