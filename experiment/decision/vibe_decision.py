import cv2
import numpy as np

from pipeline.tiler import Tile
from .base import DecisionBlock
from .vibe import ViBeSequential


class ViBeDecision(DecisionBlock):
    def __init__(self, debug: bool = False, min_pixels: int = 150):
        """
        Método de decisão baseado no extrator de fundo ViBe.
        
        Args:
            debug: Se True, abre uma janela própria mostrando a máscara preta e branca.
            min_pixels: Quantidade mínima de pixels em movimento para ativar o tile.
        """
        self.debug = debug
        self.min_pixels = min_pixels
        self.window_name = "Debug ViBe Mask"
        
        # O ViBe precisa do primeiro frame para inicializar, então começamos com None
        self.vibe = None

    def select(self, frame: np.ndarray, tiles: list[Tile], frame_index: int) -> list[int]:
        # --- PRÉ-PROCESSAMENTO ---
        # Converte para tons de cinza e faz um blur pra reduzir ruídos de compressão
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        blurred_frame = gray_frame

        # --- INICIALIZAÇÃO DO VIBE (Apenas no frame 0) ---
        if self.vibe is None:
            h, w = blurred_frame.shape[:2]
            # O ViBe precisa ser inicializado com a altura, largura e o primeiro frame
            self.vibe = ViBeSequential(h, w, blurred_frame)

        # --- EXECUÇÃO DO VIBE ---
        # h_frame, w_frame = frame.shape[:2]
        # segmentation_map = np.empty((h_frame, w_frame), dtype=np.uint8)
        segmentation_map = np.empty(blurred_frame.shape, dtype=np.uint8)
        self.vibe.segmentation(blurred_frame, segmentation_map)
        
        # Manda o ViBe atualizar o modelo de fundo usando a máscara que ele acabou de gerar
        self.vibe.update(blurred_frame, segmentation_map)

        # --- LIMPEZA DA MÁSCARA ---
        # Operação Morfológica (OPEN) pra tirar ruídos pequenininhos (pingos)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(segmentation_map, cv2.MORPH_OPEN, kernel)

        selected_indices = []
        
        # Máscara separada para rabiscar textos (debug) sem contaminar a contagem
        debug_mask = fg_mask.copy() if self.debug else None

        # --- AVALIAÇÃO DOS TILES ---
        for t in tiles:
            # Recorta a máscara
            tile_mask = fg_mask[t.y : t.y + t.h, t.x : t.x + t.w]
            
            # Conta os pixels detectados
            moving_pixels = cv2.countNonZero(tile_mask)
            
            if moving_pixels > self.min_pixels:
                selected_indices.append(t.index)
                
            if debug_mask is not None and moving_pixels > self.min_pixels:
                cv2.rectangle(debug_mask, (t.x, t.y), (t.x + t.w, t.y + t.h), (255, 255, 255), 2)
                cv2.putText(debug_mask, f"{moving_pixels} px", (t.x + 5, t.y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if debug_mask is not None:
            h, w = debug_mask.shape[:2]
            cv2.imshow(self.window_name, cv2.resize(debug_mask, (w // 2, h // 2)))

        return selected_indices

