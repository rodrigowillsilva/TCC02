from dataclasses import dataclass
import numpy as np


@dataclass
class Tile:
    """Representa um pedaço fatiado do frame original."""
    image: np.ndarray
    index: int
    x: int
    y: int
    w: int
    h: int


class Tiler:
    def __init__(self, tile_size: tuple[int, int], grid_size: tuple[int, int]):
        """
        Fatia o frame em uma grade fixa, calculando o overlap automaticamente.
        
        Args:
            tile_size: (largura, altura) em pixels. Ex: (320, 320)
            grid_size: (colunas, linhas). Ex: (5, 3)
        """
        self.tile_w, self.tile_h = tile_size
        self.cols, self.rows = grid_size

    def split(self, frame: np.ndarray) -> list[Tile]:
        frame_h, frame_w = frame.shape[:2]

        if frame_w < self.tile_w or frame_h < self.tile_h:
            raise ValueError(f"Frame ({frame_w}x{frame_h}) é menor que o tile ({self.tile_w}x{self.tile_h})")

        # np.linspace calcula as posições iniciais distribuindo o overlap de forma igual
        # O último tile sempre termina exatamenta na borda: frame_w - tile_w
        x_positions = np.linspace(0, frame_w - self.tile_w, self.cols, dtype=int) if self.cols > 1 else [0]
        y_positions = np.linspace(0, frame_h - self.tile_h, self.rows, dtype=int) if self.rows > 1 else [0]

        tiles = []
        index = 0

        for y in y_positions:
            for x in x_positions:
                # O recorte tem sempre o tamanho exato de tile_w x tile_h
                crop = frame[y : y + self.tile_h, x : x + self.tile_w]
                
                tiles.append(Tile(
                    image=crop,
                    index=index,
                    x=int(x),
                    y=int(y),
                    w=self.tile_w,
                    h=self.tile_h
                ))
                index += 1

        return tiles

