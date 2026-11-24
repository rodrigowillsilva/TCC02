from ultralytics import YOLO
import numpy as np

from pipeline.tiler import Tile
from pipeline.annotations import BBox


class YOLODetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf_thresh: float = 0.25, classes: list[int] | None = None, imgsz: int = 640):
        """
        Wrapper para o YOLO rodar a inferência nos tiles selecionados.
        
        Args:
            classes: Lista de IDs das classes permitidas (0 = pessoa no COCO). Se None, pega todas.
            imgsz: Tamanho da imagem que o YOLO vai usar na rede (640 é o padrão otimizado).
        """
        self.model = YOLO(model_name)
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        # Por padrão, filtra apenas pessoas (classe 0) para o TCC
        self.classes = classes if classes is not None else [0]

    def detect(self, tiles: list[Tile]) -> dict[int, list[BBox]]:
        """
        Roda a inferência (YOLO) em uma lista de tiles usando processamento em Batch (mais rápido).
        Retorna um dicionário onde a chave é o número (índice) do tile, e o valor é a lista de caixas detectadas.
        """
        results_dict = {}
        if not tiles:
            return results_dict

        # Processamento em lote (Batch) do Ultralytics
        images = [t.image for t in tiles]
        
        # verbose=False para não sujar o terminal
        # classes=self.classes para descartar carros/objetos
        # imgsz=self.imgsz previne upscale desnecessário (otimização de CPU)
        results = self.model(images, verbose=False, conf=self.conf_thresh, classes=self.classes, imgsz=self.imgsz)

        for tile, result in zip(tiles, results):
            boxes = []
            # result.boxes tem as caixas detectadas na imagem daquele tile específico
            for box in result.boxes:
                # Coordenadas relativas ao tile (0 a 320)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                
                boxes.append(BBox(
                    x=int(x1), y=int(y1), 
                    w=int(x2 - x1), h=int(y2 - y1),
                    label=f"{class_name} {conf:.2f}",
                    class_name=class_name,
                    conf=conf,
                    color=(0, 255, 255) # Ciano/Amarelo
                ))
            results_dict[tile.index] = boxes

        return results_dict

