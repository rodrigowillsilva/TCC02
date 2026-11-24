import cv2
import numpy as np

from .annotations import BBox, FrameAnnotations


class FrameAnnotator:
    """Recebe um frame e anotações, devolve o frame com tudo desenhado."""

    def draw(self, frame: np.ndarray, annotations: FrameAnnotations) -> np.ndarray:
        output = frame.copy()

        for box in annotations.boxes:
            pt1 = (box.x, box.y)
            pt2 = (box.x + box.w, box.y + box.h)
            cv2.rectangle(output, pt1, pt2, box.color, box.thickness)

            if box.label:
                font_scale = 0.5
                text_thickness = 1
                (tw, th), _ = cv2.getTextSize(box.label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)

                # Sempre desenha o rótulo para DENTRO da caixa para ficar padronizado
                cv2.rectangle(output, (box.x, box.y), (box.x + tw + 4, box.y + th + 6), box.color, -1)
                cv2.putText(output, box.label, (box.x + 2, box.y + th + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness)

        return output

