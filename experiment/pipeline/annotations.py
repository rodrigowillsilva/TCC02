from dataclasses import dataclass, field


@dataclass
class BBox:
    """Um retângulo para desenhar no frame. Serve pra YOLO, bordas de tiles, ou qualquer coisa."""
    x: int
    y: int
    w: int
    h: int
    label: str = ""
    class_name: str = ""
    conf: float = 0.0
    color: tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2


@dataclass
class FrameAnnotations:
    """Tudo que precisa ser desenhado em cima do frame."""
    boxes: list[BBox] = field(default_factory=list)

