from pipeline.tiler import Tile
from pipeline.annotations import BBox


class Consolidator:
    def __init__(self, iou_threshold: float = 0.40, iom_threshold: float = 0.50):
        """
        Responsável por mapear caixas (locais) do tile para o frame inteiro (global),
        unificando detecções duplicadas ou fatiadas nas bordas de overlap entre tiles.

        Usa dois critérios combinados (padrão SAHI para inferência em tiles):
        - IoU (Intersection over Union): para caixas de tamanho parecido.
        - IoM (Intersection over Minimum Area): para quando uma pessoa é cortada pela
          borda de um tile (gerando meia-caixa num tile e caixa inteira no vizinho).
        """
        self.iou_threshold = iou_threshold
        self.iom_threshold = iom_threshold

    @staticmethod
    def _compute_iou_and_iom(box_a: list[int], box_b: list[int]) -> tuple[float, float]:
        xa1, ya1, wa, ha = box_a
        xb1, yb1, wb, hb = box_b
        xa2, ya2 = xa1 + wa, ya1 + ha
        xb2, yb2 = xb1 + wb, yb1 + hb

        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0, 0.0

        inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
        area_a = float(wa * ha)
        area_b = float(wb * hb)

        min_area = min(area_a, area_b)
        union_area = area_a + area_b - inter_area

        iou = (inter_area / union_area) if union_area > 0 else 0.0
        iom = (inter_area / min_area) if min_area > 0 else 0.0
        return iou, iom

    @staticmethod
    def _touches_tile_border(local_box: BBox, tile: Tile, margin: int = 8) -> bool:
        """Verifica se a detecção encosta na borda do tile (pessoa potencialmente cortada ao meio)."""
        return (
            local_box.x <= margin
            or local_box.y <= margin
            or (local_box.x + local_box.w) >= (tile.w - margin)
            or (local_box.y + local_box.h) >= (tile.h - margin)
        )

    def consolidate(self, tile_results: dict[int, list[BBox]], tiles: list[Tile]) -> list[BBox]:
        if not tile_results:
            return []

        tile_map = {t.index: t for t in tiles}
        candidates = []

        # 1. Transformação Local -> Global + detecção de corte na borda do tile
        for t_idx, bboxes in tile_results.items():
            t = tile_map.get(t_idx)
            if not t:
                continue

            for b in bboxes:
                global_x = int(b.x + t.x)
                global_y = int(b.y + t.y)
                is_border = self._touches_tile_border(b, t)

                candidates.append({
                    "box": [global_x, global_y, int(b.w), int(b.h)],
                    "conf": float(b.conf),
                    "class_name": b.class_name,
                    "label": b.label,
                    "color": b.color,
                    "tile_idx": t_idx,
                    "is_border": is_border,
                })

        if not candidates:
            return []

        # Ordena pelas caixas mais confiáveis (priorizando caixas inteiras que não estão cortadas na borda)
        candidates.sort(key=lambda c: (not c["is_border"], c["conf"]), reverse=True)

        consolidated: list[BBox] = []
        used = [False] * len(candidates)

        # 2. Agrupamento e Fusão (IoU + IoM para costurar pessoas divididas entre dois tiles)
        for i, base in enumerate(candidates):
            if used[i]:
                continue
            used[i] = True

            merged_box = list(base["box"])
            best_conf = base["conf"]

            for j in range(i + 1, len(candidates)):
                if used[j]:
                    continue

                other = candidates[j]
                iou, iom = self._compute_iou_and_iom(merged_box, other["box"])

                # Se ambas as caixas estão cortadas na borda de tiles diferentes, basta uma sobreposição menor (costura)
                both_on_seam = (
                    base["tile_idx"] != other["tile_idx"]
                    and base["is_border"]
                    and other["is_border"]
                    and (iom >= 0.25 or iou >= 0.15)
                )

                if iou >= self.iou_threshold or iom >= self.iom_threshold or both_on_seam:
                    used[j] = True
                    # Se vieram de tiles diferentes e estavam cortadas na costura do overlap,
                    # une as duas metades na caixa envolvente (envelope da pessoa inteira)
                    if base["tile_idx"] != other["tile_idx"] and (base["is_border"] or other["is_border"]):
                        x1 = min(merged_box[0], other["box"][0])
                        y1 = min(merged_box[1], other["box"][1])
                        x2 = max(merged_box[0] + merged_box[2], other["box"][0] + other["box"][2])
                        y2 = max(merged_box[1] + merged_box[3], other["box"][1] + other["box"][3])
                        merged_box = [x1, y1, x2 - x1, y2 - y1]
                    best_conf = max(best_conf, other["conf"])

            consolidated.append(
                BBox(
                    x=merged_box[0],
                    y=merged_box[1],
                    w=merged_box[2],
                    h=merged_box[3],
                    label=base["label"],
                    class_name=base["class_name"],
                    conf=best_conf,
                    color=base["color"],
                )
            )

        return consolidated
