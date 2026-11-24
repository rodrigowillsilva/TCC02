from dataclasses import dataclass
from typing import List, Tuple
from .annotations import BBox


@dataclass
class TrackedTarget:
    """Representa um indivíduo rastreado pelo pipeline."""
    track_id: int
    x: float
    y: float
    w: float
    h: float
    vx: float = 0.0
    vy: float = 0.0
    missed_frames: int = 0
    total_detections: int = 1
    class_name: str = "person"
    conf: float = 1.0


def calculate_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Calcula IoU entre duas caixas (x, y, w, h)."""
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b

    x1 = max(xa, xb)
    y1 = max(ya, yb)
    x2 = min(xa + wa, xb + wb)
    y2 = min(ya + ha, yb + hb)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = wa * ha
    area_b = wb * hb
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return intersection / union


class SimpleVelocityTracker:
    """
    Rastreador leve e agnóstico de múltiplos objetos baseado em IoU e Cinemática Simples.
    
    Projetado para:
    - Fazer data-association (matching por IoU).
    - Estimar velocidade (vx, vy) após >= 2 observações.
    - Manter objetos 'fantasmas' com interpolação de movimento se o YOLO falhar/pular tiles.
    - Descartar alvos perdidos após max_missed_frames.
    """

    def __init__(self, iou_thresh: float = 0.3, max_missed_frames: int = 5, velocity_alpha: float = 0.6):
        self.iou_thresh = iou_thresh
        self.max_missed_frames = max_missed_frames
        self.velocity_alpha = velocity_alpha  # Média móvel suave da velocidade
        self.next_id = 1
        self.tracks: List[TrackedTarget] = []

    def update(self, detected_boxes: List[BBox]) -> List[BBox]:
        """
        Atualiza o estado dos alvos e retorna as caixas unificadas (detectadas + extrapoladas).
        """
        unmatched_detections = list(range(len(detected_boxes)))
        matched_pairs = []

        # 1. Matching com tracks existentes usando IoU
        for t_idx, track in enumerate(self.tracks):
            best_iou = 0.0
            best_d_idx = -1
            track_box = (track.x, track.y, track.w, track.h)

            for d_idx in unmatched_detections:
                det = detected_boxes[d_idx]
                det_box = (float(det.x), float(det.y), float(det.w), float(det.h))
                iou = calculate_iou(track_box, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_d_idx = d_idx

            if best_iou >= self.iou_thresh and best_d_idx != -1:
                matched_pairs.append((t_idx, best_d_idx))
                unmatched_detections.remove(best_d_idx)

        # 2. Atualizar tracks casados
        matched_track_indices = set()
        for t_idx, d_idx in matched_pairs:
            matched_track_indices.add(t_idx)
            track = self.tracks[t_idx]
            det = detected_boxes[d_idx]

            # Novo centro vs centro anterior para calcular velocidade instantânea
            old_cx = track.x + track.w / 2.0
            old_cy = track.y + track.h / 2.0
            new_cx = det.x + det.w / 2.0
            new_cy = det.y + det.h / 2.0

            inst_vx = new_cx - old_cx
            inst_vy = new_cy - old_cy

            if track.total_detections >= 1:
                track.vx = self.velocity_alpha * inst_vx + (1.0 - self.velocity_alpha) * track.vx
                track.vy = self.velocity_alpha * inst_vy + (1.0 - self.velocity_alpha) * track.vy
            else:
                track.vx = inst_vx
                track.vy = inst_vy

            track.x = float(det.x)
            track.y = float(det.y)
            track.w = float(det.w)
            track.h = float(det.h)
            track.missed_frames = 0
            track.total_detections += 1
            track.conf = det.conf
            track.class_name = det.class_name

        # 3. Tratar tracks não casados (extrapolação de velocidade / fantasma)
        surviving_tracks = []
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_track_indices:
                track.missed_frames += 1
                if track.missed_frames <= self.max_missed_frames:
                    # Só extrapola se tiver tido ao menos 2 confirmações prévias
                    if track.total_detections >= 2:
                        track.x += track.vx
                        track.y += track.vy
                    surviving_tracks.append(track)
            else:
                surviving_tracks.append(track)

        self.tracks = surviving_tracks

        # 4. Criar novas tracks para detecções não casadas
        for d_idx in unmatched_detections:
            det = detected_boxes[d_idx]
            new_track = TrackedTarget(
                track_id=self.next_id,
                x=float(det.x),
                y=float(det.y),
                w=float(det.w),
                h=float(det.h),
                vx=0.0,
                vy=0.0,
                missed_frames=0,
                total_detections=1,
                class_name=det.class_name,
                conf=det.conf
            )
            self.next_id += 1
            self.tracks.append(new_track)

        # 4b. Remove tracks duplicadas que se sobrepuseram na região de overlap entre tiles
        self.tracks.sort(key=lambda t: (t.missed_frames == 0, t.total_detections, t.conf), reverse=True)
        deduped_tracks: List[TrackedTarget] = []
        for candidate in self.tracks:
            c_box = (candidate.x, candidate.y, candidate.w, candidate.h)
            is_dup = False
            for kept in deduped_tracks:
                k_box = (kept.x, kept.y, kept.w, kept.h)
                if calculate_iou(c_box, k_box) >= 0.35:
                    is_dup = True
                    break
            if not is_dup:
                deduped_tracks.append(candidate)
        self.tracks = deduped_tracks

        # 5. Converter estado atual em BBoxes para o pipeline
        output_boxes: List[BBox] = []
        for track in self.tracks:
            is_interpolated = track.missed_frames > 0
            # Caixa extrapolada/fantasma pode usar cor ciano/amarela e espessura 1
            color = (255, 255, 0) if is_interpolated else (0, 255, 0)
            tag = f"ID:{track.track_id} [P]" if is_interpolated else f"ID:{track.track_id}"
            
            output_boxes.append(
                BBox(
                    x=int(round(track.x)),
                    y=int(round(track.y)),
                    w=int(round(track.w)),
                    h=int(round(track.h)),
                    label=tag,
                    class_name=track.class_name,
                    conf=track.conf,
                    color=color,
                    thickness=1 if is_interpolated else 2
                )
            )

        return output_boxes

