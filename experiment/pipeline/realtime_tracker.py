import threading
from dataclasses import dataclass
from typing import List, Tuple
from .annotations import BBox


@dataclass
class RealtimeTrack:
    """Representa um alvo rastreado no tempo real do vídeo (compensando frames descartados)."""
    track_id: int
    base_x: float          # Posição X exata na última vez que o YOLO viu
    base_y: float          # Posição Y exata na última vez que o YOLO viu
    w: float
    h: float
    last_frame_idx: int    # Índice do frame do vídeo em que o YOLO capturou esta caixa
    vx: float = 0.0        # Velocidade em pixels POR FRAME DO VÍDEO
    vy: float = 0.0        # Velocidade em pixels POR FRAME DO VÍDEO
    missed_updates: int = 0
    total_detections: int = 1
    class_name: str = "person"
    conf: float = 1.0


def _iou_or_center_match(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
    iou_thresh: float,
    max_center_dist: float,
) -> float:
    """
    Retorna um score de casamento combinando IoU e distância de centro.
    Em tempo real (com pulo de vários frames), a pessoa pode ter andado mais do que a largura
    da própria caixa, então a distância máxima entre centros evita perder o rastro.
    """
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b

    x1 = max(xa, xb)
    y1 = max(ya, yb)
    x2 = min(xa + wa, xb + wb)
    y2 = min(ya + ha, yb + hb)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h
    union = (wa * ha) + (wb * hb) - intersection
    iou = (intersection / union) if union > 0 else 0.0

    if iou >= iou_thresh:
        return 1.0 + iou

    # Fallback por proximidade de centro quando muitos frames foram pulados
    cxa, cya = xa + wa / 2.0, ya + ha / 2.0
    cxb, cyb = xb + wb / 2.0, yb + hb / 2.0
    dist = ((cxa - cxb) ** 2 + (cya - cyb) ** 2) ** 0.5
    if dist <= max_center_dist:
        return 1.0 - (dist / max_center_dist) * 0.9

    return 0.0


class RealtimeVelocityTracker:
    """
    Tracker exclusivo para o modo Tempo Real (Feed Ao Vivo):
    1. Calcula a velocidade real por frame do vídeo: vx = (x_novo - x_antigo) / (frame_novo - frame_antigo).
    2. Compensação de Atraso (Forward Shift): Assim que o YOLO termina de processar o frame N,
       o vídeo ao vivo já está no frame N + K. O tracker já aplica imediatamente o shift de
       +K * (vx, vy) para a caixa nascer na posição atual da pessoa (sem começar atrasada!).
    3. Interpolação Contínua no Feed: A cada frame exibido no vídeo ao vivo (30 FPS), avança
       suavemente a caixa com base no frame atual do vídeo.
    """

    def __init__(
        self,
        iou_thresh: float = 0.2,
        max_center_dist: float = 80.0,
        max_missed_updates: int = 3,
        max_extrapolation_frames: int = 45,
        velocity_alpha: float = 0.65,
        max_speed_px_per_frame: float = 12.0,
    ):
        self.iou_thresh = iou_thresh
        self.max_center_dist = max_center_dist
        self.max_missed_updates = max_missed_updates
        self.max_extrapolation_frames = max_extrapolation_frames
        self.velocity_alpha = velocity_alpha
        self.max_speed = max_speed_px_per_frame

        self._lock = threading.Lock()
        self.next_id = 1
        self.tracks: List[RealtimeTrack] = []

    def update(self, detected_boxes: List[BBox], captured_frame_idx: int = 0) -> List[BBox]:
        """
        Chamado pela thread da IA quando o YOLO termina de processar o frame `captured_frame_idx`.
        Atualiza as posições base e calcula a velocidade por frame do vídeo.
        """
        with self._lock:
            unmatched_dets = list(range(len(detected_boxes)))
            matched_pairs = []

            # 1. Projeta onde cada track deveria estar no frame `captured_frame_idx` para fazer o casamento
            for t_idx, track in enumerate(self.tracks):
                dt = max(0, captured_frame_idx - track.last_frame_idx)
                pred_x = track.base_x + track.vx * dt if track.total_detections >= 2 else track.base_x
                pred_y = track.base_y + track.vy * dt if track.total_detections >= 2 else track.base_y
                track_box = (pred_x, pred_y, track.w, track.h)

                best_score = 0.0
                best_d_idx = -1

                for d_idx in unmatched_dets:
                    det = detected_boxes[d_idx]
                    det_box = (float(det.x), float(det.y), float(det.w), float(det.h))
                    score = _iou_or_center_match(track_box, det_box, self.iou_thresh, self.max_center_dist)
                    if score > best_score:
                        best_score = score
                        best_d_idx = d_idx

                if best_score > 0.0 and best_d_idx != -1:
                    matched_pairs.append((t_idx, best_d_idx))
                    unmatched_dets.remove(best_d_idx)

            # 2. Atualiza tracks casadas e calcula velocidade real por frame do vídeo
            matched_track_indices = set()
            for t_idx, d_idx in matched_pairs:
                matched_track_indices.add(t_idx)
                track = self.tracks[t_idx]
                det = detected_boxes[d_idx]

                dt_frames = max(1, captured_frame_idx - track.last_frame_idx)

                old_cx = track.base_x + track.w / 2.0
                old_cy = track.base_y + track.h / 2.0
                new_cx = det.x + det.w / 2.0
                new_cy = det.y + det.h / 2.0

                # Velocidade em pixels por frame do vídeo
                inst_vx = (new_cx - old_cx) / dt_frames
                inst_vy = (new_cy - old_cy) / dt_frames

                # Limita velocidades absurdas por falso casamento
                inst_vx = max(-self.max_speed, min(self.max_speed, inst_vx))
                inst_vy = max(-self.max_speed, min(self.max_speed, inst_vy))

                if track.total_detections >= 2:
                    track.vx = self.velocity_alpha * inst_vx + (1.0 - self.velocity_alpha) * track.vx
                    track.vy = self.velocity_alpha * inst_vy + (1.0 - self.velocity_alpha) * track.vy
                else:
                    track.vx = inst_vx
                    track.vy = inst_vy

                track.base_x = float(det.x)
                track.base_y = float(det.y)
                track.w = float(det.w)
                track.h = float(det.h)
                track.last_frame_idx = captured_frame_idx
                track.missed_updates = 0
                track.total_detections += 1
                track.conf = det.conf
                track.class_name = det.class_name

            # 3. Mantém por algumas atualizações quem o YOLO piscou
            surviving: List[RealtimeTrack] = []
            for t_idx, track in enumerate(self.tracks):
                if t_idx not in matched_track_indices:
                    track.missed_updates += 1
                    if track.missed_updates <= self.max_missed_updates:
                        surviving.append(track)
                else:
                    surviving.append(track)

            self.tracks = surviving

            # 4. Cria novas tracks para novas detecções
            for d_idx in unmatched_dets:
                det = detected_boxes[d_idx]
                self.tracks.append(
                    RealtimeTrack(
                        track_id=self.next_id,
                        base_x=float(det.x),
                        base_y=float(det.y),
                        w=float(det.w),
                        h=float(det.h),
                        last_frame_idx=captured_frame_idx,
                        vx=0.0,
                        vy=0.0,
                        missed_updates=0,
                        total_detections=1,
                        class_name=det.class_name,
                        conf=det.conf,
                    )
                )
                self.next_id += 1

            # 4b. Remove tracks duplicadas na zona de overlap (prioriza ativas e mais antigas)
            self.tracks.sort(key=lambda t: (t.missed_updates == 0, t.total_detections, t.conf), reverse=True)
            deduped: List[RealtimeTrack] = []
            for cand in self.tracks:
                dt_c = max(0, captured_frame_idx - cand.last_frame_idx)
                cx = cand.base_x + (cand.vx * dt_c if cand.total_detections >= 2 else 0.0)
                cy = cand.base_y + (cand.vy * dt_c if cand.total_detections >= 2 else 0.0)
                c_box = (cx, cy, cand.w, cand.h)
                is_dup = False
                for kept in deduped:
                    dt_k = max(0, captured_frame_idx - kept.last_frame_idx)
                    kx = kept.base_x + (kept.vx * dt_k if kept.total_detections >= 2 else 0.0)
                    ky = kept.base_y + (kept.vy * dt_k if kept.total_detections >= 2 else 0.0)
                    k_box = (kx, ky, kept.w, kept.h)
                    if _iou_or_center_match(c_box, k_box, iou_thresh=0.30, max_center_dist=25.0) > 0.0:
                        is_dup = True
                        break
                if not is_dup:
                    deduped.append(cand)
            self.tracks = deduped

        return self.predict_at_frame(captured_frame_idx)

    def predict_at_frame(self, live_frame_idx: int) -> List[BBox]:
        """
        Chamado a cada frame do vídeo ao vivo (30 FPS).
        Desloca todas as caixas por `dt = live_frame_idx - track.last_frame_idx` vezes a velocidade:
        - Compensa instantaneamente o atraso de processamento do YOLO (o quadrado já nasce adiantado!).
        - Move suavemente os quadrados durante todos os frames descartados entre uma inferência e outra.
        """
        output_boxes: List[BBox] = []
        with self._lock:
            for track in self.tracks:
                dt = max(0, live_frame_idx - track.last_frame_idx)
                if dt > self.max_extrapolation_frames and track.missed_updates > 0:
                    continue

                # Aplica o shift de velocidade proporcional aos frames transcorridos desde a captura
                if track.total_detections >= 2:
                    shift_x = track.vx * dt
                    shift_y = track.vy * dt
                else:
                    shift_x = 0.0
                    shift_y = 0.0

                curr_x = int(round(track.base_x + shift_x))
                curr_y = int(round(track.base_y + shift_y))

                is_missed = track.missed_updates > 0
                color = (255, 255, 0) if is_missed else (0, 255, 0)
                label = f"ID:{track.track_id}" if not is_missed else f"ID:{track.track_id} [P]"

                output_boxes.append(
                    BBox(
                        x=curr_x,
                        y=curr_y,
                        w=int(round(track.w)),
                        h=int(round(track.h)),
                        label=label,
                        class_name=track.class_name,
                        conf=track.conf,
                        color=color,
                        thickness=2,
                    )
                )

        return output_boxes

