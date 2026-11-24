import json
import time
import threading
import cv2
import numpy as np

from .video_reader import VideoReader, FrameData
from .frame_annotator import FrameAnnotator
from .display import Display
from .annotations import FrameAnnotations, BBox
from .tiler import Tiler
from .profiler import Profiler


class RealtimePipelineRunner:
    """
    Simula um feed de câmera ao vivo (Tempo Real):
    - Thread Principal (Feed Ao Vivo): Reproduz o vídeo continuamente na taxa nativa (ex: 30 FPS),
      desenhando na tela as anotações mais recentes produzidas pela IA.
    - Thread de Processamento (Pipeline Edge): Pega sempre o frame mais recente disponível no feed,
      descartando os frames intermediários que passaram enquanto o frame anterior era processado.
    """

    def __init__(
        self,
        video_reader: VideoReader,
        tiler: Tiler | None = None,
        decision_block=None,
        detector=None,
        consolidator=None,
        tracker=None,
        annotator: FrameAnnotator | None = None,
        display: Display | None = None,
        profiler: Profiler | None = None,
        output_json: str | None = None,
        video_path: str = "",
    ):
        self.video_reader = video_reader
        self.tiler = tiler
        self.decision_block = decision_block
        self.detector = detector
        self.consolidator = consolidator
        self.tracker = tracker
        self.annotator = annotator or FrameAnnotator()
        self.display = display or Display(start_paused=False)
        self.profiler = profiler or Profiler()
        self.output_json = output_json

        # Estado compartilhado entre o Feed de Vídeo (Main Thread) e o Pipeline (Worker Thread)
        self._lock = threading.Lock()
        self._latest_frame: FrameData | None = None
        self._latest_tile_boxes: list[BBox] = []
        self._latest_boxes: list[BBox] = []
        self._last_processed_index: int = -1
        self._last_latency_ms: float = 0.0
        self._running = False

        # Contadores de tempo real (frames processados vs descartados)
        self.frames_displayed = 0
        self.frames_processed = 0
        self.frames_dropped = 0

        self.results_data = {"video": video_path, "realtime": True, "frames": {}}

    def _worker_loop(self):
        """Loop em background que processa sempre o frame mais recente e descarta os atrasados."""
        last_seen_idx = -1

        while self._running:
            with self._lock:
                frame_data = self._latest_frame

            # Se ainda não há frame novo (ou o vídeo está pausado no mesmo frame), aguarda um pouquinho
            if frame_data is None or frame_data.index == last_seen_idx:
                time.sleep(0.002)
                continue

            # Quantos frames passaram no vídeo enquanto estávamos ocupados no frame anterior?
            if last_seen_idx >= 0 and frame_data.index > last_seen_idx + 1:
                dropped_now = (frame_data.index - last_seen_idx) - 1
                with self._lock:
                    self.frames_dropped += dropped_now

            last_seen_idx = frame_data.index
            self.profiler.set_frame(self.frames_processed)

            t_start = time.perf_counter()
            with self.profiler.measure("00_Pipeline_Total (por frame processado)"):
                frame = frame_data.image
                tile_boxes: list[BBox] = []

                # ── 1. Tiling ──
                tiles = None
                if self.tiler:
                    with self.profiler.measure("01_Fatiamento (Tiler)"):
                        tiles = self.tiler.split(frame)

                # ── 2. Decisão ──
                selected_indices = []
                if self.decision_block and tiles:
                    with self.profiler.measure("02_Filtro (DecisionBlock)"):
                        selected_indices = self.decision_block.select(frame, tiles, frame_data.index)
                elif tiles:
                    selected_indices = [t.index for t in tiles]

                # Bordas dos tiles (Verde = Selecionado, Vermelho = Poupado)
                if tiles:
                    for t in tiles:
                        if t.index in selected_indices:
                            tile_boxes.append(
                                BBox(x=t.x, y=t.y, w=t.w, h=t.h, label=f"T{t.index}", color=(0, 255, 0), thickness=2)
                            )
                        else:
                            tile_boxes.append(
                                BBox(x=t.x, y=t.y, w=t.w, h=t.h, label=f"T{t.index}", color=(0, 0, 255), thickness=1)
                            )

                # ── 3. Detecção (YOLO) ──
                global_boxes = []
                tile_detections = {}
                if self.detector and tiles and selected_indices:
                    selected_tiles = [t for t in tiles if t.index in selected_indices]
                    with self.profiler.measure("03_IA_Inferência (YOLO)"):
                        tile_detections = self.detector.detect(selected_tiles)

                    if hasattr(self.decision_block, "update_state"):
                        self.decision_block.update_state(selected_tiles, tile_detections)

                # ── Recuperação de Cache (Modo HASH) ──
                if hasattr(self.decision_block, "get_cached_detections"):
                    cached = self.decision_block.get_cached_detections()
                    if cached:
                        tile_detections.update(cached)

                # ── 4. Consolidação (NMS) ──
                if self.consolidator and tiles and tile_detections:
                    with self.profiler.measure("04_Limpeza (Consolidator NMS)"):
                        global_boxes = self.consolidator.consolidate(tile_detections, tiles)

                # ── 4b. Rastreamento (Tracker Opcional) ──
                final_boxes = global_boxes
                if self.tracker:
                    with self.profiler.measure("04b_Rastreamento (Tracker)"):
                        if hasattr(self.tracker, "predict_at_frame"):
                            final_boxes = self.tracker.update(global_boxes, frame_data.index)
                        else:
                            final_boxes = self.tracker.update(global_boxes)

            dt_ms = (time.perf_counter() - t_start) * 1000.0
            self.profiler.record_frame_cpu()

            # Publica as novas caixas para a thread de vídeo desenhar imediatamente
            with self._lock:
                self._latest_tile_boxes = tile_boxes
                self._latest_boxes = final_boxes
                self._last_processed_index = frame_data.index
                self._last_latency_ms = dt_ms
                self.frames_processed += 1

    def _draw_hud(self, image: np.ndarray, live_idx: int, proc_idx: int, latency_ms: float) -> np.ndarray:
        """Desenha um painel no topo do vídeo mostrando o status em tempo real."""
        lag_frames = max(0, live_idx - proc_idx) if proc_idx >= 0 else live_idx
        proc_fps = (1000.0 / latency_ms) if latency_ms > 0 else 0.0

        hud_text = (
            f"FEED AO VIVO: Frame {live_idx} | "
            f"IA no Frame: {proc_idx} (Atraso: {lag_frames}f) | "
            f"Processados: {self.frames_processed} | Descartados: {self.frames_dropped} | "
            f"Latencia IA: {latency_ms:.0f}ms ({proc_fps:.1f} FPS)"
        )

        cv2.rectangle(image, (0, 0), (image.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(
            image, hud_text, (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
        )
        return image

    def run(self):
        fps = self.video_reader.fps
        frame_interval = 1.0 / fps

        self._running = True
        worker = threading.Thread(target=self._worker_loop, daemon=True)
        worker.start()

        try:
            next_frame_target_time = time.perf_counter()

            for frame_data in self.video_reader:
                t_loop_start = time.perf_counter()

                # Publica o frame atual da câmera para a thread do pipeline
                with self._lock:
                    self._latest_frame = frame_data
                    tile_boxes = list(self._latest_tile_boxes)
                    current_boxes = list(self._latest_boxes)
                    proc_idx = self._last_processed_index
                    latency_ms = self._last_latency_ms

                # Se o RealtimeVelocityTracker estiver ativo, projeta as caixas para o frame AO VIVO atual
                # (Isso já dá o shift inicial da velocidade quando o YOLO termina atrasado e move nos frames descartados!)
                if self.tracker and hasattr(self.tracker, "predict_at_frame") and proc_idx >= 0:
                    current_boxes = self.tracker.predict_at_frame(frame_data.index)

                current_annotations = FrameAnnotations(boxes=tile_boxes + current_boxes)
                self.frames_displayed += 1

                # Salva no JSON o estado ativo que o operador/sistema tinha neste frame do vídeo
                if self.output_json:
                    self.results_data["frames"][frame_data.index] = {
                        "timestamp_ms": frame_data.timestamp_ms,
                        "source_processed_frame": proc_idx,
                        "consolidated": [
                            {"x": b.x, "y": b.y, "w": b.w, "h": b.h, "class": b.class_name, "conf": b.conf}
                            for b in current_boxes
                        ],
                    }

                # Desenha as caixas mais recentes por cima do vídeo ao vivo
                with self.profiler.measure("05_Renderização_Video"):
                    output = self.annotator.draw(frame_data.image, current_annotations)
                    output = self._draw_hud(output, frame_data.index, proc_idx, latency_ms)

                # Sincroniza com o relógio real (ex: 33.3ms por frame a 30 FPS)
                elapsed = time.perf_counter() - t_loop_start
                remaining_ms = max(1, int((frame_interval - elapsed) * 1000))
                self.display.delay_ms = remaining_ms

                was_paused = self.display.paused
                if not self.display.show(output):
                    break
                if was_paused:
                    next_frame_target_time = time.perf_counter()
                else:
                    next_frame_target_time += frame_interval

        finally:
            self._running = False
            worker.join(timeout=2.0)
            self.video_reader.release()
            self.display.close()

            print("\n" + "=" * 55)
            print(f"{' RESUMO DO FEED EM TEMPO REAL ':^55}")
            print("=" * 55)
            print(f"📹 Frames Exibidos no Feed : {self.frames_displayed}")
            print(f"✅ Frames Processados (IA) : {self.frames_processed}")
            print(f"🗑️  Frames Descartados      : {self.frames_dropped}")
            if self.frames_displayed > 0:
                pct = (self.frames_processed / self.frames_displayed) * 100
                print(f"⚡ Taxa de Aproveitamento  : {pct:.1f}% dos frames do vídeo")
            print("=" * 55)

            if getattr(self.display, "abort_save", False):
                print("\n[!] Salvamento de resultados cancelado pelo usuário ([ESC] / [X]).")
            elif self.output_json:
                self.results_data["config"] = getattr(self.decision_block, "config", {"method": "BASELINE"})
                telemetry = self.profiler.get_summary_dict()
                telemetry["realtime_stats"] = {
                    "frames_displayed": self.frames_displayed,
                    "frames_processed": self.frames_processed,
                    "frames_dropped": self.frames_dropped,
                    "processed_ratio_percent": round(
                        (self.frames_processed / self.frames_displayed * 100) if self.frames_displayed > 0 else 0.0, 2
                    ),
                }
                self.results_data["telemetry"] = telemetry

                with open(self.output_json, "w") as f:
                    json.dump(self.results_data, f, indent=2)
                print(f"\n[+] Resultados Tempo Real exportados para: {self.output_json}")
            else:
                print("\n[INFO] Salvamento de resultados desativado (SAVE_RESULTS=False).")

