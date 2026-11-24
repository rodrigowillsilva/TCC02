from .video_reader import VideoReader
from .frame_annotator import FrameAnnotator
from .display import Display
from .annotations import FrameAnnotations, BBox
from .tiler import Tiler
from .profiler import Profiler


class PipelineRunner:
    """
    Orquestra o pipeline. Componentes que ainda não existem ficam None.

    Fluxo:
        VideoReader → (Tiler) → (DecisionBlock) → (Detector) → Annotator → Display
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
        self.display = display or Display()
        self.profiler = profiler or Profiler()
        self.output_json = output_json
        
        # Estrutura para salvar o JSON de resultados
        self.results_data = {"video": video_path, "frames": {}}

    def run(self):
        try:
            for frame_data in self.video_reader:
                self.profiler.set_frame(frame_data.index)
                with self.profiler.measure("00_Pipeline_Total (por frame)"):
                    frame = frame_data.image
                    annotations = FrameAnnotations()

                    # ── Tiling ──
                    tiles = None
                    if self.tiler:
                        with self.profiler.measure("01_Fatiamento (Tiler)"):
                            tiles = self.tiler.split(frame)
                        
                        # Desenha as bordas dos tiles e anota com o índice deles
                        for t in tiles:
                            annotations.boxes.append(
                                BBox(
                                    x=t.x, y=t.y, w=t.w, h=t.h, 
                                    label=f"T{t.index}",  # Aqui juntamos o metadado na imagem!
                                    color=(0, 0, 255),    # Vermelho do OpenCV (BGR)
                                    thickness=1
                                )
                            )

                    # ── Decisão ──
                    selected_indices = []
                    if self.decision_block and tiles:
                        with self.profiler.measure("02_Filtro (DecisionBlock)"):
                            selected_indices = self.decision_block.select(frame, tiles, frame_data.index)
                    elif tiles:
                        # Se não tem filtro inteligente, seleciona todos os tiles (baseline)
                        selected_indices = [t.index for t in tiles]

                    # Desenha os quadrados na tela (Verde = Vai pro YOLO, Vermelho = Jogado Fora)
                    if tiles:
                        for t in tiles:
                            is_selected = t.index in selected_indices
                            if is_selected:
                                annotations.boxes.append(
                                    BBox(x=t.x, y=t.y, w=t.w, h=t.h, label=f"T{t.index}", color=(0, 255, 0), thickness=2)
                                )
                            else:
                                annotations.boxes.append(
                                    BBox(x=t.x, y=t.y, w=t.w, h=t.h, label=f"T{t.index}", color=(0, 0, 255), thickness=1)
                                )

                    # ── Detecção (A Mágica da IA) ──
                    global_boxes = []
                    tile_detections = {}
                    if self.detector and tiles and selected_indices:
                        # Pega apenas os objetos de tile que foram aprovados
                        selected_tiles = [t for t in tiles if t.index in selected_indices]
                        
                        # Manda pro YOLO apenas o que importa
                        with self.profiler.measure("03_IA_Inferência (YOLO)"):
                            tile_detections = self.detector.detect(selected_tiles)
                            
                        # ── Feedback Loop (Aprendizado) ──
                        # Se o nosso algoritmo de decisão for inteligente (como o Scheduler ou HASH),
                        # nós alimentamos ele com o resultado do YOLO para ele atualizar seu banco/estado.
                        if hasattr(self.decision_block, "update_state"):
                            self.decision_block.update_state(selected_tiles, tile_detections)

                    # ── Recuperação de Cache (Ex: Modo HASH) ──
                    if hasattr(self.decision_block, "get_cached_detections"):
                        cached = self.decision_block.get_cached_detections()
                        if cached:
                            tile_detections.update(cached)

                    # ── Consolidação (NMS) ──
                    if self.consolidator and tiles and tile_detections:
                        # Converte de coords locais pra globais e mata detecções repetidas
                        with self.profiler.measure("04_Limpeza (Consolidator NMS)"):
                            global_boxes = self.consolidator.consolidate(tile_detections, tiles)

                    # ── Rastreamento & Interpolação (Opcional) ──
                    final_boxes = global_boxes
                    if self.tracker:
                        with self.profiler.measure("04b_Rastreamento (Tracker)"):
                            final_boxes = self.tracker.update(global_boxes)

                    # Adiciona os bonequinhos detectados/rastreados pra desenhar na tela final!
                    for box in final_boxes:
                        annotations.boxes.append(box)

                    # ── Salvar Resultados (JSON Export) ──
                    if self.output_json:
                        self.results_data["frames"][frame_data.index] = {
                            "timestamp_ms": frame_data.timestamp_ms,
                            "consolidated": [{"x": b.x, "y": b.y, "w": b.w, "h": b.h, "class": b.class_name, "conf": b.conf} for b in final_boxes]
                        }

                    # ── Anotação + Display ──
                    with self.profiler.measure("05_Renderização_Video"):
                        output = self.annotator.draw(frame, annotations)

                # ── Interface e Pausa (NÃO CONTA NO TEMPO DO PIPELINE) ──
                if not self.display.show(output):
                    break

                # ── Rastreamento de CPU ──
                # Mede a CPU que gastamos exatamente nesse frame
                self.profiler.record_frame_cpu()

        finally:
            self.video_reader.release()
            self.display.close()
            
            # Ao final do programa, salva o arquivo JSON (se não foi cancelado nem desativado)
            if getattr(self.display, "abort_save", False):
                print("\n[!] Salvamento de resultados cancelado pelo usuário ([ESC] / [X]).")
            elif self.output_json:
                import json
                
                # Coleta os metadados (Parâmetros) do método de decisão
                self.results_data["config"] = getattr(self.decision_block, "config", {"method": "BASELINE"})
                
                # Coleta os tempos de CPU e Profiler
                self.results_data["telemetry"] = self.profiler.get_summary_dict()
                
                with open(self.output_json, 'w') as f:
                    json.dump(self.results_data, f, indent=2)
                print(f"\n[+] Resultados exportados para: {self.output_json}")
            else:
                print("\n[INFO] Salvamento de resultados desativado (SAVE_RESULTS=False).")
