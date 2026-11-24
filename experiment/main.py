import sys
from pathlib import Path
import cv2
import numpy as np

# adiciona o diretório experiment ao path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.video_reader import VideoReader
from pipeline.display import Display
from pipeline.tiler import Tiler
from pipeline.pipeline_runner import PipelineRunner
from pipeline.detector import YOLODetector
from pipeline.consolidator import Consolidator
from pipeline.profiler import Profiler
from decision.knn_decision import KNNDecision


def teste_visual_tiles(video_path, tiler):
    """(FÁCIL DE APAGAR DEPOIS) Mostra um tile por vez em um fundo preto."""
    pass # Removido para usar o pipeline principal


def main():
    videos_dir = Path(__file__).parent.parent / "videos"
    video_path = videos_dir / "VIRAT_S_010203_08_000895_000975.mp4"

    MODE = "HASH"

    reader = VideoReader(video_path)
    display = Display(window_name=f"TCC02 - Pipeline ({MODE})", scale=0.5)
    
    tiler = Tiler(tile_size=(320, 320), grid_size=(5, 3))
    
    # Instancia o algoritmo correto dependendo do modo escolhido
    if MODE == "BASELINE":
        decision = None  # None = Sem filtro, roda YOLO em tudo!
    
    elif MODE == "KNN":
        from decision.knn_decision import KNNDecision
        # Campeão Bateria B: KNN (p=150, dist2=200, sf=0.50)
        decision = KNNDecision(debug=True, history=500, dist2Threshold=200.0, min_pixels=150, scale_factor=0.50)
        
    elif MODE == "FRAMEDIFF":
        from decision.frame_diff_decision import FrameDiffDecision
        # Campeão Bateria B: FrameDiff (p=75, thr=20, step=3)
        decision = FrameDiffDecision(debug=True, min_pixels=75, threshold=20, scale_factor=0.75, frame_step=3)
        
    elif MODE == "SCHEDULER":
        from decision.scheduler_decision import SchedulerDecision
        decision = SchedulerDecision(budget=6, hunger_weight=1.0, heat_weight=5.0, decay=0.9)

    elif MODE == "HASH":
        from decision.hash_decision import HashDecision
        # Campeão Bateria B: Wavelet (5..40, tol=1.5%, TTL=6/30)
        HASH_TYPE = "WAVELET"
        decision = HashDecision(
            levels=(5, 10, 20, 40),
            hash_type=HASH_TYPE,
            tolerance_ratio=0.015,
            ttl_positive=6,
            ttl_empty=30,
            debug=True
        )
        
    elif MODE == "VIBE":
        from decision.vibe_decision import ViBeDecision
        decision = ViBeDecision(debug=True, min_pixels=150)
        
    elif MODE == "MOG2":
        from decision.bgs import BGSDecision
        # Campeão Bateria B: MOG2 (p=100, var=25)
        decision = BGSDecision(debug=True, history=500, var_threshold=25.0, min_pixels=100)
        
    else:
        raise ValueError(f"Modo de experimento inválido: {MODE}")
    
    # Inicia a nossa IA e o Consolidador de Overlap
    detector = YOLODetector(model_name="yolov8n.pt", conf_thresh=0.25, imgsz=640)
    consolidator = Consolidator(iou_threshold=0.45)

    # Tracker opcional (Defina como None para desligar completamente sem alterar o resto)
    USE_TRACKER = False
    if USE_TRACKER:
        from pipeline.tracker import SimpleVelocityTracker
        tracker = SimpleVelocityTracker(iou_thresh=0.25, max_missed_frames=8, velocity_alpha=0.6)
    else:
        tracker = None

    # --- GERENCIAMENTO DE RESULTADOS (PASTAS E ENUMERAÇÃO) ---
    SAVE_RESULTS = True  # Mude para False para não salvar JSON (ou aperte [ESC]/[X] na janela para cancelar)
    if SAVE_RESULTS:
        results_dir = Path(__file__).parent.parent / "results" / MODE
        results_dir.mkdir(parents=True, exist_ok=True)
        existing_runs = list(results_dir.glob("run_*.json"))
        run_number = len(existing_runs) + 1
        output_filename = str(results_dir / f"run_{run_number:03d}.json")
        print(f"\n[INFO] Experimento atual: {MODE}")
        print(f"[INFO] Os resultados serão salvos em: {output_filename}\n")
    else:
        output_filename = None
        print(f"\n[INFO] Experimento atual: {MODE} (Salvamento desativado)\n")

    # Instancia o Profiler (warmup_frames descarta os N primeiros frames para estabilização)
    WARMUP_FRAMES = 10  # Defina como 0 para desligar o descarte e medir desde o início
    profiler = Profiler(warmup_frames=WARMUP_FRAMES)

    # --- PIPELINE NORMAL ---
    pipeline = PipelineRunner(
        video_reader=reader,
        tiler=tiler,
        decision_block=decision,
        detector=detector,
        consolidator=consolidator,
        tracker=tracker,
        display=display,
        profiler=profiler,
        output_json=output_filename,
        video_path=str(video_path)
    )
    pipeline.run()
    
    # Imprime a bela tabela de performance!
    pipeline.profiler.print_stats()

if __name__ == "__main__":
    main()
