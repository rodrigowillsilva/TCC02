import sys
from pathlib import Path

# adiciona o diretório experiment ao path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.video_reader import VideoReader
from pipeline.display import Display
from pipeline.tiler import Tiler
from pipeline.realtime_runner import RealtimePipelineRunner
from pipeline.detector import YOLODetector
from pipeline.consolidator import Consolidator
from pipeline.profiler import Profiler


def main():
    videos_dir = Path(__file__).parent.parent / "videos"
    video_path = videos_dir / "teste_1_quiosque.mp4"

    # Modos disponíveis: "BASELINE", "KNN", "FRAMEDIFF", "SCHEDULER", "HASH", "VIBE", "MOG2"
    MODE = "HASH"

    reader = VideoReader(video_path)
    # Começa tocando direto (start_paused=False) na velocidade nativa do vídeo
    display = Display(
        window_name=f"TCC02 - FEED AO VIVO ({MODE})",
        scale=0.5,
        fps=int(reader.fps),
        start_paused=False,
    )

    tiler = Tiler(tile_size=(320, 320), grid_size=(5, 3))

    # OBS: Em modo multithread (Tempo Real), mantemos debug=False nas janelas secundárias do OpenCV
    if MODE == "BASELINE":
        decision = None

    elif MODE == "KNN":
        from decision.knn_decision import KNNDecision
        # Campeão Bateria B: KNN (p=150, dist2=200, sf=0.50)
        decision = KNNDecision(debug=False, history=500, dist2Threshold=200.0, min_pixels=150, scale_factor=0.50)

    elif MODE == "FRAMEDIFF":
        from decision.frame_diff_decision import FrameDiffDecision
        # Campeão Bateria B: FrameDiff (p=75, thr=20, step=3)
        decision = FrameDiffDecision(debug=False, min_pixels=75, threshold=20, scale_factor=0.75, frame_step=3)

    elif MODE == "SCHEDULER":
        from decision.scheduler_decision import SchedulerDecision
        decision = SchedulerDecision(budget=6, hunger_weight=1.0, heat_weight=5.0, decay=0.9)

    elif MODE == "HASH":
        from decision.hash_decision import HashDecision
        # Campeão Bateria B: Wavelet (5..40, tol=1.5%, TTL=6/30)
        HASH_TYPE = "WAVELET"  # "VALUE", "EDGE", "HYBRID" ou "WAVELET"
        decision = HashDecision(
            levels=(5, 10, 20, 40),
            hash_type=HASH_TYPE,
            tolerance_ratio=0.015,
            ttl_positive=6,
            ttl_empty=30,
            debug=True,
        )

    elif MODE == "VIBE":
        from decision.vibe_decision import ViBeDecision
        decision = ViBeDecision(debug=False, min_pixels=150)

    elif MODE == "MOG2":
        from decision.bgs import BGSDecision
        # Campeão Bateria B: MOG2 (p=100, var=25)
        decision = BGSDecision(debug=False, history=500, var_threshold=25.0, min_pixels=100)

    else:
        raise ValueError(f"Modo de experimento inválido: {MODE}")

    detector = YOLODetector(model_name="yolov8n.pt", conf_thresh=0.25, imgsz=640)
    consolidator = Consolidator(iou_threshold=0.45)

    USE_TRACKER = False
    if USE_TRACKER:
        from pipeline.realtime_tracker import RealtimeVelocityTracker
        tracker = RealtimeVelocityTracker(
            iou_thresh=0.2,
            max_center_dist=80.0,
            max_missed_updates=3,
            velocity_alpha=0.65,
        )
    else:
        tracker = None

    # Salva os resultados na pasta <MODE>_REALTIME para não misturar com os testes frame-a-frame
    SAVE_RESULTS = False  # Mude para False para não salvar JSON (ou aperte [ESC]/[X] na janela para cancelar)
    if SAVE_RESULTS:
        results_dir = Path(__file__).parent.parent / "results" / f"{MODE}_REALTIME"
        results_dir.mkdir(parents=True, exist_ok=True)
        existing_runs = list(results_dir.glob("run_*.json"))
        run_number = len(existing_runs) + 1
        output_filename = str(results_dir / f"run_{run_number:03d}.json")
        print(f"\n[INFO] Experimento TEMPO REAL: {MODE}")
        print(f"[INFO] FPS Nativo do Vídeo: {reader.fps:.1f} FPS")
        print(f"[INFO] Resultados serão salvos em: {output_filename}\n")
    else:
        output_filename = None
        print(f"\n[INFO] Experimento TEMPO REAL: {MODE} (Salvamento desativado)")
        print(f"[INFO] FPS Nativo do Vídeo: {reader.fps:.1f} FPS\n")

    WARMUP_FRAMES = 0
    profiler = Profiler(warmup_frames=WARMUP_FRAMES)

    pipeline = RealtimePipelineRunner(
        video_reader=reader,
        tiler=tiler,
        decision_block=decision,
        detector=detector,
        consolidator=consolidator,
        tracker=tracker,
        display=display,
        profiler=profiler,
        output_json=output_filename,
        video_path=str(video_path),
    )

    pipeline.run()
    pipeline.profiler.print_stats()


if __name__ == "__main__":
    main()

