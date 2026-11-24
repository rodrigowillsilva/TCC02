import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Adiciona o diretório experiment ao path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.video_reader import VideoReader
from pipeline.tiler import Tiler
from pipeline.detector import YOLODetector
from pipeline.consolidator import Consolidator
from pipeline.frame_annotator import FrameAnnotator
from pipeline.annotations import FrameAnnotations, BBox
from pipeline.profiler import Profiler
from evaluator.compare import calculate_iou

from decision.knn_decision import KNNDecision
from decision.frame_diff_decision import FrameDiffDecision
from decision.bgs import BGSDecision
from decision.hash_decision import HashDecision


def create_browser_video_writer(
    output_path: Path, fps: float, size: tuple[int, int]
) -> cv2.VideoWriter:
    """
    Cria um VideoWriter em H.264 compatível com navegadores HTML5 (Chrome/Edge/Firefox).
    Usa o encoder nativo do Windows (CAP_MSMF + H264) e faz fallback caso necessário.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size

    # 1ª tentativa: Windows Media Foundation H.264 (abre direto no navegador sem avisos de DLL)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.CAP_MSMF,
        cv2.VideoWriter_fourcc(*"H264"),
        fps,
        (w, h),
    )
    if writer.isOpened():
        return writer

    # 2ª tentativa: avc1
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (w, h),
    )
    if writer.isOpened():
        return writer

    # 3ª tentativa: mp4v padrão
    return cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )


def render_mask_frame(
    mask_gray: np.ndarray | None,
    tiles: list,
    selected_indices: list[int],
    scale_factor: float,
    target_size: tuple[int, int],
) -> np.ndarray:
    """
    Formata a máscara binária P&B (Slide 7) com os retângulos dos tiles ativos
    e a contagem de pixels brancos em cada tile.
    """
    target_w, target_h = target_size
    if mask_gray is None:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)

    for t in tiles:
        sx = int(t.x * scale_factor)
        sy = int(t.y * scale_factor)
        sw = int(t.w * scale_factor)
        sh = int(t.h * scale_factor)

        tile_crop = mask_gray[sy : sy + sh, sx : sx + sw]
        white_px = int(cv2.countNonZero(tile_crop)) if tile_crop.size > 0 else 0

        if t.index in selected_indices:
            cv2.rectangle(mask_bgr, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            cv2.putText(
                mask_bgr,
                f"T{t.index}: {white_px}px",
                (sx + 6, sy + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                1,
            )
        else:
            cv2.rectangle(mask_bgr, (sx, sy), (sx + sw, sy + sh), (60, 60, 140), 1)

    if (mask_bgr.shape[1], mask_bgr.shape[0]) != (target_w, target_h):
        mask_bgr = cv2.resize(
            mask_bgr, (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )

    return mask_bgr


def evaluate_against_baseline(
    base_data: dict, exp_data: dict, iou_threshold: float = 0.40
) -> dict:
    """Calcula TP, FN, FP, Recall, Precision e F1 comparando um método contra o Baseline."""
    gt_frames = base_data.get("frames", {})
    exp_frames = exp_data.get("frames", {})

    total_gt = 0
    tp = 0
    fp = 0
    frames_eval = 0

    for frame_idx, exp_info in exp_frames.items():
        if str(frame_idx) not in gt_frames:
            continue
        gt_info = gt_frames[str(frame_idx)]
        frames_eval += 1

        gt_boxes = gt_info.get("consolidated", [])
        exp_boxes = exp_info.get("consolidated", [])
        total_gt += len(gt_boxes)

        gt_matched = [False] * len(gt_boxes)
        for eb in exp_boxes:
            best_iou = 0.0
            best_j = -1
            for j, gb in enumerate(gt_boxes):
                if not gt_matched[j]:
                    iou = calculate_iou(eb, gb)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
            if best_iou >= iou_threshold and best_j != -1:
                gt_matched[best_j] = True
                tp += 1
            else:
                fp += 1

    fn = total_gt - tp
    recall = (tp / total_gt * 100.0) if total_gt > 0 else 0.0
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "iou_threshold": iou_threshold,
        "frames_evaluated": frames_eval,
        "gt_total": total_gt,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "recall": round(recall, 2),
        "precision": round(precision, 2),
        "f1": round(f1, 2),
    }


def run_single_method(
    method_key: str,
    mode_folder_name: str,
    decision_block,
    has_mask_video: bool,
    video_path: Path,
    clips_dir: Path,
    suite_results_dir: Path,
    tiler: Tiler,
    detector: YOLODetector,
    consolidator: Consolidator,
    warmup_frames: int = 10,
    video_scale: float = 0.75,
    record_clips: bool = True,
) -> dict:
    """
    Roda um método do início ao fim de forma 100% autônoma (sem abrir janela na tela):
    - Mede o tempo puro do pipeline (sem contar a gravação de vídeo).
    - Salva os vídeos `_grid.mp4`, `_clean.mp4` e (se aplicável) `_mask.mp4` em `apresentacao/clips/`.
    - Salva o JSON completo de resultados e telemetria.
    """
    print("\n" + "=" * 70)
    print(
        f"🚀 INICIANDO MÉTODO: {mode_folder_name} ({method_key}) | Vídeo: {video_path.name}"
    )
    print("=" * 70)

    reader = VideoReader(video_path)
    fps = reader.fps
    orig_w, orig_h = reader.resolution
    out_w = (int(orig_w * video_scale) // 2) * 2
    out_h = (int(orig_h * video_scale) // 2) * 2
    out_size = (out_w, out_h)

    annotator = FrameAnnotator()
    profiler = Profiler(warmup_frames=warmup_frames)

    writer_grid = None
    writer_clean = None
    writer_mask = None

    if record_clips:
        grid_path = clips_dir / f"{method_key}_grid.mp4"
        clean_path = clips_dir / f"{method_key}_clean.mp4"
        writer_grid = create_browser_video_writer(grid_path, fps, out_size)
        writer_clean = create_browser_video_writer(clean_path, fps, out_size)
        if has_mask_video:
            mask_path = clips_dir / f"{method_key}_mask.mp4"
            writer_mask = create_browser_video_writer(mask_path, fps, out_size)

    results_data = {
        "video": str(video_path),
        "method_key": method_key,
        "frames": {},
    }

    total_tiles_possible = 0
    total_tiles_selected = 0
    total_frames = reader.total_frames
    t_run_start = time.perf_counter()

    try:
        for frame_data in reader:
            profiler.set_frame(frame_data.index)
            frame = frame_data.image

            # ── MEDIÇÃO PURA DO PIPELINE (Sem contar gravação de vídeo em disco) ──
            with profiler.measure("00_Pipeline_Total (por frame)"):
                with profiler.measure("01_Fatiamento (Tiler)"):
                    tiles = tiler.split(frame)

                if decision_block is not None:
                    with profiler.measure("02_Filtro (DecisionBlock)"):
                        selected_indices = decision_block.select(
                            frame, tiles, frame_data.index
                        )
                else:
                    selected_indices = [t.index for t in tiles]

                if frame_data.index >= warmup_frames:
                    total_tiles_possible += len(tiles)
                    total_tiles_selected += len(selected_indices)

                global_boxes = []
                tile_detections = {}
                if selected_indices:
                    selected_tiles = [t for t in tiles if t.index in selected_indices]
                    with profiler.measure("03_IA_Inferência (YOLO)"):
                        tile_detections = detector.detect(selected_tiles)

                    if hasattr(decision_block, "update_state"):
                        decision_block.update_state(selected_tiles, tile_detections)

                if hasattr(decision_block, "get_cached_detections"):
                    cached = decision_block.get_cached_detections()
                    if cached:
                        tile_detections.update(cached)

                if tile_detections:
                    with profiler.measure("04_Limpeza (Consolidator NMS)"):
                        global_boxes = consolidator.consolidate(tile_detections, tiles)

            profiler.record_frame_cpu()

            # Salva detecções do frame no dicionário de resultados
            results_data["frames"][str(frame_data.index)] = {
                "timestamp_ms": frame_data.timestamp_ms,
                "selected_tiles_count": len(selected_indices),
                "consolidated": [
                    {
                        "x": b.x,
                        "y": b.y,
                        "w": b.w,
                        "h": b.h,
                        "class": b.class_name,
                        "conf": b.conf,
                    }
                    for b in global_boxes
                ],
            }

            # ── GRAVAÇÃO DOS VÍDEOS DA APRESENTAÇÃO (FORA DO PROFILER!) ──
            if record_clips:
                # 1. Vídeo Limpo (Só as caixas das pessoas consolidadas)
                clean_annotations = FrameAnnotations(
                    boxes=[
                        BBox(
                            x=b.x,
                            y=b.y,
                            w=b.w,
                            h=b.h,
                            label=f"{b.conf:.2f}",
                            class_name=b.class_name,
                            conf=b.conf,
                            color=(0, 255, 0),
                            thickness=2,
                        )
                        for b in global_boxes
                    ]
                )
                clean_img = annotator.draw(frame, clean_annotations)

                # 2. Vídeo Com Grade (Tiles Verdes/Vermelhos + Pessoas)
                grid_boxes = []
                selected_set = set(selected_indices)
                for t in tiles:
                    if t.index in selected_set:
                        grid_boxes.append(
                            BBox(
                                x=t.x,
                                y=t.y,
                                w=t.w,
                                h=t.h,
                                label=f"T{t.index}",
                                color=(0, 255, 0),
                                thickness=2,
                            )
                        )
                    else:
                        grid_boxes.append(
                            BBox(
                                x=t.x,
                                y=t.y,
                                w=t.w,
                                h=t.h,
                                label=f"T{t.index}",
                                color=(0, 0, 255),
                                thickness=1,
                            )
                        )
                grid_annotations = FrameAnnotations(
                    boxes=grid_boxes + clean_annotations.boxes
                )
                grid_img = annotator.draw(frame, grid_annotations)

                if (out_w, out_h) != (orig_w, orig_h):
                    clean_img = cv2.resize(
                        clean_img, out_size, interpolation=cv2.INTER_AREA
                    )
                    grid_img = cv2.resize(
                        grid_img, out_size, interpolation=cv2.INTER_AREA
                    )

                writer_clean.write(clean_img)
                writer_grid.write(grid_img)

                # 3. Vídeo da Máscara P&B (Slide 7)
                if writer_mask is not None:
                    last_mask = getattr(decision_block, "last_mask", None)
                    sf = getattr(decision_block, "scale_factor", 1.0)
                    mask_img = render_mask_frame(
                        last_mask, tiles, selected_indices, sf, out_size
                    )
                    writer_mask.write(mask_img)

            # Log de progresso a cada 50 frames
            if (frame_data.index + 1) % 50 == 0 or (
                frame_data.index + 1
            ) == total_frames:
                elapsed = time.perf_counter() - t_run_start
                pct = (
                    ((frame_data.index + 1) / total_frames * 100)
                    if total_frames > 0
                    else 0.0
                )
                print(
                    f"  [{mode_folder_name}] Frame {frame_data.index + 1}/{total_frames} "
                    f"({pct:.1f}%) | Selecionados: {len(selected_indices)}/15 | Tempo decorrido: {elapsed:.1f}s"
                )

    finally:
        reader.release()
        if writer_grid:
            writer_grid.release()
        if writer_clean:
            writer_clean.release()
        if writer_mask:
            writer_mask.release()

    # Monta metadados e telemetria final
    results_data["config"] = getattr(decision_block, "config", {"method": "BASELINE"})
    telemetry = profiler.get_summary_dict()
    tiles_ratio = (
        (total_tiles_selected / total_tiles_possible * 100.0)
        if total_tiles_possible > 0
        else 100.0
    )
    telemetry["tile_selection"] = {
        "total_tiles_evaluated": total_tiles_possible,
        "total_tiles_sent_to_yolo": total_tiles_selected,
        "selected_percent": round(tiles_ratio, 2),
        "saved_percent": round(100.0 - tiles_ratio, 2),
    }
    results_data["telemetry"] = telemetry

    # Salva na pasta oficial da suíte e também na pasta enumerada `results/<MODE>/`
    suite_results_dir.mkdir(parents=True, exist_ok=True)
    suite_json_path = suite_results_dir / f"{video_path.stem}_{method_key}.json"
    with open(suite_json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    mode_dir = Path(__file__).parent.parent / "results" / mode_folder_name
    mode_dir.mkdir(parents=True, exist_ok=True)
    run_num = len(list(mode_dir.glob("run_*.json"))) + 1
    enum_json_path = mode_dir / f"run_{run_num:03d}.json"
    with open(enum_json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    profiler.print_stats()
    print(f"[+] JSON salvo em: {suite_json_path}")
    return results_data


def build_reports(
    all_video_results: dict[str, dict[str, dict]],
    report_md_path: Path,
    report_json_path: Path,
):
    """
    Gera o relatório consolidado em Markdown (com tabelas prontas para a Apresentação e LaTeX para o artigo)
    e em JSON.
    """
    summary_payload = {}
    md_lines = [
        "# Relatório Quantitativo Automático — Apresentação & Artigo TCC02",
        "",
        "Este arquivo foi gerado automaticamente por `experiment/gerar_tudo_apresentacao.py`.",
        "",
    ]

    for video_name, methods_dict in all_video_results.items():
        base_data = methods_dict["baseline"]
        base_total_ms = (
            base_data["telemetry"]["timings"]
            .get("00_Pipeline_Total (por frame)", {})
            .get("avg_ms", 1.0)
        )
        base_yolo_ms = (
            base_data["telemetry"]["timings"]
            .get("03_IA_Inferência (YOLO)", {})
            .get("avg_ms", 1.0)
        )

        summary_payload[video_name] = {}

        md_lines.append(f"## Vídeo: `{video_name}`")
        md_lines.append("")
        md_lines.append(
            "| Método | Filtro (ms) | YOLO (ms) | Total (ms/frame) | FPS | Redução Tempo | Speedup | Tiles no YOLO (%) | Recall @0.40 | Precision @0.40 | F1 @0.40 | Recall @0.45 | Precision @0.45 |"
        )
        md_lines.append(
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        )

        latex_rows = []

        for m_key, m_label in [
            ("baseline", "Baseline (15 tiles)"),
            ("framediff", "FrameDiff (t-3)"),
            ("knn", "KNN"),
            ("mog2", "MOG2"),
            ("hash", "Hash Wavelet"),
        ]:
            if m_key not in methods_dict:
                continue
            m_data = methods_dict[m_key]
            timings = m_data["telemetry"]["timings"]

            total_ms = timings.get("00_Pipeline_Total (por frame)", {}).get(
                "avg_ms", 0.0
            )
            filter_ms = timings.get("02_Filtro (DecisionBlock)", {}).get("avg_ms", 0.0)
            yolo_ms = timings.get("03_IA_Inferência (YOLO)", {}).get("avg_ms", 0.0)
            fps_equiv = (1000.0 / total_ms) if total_ms > 0 else 0.0
            reduction_pct = (
                ((base_total_ms - total_ms) / base_total_ms * 100.0)
                if base_total_ms > 0
                else 0.0
            )
            speedup = (base_total_ms / total_ms) if total_ms > 0 else 1.0
            tiles_pct = (
                m_data["telemetry"]
                .get("tile_selection", {})
                .get("selected_percent", 100.0)
            )
            cpu_avg = m_data["telemetry"].get("cpu", {}).get("average_percent", 0.0)

            eval_040 = evaluate_against_baseline(base_data, m_data, iou_threshold=0.40)
            eval_045 = evaluate_against_baseline(base_data, m_data, iou_threshold=0.45)

            summary_payload[video_name][m_key] = {
                "label": m_label,
                "filter_ms": filter_ms,
                "yolo_ms": yolo_ms,
                "total_ms": total_ms,
                "fps": round(fps_equiv, 2),
                "time_reduction_percent": round(reduction_pct, 2),
                "speedup": round(speedup, 2),
                "tiles_sent_percent": tiles_pct,
                "cpu_avg_percent": cpu_avg,
                "eval_iou_040": eval_040,
                "eval_iou_045": eval_045,
            }

            filter_str = f"{filter_ms:.2f}" if m_key != "baseline" else "—"
            md_lines.append(
                f"| **{m_label}** | {filter_str} | {yolo_ms:.2f} | **{total_ms:.2f}** | "
                f"{fps_equiv:.2f} | -{reduction_pct:.1f}% | {speedup:.2f}x | {tiles_pct:.1f}% | "
                f"**{eval_040['recall']:.2f}%** | {eval_040['precision']:.2f}% | {eval_040['f1']:.2f}% | "
                f"{eval_045['recall']:.2f}% | {eval_045['precision']:.2f}% |"
            )

            latex_rows.append(
                f"{m_label} & {filter_str} & {yolo_ms:.2f} & {total_ms:.2f} & {fps_equiv:.2f} & "
                f"{tiles_pct:.1f}\\% & {eval_040['precision']:.2f}\\% & {eval_040['recall']:.2f}\\% & {eval_040['f1']:.2f}\\% \\\\"
            )

        md_lines.append("")
        md_lines.append(
            "### Linhas prontas para a Tabela II em LaTeX (`access.tex` - IoU >= 0.40):"
        )
        md_lines.append("```latex")
        md_lines.extend(latex_rows)
        md_lines.append("```")
        md_lines.append("")

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n" + "=" * 70)
    print(f"✅ RELATÓRIO QUANTITATIVO GERADO COM SUCESSO:")
    print(f"   📄 Markdown + LaTeX : {report_md_path}")
    print(f"   📊 JSON Consolidado : {report_json_path}")
    print("=" * 70)


def main():
    base_dir = Path(__file__).parent.parent
    videos_dir = base_dir / "videos"
    clips_dir = base_dir / "apresentacao" / "clips"
    suite_results_dir = base_dir / "results" / "SUITE_APRESENTACAO"

    # Lista de vídeos para rodar na suíte:
    # O 1º vídeo (507 frames) gera todos os 13 clipes .mp4 da apresentação + métricas.
    # O 2º vídeo (1871 frames) roda apenas para métricas quantitativas (sem reescrever os clipes).
    videos_to_run = [
        (videos_dir / "VIRAT_S_010203_08_000895_000975.mp4", True),
        (videos_dir / "VIRAT_S_010004_01_000163_000188.mp4", False),
    ]

    WARMUP_FRAMES = 10
    tiler = Tiler(tile_size=(320, 320), grid_size=(5, 3))
    detector = YOLODetector(model_name="yolov8n.pt", conf_thresh=0.25, imgsz=640)
    consolidator = Consolidator(iou_threshold=0.45, iom_threshold=0.50)

    all_video_results: dict[str, dict[str, dict]] = {}

    for video_path, should_record_clips in videos_to_run:
        if not video_path.exists():
            print(f"[AVISO] Vídeo não encontrado, pulando: {video_path}")
            continue

        all_video_results[video_path.name] = {}

        # Fábrica dos 5 métodos na ordem de execução (usando as configurações campeãs da Bateria B)
        methods_spec = [
            ("baseline", "BASELINE", None, False),
            (
                "knn",
                "KNN",
                KNNDecision(
                    debug=False,
                    history=500,
                    dist2Threshold=200.0,
                    min_pixels=150,
                    scale_factor=0.50,
                ),
                True,
            ),
            (
                "framediff",
                "FRAMEDIFF",
                FrameDiffDecision(
                    debug=False,
                    min_pixels=75,
                    threshold=20,
                    scale_factor=0.75,
                    frame_step=3,
                ),
                True,
            ),
            (
                "mog2",
                "MOG2",
                BGSDecision(
                    debug=False, history=500, var_threshold=25.0, min_pixels=100
                ),
                True,
            ),
            (
                "hash",
                "HASH",
                HashDecision(
                    levels=(5, 10, 20, 40),
                    hash_type="WAVELET",
                    tolerance_ratio=0.015,
                    ttl_positive=6,
                    ttl_empty=30,
                    debug=False,
                ),
                False,
            ),
        ]

        for method_key, mode_folder, decision_instance, has_mask in methods_spec:
            res = run_single_method(
                method_key=method_key,
                mode_folder_name=mode_folder,
                decision_block=decision_instance,
                has_mask_video=has_mask,
                video_path=video_path,
                clips_dir=clips_dir,
                suite_results_dir=suite_results_dir,
                tiler=tiler,
                detector=detector,
                consolidator=consolidator,
                warmup_frames=WARMUP_FRAMES,
                record_clips=should_record_clips,
            )
            all_video_results[video_path.name][method_key] = res

    report_md = base_dir / "apresentacao" / "relatorio_quantitativo.md"
    report_json = base_dir / "apresentacao" / "relatorio_quantitativo.json"
    build_reports(all_video_results, report_md, report_json)


if __name__ == "__main__":
    main()
