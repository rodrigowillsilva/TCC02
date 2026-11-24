import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# Adiciona o diretório experiment ao path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.video_reader import VideoReader
from pipeline.tiler import Tiler
from pipeline.detector import YOLODetector
from pipeline.consolidator import Consolidator
from pipeline.profiler import Profiler
from evaluator.compare import calculate_iou

from decision.knn_decision import KNNDecision
from decision.frame_diff_decision import FrameDiffDecision
from decision.bgs import BGSDecision
from decision.hash_decision import HashDecision

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_hardware_info() -> dict:
    """Detecta automaticamente modelo da CPU, núcleos e RAM total da máquina."""
    cpu_name = platform.processor()
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                text=True,
                timeout=5,
            ).strip()
            if out:
                cpu_name = out.splitlines()[0].strip()
        except Exception:
            pass

    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": platform.python_version(),
        "cpu_model": cpu_name,
        "logical_cores": os.cpu_count() or 0,
    }
    if HAS_PSUTIL:
        info["physical_cores"] = psutil.cpu_count(logical=False) or 0
        info["total_ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    return info


def evaluate_predictions(base_frames: dict, exp_frames: dict, iou_threshold: float, warmup_frames: int = 10) -> dict:
    """
    Compara as caixas consolidadas do experimento contra o Baseline quadro a quadro
    (após os quadros de warmup para manter consistência com a janela medida).
    Retorna TP, FN, FP, Precision, Recall, F1 e IoU Médio dos TPs.
    """
    total_gt = 0
    tp = 0
    fp = 0
    frames_eval = 0
    matched_ious: list[float] = []

    for frame_idx_str, exp_info in exp_frames.items():
        frame_idx = int(frame_idx_str)
        if frame_idx < warmup_frames:
            continue
        if frame_idx_str not in base_frames:
            continue

        gt_info = base_frames[frame_idx_str]
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
                matched_ious.append(best_iou)
            else:
                fp += 1

    fn = total_gt - tp
    recall = (tp / total_gt * 100.0) if total_gt > 0 else 100.0
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 100.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = (sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0

    return {
        "iou_threshold": iou_threshold,
        "frames_evaluated": frames_eval,
        "gt_total": total_gt,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
        "mean_iou_tp": round(mean_iou, 4),
    }


def extract_stage_metrics(telemetry: dict, measured_frames: int, base_total_ms: float | None = None) -> dict:
    """
    Extrai as latências médias por quadro do vídeo (amortizando o YOLO pelo total de quadros medidos).
    """
    timings = telemetry.get("timings", {})

    def _stage_total_s(key: str) -> float:
        return float(timings.get(key, {}).get("total_seconds", 0.0))

    def _stage_calls(key: str) -> int:
        return int(timings.get(key, {}).get("calls", 0))

    denom = max(1, measured_frames)

    total_ms = (_stage_total_s("00_Pipeline_Total (por frame)") * 1000.0) / denom
    tiler_ms = (_stage_total_s("01_Fatiamento (Tiler)") * 1000.0) / denom
    filter_ms = (_stage_total_s("02_Filtro (DecisionBlock)") * 1000.0) / denom
    yolo_amortized_ms = (_stage_total_s("03_IA_Inferência (YOLO)") * 1000.0) / denom
    yolo_calls = _stage_calls("03_IA_Inferência (YOLO)")
    yolo_per_active_call_ms = (
        (_stage_total_s("03_IA_Inferência (YOLO)") * 1000.0) / yolo_calls if yolo_calls > 0 else 0.0
    )
    consolidator_ms = (_stage_total_s("04_Limpeza (Consolidator NMS)") * 1000.0) / denom

    fps = (1000.0 / total_ms) if total_ms > 0 else 0.0
    tile_sel = telemetry.get("tile_selection", {})
    tiles_sent_pct = float(tile_sel.get("selected_percent", 100.0))
    tiles_saved_pct = float(tile_sel.get("saved_percent", 0.0))
    cpu_avg = float(telemetry.get("cpu", {}).get("average_percent", 0.0))

    if base_total_ms and base_total_ms > 0:
        reduction_pct = ((base_total_ms - total_ms) / base_total_ms) * 100.0
        speedup = base_total_ms / total_ms if total_ms > 0 else 1.0
    else:
        reduction_pct = 0.0
        speedup = 1.0

    return {
        "measured_frames": measured_frames,
        "tiler_ms": round(tiler_ms, 2),
        "filter_ms": round(filter_ms, 2),
        "yolo_ms": round(yolo_amortized_ms, 2),
        "yolo_per_active_call_ms": round(yolo_per_active_call_ms, 2),
        "yolo_active_frames": yolo_calls,
        "consolidator_ms": round(consolidator_ms, 2),
        "total_ms": round(total_ms, 2),
        "fps": round(fps, 2),
        "time_reduction_percent": round(reduction_pct, 2),
        "speedup": round(speedup, 2),
        "tiles_sent_percent": round(tiles_sent_pct, 2),
        "tiles_saved_percent": round(tiles_saved_pct, 2),
        "cpu_avg_percent": round(cpu_avg, 2),
    }


def instantiate_decision(family: str, params: dict):
    """Cria uma nova instância limpa do bloco de decisão a partir do dicionário de hiperparâmetros."""
    if family == "baseline":
        return None
    elif family == "framediff":
        return FrameDiffDecision(debug=False, **params)
    elif family == "knn":
        return KNNDecision(debug=False, **params)
    elif family == "mog2":
        return BGSDecision(debug=False, **params)
    elif family == "hash":
        return HashDecision(debug=False, **params)
    else:
        raise ValueError(f"Família desconhecida: {family}")


def get_ablation_candidates() -> dict[str, list[dict]]:
    """
    Define as variações candidatas da Bateria B para cada método (usando apenas WAVELET para o Hash).
    """
    return {
        "framediff": [
            {
                "cand_id": "fdiff_p75_t20_s3",
                "label": "FrameDiff (p=75, thr=20, step=3)",
                "params": {"min_pixels": 75, "threshold": 20, "scale_factor": 0.75, "frame_step": 3},
            },
            {
                "cand_id": "fdiff_p100_t25_s3",
                "label": "FrameDiff (p=100, thr=25, step=3)",
                "params": {"min_pixels": 100, "threshold": 25, "scale_factor": 0.75, "frame_step": 3},
            },
            {
                "cand_id": "fdiff_p150_t25_s3",
                "label": "FrameDiff (p=150, thr=25, step=3)",
                "params": {"min_pixels": 150, "threshold": 25, "scale_factor": 0.75, "frame_step": 3},
            },
            {
                "cand_id": "fdiff_p250_t25_s3",
                "label": "FrameDiff (p=250, thr=25, step=3)",
                "params": {"min_pixels": 250, "threshold": 25, "scale_factor": 0.75, "frame_step": 3},
            },
            {
                "cand_id": "fdiff_p150_t25_s2",
                "label": "FrameDiff (p=150, thr=25, step=2)",
                "params": {"min_pixels": 150, "threshold": 25, "scale_factor": 0.75, "frame_step": 2},
            },
            {
                "cand_id": "fdiff_p150_t25_s5",
                "label": "FrameDiff (p=150, thr=25, step=5)",
                "params": {"min_pixels": 150, "threshold": 25, "scale_factor": 0.75, "frame_step": 5},
            },
        ],
        "knn": [
            {
                "cand_id": "knn_p75_d150_sf075",
                "label": "KNN (p=75, dist2=150, sf=0.75)",
                "params": {"history": 500, "dist2Threshold": 150.0, "min_pixels": 75, "scale_factor": 0.75},
            },
            {
                "cand_id": "knn_p100_d200_sf075",
                "label": "KNN (p=100, dist2=200, sf=0.75)",
                "params": {"history": 500, "dist2Threshold": 200.0, "min_pixels": 100, "scale_factor": 0.75},
            },
            {
                "cand_id": "knn_p150_d200_sf075",
                "label": "KNN (p=150, dist2=200, sf=0.75)",
                "params": {"history": 500, "dist2Threshold": 200.0, "min_pixels": 150, "scale_factor": 0.75},
            },
            {
                "cand_id": "knn_p250_d200_sf075",
                "label": "KNN (p=250, dist2=200, sf=0.75)",
                "params": {"history": 500, "dist2Threshold": 200.0, "min_pixels": 250, "scale_factor": 0.75},
            },
            {
                "cand_id": "knn_p150_d300_sf075",
                "label": "KNN (p=150, dist2=300, sf=0.75)",
                "params": {"history": 500, "dist2Threshold": 300.0, "min_pixels": 150, "scale_factor": 0.75},
            },
            {
                "cand_id": "knn_p150_d200_sf050",
                "label": "KNN (p=150, dist2=200, sf=0.50)",
                "params": {"history": 500, "dist2Threshold": 200.0, "min_pixels": 150, "scale_factor": 0.50},
            },
        ],
        "mog2": [
            {
                "cand_id": "mog2_p100_v25",
                "label": "MOG2 (p=100, var=25)",
                "params": {"history": 500, "var_threshold": 25.0, "min_pixels": 100},
            },
            {
                "cand_id": "mog2_p100_v36",
                "label": "MOG2 (p=100, var=36)",
                "params": {"history": 500, "var_threshold": 36.0, "min_pixels": 100},
            },
            {
                "cand_id": "mog2_p150_v50",
                "label": "MOG2 (p=150, var=50)",
                "params": {"history": 500, "var_threshold": 50.0, "min_pixels": 150},
            },
            {
                "cand_id": "mog2_p250_v50",
                "label": "MOG2 (p=250, var=50)",
                "params": {"history": 500, "var_threshold": 50.0, "min_pixels": 250},
            },
            {
                "cand_id": "mog2_p150_v75",
                "label": "MOG2 (p=150, var=75)",
                "params": {"history": 500, "var_threshold": 75.0, "min_pixels": 150},
            },
        ],
        "hash": [
            {
                "cand_id": "wavelet_L80_tol010_ttl6_30",
                "label": "Wavelet (5..80, tol=1.0%, TTL=6/30)",
                "params": {"levels": (5, 10, 20, 40, 80), "hash_type": "WAVELET", "tolerance_ratio": 0.010, "ttl_positive": 6, "ttl_empty": 30},
            },
            {
                "cand_id": "wavelet_L80_tol015_ttl6_30",
                "label": "Wavelet (5..80, tol=1.5%, TTL=6/30)",
                "params": {"levels": (5, 10, 20, 40, 80), "hash_type": "WAVELET", "tolerance_ratio": 0.015, "ttl_positive": 6, "ttl_empty": 30},
            },
            {
                "cand_id": "wavelet_L80_tol025_ttl6_30",
                "label": "Wavelet (5..80, tol=2.5%, TTL=6/30)",
                "params": {"levels": (5, 10, 20, 40, 80), "hash_type": "WAVELET", "tolerance_ratio": 0.025, "ttl_positive": 6, "ttl_empty": 30},
            },
            {
                "cand_id": "wavelet_L80_tol015_ttl3_15",
                "label": "Wavelet (5..80, tol=1.5%, TTL=3/15)",
                "params": {"levels": (5, 10, 20, 40, 80), "hash_type": "WAVELET", "tolerance_ratio": 0.015, "ttl_positive": 3, "ttl_empty": 15},
            },
            {
                "cand_id": "wavelet_L80_tol015_ttl12_60",
                "label": "Wavelet (5..80, tol=1.5%, TTL=12/60)",
                "params": {"levels": (5, 10, 20, 40, 80), "hash_type": "WAVELET", "tolerance_ratio": 0.015, "ttl_positive": 12, "ttl_empty": 60},
            },
            {
                "cand_id": "wavelet_L80_tol015_ttl0_0",
                "label": "Wavelet (5..80, tol=1.5%, Sem TTL 0/0)",
                "params": {"levels": (5, 10, 20, 40, 80), "hash_type": "WAVELET", "tolerance_ratio": 0.015, "ttl_positive": 0, "ttl_empty": 0},
            },
            {
                "cand_id": "wavelet_L40_tol015_ttl6_30",
                "label": "Wavelet (5..40, tol=1.5%, TTL=6/30)",
                "params": {"levels": (5, 10, 20, 40), "hash_type": "WAVELET", "tolerance_ratio": 0.015, "ttl_positive": 6, "ttl_empty": 30},
            },
            {
                "cand_id": "wavelet_L40_tol010_ttl3_20",
                "label": "Wavelet (5..40, tol=1.0%, TTL=3/20)",
                "params": {"levels": (5, 10, 20, 40), "hash_type": "WAVELET", "tolerance_ratio": 0.010, "ttl_positive": 3, "ttl_empty": 20},
            },
        ],
    }


def run_headless_benchmark(
    run_id: str,
    method_label: str,
    decision_block,
    video_path: Path,
    raw_dir: Path,
    tiler: Tiler,
    detector: YOLODetector,
    consolidator: Consolidator,
    warmup_frames: int = 10,
    resume_existing: bool = True,
) -> dict:
    """
    Executa um método sobre um vídeo sem gravar MP4 e sem abrir janelas GUI.
    Se o arquivo JSON bruto já existir em `raw_dir` e `resume_existing=True`, carrega do disco.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_json_path = raw_dir / f"{video_path.stem}__{run_id}.json"
    tmp_json_path = raw_dir / f"{video_path.stem}__{run_id}.json.tmp"

    # Remove arquivo temporário caso uma execução anterior tenha sido interrompida durante a gravação
    if tmp_json_path.exists():
        try:
            tmp_json_path.unlink()
        except OSError:
            pass

    if resume_existing and raw_json_path.exists():
        try:
            with open(raw_json_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            # Verifica se o JSON tem todas as chaves e frames completos
            if (
                isinstance(cached_data, dict)
                and "frames" in cached_data
                and "telemetry" in cached_data
                and len(cached_data["frames"]) > 0
            ):
                print(f"  [CACHE DISCO] Reaproveitando execução salva: {raw_json_path.name}")
                return cached_data
            else:
                print(f"  [LIMPEZA] Arquivo incompleto detectado e removido: {raw_json_path.name}")
                raw_json_path.unlink(missing_ok=True)
        except Exception:
            print(f"  [LIMPEZA] Arquivo corrompido detectado e removido: {raw_json_path.name}")
            raw_json_path.unlink(missing_ok=True)

    print(f"\n▶ Rodando [{method_label}] em {video_path.name} ...")
    reader = VideoReader(video_path)
    total_frames = reader.total_frames
    profiler = Profiler(warmup_frames=warmup_frames)

    frames_dict = {}
    total_tiles_possible = 0
    total_tiles_selected = 0
    measured_frames = 0
    t0 = time.perf_counter()

    try:
        for frame_data in reader:
            profiler.set_frame(frame_data.index)
            frame = frame_data.image

            with profiler.measure("00_Pipeline_Total (por frame)"):
                with profiler.measure("01_Fatiamento (Tiler)"):
                    tiles = tiler.split(frame)

                if decision_block is not None:
                    with profiler.measure("02_Filtro (DecisionBlock)"):
                        selected_indices = decision_block.select(frame, tiles, frame_data.index)
                else:
                    selected_indices = [t.index for t in tiles]

                if frame_data.index >= warmup_frames:
                    measured_frames += 1
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

            frames_dict[str(frame_data.index)] = {
                "timestamp_ms": frame_data.timestamp_ms,
                "selected_tiles_count": len(selected_indices),
                "consolidated": [
                    {"x": b.x, "y": b.y, "w": b.w, "h": b.h, "class": b.class_name, "conf": round(float(b.conf), 4)}
                    for b in global_boxes
                ],
            }

            if (frame_data.index + 1) % 100 == 0 or (frame_data.index + 1) == total_frames:
                elapsed = time.perf_counter() - t0
                pct = ((frame_data.index + 1) / total_frames * 100.0) if total_frames > 0 else 0.0
                print(
                    f"    Frame {frame_data.index + 1}/{total_frames} ({pct:.1f}%) | "
                    f"Tiles: {len(selected_indices)}/15 | Decorrido: {elapsed:.1f}s"
                )
    finally:
        reader.release()

    telemetry = profiler.get_summary_dict()
    telemetry["measured_frames"] = measured_frames
    tiles_ratio = (total_tiles_selected / total_tiles_possible * 100.0) if total_tiles_possible > 0 else 100.0
    telemetry["tile_selection"] = {
        "total_tiles_evaluated": total_tiles_possible,
        "total_tiles_sent_to_yolo": total_tiles_selected,
        "selected_percent": round(tiles_ratio, 2),
        "saved_percent": round(100.0 - tiles_ratio, 2),
    }

    payload = {
        "video": video_path.name,
        "run_id": run_id,
        "method_label": method_label,
        "config": getattr(decision_block, "config", {"method": "BASELINE"}),
        "telemetry": telemetry,
        "frames": frames_dict,
    }

    with open(tmp_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_json_path, raw_json_path)

    return payload


def select_best_candidates(
    candidates_by_family: dict[str, list[dict]],
    ablation_results_by_video: dict[str, dict[str, dict]],
) -> dict[str, dict]:
    """
    Avalia o desempenho médio de cada candidato nos 2 vídeos de ablação (7º e 13º)
    e elege o melhor de cada família usando um Score Combinado (Eficácia F1 + Eficiência Redução de Tempo):
      - Para métodos de movimento (FrameDiff, KNN, MOG2): 0.70 * F1(@0.40) + 0.30 * Redução_Tempo(%)
      - Para Hash Wavelet: 0.65 * F1(@0.40) + 0.35 * Redução_Tempo(%)
    """
    winners = {}
    ablation_videos = list(ablation_results_by_video.keys())

    print("\n" + "=" * 80)
    print("🏆 ELEIÇÃO DAS MELHORES CONFIGURAÇÕES (BATERIA B -> BATERIA A)")
    print("=" * 80)

    for family, cand_list in candidates_by_family.items():
        w_f1 = 0.65 if family == "hash" else 0.70
        w_red = 0.35 if family == "hash" else 0.30

        best_cand = None
        best_score = -1e9

        for cand in cand_list:
            cid = cand["cand_id"]
            entries = [
                ablation_results_by_video[v][cid]
                for v in ablation_videos
                if cid in ablation_results_by_video[v]
            ]
            if not entries:
                continue

            avg_f1 = sum(e["eval_iou_040"]["f1"] for e in entries) / len(entries)
            avg_rec = sum(e["eval_iou_040"]["recall"] for e in entries) / len(entries)
            avg_prec = sum(e["eval_iou_040"]["precision"] for e in entries) / len(entries)
            avg_red = sum(max(0.0, e["efficiency"]["time_reduction_percent"]) for e in entries) / len(entries)
            avg_ms = sum(e["efficiency"]["total_ms"] for e in entries) / len(entries)

            score = (w_f1 * avg_f1) + (w_red * avg_red)
            cand["ablation_avg"] = {
                "avg_f1_040": round(avg_f1, 2),
                "avg_recall_040": round(avg_rec, 2),
                "avg_precision_040": round(avg_prec, 2),
                "avg_time_reduction_percent": round(avg_red, 2),
                "avg_total_ms": round(avg_ms, 2),
                "selection_score": round(score, 2),
            }

            if score > best_score:
                best_score = score
                best_cand = cand

        winners[family] = best_cand
        avg_info = best_cand["ablation_avg"]
        print(
            f"  ★ [{family.upper():<9}] Vencedor: {best_cand['label']:<38} | "
            f"F1: {avg_info['avg_f1_040']:.2f}% (Rec: {avg_info['avg_recall_040']:.2f}%) | "
            f"Redução: -{avg_info['avg_time_reduction_percent']:.1f}% ({avg_info['avg_total_ms']:.1f}ms) | "
            f"Score: {avg_info['selection_score']:.2f}"
        )

    print("=" * 80)
    return winners


def aggregate_bateria_a(per_video_summary: dict, method_order: list[tuple[str, str]]) -> dict:
    """Calcula a média global e totais acumulados (micro e macro) entre todos os vídeos da Bateria A."""
    agg = {}
    video_names = list(per_video_summary.keys())
    if not video_names:
        return agg

    base_totals = [
        per_video_summary[v]["baseline"]["efficiency"]["total_ms"]
        for v in video_names
        if "baseline" in per_video_summary[v]
    ]
    global_base_ms = (sum(base_totals) / len(base_totals)) if base_totals else 1.0

    for m_key, m_label in method_order:
        v_entries = [per_video_summary[v][m_key] for v in video_names if m_key in per_video_summary[v]]
        if not v_entries:
            continue

        n = len(v_entries)
        avg_tiler = sum(e["efficiency"]["tiler_ms"] for e in v_entries) / n
        avg_filter = sum(e["efficiency"]["filter_ms"] for e in v_entries) / n
        avg_yolo = sum(e["efficiency"]["yolo_ms"] for e in v_entries) / n
        avg_cons = sum(e["efficiency"]["consolidator_ms"] for e in v_entries) / n
        avg_total = sum(e["efficiency"]["total_ms"] for e in v_entries) / n
        avg_tiles_sent = sum(e["efficiency"]["tiles_sent_percent"] for e in v_entries) / n
        avg_tiles_saved = 100.0 - avg_tiles_sent
        avg_cpu = sum(e["efficiency"]["cpu_avg_percent"] for e in v_entries) / n

        fps_global = (1000.0 / avg_total) if avg_total > 0 else 0.0
        red_global = ((global_base_ms - avg_total) / global_base_ms * 100.0) if global_base_ms > 0 else 0.0
        speedup_global = (global_base_ms / avg_total) if avg_total > 0 else 1.0

        def _agg_eval(eval_key: str) -> dict:
            tot_gt = sum(e[eval_key]["gt_total"] for e in v_entries)
            tot_tp = sum(e[eval_key]["tp"] for e in v_entries)
            tot_fn = sum(e[eval_key]["fn"] for e in v_entries)
            tot_fp = sum(e[eval_key]["fp"] for e in v_entries)
            tot_frames = sum(e[eval_key]["frames_evaluated"] for e in v_entries)

            micro_rec = (tot_tp / tot_gt * 100.0) if tot_gt > 0 else 100.0
            micro_prec = (tot_tp / (tot_tp + tot_fp) * 100.0) if (tot_tp + tot_fp) > 0 else 100.0
            micro_f1 = (
                (2 * micro_prec * micro_rec / (micro_prec + micro_rec))
                if (micro_prec + micro_rec) > 0
                else 0.0
            )

            macro_rec = sum(e[eval_key]["recall"] for e in v_entries) / n
            macro_prec = sum(e[eval_key]["precision"] for e in v_entries) / n
            macro_f1 = sum(e[eval_key]["f1"] for e in v_entries) / n
            mean_iou = sum(e[eval_key]["mean_iou_tp"] for e in v_entries) / n

            return {
                "frames_evaluated": tot_frames,
                "gt_total": tot_gt,
                "tp": tot_tp,
                "fn": tot_fn,
                "fp": tot_fp,
                "micro_precision": round(micro_prec, 2),
                "micro_recall": round(micro_rec, 2),
                "micro_f1": round(micro_f1, 2),
                "macro_precision": round(macro_prec, 2),
                "macro_recall": round(macro_rec, 2),
                "macro_f1": round(macro_f1, 2),
                "mean_iou_tp": round(mean_iou, 4),
            }

        agg[m_key] = {
            "label": m_label,
            "videos_count": n,
            "efficiency": {
                "tiler_ms": round(avg_tiler, 2),
                "filter_ms": round(avg_filter, 2),
                "yolo_ms": round(avg_yolo, 2),
                "consolidator_ms": round(avg_cons, 2),
                "total_ms": round(avg_total, 2),
                "fps": round(fps_global, 2),
                "time_reduction_percent": round(red_global, 2),
                "speedup": round(speedup_global, 2),
                "tiles_sent_percent": round(avg_tiles_sent, 2),
                "tiles_saved_percent": round(avg_tiles_saved, 2),
                "cpu_avg_percent": round(avg_cpu, 2),
            },
            "eval_iou_040": _agg_eval("eval_iou_040"),
            "eval_iou_050": _agg_eval("eval_iou_050"),
        }

    return agg


def write_official_outputs(
    out_dir: Path,
    hw_info: dict,
    winners: dict,
    candidates_by_family: dict,
    ablation_results_by_video: dict,
    per_video_summary: dict,
    global_aggregate: dict,
    method_order: list[tuple[str, str]],
):
    """Salva `official_results.json` e `official_report.md`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "official_results.json"
    md_path = out_dir / "official_report.md"

    winners_clean = {
        fam: {
            "cand_id": w["cand_id"],
            "label": w["label"],
            "params": {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in w["params"].items()
            },
            "ablation_avg": w.get("ablation_avg", {}),
        }
        for fam, w in winners.items()
        if w is not None
    }

    full_json = {
        "hardware": hw_info,
        "elected_champions_from_bateria_b": winners_clean,
        "global_aggregate_bateria_a": global_aggregate,
        "per_video_bateria_a": per_video_summary,
        "ablation_bateria_b": ablation_results_by_video,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_json, f, indent=2)

    lines = [
        "# Relatório Oficial de Experimentos — Artigo TCC02",
        "",
        "## 1. Ambiente de Hardware Detectado",
        f"- **CPU:** `{hw_info.get('cpu_model')}` ({hw_info.get('physical_cores', '?')} núcleos físicos / {hw_info.get('logical_cores', '?')} lógicos)",
        f"- **Memória RAM Total:** `{hw_info.get('total_ram_gb', '?')} GB`",
        f"- **Sistema Operacional:** `{hw_info.get('os')}` | Python `{hw_info.get('python_version')}`",
        "",
        "---",
        "",
        "## 2. Fase 1 (Bateria B) — Calibração e Estudo de Ablação (Vídeos #7 e #13)",
        "",
    ]

    if winners_clean:
        lines.append("### 2.1. Configurações Campeãs Eleitas para a Bateria Principal (Tabela I do Artigo)")
        lines.append("")
        lines.append("| Família | Configuração Eleita | Parâmetros | F1 Médio (@0.40) | Recall Médio | Redução de Tempo | Score |")
        lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: |")
        for fam, w in winners_clean.items():
            avg = w.get("ablation_avg", {})
            p_str = ", ".join(f"`{k}={v}`" for k, v in w["params"].items())
            lines.append(
                f"| **{fam.upper()}** | **{w['label']}** | {p_str} | "
                f"**{avg.get('avg_f1_040', 0):.2f}%** | {avg.get('avg_recall_040', 0):.2f}% | "
                f"-{avg.get('avg_time_reduction_percent', 0):.1f}% | **{avg.get('selection_score', 0):.2f}** |"
            )
        lines.append("")

    for v_name, ab_dict in ablation_results_by_video.items():
        lines.append(f"### 2.2. Resultados de Todas as Variações no Vídeo de Ablação: `{v_name}`")
        lines.append("")
        lines.append(
            "| Candidato | Filtro (ms) | YOLO (ms) | Total (ms) | FPS | Redução (%) | Speedup | Tiles (%) | Precision @0.40 | Recall @0.40 | F1 @0.40 |"
        )
        lines.append(
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        )
        for cid, item in ab_dict.items():
            eff = item["efficiency"]
            e40 = item["eval_iou_040"]
            f_str = f"{eff['filter_ms']:.2f}" if cid != "baseline" else "—"
            is_winner = any(w.get("cand_id") == cid for w in winners_clean.values())
            prefix = "🏆 " if is_winner else ""
            lines.append(
                f"| {prefix}**{item['label']}** | {f_str} | {eff['yolo_ms']:.2f} | "
                f"**{eff['total_ms']:.2f}** | {eff['fps']:.2f} | -{eff['time_reduction_percent']:.1f}% | "
                f"{eff['speedup']:.2f}x | {eff['tiles_sent_percent']:.1f}% | "
                f"{e40['precision']:.2f}% | **{e40['recall']:.2f}%** | **{e40['f1']:.2f}%** |"
            )
        lines.append("")

    if global_aggregate:
        lines.extend([
            "---",
            "",
            "## 3. Fase 2 (Bateria A) — Tabela Global Consolidada (Todos os Vídeos com as Melhores Configurações)",
            "",
            "| Estratégia | Filtro (ms) | YOLO (ms) | Total (ms) | FPS | Redução (%) | Speedup | Tiles Enviados (%) | Precision @0.40 | Recall @0.40 | F1 @0.40 | IoU Médio (TP) | Recall @0.50 | F1 @0.50 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        latex_global_rows = []
        for m_key, m_label in method_order:
            if m_key not in global_aggregate:
                continue
            g = global_aggregate[m_key]
            eff = g["efficiency"]
            e40 = g["eval_iou_040"]
            e50 = g["eval_iou_050"]

            f_str = f"{eff['filter_ms']:.2f}" if m_key != "baseline" else "—"
            lines.append(
                f"| **{m_label}** | {f_str} | {eff['yolo_ms']:.2f} | **{eff['total_ms']:.2f}** | "
                f"{eff['fps']:.2f} | -{eff['time_reduction_percent']:.1f}% | {eff['speedup']:.2f}x | "
                f"{eff['tiles_sent_percent']:.1f}% | {e40['micro_precision']:.2f}% | "
                f"**{e40['micro_recall']:.2f}%** | **{e40['micro_f1']:.2f}%** | {e40['mean_iou_tp']:.3f} | "
                f"{e50['micro_recall']:.2f}% | {e50['micro_f1']:.2f}% |"
            )

            latex_global_rows.append(
                f"{m_label} & {f_str} & {eff['yolo_ms']:.2f} & {eff['total_ms']:.2f} & "
                f"{eff['fps']:.2f} & {eff['tiles_sent_percent']:.1f}\\% & "
                f"{e40['micro_precision']:.2f}\\% & {e40['micro_recall']:.2f}\\% & {e40['micro_f1']:.2f}\\% \\\\"
            )

        lines.extend([
            "",
            "### 3.1. Código LaTeX pronto para a Tabela II do `access.tex` (Agregado Global):",
            "```latex",
            "% Colunas: Estratégia | Filtro (ms) | YOLO (ms) | Total (ms) | FPS | Tiles (%) | Precision | Recall | F1",
            *latex_global_rows,
            "```",
            "",
            "---",
            "",
            "## 4. Resultados Individuais por Vídeo (Bateria A)",
            "",
        ])

        for v_name, m_dict in per_video_summary.items():
            lines.append(f"### Vídeo: `{v_name}`")
            lines.append("")
            lines.append(
                "| Estratégia | Filtro (ms) | YOLO (ms) | Total (ms) | FPS | Speedup | Tiles Enviados (%) | TP | FN | FP | Precision @0.40 | Recall @0.40 | F1 @0.40 |"
            )
            lines.append(
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
            )
            for m_key, m_label in method_order:
                if m_key not in m_dict:
                    continue
                entry = m_dict[m_key]
                eff = entry["efficiency"]
                e40 = entry["eval_iou_040"]
                f_str = f"{eff['filter_ms']:.2f}" if m_key != "baseline" else "—"
                lines.append(
                    f"| **{m_label}** | {f_str} | {eff['yolo_ms']:.2f} | **{eff['total_ms']:.2f}** | "
                    f"{eff['fps']:.2f} | {eff['speedup']:.2f}x | {eff['tiles_sent_percent']:.1f}% | "
                    f"{e40['tp']} | {e40['fn']} | {e40['fp']} | "
                    f"{e40['precision']:.2f}% | **{e40['recall']:.2f}%** | {e40['f1']:.2f}% |"
                )
            lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    base_dir = Path(__file__).parent.parent
    exp_videos_dir = base_dir / "videos" / "videos_experimento"
    if not exp_videos_dir.exists():
        exp_videos_dir = base_dir / "videos_experimento"

    out_dir = base_dir / "results" / "OFFICIAL_SUITE"
    raw_dir = out_dir / "raw"

    video_files = sorted(exp_videos_dir.glob("*.mp4"))
    if not video_files:
        print(f"[ERRO] Nenhum vídeo .mp4 encontrado em: {exp_videos_dir}")
        return

    WARMUP_FRAMES = 10
    RESUME_EXISTING = True

    # Seleciona exatamente o 7º (índice 6) e o 13º (índice 12) vídeo da pasta ordenada para a Bateria B
    idx_7 = min(6, len(video_files) - 1)
    idx_13 = min(12, len(video_files) - 1)
    ablation_videos = [video_files[idx_7], video_files[idx_13]] if idx_7 != idx_13 else [video_files[idx_7]]

    hw_info = get_hardware_info()
    print("=" * 80)
    print("🔬 SUÍTE OFICIAL DE EXPERIMENTOS — ARTIGO TCC02 (BATERIA B -> ELEIÇÃO -> BATERIA A)")
    print("=" * 80)
    print(f"🖥️  CPU Detectada   : {hw_info.get('cpu_model')} ({hw_info.get('logical_cores')} threads)")
    print(f"💾 RAM Total       : {hw_info.get('total_ram_gb', '?')} GB")
    print(f"📂 Pasta Vídeos    : {exp_videos_dir} ({len(video_files)} vídeos)")
    print(f"🧪 Vídeos Ablação  : 7º = {ablation_videos[0].name} | 13º = {ablation_videos[-1].name}")
    print(f"📊 Saída Oficial   : {out_dir}")
    print("=" * 80)

    tiler = Tiler(tile_size=(320, 320), grid_size=(5, 3))
    detector = YOLODetector(model_name="yolov8n.pt", conf_thresh=0.25, imgsz=640)
    consolidator = Consolidator(iou_threshold=0.45, iom_threshold=0.50)

    candidates_by_family = get_ablation_candidates()
    ablation_results_by_video: dict[str, dict[str, dict]] = {}
    per_video_summary: dict[str, dict] = {}
    winners: dict[str, dict] = {}

    method_order = [
        ("baseline", "Baseline (15 tiles)"),
        ("framediff", "FrameDiff"),
        ("knn", "KNN"),
        ("mog2", "MOG2"),
        ("hash", "Hash Wavelet"),
    ]

    # =========================================================================
    # FASE 1: BATERIA B (ABLAÇÃO NOS VÍDEOS 7º E 13º DA PASTA)
    # =========================================================================
    print("\n" + "=" * 80)
    print("🧪 FASE 1: EXECUTANDO BATERIA B (ESTUDO DE ABLAÇÃO NOS VÍDEOS 7º E 13º)")
    print("=" * 80)

    for v_pos, video_path in zip([7, 13], ablation_videos):
        print(f"\n[BATERIA B - {v_pos}º Vídeo da Pasta] 🎬 {video_path.name}")
        ablation_results_by_video[video_path.name] = {}

        # 1. Roda o Baseline neste vídeo
        base_payload = run_headless_benchmark(
            run_id="cand__baseline",
            method_label="Baseline (15 tiles)",
            decision_block=None,
            video_path=video_path,
            raw_dir=raw_dir,
            tiler=tiler,
            detector=detector,
            consolidator=consolidator,
            warmup_frames=WARMUP_FRAMES,
            resume_existing=RESUME_EXISTING,
        )
        measured_base = base_payload["telemetry"].get(
            "measured_frames", max(1, len(base_payload["frames"]) - WARMUP_FRAMES)
        )
        base_eff = extract_stage_metrics(base_payload["telemetry"], measured_base, base_total_ms=None)
        base_total_ms = base_eff["total_ms"]

        ablation_results_by_video[video_path.name]["baseline"] = {
            "cand_id": "baseline",
            "label": "Baseline (15 tiles)",
            "config": base_payload["config"],
            "efficiency": base_eff,
            "eval_iou_040": evaluate_predictions(base_payload["frames"], base_payload["frames"], 0.40, WARMUP_FRAMES),
            "eval_iou_050": evaluate_predictions(base_payload["frames"], base_payload["frames"], 0.50, WARMUP_FRAMES),
        }

        # 2. Roda todas as variações candidatas de FrameDiff, KNN, MOG2 e Hash Wavelet
        for family, cand_list in candidates_by_family.items():
            for cand in cand_list:
                cid = cand["cand_id"]
                clabel = cand["label"]
                decision_instance = instantiate_decision(family, cand["params"])

                payload = run_headless_benchmark(
                    run_id=f"cand__{cid}",
                    method_label=clabel,
                    decision_block=decision_instance,
                    video_path=video_path,
                    raw_dir=raw_dir,
                    tiler=tiler,
                    detector=detector,
                    consolidator=consolidator,
                    warmup_frames=WARMUP_FRAMES,
                    resume_existing=RESUME_EXISTING,
                )
                measured = payload["telemetry"].get(
                    "measured_frames", max(1, len(payload["frames"]) - WARMUP_FRAMES)
                )
                eff = extract_stage_metrics(payload["telemetry"], measured, base_total_ms=base_total_ms)
                eval_040 = evaluate_predictions(base_payload["frames"], payload["frames"], 0.40, WARMUP_FRAMES)
                eval_050 = evaluate_predictions(base_payload["frames"], payload["frames"], 0.50, WARMUP_FRAMES)

                ablation_results_by_video[video_path.name][cid] = {
                    "cand_id": cid,
                    "family": family,
                    "label": clabel,
                    "config": payload["config"],
                    "efficiency": eff,
                    "eval_iou_040": eval_040,
                    "eval_iou_050": eval_050,
                }

        # Salva checkpoint parcial da Bateria B
        write_official_outputs(
            out_dir, hw_info, winners, candidates_by_family,
            ablation_results_by_video, per_video_summary, {}, method_order
        )

    # =========================================================================
    # FASE 2: ELEIÇÃO AUTOMÁTICA DA MELHOR CONFIGURAÇÃO DE CADA MÉTODO
    # =========================================================================
    winners = select_best_candidates(candidates_by_family, ablation_results_by_video)
    method_order = [
        ("baseline", "Baseline (15 tiles)"),
        ("framediff", winners["framediff"]["label"]),
        ("knn", winners["knn"]["label"]),
        ("mog2", winners["mog2"]["label"]),
        ("hash", winners["hash"]["label"]),
    ]

    # =========================================================================
    # FASE 3: BATERIA A (TODOS OS 20 VÍDEOS USANDO AS CONFIGURAÇÕES CAMPEÃS!)
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"🚀 FASE 3: EXECUTANDO BATERIA A EM TODOS OS {len(video_files)} VÍDEOS COM OS CAMPEÕES ELEITOS")
    print("=" * 80)

    for v_idx, video_path in enumerate(video_files, start=1):
        print(f"\n[BATERIA A - Vídeo {v_idx}/{len(video_files)}] 🎬 {video_path.name}")
        per_video_summary[video_path.name] = {}

        # 1. Baseline (se já rodou na Bateria B para o 7º ou 13º vídeo, carrega direto do cache `cand__baseline`!)
        base_payload = run_headless_benchmark(
            run_id="cand__baseline",
            method_label="Baseline (15 tiles)",
            decision_block=None,
            video_path=video_path,
            raw_dir=raw_dir,
            tiler=tiler,
            detector=detector,
            consolidator=consolidator,
            warmup_frames=WARMUP_FRAMES,
            resume_existing=RESUME_EXISTING,
        )
        measured_base = base_payload["telemetry"].get(
            "measured_frames", max(1, len(base_payload["frames"]) - WARMUP_FRAMES)
        )
        base_eff = extract_stage_metrics(base_payload["telemetry"], measured_base, base_total_ms=None)
        base_total_ms = base_eff["total_ms"]

        per_video_summary[video_path.name]["baseline"] = {
            "label": "Baseline (15 tiles)",
            "config": base_payload["config"],
            "efficiency": base_eff,
            "eval_iou_040": evaluate_predictions(base_payload["frames"], base_payload["frames"], 0.40, WARMUP_FRAMES),
            "eval_iou_050": evaluate_predictions(base_payload["frames"], base_payload["frames"], 0.50, WARMUP_FRAMES),
        }

        # 2. Roda os 4 Campeões Eleitos (nos vídeos 7º e 13º o `cand__<cid>` já existe no disco e é carregado em 0s!)
        for family in ["framediff", "knn", "mog2", "hash"]:
            win_cand = winners[family]
            cid = win_cand["cand_id"]
            clabel = win_cand["label"]
            decision_instance = instantiate_decision(family, win_cand["params"])

            payload = run_headless_benchmark(
                run_id=f"cand__{cid}",
                method_label=clabel,
                decision_block=decision_instance,
                video_path=video_path,
                raw_dir=raw_dir,
                tiler=tiler,
                detector=detector,
                consolidator=consolidator,
                warmup_frames=WARMUP_FRAMES,
                resume_existing=RESUME_EXISTING,
            )
            measured = payload["telemetry"].get(
                "measured_frames", max(1, len(payload["frames"]) - WARMUP_FRAMES)
            )
            eff = extract_stage_metrics(payload["telemetry"], measured, base_total_ms=base_total_ms)
            eval_040 = evaluate_predictions(base_payload["frames"], payload["frames"], 0.40, WARMUP_FRAMES)
            eval_050 = evaluate_predictions(base_payload["frames"], payload["frames"], 0.50, WARMUP_FRAMES)

            per_video_summary[video_path.name][family] = {
                "label": clabel,
                "cand_id": cid,
                "config": payload["config"],
                "efficiency": eff,
                "eval_iou_040": eval_040,
                "eval_iou_050": eval_050,
            }

        global_agg = aggregate_bateria_a(per_video_summary, method_order)
        write_official_outputs(
            out_dir, hw_info, winners, candidates_by_family,
            ablation_results_by_video, per_video_summary, global_agg, method_order
        )
        print(f"  [✓] Checkpoint atualizado ({v_idx}/{len(video_files)} vídeos) em: {out_dir / 'official_report.md'}")

    print("\n" + "=" * 80)
    print("🎉 SUÍTE OFICIAL COMPLETA (BATERIA B + ELEIÇÃO + BATERIA A) CONCLUÍDA!")
    print(f"📄 Relatório Markdown + LaTeX : {out_dir / 'official_report.md'}")
    print(f"📊 Dados Completos JSON       : {out_dir / 'official_results.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
