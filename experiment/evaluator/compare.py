import json
import argparse
from pathlib import Path


def calculate_iou(box1, box2):
    """
    Calcula a métrica IoU (Intersection over Union) entre duas caixas.
    IoU = Área de Interseção / Área de União.
    Retorna um valor entre 0.0 (nenhuma sobreposição) e 1.0 (sobreposição perfeita).
    """
    # Coordenadas: x_esqueda, y_topo, x_direita, y_baixo
    b1_x1, b1_y1 = box1['x'], box1['y']
    b1_x2, b1_y2 = b1_x1 + box1['w'], b1_y1 + box1['h']
    
    b2_x1, b2_y1 = box2['x'], box2['y']
    b2_x2, b2_y2 = b2_x1 + box2['w'], b2_y1 + box2['h']
    
    # Descobre o retângulo de interseção
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    # Se não se cruzam
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
        
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    box1_area = box1['w'] * box1['h']
    box2_area = box2['w'] * box2['h']
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0:
        return 0.0
        
    return inter_area / float(union_area)


def compare_jsons(gt_path: str, exp_path: str, iou_threshold: float = 0.4):
    print(f"=== AVALIADOR DE RESULTADOS ===")
    print(f"Ground Truth (Base) : {gt_path}")
    print(f"Experimento         : {exp_path}")
    print(f"Tolerância (IoU)    : {iou_threshold*100}% de sobreposição mínima\n")
    
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
        
    with open(exp_path, 'r') as f:
        exp_data = json.load(f)
        
    gt_frames = gt_data.get("frames", {})
    exp_frames = exp_data.get("frames", {})
    
    total_pessoas_gt = 0      # Quantas pessoas realmente existiam
    total_pessoas_achadas = 0 # Quantas o experimento achou corretamente (True Positives)
    total_falsos_positivos = 0 # Quantas o experimento inventou (False Positives)
    
    # Vamos avaliar até o último frame do experimento (caso você feche a janela antes do fim)
    avaliados = 0
    
    for frame_idx, exp_frame_info in exp_frames.items():
        if str(frame_idx) not in gt_frames:
            continue # Se o GT não tem esse frame, pula
            
        gt_frame_info = gt_frames[str(frame_idx)]
        avaliados += 1
        
        gt_boxes = gt_frame_info.get("consolidated", [])
        exp_boxes = exp_frame_info.get("consolidated", [])
        
        total_pessoas_gt += len(gt_boxes)
        
        # Para cada caixa do experimento, vamos ver se ela "bate" com alguma do GT
        exp_matched = [False] * len(exp_boxes)
        gt_matched = [False] * len(gt_boxes)
        
        for i, exp_b in enumerate(exp_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            # Procura a caixa do GT que mais se sobrepõe
            for j, gt_b in enumerate(gt_boxes):
                if not gt_matched[j]:
                    iou = calculate_iou(exp_b, gt_b)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                        
            # Se a sobreposição foi maior que a nossa tolerância matemática
            if best_iou >= iou_threshold:
                exp_matched[i] = True
                gt_matched[best_gt_idx] = True
                total_pessoas_achadas += 1
            else:
                # O experimento desenhou uma caixa, mas não bateu com ninguém do GT
                total_falsos_positivos += 1
                
    # ==== MATEMÁTICA FINAL (MÉTRICAS) ====
    total_perdidas = total_pessoas_gt - total_pessoas_achadas # Falsos Negativos
    
    recall = (total_pessoas_achadas / total_pessoas_gt) * 100 if total_pessoas_gt > 0 else 0
    precision = (total_pessoas_achadas / (total_pessoas_achadas + total_falsos_positivos)) * 100 if (total_pessoas_achadas + total_falsos_positivos) > 0 else 0
    
    print(f"Frames Avaliados: {avaliados}")
    print("-" * 30)
    print(f"Pessoas na Vida Real (GT) : {total_pessoas_gt}")
    print(f"Pessoas Encontradas (TP)  : {total_pessoas_achadas}")
    print(f"Pessoas Perdidas (FN)     : {total_perdidas}")
    print(f"Alucinações (FP)          : {total_falsos_positivos}")
    print("-" * 30)
    print(f"RECALL (Taxa de Acerto)   : {recall:.2f}% (Quantas pessoas verdadeiras nós detectamos)")
    print(f"PRECISION (Precisão)      : {precision:.2f}% (Quando ele apita, qual a chance de estar certo)")
    print("-" * 30)

    # ==== TELEMETRIA & COMPARAÇÃO DE LATÊNCIA ====
    gt_telemetry = gt_data.get("telemetry", {})
    exp_telemetry = exp_data.get("telemetry", {})

    gt_timings = gt_telemetry.get("timings", {})
    exp_timings = exp_telemetry.get("timings", {})

    if exp_timings:
        print("\n=== LATÊNCIA E DESEMPENHO (MÉDIA POR ETAPA) ===")
        all_steps = sorted(set(list(gt_timings.keys()) + list(exp_timings.keys())))
        
        has_gt_timings = bool(gt_timings)
        if has_gt_timings:
            print(f"{'Etapa':<32} | {'Exp (ms)':<10} | {'GT/Base (ms)':<12} | {'Ganho / Variação':<18}")
            print("-" * 80)
        else:
            print(f"{'Etapa':<32} | {'Exp (ms)':<10}")
            print("-" * 46)

        for step in all_steps:
            exp_step = exp_timings.get(step, {})
            exp_avg = exp_step.get("avg_ms", None)

            if not has_gt_timings:
                if exp_avg is not None:
                    print(f"{step:<32} | {exp_avg:>8.2f} ms")
                continue

            gt_step = gt_timings.get(step, {})
            gt_avg = gt_step.get("avg_ms", None)

            exp_str = f"{exp_avg:>8.2f} ms" if exp_avg is not None else "     N/A  "
            gt_str = f"{gt_avg:>8.2f} ms" if gt_avg is not None else "     N/A  "

            delta_str = ""
            if exp_avg is not None and gt_avg is not None and gt_avg > 0:
                diff = gt_avg - exp_avg
                pct = (diff / gt_avg) * 100
                if diff > 0:
                    delta_str = f"▼ {diff:>7.2f} ms (-{pct:.1f}% mais rápido)"
                elif diff < 0:
                    delta_str = f"▲ {abs(diff):>7.2f} ms (+{abs(pct):.1f}% mais lento)"
                else:
                    delta_str = "= 0.00 ms (igual)"
            else:
                delta_str = "-"

            print(f"{step:<32} | {exp_str} | {gt_str} | {delta_str}")

        # Comparação de CPU
        gt_cpu = gt_telemetry.get("cpu", {}).get("average_percent")
        exp_cpu = exp_telemetry.get("cpu", {}).get("average_percent")
        if exp_cpu is not None:
            print("-" * (80 if has_gt_timings else 46))
            if gt_cpu is not None:
                cpu_diff = gt_cpu - exp_cpu
                signal = "-" if cpu_diff >= 0 else "+"
                print(f"🖥️  Uso Médio de CPU: Exp = {exp_cpu:.1f}% | Base = {gt_cpu:.1f}% | Diferença = {signal}{abs(cpu_diff):.1f}%")
            else:
                print(f"🖥️  Uso Médio de CPU (Exp): {exp_cpu:.1f}%")
        print("=" * (80 if has_gt_timings else 46) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compara o JSON do experimento com o Ground Truth (GT) usando IoU.")
    parser.add_argument("--gt", type=str, help="Caminho para o arquivo JSON do Ground Truth")
    parser.add_argument("--exp", type=str, help="Caminho para o arquivo JSON do Experimento (Ex: results_KNN.json)")
    parser.add_argument("--iou", type=float, default=0.4, help="Tolerância IoU (0.0 a 1.0). Padrão: 0.4")
    
    args = parser.parse_args()

    # Define os caminhos padrão caso o usuário não passe os argumentos no terminal
    base_dir = Path(__file__).parent.parent
    gt_path = args.gt if args.gt else str(base_dir.parent / "ground_truth" / "gt_VIRAT_S_010004.json")
    exp_path = args.exp if args.exp else str(base_dir / "results_KNN.json")
    
    # Corrige a busca pela pasta "ground_truth" baseada na estrutura do projeto
    if not args.gt:
        gt_path = str(base_dir.parent / "ground_truth" / "gt_VIRAT_S_010004.json")
        # Fallback caso rode de outra pasta
        if not Path(gt_path).exists():
             gt_path = str(Path(__file__).parent.parent.parent / "ground_truth" / "gt_VIRAT_S_010004.json")

    gt_file = Path(gt_path)
    exp_file = Path(exp_path)
    
    if not gt_file.exists():
        print(f"Erro: Ground Truth não encontrado em {gt_file}")
    elif not exp_file.exists():
        print(f"Erro: Resultado do experimento não encontrado em {exp_file}")
        print("Dica: Você rodou o 'main.py' pra gerar o arquivo de resultados?")
    else:
        compare_jsons(str(gt_file), str(exp_file), iou_threshold=args.iou)

