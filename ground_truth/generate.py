import sys
import json
import time
from pathlib import Path

# Adiciona a pasta "experiment" no PATH para podermos importar tudo do pipeline
base_dir = Path(__file__).parent.parent
experiment_dir = base_dir / "experiment"
sys.path.insert(0, str(experiment_dir))

from pipeline.video_reader import VideoReader
from pipeline.tiler import Tiler
from pipeline.detector import YOLODetector
from pipeline.consolidator import Consolidator


def generate_ground_truth(video_path: str, output_json: str):
    print(f"=== GERADOR DE GROUND TRUTH ===")
    print(f"Vídeo: {video_path}")
    
    reader = VideoReader(video_path)
    # A base de comparação recorta e processa absolutamente TUDO (sem pular nada)
    # Nova Grade otimizada: 640x640 com 4 colunas e 2 linhas (8 tiles totais em vez de 24)
    tiler = Tiler(tile_size=(640, 640), grid_size=(4, 2))
    
    # ⚠️ ATUALIZADO: Usando o YOLOv8x (Extra-Large). 
    # É super pesado, mas garantirá a melhor precisão possível para ser o "Gabarito" (Ground Truth).
    detector = YOLODetector(model_name="yolov8x.pt", conf_thresh=0.25, imgsz=640)
    consolidator = Consolidator(iou_threshold=0.45)
    
    results = {
        "video": str(video_path),
        "frames": {}
    }
    
    for frame_data in reader:
        frame = frame_data.image
        
        # 1. Fatia
        tiles = tiler.split(frame)
        
        # 2. Detecta em TODOS os tiles (com medição de velocidade)
        t0 = time.perf_counter()
        tile_detections = detector.detect(tiles)
        dt = time.perf_counter() - t0
        
        tps = len(tiles) / dt if dt > 0 else 0
        
        # Mostra o progresso sem pular linha
        print(f"Processando frame {frame_data.index}/{reader.total_frames} | Velocidade: {tps:.1f} tiles/segundo", end='\r')
        
        # 3. Consolida globalmente
        global_boxes = consolidator.consolidate(tile_detections, tiles)
        
        # 4. Estruturação dos dados
        frame_result = {
            "timestamp_ms": frame_data.timestamp_ms,
            "tiles": {},           # Guardamos a detecção local de cada tile (útil para debug)
            "consolidated": []     # Guardamos a detecção final limpa no frame todo
        }
        
        for t_idx, boxes in tile_detections.items():
            frame_result["tiles"][t_idx] = [
                {"x": b.x, "y": b.y, "w": b.w, "h": b.h, "class": b.class_name, "conf": b.conf}
                for b in boxes
            ]
        
        for b in global_boxes:
            frame_result["consolidated"].append(
                {"x": b.x, "y": b.y, "w": b.w, "h": b.h, "class": b.class_name, "conf": b.conf}
            )
            
        results["frames"][frame_data.index] = frame_result
        
    reader.release()
    
    # 5. Salva o resultado final num JSON formatadinho
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n\nGround Truth salvo com sucesso em:\n{output_json}")


if __name__ == "__main__":
    # Caminhos padrão do projeto
    video_file = base_dir / "videos" / "VIRAT_S_010004_01_000163_000188.mp4"
    output_folder = base_dir / "ground_truth"
    output_file = output_folder / "gt_VIRAT_S_010004.json"
    
    # Cria a pasta ground_truth caso não exista
    output_folder.mkdir(parents=True, exist_ok=True)
    
    generate_ground_truth(str(video_file), str(output_file))

