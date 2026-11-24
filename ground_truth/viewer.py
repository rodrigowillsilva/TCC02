import sys
import json
import cv2
import argparse
from pathlib import Path

base_dir = Path(__file__).parent.parent
experiment_dir = base_dir / "experiment"
sys.path.insert(0, str(experiment_dir))

from pipeline.video_reader import VideoReader


def view_jsons(video_path: str, json1_path: str, json2_path: str = None):
    print(f"Abrindo vídeo: {video_path}")
    if not Path(video_path).exists():
        print(f"Erro: Vídeo não encontrado no caminho: {video_path}")
        return

    print(f"Lendo JSON 1: {json1_path}")
    with open(json1_path, 'r') as f:
        data1 = json.load(f)
        
    data2 = None
    if json2_path:
        print(f"Lendo JSON 2: {json2_path}")
        if Path(json2_path).exists():
            with open(json2_path, 'r') as f:
                data2 = json.load(f)
        else:
            print(f"Aviso: Arquivo de experimento {json2_path} não encontrado.")
            
    reader = VideoReader(video_path)
    paused = False
    
    print("\nControles do Visualizador:")
    print(" - [ESPAÇO]: Pausar / Continuar")
    print(" - [D]     : Avançar 1 frame (quando pausado)")
    print(" - [Q]     : Sair\n")
    print("Legenda:")
    print(" - VERDE: JSON 1 (Ex: Ground Truth)")
    if data2:
        print(" - AZUL : JSON 2 (Ex: Experimento)")
    
    for frame_data in reader:
        frame = frame_data.image.copy()
        frame_idx_str = str(frame_data.index)
        
        # 1. Desenha o JSON 1 (Verde)
        if frame_idx_str in data1.get("frames", {}):
            frame_info = data1["frames"][frame_idx_str]
            for box in frame_info.get("consolidated", []):
                x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"J1 {box['conf']:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 2. Desenha o JSON 2 (Azul)
        if data2 and frame_idx_str in data2.get("frames", {}):
            exp_info = data2["frames"][frame_idx_str]
            for box in exp_info.get("consolidated", []):
                x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"J2 {box['conf']:.2f}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Redimensiona para caber na tela
        h_frame, w_frame = frame.shape[:2]
        frame_resized = cv2.resize(frame, (w_frame // 2, h_frame // 2))
        
        cv2.imshow("JSON Multi-Viewer", frame_resized)
        
        while True:
            delay = 0 if paused else 30
            key = cv2.waitKey(delay) & 0xFF
            
            if key == ord('q') or key == 27:
                reader.release()
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                paused = not paused
                if not paused:
                    break
            elif key == ord('d'):
                if paused:
                    break
            elif not paused:
                break
                
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizador de Caixas (JSON) em Vídeo")
    # Opcionais viraram argumentos obrigatórios na ordem pedida (o terceiro tem nargs='?' para ser opcional)
    parser.add_argument("video", type=str, help="Caminho para o vídeo (Ex: videos/video.mp4)")
    parser.add_argument("json1", type=str, help="Caminho para o JSON Principal (Ficará Verde)")
    parser.add_argument("json2", type=str, nargs='?', default=None, help="[Opcional] Caminho para o JSON de comparação (Ficará Azul)")
    
    args = parser.parse_args()
    
    if not Path(args.json1).exists():
        print(f"Arquivo JSON 1 não encontrado: {args.json1}")
    else:
        view_jsons(args.video, args.json1, args.json2)
