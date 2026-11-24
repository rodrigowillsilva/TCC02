import cv2
import numpy as np


class Display:
    """Mostra um frame na tela e gerencia controles de reprodução."""

    def __init__(self, window_name: str = "TCC02 Pipeline", scale: float = 1.0, fps: int = 30, start_paused: bool = True):
        self.window_name = window_name
        self.scale = scale
        self.delay_ms = int(1000 / fps)
        self.paused = start_paused
        self.abort_save = False
        
        print("Controles de Reprodução:")
        print("  [Espaço] - Pausar / Continuar")
        print("  [N] ou [D] - Próximo frame (quando pausado)")
        print("  [Q] ou Fechar Janela - Sair e Salvar")
        print("  [ESC] ou [X] - Sair SEM Salvar resultados")
        print("-" * 30)

    def show(self, frame: np.ndarray) -> bool:
        """Mostra o frame. Retorna False se o usuário apertou 'q' ou fechou a janela."""
        if self.scale != 1.0:
            w = int(frame.shape[1] * self.scale)
            h = int(frame.shape[0] * self.scale)
            frame = cv2.resize(frame, (w, h))

        cv2.imshow(self.window_name, frame)

        while True:
            # Checa se o usuário fechou a janela no 'X'
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    return False
            except cv2.error:
                return False

            # Se pausado, espera indefinidamente (0). Se tocando, espera delay_ms.
            wait_time = 0 if self.paused else self.delay_ms
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord('q'):
                return False
            elif key == 27 or key == ord('x'):
                self.abort_save = True
                return False
            elif key == ord(' '):
                self.paused = not self.paused
            elif key == ord('n') or key == ord('d'):
                self.paused = True  # Garante que continua pausado ao avançar 1 frame
                return True
            
            # Se não apertou nada (ou apertou tecla inútil) e não está pausado, segue pro próximo
            if not self.paused:
                return True

    def close(self):
        cv2.destroyAllWindows()
