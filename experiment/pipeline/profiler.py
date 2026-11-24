import time
from contextlib import contextmanager

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

class Profiler:
    def __init__(self, warmup_frames: int = 0):
        self.warmup_frames = warmup_frames
        self.current_frame = 0
        self.is_active = True
        self.total_times = {}
        self.call_counts = {}
        self.cpu_records = []
        
        if HAS_PSUTIL:
            self.process = psutil.Process()
            # Chama uma vez vazio para iniciar a janela de medição de CPU
            self.process.cpu_percent(interval=None)

    def set_frame(self, frame_index: int):
        """Atualiza o índice do frame para controlar o período de warmup."""
        self.current_frame = frame_index
        # Só mede se já tiver passado do período de warmup
        self.is_active = (frame_index >= self.warmup_frames)

    @contextmanager
    def measure(self, name: str):
        """Mede automaticamente o tempo do bloco de código (ignora se estiver no warmup)."""
        if not self.is_active:
            yield
            return

        t0 = time.perf_counter()
        yield
        dt = time.perf_counter() - t0
        
        if name not in self.total_times:
            self.total_times[name] = 0.0
            self.call_counts[name] = 0
            
        self.total_times[name] += dt
        self.call_counts[name] += 1

    def record_frame_cpu(self):
        """Registra a porcentagem de CPU gasta pelo seu TCC neste frame (ignora se estiver no warmup)."""
        if not self.is_active:
            return
            
        if HAS_PSUTIL:
            # cpu_percent retorna o total somado dos nucleos (pode ser > 100).
            # Dividindo pelo número de núcleos, temos a % global do computador.
            usage = self.process.cpu_percent(interval=None) / psutil.cpu_count()
            self.cpu_records.append(usage)

    def simulate_delay(self, delay_ms: float):
        """Injeta um atraso artificial."""
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    def get_summary_dict(self) -> dict:
        """Exporta os tempos e uso de CPU em um formato de dicionário para o JSON."""
        summary = {
            "warmup_frames": self.warmup_frames,
            "timings": {}
        }
        for name in sorted(self.total_times.keys()):
            total = self.total_times[name]
            count = self.call_counts[name]
            avg_ms = (total / count) * 1000 if count > 0 else 0
            summary["timings"][name] = {
                "calls": count,
                "total_seconds": round(total, 4),
                "avg_ms": round(avg_ms, 2)
            }
            
        if HAS_PSUTIL and self.cpu_records:
            summary["cpu"] = {
                "average_percent": round(sum(self.cpu_records) / len(self.cpu_records), 2),
                "max_percent": round(max(self.cpu_records), 2)
            }
        else:
            summary["cpu"] = {"error": "Biblioteca psutil não instalada ou sem dados"}
            
        return summary

    def print_stats(self):
        """Imprime uma tabela elegante com os tempos no terminal."""
        print("\n" + "="*50)
        print(f"{' RELATÓRIO DE TELEMETRIA (PROFILER) ':^50}")
        print("="*50)
        
        if HAS_PSUTIL and self.cpu_records:
            avg_cpu = sum(self.cpu_records) / len(self.cpu_records)
            print(f"🖥️  CPU Médio do Sistema : {avg_cpu:.1f}%")
            print("="*50)
            
        for name in sorted(self.total_times.keys()):
            total = self.total_times[name]
            count = self.call_counts[name]
            avg_ms = (total / count) * 1000 if count > 0 else 0
            
            print(f"🔹 {name}")
            print(f"    Chamadas  : {count}")
            print(f"    Tempo Total: {total:.2f} s")
            print(f"    Média      : {avg_ms:.2f} ms/chamada")
        print("="*50 + "\n")

