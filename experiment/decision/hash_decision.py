import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
from .base import DecisionBlock
from pipeline.tiler import Tile
from pipeline.annotations import BBox


class HashDecision(DecisionBlock):
    """
    Selecionador de Tiles baseado em Árvore de Hashes Visuais
    (8x8 -> 10x10 -> 13x13 -> 16x16 -> 20x20 -> 24x24 -> 32x32).
    
    Suporta diferentes tipos de hash via string (`hash_type`):
    - "VALUE"   : Baseado no valor/luminância dos pixels (média de cor/brilho).
    - "EDGE"    : Baseado puramente nas bordas/gradientes (imune a sombras suaves).
    - "HYBRID"  : Fusão de luminância + realce de bordas.
    - "WAVELET" : Decomposição Wavelet de Haar (quadrante LL sucessivo sem resize comum: 40->20->10->5).
    """

    def __init__(
        self,
        levels: Tuple[int, ...] = (8, 10, 13, 16, 20, 24, 32),
        hash_type: str = "VALUE",
        tolerance_ratio: float = 0.015,
        ttl_positive: int = 3,
        ttl_empty: int = 90,
        debug: bool = False,
    ):
        """
        Args:
            levels: Tamanhos progressivos da árvore de hash (no modo WAVELET, usa (5, 10, 20, 40) por divisão exata de 320).
            hash_type: Tipo do hash ("VALUE", "EDGE", "HYBRID" ou "WAVELET").
            tolerance_ratio: Proporção de bits que podem variar por ruído da câmera em cada nível
                             (ex: 0.015 = 1.5% dos bits: ~1 bit no 8x8 até ~15 bits no 32x32).
            ttl_positive: Validade máxima (em frames) de um hash que contém pessoas detectadas.
            ttl_empty: Validade máxima (em frames) de um hash vazio (sem pessoas).
            debug: Se True, exibe estatísticas de Cache Hit/Miss no console.
        """
        self.hash_type = hash_type.upper()
        if self.hash_type not in ("VALUE", "EDGE", "HYBRID", "WAVELET"):
            raise ValueError(f"hash_type inválido: {self.hash_type}. Use 'VALUE', 'EDGE', 'HYBRID' ou 'WAVELET'.")

        # Se for WAVELET, usa as divisões diádicas exatas de 320x320 (320 -> 160 -> 80 -> 40 -> 20 -> 10 -> 5)
        if self.hash_type == "WAVELET" and levels == (8, 10, 13, 16, 20, 24, 32):
            self.levels = (5, 10, 20, 40, 80)
        else:
            self.levels = levels

        self.tolerance_ratio = tolerance_ratio
        self.ttl_positive = ttl_positive
        self.ttl_empty = ttl_empty
        self.debug = debug

        # Calcula a tolerância de bits proporcional ao total de bits (size * size) de cada nível
        # Ex com 0.015: 8x8(64b)->1, 10x10(100b)->2, 13x13(169b)->3, 16x16(256b)->4, 20x20(400b)->6, 24x24(576b)->9, 32x32(1024b)->15
        self.level_tolerances: Dict[int, int] = {
            size: max(1, int(round((size * size) * self.tolerance_ratio)))
            if self.tolerance_ratio > 0 else 0
            for size in self.levels
        }

        # Kernel 3x3 pré-alocado para extração de bordas
        self._edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Banco em RAM: chave = coordenada (x, y) do tile -> raiz da árvore daquele tile
        self.tile_db: Dict[Tuple[int, int], Dict[int, Any]] = {}

        # Armazena (coord, tile_hashes, frame_index) dos tiles que deram Cache Miss no frame atual
        self.pending_hashes: Dict[int, Tuple[Tuple[int, int], List[int], int]] = {}

        # Armazena as detecções recuperadas do banco para os tiles que deram Cache Hit no frame atual
        self.current_cached_detections: Dict[int, List[BBox]] = {}

        # Estatísticas acumuladas para telemetria/config
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.expired_hits = 0
        self.total_entries = 0

    @property
    def config(self) -> dict:
        hit_rate = (self.cache_hits / self.total_queries * 100) if self.total_queries > 0 else 0.0
        return {
            "method": "HASH",
            "hash_type": self.hash_type,
            "levels": list(self.levels),
            "tolerance_ratio": self.tolerance_ratio,
            "level_tolerances_bits": self.level_tolerances,
            "ttl_positive": self.ttl_positive,
            "ttl_empty": self.ttl_empty,
            "stats": {
                "total_queries": self.total_queries,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "expired_revalidations": self.expired_hits,
                "hit_rate_percent": round(hit_rate, 2),
                "unique_paths_stored": self.total_entries,
            }
        }

    def _preprocess_tile(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Prepara a imagem do tile conforme a modalidade escolhida (`hash_type`):
        - VALUE  : Escala de cinza direta (o INTER_AREA no resize já faz a média dos blocos).
        - EDGE   : Gradiente morfológico de bordas (ignora sombras e iluminação).
        - HYBRID : Combinação de valor + bordas realçadas.
        """
        gray_tile = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY) if bgr_img.ndim == 3 else bgr_img
        if self.hash_type in ("VALUE", "WAVELET"):
            return gray_tile

        blurred = cv2.GaussianBlur(gray_tile, (3, 3), 0)
        edges = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, self._edge_kernel)
        if self.hash_type == "EDGE":
            return edges

        # HYBRID
        return cv2.addWeighted(blurred, 0.5, edges, 2.0, 0)

    def _compute_wavelet_ll_hashes(self, gray_tile: np.ndarray) -> Dict[int, int]:
        """
        Aplica a Transformada Wavelet Discreta de Haar 2D (separável) sucessivamente sobre o
        quadrante LL (Canto Superior Esquerdo: 320 -> 160 -> 80 -> 40 -> 20 -> 10 -> 5) sem resize.
        Retorna os hashes binários para cada tamanho presente em self.levels.
        """
        target_set = set(self.levels)
        min_target = min(self.levels)
        hashes_by_size: Dict[int, int] = {}

        # Passo 1 da Wavelet de Haar direto do uint8 para uint16 (evita cópia de 320x320)
        h, w = gray_tile.shape
        h2, w2 = (h // 2) * 2, (w // 2) * 2
        row_sum = np.add(gray_tile[0:h2:2, :w2], gray_tile[1:h2:2, :w2], dtype=np.uint16)
        curr = (row_sum[:, 0:w2:2] + row_sum[:, 1:w2:2]) >> 2

        while True:
            sz = curr.shape[0]
            if sz in target_set:
                mean_val = float(curr.mean())
                binary_bits = (curr >= mean_val).astype(np.uint8)
                packed = np.packbits(binary_bits).tobytes()
                hashes_by_size[sz] = int.from_bytes(packed, byteorder="big")

            if sz <= min_target or sz < 2:
                break

            h, w = curr.shape
            h2, w2 = (h // 2) * 2, (w // 2) * 2
            row_sum = curr[0:h2:2, :w2] + curr[1:h2:2, :w2]
            curr = (row_sum[:, 0:w2:2] + row_sum[:, 1:w2:2]) >> 2

        return hashes_by_size

    def _compute_hash(self, base_small: np.ndarray, size: int) -> int:
        """
        Redimensiona a partir da base (ex: 32x32) para (size x size) e binariza pela média dos pixels.
        """
        if base_small.shape[0] == size and base_small.shape[1] == size:
            resized = base_small
        else:
            resized = cv2.resize(base_small, (size, size), interpolation=cv2.INTER_AREA)
        mean_val = float(resized.mean())
        binary_bits = (resized >= mean_val).astype(np.uint8)
        packed = np.packbits(binary_bits).tobytes()
        return int.from_bytes(packed, byteorder="big")

    def _find_matching_key(self, node: Dict[int, Any], target_hash: int, level_size: int) -> int | None:
        """Busca a chave na árvore em O(1) exato ou usando a tolerância proporcional do nível."""
        if target_hash in node:
            return target_hash

        tol = self.level_tolerances.get(level_size, 0)
        if tol <= 0 or not node:
            return None

        best_key = None
        best_dist = tol + 1

        for existing_key in node.keys():
            dist = (target_hash ^ existing_key).bit_count()
            if dist <= tol and dist < best_dist:
                best_dist = dist
                best_key = existing_key
                if dist == 1:
                    break

        return best_key

    def select(self, frame: np.ndarray, tiles: List[Tile], frame_index: int) -> List[int]:
        selected_indices: List[int] = []
        self.pending_hashes.clear()
        self.current_cached_detections.clear()

        for tile in tiles:
            self.total_queries += 1
            coord = (tile.x, tile.y)
            tree_root = self.tile_db.setdefault(coord, {})

            # 1. Pré-processa o tile
            proc_img = self._preprocess_tile(tile.image)

            # 2. Gera os hashes e percorre a árvore nível a nível
            curr_node = tree_root
            tile_hashes: List[int] = []
            is_hit = True

            if self.hash_type == "WAVELET":
                wavelet_map = self._compute_wavelet_ll_hashes(proc_img)
                for size in self.levels:
                    h_int = wavelet_map[size]
                    tile_hashes.append(h_int)
                    if is_hit:
                        matched_key = self._find_matching_key(curr_node, h_int, size)
                        if matched_key is not None:
                            curr_node = curr_node[matched_key]
                        else:
                            is_hit = False
            else:
                max_size = max(self.levels)
                base_small = cv2.resize(proc_img, (max_size, max_size), interpolation=cv2.INTER_AREA)
                for size in self.levels:
                    h_int = self._compute_hash(base_small, size)
                    tile_hashes.append(h_int)
                    if is_hit:
                        matched_key = self._find_matching_key(curr_node, h_int, size)
                        if matched_key is not None:
                            curr_node = curr_node[matched_key]
                        else:
                            is_hit = False

            # 3. Verifica se chegou na folha (32x32) e se o TTL (validade) ainda está ativo
            valid_hit = False
            if is_hit and isinstance(curr_node, dict) and "detections" in curr_node:
                saved_dets = curr_node["detections"]
                max_ttl = self.ttl_positive if len(saved_dets) > 0 else self.ttl_empty

                # Se max_ttl for None ou <= 0, o hash nunca expira (memória eterna)
                if max_ttl is None or max_ttl <= 0:
                    valid_hit = True
                else:
                    age = frame_index - curr_node.get("last_frame", 0)
                    if age <= max_ttl:
                        valid_hit = True
                    else:
                        self.expired_hits += 1

            if valid_hit:
                self.cache_hits += 1
                curr_node["hits"] += 1
                cached_boxes = [
                    BBox(
                        x=b.x, y=b.y, w=b.w, h=b.h,
                        label=b.label, class_name=b.class_name,
                        conf=b.conf, color=b.color, thickness=b.thickness
                    )
                    for b in curr_node["detections"]
                ]
                self.current_cached_detections[tile.index] = cached_boxes
            else:
                # 4. Cache Miss (ou TTL expirado): manda pro YOLO e agenda gravação no banco
                self.cache_misses += 1
                selected_indices.append(tile.index)
                self.pending_hashes[tile.index] = (coord, tile_hashes, frame_index)

        if self.debug and frame_index % 30 == 0:
            hit_rate = (self.cache_hits / self.total_queries * 100) if self.total_queries > 0 else 0.0
            print(
                f"[HASH-{self.hash_type} Frame {frame_index}] Selecionados p/ YOLO: {len(selected_indices)}/{len(tiles)} | "
                f"Hit Rate: {hit_rate:.1f}% | Expirados: {self.expired_hits} | Banco: {self.total_entries}"
            )

        return selected_indices

    def update_state(self, selected_tiles: List[Tile], tile_detections: Dict[int, List[BBox]]):
        """Salva ou atualiza na árvore os hashes dos tiles que passaram pelo YOLO."""
        for tile in selected_tiles:
            if tile.index not in self.pending_hashes:
                continue

            coord, tile_hashes, frame_index = self.pending_hashes[tile.index]
            detections = tile_detections.get(tile.index, [])

            curr_node = self.tile_db.setdefault(coord, {})
            for size, h_int in zip(self.levels[:-1], tile_hashes[:-1]):
                matched_key = self._find_matching_key(curr_node, h_int, size)
                if matched_key is not None:
                    curr_node = curr_node[matched_key]
                else:
                    curr_node[h_int] = {}
                    curr_node = curr_node[h_int]

            leaf_target = tile_hashes[-1]
            leaf_size = self.levels[-1]
            matched_leaf = self._find_matching_key(curr_node, leaf_target, leaf_size)
            leaf_key = matched_leaf if matched_leaf is not None else leaf_target

            if leaf_key not in curr_node:
                self.total_entries += 1

            curr_node[leaf_key] = {
                "detections": [
                    BBox(
                        x=b.x, y=b.y, w=b.w, h=b.h,
                        label=b.label, class_name=b.class_name,
                        conf=b.conf, color=b.color, thickness=b.thickness
                    )
                    for b in detections
                ],
                "hits": 0,
                "last_frame": frame_index,
            }

    def get_cached_detections(self) -> Dict[int, List[BBox]]:
        """Retorna as detecções resgatadas do banco para os tiles que deram Cache Hit."""
        return self.current_cached_detections
