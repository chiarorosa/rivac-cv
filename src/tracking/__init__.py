"""
Módulo de tracking para rastreamento de objetos
Implementa algoritmos de tracking para acompanhar pessoas ao longo do tempo
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.config import get_config
from ..utils.logger import LoggerMixin


class TrackState(Enum):
    """Estados possíveis de um track."""

    NEW = "new"
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


@dataclass
class Track:
    """
    Representa um objeto rastreado ao longo do tempo.
    """

    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    state: TrackState
    age: int = 0  # Número de frames desde criação
    hits: int = 0  # Número de detecções confirmadas
    time_since_update: int = 0  # Frames desde última atualização
    features: Optional[np.ndarray] = None  # Features para re-identificação

    @property
    def center(self) -> Tuple[float, float]:
        """Retorna centro do bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        """Retorna área do bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    def to_dict(self) -> Dict[str, Any]:
        """Converte track para dicionário."""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "state": self.state.value,
            "age": self.age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
            "center": self.center,
            "area": self.area,
        }


class BaseTracker(ABC, LoggerMixin):
    """
    Classe base abstrata para implementação de trackers.
    Define interface comum para todos os algoritmos de tracking.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa tracker base.

        Args:
            config: Configurações específicas do tracker
        """
        self.config = config or get_config().get("tracking", {})
        self.tracks: List[Track] = []
        self.next_id = 1
        self.frame_count = 0

        # Parâmetros configuráveis
        self.max_age = self.config.get("max_age", 30)
        self.min_hits = self.config.get("min_hits", 3)
        self.iou_threshold = self.config.get("iou_threshold", 0.3)

    @abstractmethod
    def update(self, detections: List[Dict[str, Any]]) -> List[Track]:
        """
        Atualiza tracker com novas detecções.

        Args:
            detections: Lista de detecções do frame atual

        Returns:
            Lista de tracks atualizados
        """
        pass

    def reset(self):
        """Reseta estado do tracker."""
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
        self.logger.info("Tracker resetado")

    def get_active_tracks(self) -> List[Track]:
        """Retorna apenas tracks ativos (não removidos)."""
        return [track for track in self.tracks if track.state != TrackState.REMOVED]

    def get_confirmed_tracks(self) -> List[Track]:
        """Retorna tracks confirmados (com hits suficientes)."""
        return [track for track in self.tracks if track.hits >= self.min_hits and track.state in [TrackState.TRACKED]]

    def _calculate_iou(
        self, bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]
    ) -> float:
        """
        Calcula Intersection over Union entre dois bounding boxes.

        Args:
            bbox1: Primeiro bounding box (x1, y1, x2, y2)
            bbox2: Segundo bounding box (x1, y1, x2, y2)

        Returns:
            Valor IoU entre 0 e 1
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calcular área de interseção
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Calcular áreas dos bounding boxes
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        # Calcular união
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _calculate_distance(self, center1: Tuple[float, float], center2: Tuple[float, float]) -> float:
        """
        Calcula distância euclidiana entre dois centros.

        Args:
            center1: Primeiro centro (x, y)
            center2: Segundo centro (x, y)

        Returns:
            Distância euclidiana
        """
        x1, y1 = center1
        x2, y2 = center2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


class SimpleTracker(BaseTracker):
    """
    Implementação simples de tracker baseado em IoU.
    Associa detecções com tracks baseado na sobreposição dos bounding boxes.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicializa SimpleTracker."""
        super().__init__(config)
        self.logger.info("SimpleTracker inicializado")

    def update(self, detections: List[Dict[str, Any]]) -> List[Track]:
        """
        Atualiza tracker com detecções simples baseado em IoU.

        Args:
            detections: Lista de detecções

        Returns:
            Lista de tracks atualizados
        """
        self.frame_count += 1

        # Converter detecções para formato interno
        detection_bboxes = []
        for det in detections:
            bbox = det.get("bbox", det.get("xyxy", [0, 0, 0, 0]))
            detection_bboxes.append(
                {
                    "bbox": tuple(bbox),
                    "confidence": det.get("confidence", 0.0),
                    "class_id": det.get("class_id", 0),
                    "class_name": det.get("class_name", "object"),
                }
            )

        # Calcular matriz de IoU entre tracks e detecções
        iou_matrix = np.zeros((len(self.tracks), len(detection_bboxes)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detection_bboxes):
                iou_matrix[i, j] = self._calculate_iou(track.bbox, det["bbox"])

        # Associação usando algoritmo húngaro simplificado
        matched_indices = self._hungarian_assignment(iou_matrix)

        # Processar matches
        matched_tracks = set()
        matched_detections = set()

        for track_idx, det_idx in matched_indices:
            if iou_matrix[track_idx, det_idx] >= self.iou_threshold:
                # Atualizar track existente
                track = self.tracks[track_idx]
                detection = detection_bboxes[det_idx]

                track.bbox = detection["bbox"]
                track.confidence = detection["confidence"]
                track.hits += 1
                track.time_since_update = 0
                track.state = TrackState.TRACKED

                matched_tracks.add(track_idx)
                matched_detections.add(det_idx)

        # Criar novos tracks para detecções não associadas
        for det_idx, detection in enumerate(detection_bboxes):
            if det_idx not in matched_detections:
                new_track = Track(
                    track_id=self.next_id,
                    bbox=detection["bbox"],
                    confidence=detection["confidence"],
                    class_id=detection["class_id"],
                    class_name=detection["class_name"],
                    state=TrackState.NEW,
                    hits=1,
                )
                self.tracks.append(new_track)
                self.next_id += 1

        # Atualizar tracks não associados
        for track_idx, track in enumerate(self.tracks):
            if track_idx not in matched_tracks:
                track.time_since_update += 1
                track.age += 1

                if track.time_since_update > self.max_age:
                    track.state = TrackState.REMOVED
                else:
                    track.state = TrackState.LOST

        # Atualizar idade de todos os tracks
        for track in self.tracks:
            track.age += 1

        # Remover tracks muito antigos
        self.tracks = [track for track in self.tracks if track.state != TrackState.REMOVED]

        active_tracks = self.get_active_tracks()
        self.logger.debug(f"Frame {self.frame_count}: {len(active_tracks)} tracks ativos")

        return active_tracks

    def _hungarian_assignment(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """
        Implementação simplificada do algoritmo húngaro.
        Para casos complexos, usar scipy.optimize.linear_sum_assignment.

        Args:
            cost_matrix: Matriz de custos (IoU neste caso)

        Returns:
            Lista de pares (track_idx, detection_idx)
        """
        matches = []

        if cost_matrix.size == 0:
            return matches

        # Implementação greedy simples (não ótima, mas funcional)
        used_tracks = set()
        used_detections = set()

        # Encontrar associações em ordem decrescente de IoU
        while True:
            max_iou = 0
            best_match = None

            for i in range(cost_matrix.shape[0]):
                if i in used_tracks:
                    continue
                for j in range(cost_matrix.shape[1]):
                    if j in used_detections:
                        continue
                    if cost_matrix[i, j] > max_iou:
                        max_iou = cost_matrix[i, j]
                        best_match = (i, j)

            if best_match is None or max_iou < self.iou_threshold:
                break

            matches.append(best_match)
            used_tracks.add(best_match[0])
            used_detections.add(best_match[1])

        return matches


try:
    from ultralytics.trackers import BOTSORT, BYTETracker

    class YOLOTracker(BaseTracker):
        """
        Wrapper para trackers da Ultralytics (BYTETracker, BoTSORT).
        """

        def __init__(self, config: Optional[Dict[str, Any]] = None):
            """Inicializa YOLOTracker."""
            super().__init__(config)

            tracker_type = self.config.get("tracker_type", "bytetrack")

            if tracker_type.lower() == "botsort":
                self.tracker = BOTSORT()
            else:
                self.tracker = BYTETracker()

            self.logger.info(f"YOLOTracker inicializado com {tracker_type}")

        def update(self, detections: List[Dict[str, Any]]) -> List[Track]:
            """
            Atualiza usando tracker da Ultralytics.

            Args:
                detections: Lista de detecções

            Returns:
                Lista de tracks atualizados
            """
            self.frame_count += 1

            if not detections:
                return self.get_active_tracks()

            # Converter detecções para formato esperado
            det_array = np.array(
                [
                    [
                        *det.get("bbox", det.get("xyxy", [0, 0, 0, 0])),
                        det.get("confidence", 0.0),
                        det.get("class_id", 0),
                    ]
                    for det in detections
                ]
            )

            # Atualizar tracker
            tracks_output = self.tracker.update(det_array)

            # Converter saída para formato interno
            current_tracks = []
            for track_data in tracks_output:
                track_id = int(track_data[4])
                bbox = tuple(track_data[:4])
                confidence = float(track_data[5]) if len(track_data) > 5 else 0.0

                track = Track(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=confidence,
                    class_id=0,  # Assumindo pessoa
                    class_name="person",
                    state=TrackState.TRACKED,
                )
                current_tracks.append(track)

            self.tracks = current_tracks

            self.logger.debug(f"Frame {self.frame_count}: {len(current_tracks)} tracks do YOLO")

            return current_tracks

except ImportError:
    # Fallback se Ultralytics não estiver disponível
    class YOLOTracker(SimpleTracker):
        """Fallback para SimpleTracker se Ultralytics não disponível."""

        def __init__(self, config: Optional[Dict[str, Any]] = None):
            super().__init__(config)
            self.logger.warning("Ultralytics não disponível, usando SimpleTracker")


def create_tracker(tracker_type: str = "simple", config: Optional[Dict[str, Any]] = None) -> BaseTracker:
    """
    Factory function para criar trackers.

    Args:
        tracker_type: Tipo do tracker ('simple', 'yolo', 'bytetrack', 'botsort')
        config: Configurações específicas

    Returns:
        Instância do tracker
    """
    if tracker_type.lower() in ["yolo", "bytetrack", "botsort"]:
        return YOLOTracker(config)
    else:
        return SimpleTracker(config)
