"""
Classe base para detectores de objetos
Define interface comum para diferentes modelos
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.logger import LoggerMixin


@dataclass
class Detection:
    """
    Classe para representar uma detecção.
    """

    # Coordenadas da bounding box (x1, y1, x2, y2)
    bbox: Tuple[float, float, float, float]

    # Confiança da detecção (0.0 a 1.0)
    confidence: float

    # ID da classe detectada
    class_id: int

    # Nome da classe (opcional)
    class_name: Optional[str] = None

    # ID único do track (se usando tracking)
    track_id: Optional[int] = None

    # Dados adicionais específicos do modelo
    extra_data: Optional[Dict[str, Any]] = None

    @property
    def x1(self) -> float:
        """Coordenada x do canto superior esquerdo."""
        return self.bbox[0]

    @property
    def y1(self) -> float:
        """Coordenada y do canto superior esquerdo."""
        return self.bbox[1]

    @property
    def x2(self) -> float:
        """Coordenada x do canto inferior direito."""
        return self.bbox[2]

    @property
    def y2(self) -> float:
        """Coordenada y do canto inferior direito."""
        return self.bbox[3]

    @property
    def center_x(self) -> float:
        """Coordenada x do centro da bbox."""
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        """Coordenada y do centro da bbox."""
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        """Largura da bbox."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Altura da bbox."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Área da bbox."""
        return self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        """Converte detecção para dicionário."""
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "width": self.width,
            "height": self.height,
            "area": self.area,
            "extra_data": self.extra_data,
        }


class BaseDetector(ABC, LoggerMixin):
    """
    Classe base para detectores de objetos.
    Define interface comum para diferentes modelos.
    """

    def __init__(self, model_path: str, **kwargs):
        """
        Inicializa o detector.

        Args:
            model_path: Caminho para o modelo
            **kwargs: Parâmetros específicos do detector
        """
        self.model_path = model_path
        self.model = None
        self.is_loaded = False

        # Parâmetros comuns
        self.confidence_threshold = kwargs.get("confidence", 0.3)
        self.iou_threshold = kwargs.get("iou_threshold", 0.5)
        self.classes = kwargs.get("classes", None)  # None = todas as classes
        self.device = kwargs.get("device", "auto")

        self.logger.info(f"Detector inicializado: {self.__class__.__name__}")

    @abstractmethod
    def load_model(self) -> bool:
        """
        Carrega o modelo de detecção.

        Returns:
            True se carregou com sucesso
        """
        pass

    @abstractmethod
    def predict(self, frame: np.ndarray) -> List[Detection]:
        """
        Executa detecção em um frame.

        Args:
            frame: Frame de vídeo (BGR)

        Returns:
            Lista de detecções encontradas
        """
        pass

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Pré-processa frame antes da detecção.
        Pode ser sobrescrito por detectores específicos.

        Args:
            frame: Frame original

        Returns:
            Frame pré-processado
        """
        return frame

    def postprocess_detections(self, detections: List[Detection]) -> List[Detection]:
        """
        Pós-processa detecções.
        Aplica filtros de confiança, classes, etc.

        Args:
            detections: Lista de detecções brutas

        Returns:
            Lista de detecções filtradas
        """
        filtered = []

        for detection in detections:
            # Filtro de confiança
            if detection.confidence < self.confidence_threshold:
                continue

            # Filtro de classes
            if self.classes is not None and detection.class_id not in self.classes:
                continue

            filtered.append(detection)

        return filtered

    def get_class_names(self) -> Dict[int, str]:
        """
        Retorna mapeamento de IDs para nomes das classes.

        Returns:
            Dicionário {class_id: class_name}
        """
        # Implementação padrão com classes COCO
        return {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            # ... adicionar outras classes conforme necessário
        }

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Método principal de detecção.
        Combina pré-processamento, predição e pós-processamento.

        Args:
            frame: Frame de entrada

        Returns:
            Lista de detecções finais
        """
        if not self.is_loaded:
            if not self.load_model():
                self.logger.error("Modelo não carregado, não é possível detectar")
                return []

        # Pré-processamento
        processed_frame = self.preprocess_frame(frame)

        # Detecção
        raw_detections = self.predict(processed_frame)

        # Pós-processamento
        final_detections = self.postprocess_detections(raw_detections)

        # Adicionar nomes das classes
        class_names = self.get_class_names()
        for detection in final_detections:
            if detection.class_name is None:
                detection.class_name = class_names.get(detection.class_id, f"class_{detection.class_id}")

        return final_detections

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo.

        Returns:
            Dicionário com informações do modelo
        """
        return {
            "model_path": self.model_path,
            "model_type": self.__class__.__name__,
            "is_loaded": self.is_loaded,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "classes": self.classes,
            "device": self.device,
        }

    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Define novo threshold de confiança.

        Args:
            threshold: Novo threshold (0.0 a 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            self.logger.info(f"Threshold de confiança atualizado para: {threshold}")
        else:
            self.logger.warning(f"Threshold inválido: {threshold}. Deve estar entre 0.0 e 1.0")

    def set_iou_threshold(self, threshold: float) -> None:
        """
        Define novo threshold de IoU.

        Args:
            threshold: Novo threshold (0.0 a 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.iou_threshold = threshold
            self.logger.info(f"Threshold de IoU atualizado para: {threshold}")
        else:
            self.logger.warning(f"Threshold inválido: {threshold}. Deve estar entre 0.0 e 1.0")

    def set_classes(self, classes: Optional[List[int]]) -> None:
        """
        Define classes a serem detectadas.

        Args:
            classes: Lista de IDs das classes ou None para todas
        """
        self.classes = classes
        if classes is None:
            self.logger.info("Detectando todas as classes")
        else:
            self.logger.info(f"Detectando apenas classes: {classes}")

    def __del__(self):
        """Cleanup ao destruir objeto."""
        if hasattr(self, "model") and self.model is not None:
            try:
                del self.model
            except:
                pass
