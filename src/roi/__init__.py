"""
Módulo de gerenciamento de ROI (Regions of Interest)
Define e gerencia áreas de interesse para análise
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from ..utils.config import get_config
from ..utils.logger import LoggerMixin


class ROIType(Enum):
    """Tipos de ROI suportados."""

    RECTANGLE = "rectangle"
    POLYGON = "polygon"
    CIRCLE = "circle"
    LINE = "line"


@dataclass
class ROI:
    """
    Representa uma região de interesse.
    """

    name: str
    roi_type: ROIType
    points: List[Tuple[int, int]]
    active: bool = True
    color: Tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validação após inicialização."""
        if self.roi_type == ROIType.RECTANGLE and len(self.points) != 2:
            raise ValueError("ROI retângulo deve ter exatamente 2 pontos")
        elif self.roi_type == ROIType.CIRCLE and len(self.points) != 2:
            raise ValueError("ROI círculo deve ter centro e ponto na circunferência")
        elif self.roi_type == ROIType.LINE and len(self.points) != 2:
            raise ValueError("ROI linha deve ter exatamente 2 pontos")
        elif self.roi_type == ROIType.POLYGON and len(self.points) < 3:
            raise ValueError("ROI polígono deve ter pelo menos 3 pontos")

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Retorna bounding box da ROI (x1, y1, x2, y2)."""
        if not self.points:
            return (0, 0, 0, 0)

        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]

        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

    @property
    def center(self) -> Tuple[int, int]:
        """Retorna centro da ROI."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> float:
        """Calcula área da ROI."""
        if self.roi_type == ROIType.RECTANGLE:
            x1, y1, x2, y2 = self.bbox
            return abs((x2 - x1) * (y2 - y1))

        elif self.roi_type == ROIType.CIRCLE:
            center, edge = self.points
            radius = np.sqrt((edge[0] - center[0]) ** 2 + (edge[1] - center[1]) ** 2)
            return np.pi * radius**2

        elif self.roi_type == ROIType.POLYGON:
            # Fórmula do shoelace
            n = len(self.points)
            if n < 3:
                return 0

            area = 0
            for i in range(n):
                j = (i + 1) % n
                area += self.points[i][0] * self.points[j][1]
                area -= self.points[j][0] * self.points[i][1]

            return abs(area) / 2

        else:  # LINE
            return 0

    def contains_point(self, point: Tuple[int, int]) -> bool:
        """
        Verifica se um ponto está dentro da ROI.

        Args:
            point: Ponto a verificar (x, y)

        Returns:
            True se o ponto estiver dentro da ROI
        """
        if not self.active:
            return False

        x, y = point

        if self.roi_type == ROIType.RECTANGLE:
            x1, y1, x2, y2 = self.bbox
            return x1 <= x <= x2 and y1 <= y <= y2

        elif self.roi_type == ROIType.CIRCLE:
            center, edge = self.points
            radius = np.sqrt((edge[0] - center[0]) ** 2 + (edge[1] - center[1]) ** 2)
            distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            return distance <= radius

        elif self.roi_type == ROIType.POLYGON:
            return cv2.pointPolygonTest(np.array(self.points, dtype=np.int32), point, False) >= 0

        else:  # LINE
            return False

    def intersects_bbox(self, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Verifica se uma bounding box intersecta com a ROI.

        Args:
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            True se houver interseção
        """
        if not self.active:
            return False

        x1, y1, x2, y2 = bbox

        # Verificar se algum canto do bbox está dentro da ROI
        corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
        for corner in corners:
            if self.contains_point(corner):
                return True

        # Verificar se o centro está dentro
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        return self.contains_point(center)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Desenha a ROI no frame.

        Args:
            frame: Frame onde desenhar

        Returns:
            Frame com ROI desenhada
        """
        if not self.active or not self.points:
            return frame

        color = self.color if self.active else (128, 128, 128)

        if self.roi_type == ROIType.RECTANGLE:
            x1, y1, x2, y2 = self.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

        elif self.roi_type == ROIType.CIRCLE:
            center, edge = self.points
            radius = int(np.sqrt((edge[0] - center[0]) ** 2 + (edge[1] - center[1]) ** 2))
            cv2.circle(frame, center, radius, color, self.thickness)

        elif self.roi_type == ROIType.POLYGON:
            points = np.array(self.points, dtype=np.int32)
            cv2.polylines(frame, [points], True, color, self.thickness)

        elif self.roi_type == ROIType.LINE:
            cv2.line(frame, self.points[0], self.points[1], color, self.thickness)

        # Desenhar nome da ROI
        x, y = self.center
        cv2.putText(frame, self.name, (x - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def to_dict(self) -> Dict[str, Any]:
        """Converte ROI para dicionário."""
        return {
            "name": self.name,
            "roi_type": self.roi_type.value,
            "points": self.points,
            "active": self.active,
            "color": self.color,
            "thickness": self.thickness,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ROI":
        """Cria ROI a partir de dicionário."""
        return cls(
            name=data["name"],
            roi_type=ROIType(data["roi_type"]),
            points=data["points"],
            active=data.get("active", True),
            color=tuple(data.get("color", (0, 255, 0))),
            thickness=data.get("thickness", 2),
            metadata=data.get("metadata", {}),
        )


class ROIManager(LoggerMixin):
    """
    Gerenciador de ROIs.
    Responsável por criar, modificar e persistir ROIs.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa gerenciador de ROI.

        Args:
            config: Configurações do ROI manager
        """
        self.config = config or get_config().get("roi", {})
        self.rois: Dict[str, ROI] = {}
        self.roi_file = Path(self.config.get("roi_file", "config/rois.json"))

        # Carregar ROIs existentes
        self.load_rois()

        self.logger.info(f"ROIManager inicializado com {len(self.rois)} ROIs")

    def add_roi(self, roi: ROI) -> bool:
        """
        Adiciona nova ROI.

        Args:
            roi: ROI a adicionar

        Returns:
            True se adicionada com sucesso
        """
        if roi.name in self.rois:
            self.logger.warning(f"ROI '{roi.name}' já existe")
            return False

        self.rois[roi.name] = roi
        self.logger.info(f"ROI '{roi.name}' adicionada")
        return True

    def remove_roi(self, name: str) -> bool:
        """
        Remove ROI pelo nome.

        Args:
            name: Nome da ROI

        Returns:
            True se removida com sucesso
        """
        if name not in self.rois:
            self.logger.warning(f"ROI '{name}' não encontrada")
            return False

        del self.rois[name]
        self.logger.info(f"ROI '{name}' removida")
        return True

    def get_roi(self, name: str) -> Optional[ROI]:
        """
        Obtém ROI pelo nome.

        Args:
            name: Nome da ROI

        Returns:
            ROI encontrada ou None
        """
        return self.rois.get(name)

    def get_active_rois(self) -> List[ROI]:
        """Retorna lista de ROIs ativas."""
        return [roi for roi in self.rois.values() if roi.active]

    def toggle_roi(self, name: str) -> bool:
        """
        Alterna estado ativo/inativo de uma ROI.

        Args:
            name: Nome da ROI

        Returns:
            True se alterada com sucesso
        """
        if name not in self.rois:
            return False

        self.rois[name].active = not self.rois[name].active
        status = "ativa" if self.rois[name].active else "inativa"
        self.logger.info(f"ROI '{name}' agora está {status}")
        return True

    def check_point_in_rois(self, point: Tuple[int, int]) -> List[str]:
        """
        Verifica quais ROIs contêm um ponto.

        Args:
            point: Ponto a verificar

        Returns:
            Lista de nomes das ROIs que contêm o ponto
        """
        containing_rois = []
        for name, roi in self.rois.items():
            if roi.contains_point(point):
                containing_rois.append(name)
        return containing_rois

    def check_bbox_in_rois(self, bbox: Tuple[int, int, int, int]) -> List[str]:
        """
        Verifica quais ROIs intersectam com uma bounding box.

        Args:
            bbox: Bounding box a verificar

        Returns:
            Lista de nomes das ROIs que intersectam
        """
        intersecting_rois = []
        for name, roi in self.rois.items():
            if roi.intersects_bbox(bbox):
                intersecting_rois.append(name)
        return intersecting_rois

    def draw_all_rois(self, frame: np.ndarray) -> np.ndarray:
        """
        Desenha todas as ROIs no frame.

        Args:
            frame: Frame onde desenhar

        Returns:
            Frame com todas as ROIs desenhadas
        """
        result_frame = frame.copy()
        for roi in self.rois.values():
            result_frame = roi.draw(result_frame)
        return result_frame

    def save_rois(self, filepath: Optional[Path] = None) -> bool:
        """
        Salva ROIs em arquivo JSON.

        Args:
            filepath: Caminho do arquivo (opcional)

        Returns:
            True se salvo com sucesso
        """
        try:
            filepath = filepath or self.roi_file
            filepath.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "rois": [roi.to_dict() for roi in self.rois.values()],
                "metadata": {"version": "1.0", "created_with": "rivac-cv"},
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"ROIs salvas em {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao salvar ROIs: {e}")
            return False

    def load_rois(self, filepath: Optional[Path] = None) -> bool:
        """
        Carrega ROIs de arquivo JSON.

        Args:
            filepath: Caminho do arquivo (opcional)

        Returns:
            True se carregado com sucesso
        """
        try:
            filepath = filepath or self.roi_file

            if not filepath.exists():
                self.logger.info(f"Arquivo de ROIs não encontrado: {filepath}")
                return True  # Não é erro, apenas não há ROIs salvas

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.rois.clear()
            for roi_data in data.get("rois", []):
                roi = ROI.from_dict(roi_data)
                self.rois[roi.name] = roi

            self.logger.info(f"Carregadas {len(self.rois)} ROIs de {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao carregar ROIs: {e}")
            return False

    def create_rectangle_roi(self, name: str, x1: int, y1: int, x2: int, y2: int, **kwargs) -> bool:
        """
        Cria ROI retângular.

        Args:
            name: Nome da ROI
            x1, y1, x2, y2: Coordenadas do retângulo
            **kwargs: Argumentos adicionais

        Returns:
            True se criada com sucesso
        """
        roi = ROI(name=name, roi_type=ROIType.RECTANGLE, points=[(x1, y1), (x2, y2)], **kwargs)
        return self.add_roi(roi)

    def create_polygon_roi(self, name: str, points: List[Tuple[int, int]], **kwargs) -> bool:
        """
        Cria ROI poligonal.

        Args:
            name: Nome da ROI
            points: Lista de pontos do polígono
            **kwargs: Argumentos adicionais

        Returns:
            True se criada com sucesso
        """
        roi = ROI(name=name, roi_type=ROIType.POLYGON, points=points, **kwargs)
        return self.add_roi(roi)

    def create_circle_roi(self, name: str, center: Tuple[int, int], radius_point: Tuple[int, int], **kwargs) -> bool:
        """
        Cria ROI circular.

        Args:
            name: Nome da ROI
            center: Centro do círculo
            radius_point: Ponto na circunferência para definir raio
            **kwargs: Argumentos adicionais

        Returns:
            True se criada com sucesso
        """
        roi = ROI(name=name, roi_type=ROIType.CIRCLE, points=[center, radius_point], **kwargs)
        return self.add_roi(roi)

    def create_line_roi(self, name: str, start: Tuple[int, int], end: Tuple[int, int], **kwargs) -> bool:
        """
        Cria ROI linear (para contagem).

        Args:
            name: Nome da ROI
            start: Ponto inicial da linha
            end: Ponto final da linha
            **kwargs: Argumentos adicionais

        Returns:
            True se criada com sucesso
        """
        roi = ROI(name=name, roi_type=ROIType.LINE, points=[start, end], **kwargs)
        return self.add_roi(roi)


def create_roi_manager(config: Optional[Dict[str, Any]] = None) -> ROIManager:
    """
    Factory function para criar ROI manager.

    Args:
        config: Configurações específicas

    Returns:
        Instância do ROI manager
    """
    return ROIManager(config)
