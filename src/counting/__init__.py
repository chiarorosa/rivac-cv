"""
Módulo de contagem de pessoas
Implementa algoritmos para contar pessoas entrando/saindo de áreas
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..roi import ROI, ROIType
from ..tracking import Track, TrackState
from ..utils.config import get_config
from ..utils.logger import LoggerMixin


class CountDirection(Enum):
    """Direções de contagem."""

    IN = "in"
    OUT = "out"
    BIDIRECTIONAL = "bidirectional"


class CrossingDirection(Enum):
    """Direções de cruzamento detectadas."""

    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    TOP_TO_BOTTOM = "top_to_bottom"
    BOTTOM_TO_TOP = "bottom_to_top"
    UNKNOWN = "unknown"


@dataclass
class CountEvent:
    """
    Representa um evento de contagem.
    """

    track_id: int
    timestamp: float
    direction: CrossingDirection
    roi_name: str
    position: Tuple[float, float]
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converte evento para dicionário."""
        return {
            "track_id": self.track_id,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "direction": self.direction.value,
            "roi_name": self.roi_name,
            "position": self.position,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class CountStatistics:
    """
    Estatísticas de contagem.
    """

    total_in: int = 0
    total_out: int = 0
    current_count: int = 0
    events: List[CountEvent] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    @property
    def total_crossings(self) -> int:
        """Total de cruzamentos detectados."""
        return self.total_in + self.total_out

    @property
    def net_count(self) -> int:
        """Contagem líquida (entradas - saídas)."""
        return self.total_in - self.total_out

    @property
    def duration_hours(self) -> float:
        """Duração em horas desde o início."""
        return (time.time() - self.start_time) / 3600

    @property
    def rate_per_hour(self) -> float:
        """Taxa de cruzamentos por hora."""
        duration = self.duration_hours
        return self.total_crossings / duration if duration > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Converte estatísticas para dicionário."""
        return {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "current_count": self.current_count,
            "net_count": self.net_count,
            "total_crossings": self.total_crossings,
            "duration_hours": self.duration_hours,
            "rate_per_hour": self.rate_per_hour,
            "events_count": len(self.events),
            "start_time": self.start_time,
        }


class BaseCounter(ABC, LoggerMixin):
    """
    Classe base abstrata para contadores.
    """

    def __init__(self, roi: ROI, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa contador base.

        Args:
            roi: ROI associada ao contador
            config: Configurações específicas
        """
        self.roi = roi
        self.config = config or get_config().get("counting", {})
        self.statistics = CountStatistics()
        self.track_history: Dict[int, List[Tuple[float, float, float]]] = {}

        # Parâmetros configuráveis
        self.min_track_length = self.config.get("min_track_length", 5)
        self.max_history_size = self.config.get("max_history_size", 100)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.3)

    @abstractmethod
    def update(self, tracks: List[Track]) -> List[CountEvent]:
        """
        Atualiza contador com tracks atuais.

        Args:
            tracks: Lista de tracks ativos

        Returns:
            Lista de novos eventos de contagem
        """
        pass

    def reset(self):
        """Reseta estatísticas do contador."""
        self.statistics = CountStatistics()
        self.track_history.clear()
        self.logger.info(f"Contador da ROI '{self.roi.name}' resetado")

    def get_statistics(self) -> CountStatistics:
        """Retorna estatísticas atuais."""
        return self.statistics

    def _update_track_history(self, tracks: List[Track]):
        """
        Atualiza histórico de posições dos tracks.

        Args:
            tracks: Lista de tracks ativos
        """
        current_time = time.time()
        active_track_ids = set()

        for track in tracks:
            if track.state in [TrackState.TRACKED, TrackState.NEW]:
                track_id = track.track_id
                center = track.center

                # Adicionar posição ao histórico
                if track_id not in self.track_history:
                    self.track_history[track_id] = []

                self.track_history[track_id].append((center[0], center[1], current_time))

                # Limitar tamanho do histórico
                if len(self.track_history[track_id]) > self.max_history_size:
                    self.track_history[track_id].pop(0)

                active_track_ids.add(track_id)

        # Remover histórico de tracks inativos
        inactive_tracks = set(self.track_history.keys()) - active_track_ids
        for track_id in inactive_tracks:
            del self.track_history[track_id]

    def _calculate_direction(self, track_id: int) -> CrossingDirection:
        """
        Calcula direção de movimento baseada no histórico.

        Args:
            track_id: ID do track

        Returns:
            Direção de movimento detectada
        """
        if track_id not in self.track_history:
            return CrossingDirection.UNKNOWN

        history = self.track_history[track_id]
        if len(history) < 2:
            return CrossingDirection.UNKNOWN

        # Comparar primeira e última posição
        start_x, start_y, _ = history[0]
        end_x, end_y, _ = history[-1]

        dx = end_x - start_x
        dy = end_y - start_y

        # Determinar direção predominante
        if abs(dx) > abs(dy):
            if dx > 0:
                return CrossingDirection.LEFT_TO_RIGHT
            else:
                return CrossingDirection.RIGHT_TO_LEFT
        else:
            if dy > 0:
                return CrossingDirection.TOP_TO_BOTTOM
            else:
                return CrossingDirection.BOTTOM_TO_TOP


class LineCounter(BaseCounter):
    """
    Contador baseado em linha de contagem.
    Detecta cruzamentos de uma linha virtual.
    """

    def __init__(self, roi: ROI, config: Optional[Dict[str, Any]] = None):
        """Inicializa LineCounter."""
        if roi.roi_type != ROIType.LINE:
            raise ValueError("LineCounter requer ROI do tipo LINE")

        super().__init__(roi, config)
        self.crossed_tracks: Set[int] = set()
        self.crossing_states: Dict[int, str] = {}  # 'approaching', 'crossed'

        self.logger.info(f"LineCounter inicializado para ROI '{roi.name}'")

    def update(self, tracks: List[Track]) -> List[CountEvent]:
        """
        Atualiza contador baseado em linha.

        Args:
            tracks: Lista de tracks ativos

        Returns:
            Lista de novos eventos de contagem
        """
        self._update_track_history(tracks)
        new_events = []

        line_start, line_end = self.roi.points

        for track in tracks:
            if track.state not in [TrackState.TRACKED, TrackState.NEW]:
                continue

            track_id = track.track_id
            center = track.center

            if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
                continue

            # Verificar se houve cruzamento da linha
            history = self.track_history[track_id]
            prev_pos = history[-2][:2]
            curr_pos = history[-1][:2]

            if self._line_intersection(prev_pos, curr_pos, line_start, line_end):
                if track_id not in self.crossed_tracks:
                    # Novo cruzamento detectado
                    direction = self._calculate_crossing_direction(prev_pos, curr_pos, line_start, line_end)

                    event = CountEvent(
                        track_id=track_id,
                        timestamp=time.time(),
                        direction=direction,
                        roi_name=self.roi.name,
                        position=center,
                        confidence=track.confidence,
                    )

                    new_events.append(event)
                    self.crossed_tracks.add(track_id)

                    # Atualizar estatísticas
                    self._update_statistics(event)

                    self.logger.debug(f"Cruzamento detectado: track {track_id}, direção {direction.value}")

        # Limpar tracks antigos
        active_track_ids = {track.track_id for track in tracks}
        self.crossed_tracks &= active_track_ids

        return new_events

    def _line_intersection(
        self, p1: Tuple[float, float], p2: Tuple[float, float], line_start: Tuple[int, int], line_end: Tuple[int, int]
    ) -> bool:
        """
        Verifica se o segmento p1-p2 intersecta com a linha.

        Args:
            p1, p2: Pontos do movimento
            line_start, line_end: Pontos da linha de contagem

        Returns:
            True se há interseção
        """

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, line_start, line_end) != ccw(p2, line_start, line_end) and ccw(p1, p2, line_start) != ccw(
            p1, p2, line_end
        )

    def _calculate_crossing_direction(
        self,
        prev_pos: Tuple[float, float],
        curr_pos: Tuple[float, float],
        line_start: Tuple[int, int],
        line_end: Tuple[int, int],
    ) -> CrossingDirection:
        """
        Calcula direção do cruzamento baseada na orientação da linha.

        Args:
            prev_pos, curr_pos: Posições anterior e atual
            line_start, line_end: Pontos da linha

        Returns:
            Direção do cruzamento
        """
        # Vetor da linha
        line_vec = (line_end[0] - line_start[0], line_end[1] - line_start[1])

        # Vetor do movimento
        move_vec = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])

        # Produto cruzado para determinar lado
        cross_product = line_vec[0] * move_vec[1] - line_vec[1] * move_vec[0]

        # Determinar direção baseada na orientação da linha
        if abs(line_vec[0]) > abs(line_vec[1]):  # Linha mais horizontal
            if cross_product > 0:
                return CrossingDirection.TOP_TO_BOTTOM
            else:
                return CrossingDirection.BOTTOM_TO_TOP
        else:  # Linha mais vertical
            if cross_product > 0:
                return CrossingDirection.LEFT_TO_RIGHT
            else:
                return CrossingDirection.RIGHT_TO_LEFT

    def _update_statistics(self, event: CountEvent):
        """Atualiza estatísticas com novo evento."""
        self.statistics.events.append(event)

        # Mapear direção para entrada/saída baseado na configuração
        direction_mapping = self.config.get(
            "direction_mapping",
            {
                CrossingDirection.LEFT_TO_RIGHT.value: "in",
                CrossingDirection.RIGHT_TO_LEFT.value: "out",
                CrossingDirection.TOP_TO_BOTTOM.value: "in",
                CrossingDirection.BOTTOM_TO_TOP.value: "out",
            },
        )

        count_type = direction_mapping.get(event.direction.value, "in")

        if count_type == "in":
            self.statistics.total_in += 1
            self.statistics.current_count += 1
        else:
            self.statistics.total_out += 1
            self.statistics.current_count -= 1

        # Garantir que contagem atual não seja negativa
        self.statistics.current_count = max(0, self.statistics.current_count)


class AreaCounter(BaseCounter):
    """
    Contador baseado em área.
    Conta objetos que entram/saem de uma região definida.
    """

    def __init__(self, roi: ROI, config: Optional[Dict[str, Any]] = None):
        """Inicializa AreaCounter."""
        if roi.roi_type == ROIType.LINE:
            raise ValueError("AreaCounter não suporta ROI do tipo LINE")

        super().__init__(roi, config)
        self.tracks_in_area: Set[int] = set()
        self.tracks_entered: Set[int] = set()

        self.logger.info(f"AreaCounter inicializado para ROI '{roi.name}'")

    def update(self, tracks: List[Track]) -> List[CountEvent]:
        """
        Atualiza contador baseado em área.

        Args:
            tracks: Lista de tracks ativos

        Returns:
            Lista de novos eventos de contagem
        """
        self._update_track_history(tracks)
        new_events = []

        current_tracks_in_area = set()

        for track in tracks:
            if track.state not in [TrackState.TRACKED, TrackState.NEW]:
                continue

            track_id = track.track_id
            center = track.center

            # Verificar se track está na área
            if self.roi.contains_point(center):
                current_tracks_in_area.add(track_id)

                # Novo track entrando na área
                if track_id not in self.tracks_in_area:
                    direction = self._calculate_direction(track_id)

                    event = CountEvent(
                        track_id=track_id,
                        timestamp=time.time(),
                        direction=CrossingDirection.LEFT_TO_RIGHT,  # Entrada na área
                        roi_name=self.roi.name,
                        position=center,
                        confidence=track.confidence,
                        metadata={"event_type": "entry"},
                    )

                    new_events.append(event)
                    self.tracks_entered.add(track_id)

                    self.statistics.total_in += 1
                    self.statistics.current_count += 1
                    self.statistics.events.append(event)

                    self.logger.debug(f"Entrada na área: track {track_id}")

        # Detectar saídas da área
        for track_id in self.tracks_in_area - current_tracks_in_area:
            if track_id in self.tracks_entered:
                # Track que estava na área saiu
                event = CountEvent(
                    track_id=track_id,
                    timestamp=time.time(),
                    direction=CrossingDirection.RIGHT_TO_LEFT,  # Saída da área
                    roi_name=self.roi.name,
                    position=(0, 0),  # Posição não disponível para saída
                    confidence=1.0,
                    metadata={"event_type": "exit"},
                )

                new_events.append(event)

                self.statistics.total_out += 1
                self.statistics.current_count -= 1
                self.statistics.events.append(event)

                self.logger.debug(f"Saída da área: track {track_id}")

        # Atualizar estado
        self.tracks_in_area = current_tracks_in_area

        # Garantir contagem não negativa
        self.statistics.current_count = max(0, self.statistics.current_count)

        return new_events


class CounterManager(LoggerMixin):
    """
    Gerenciador de contadores.
    Coordena múltiplos contadores para diferentes ROIs.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa gerenciador de contadores.

        Args:
            config: Configurações do manager
        """
        self.config = config or get_config().get("counting", {})
        self.counters: Dict[str, BaseCounter] = {}

        self.logger.info("CounterManager inicializado")

    def add_counter(self, roi: ROI, counter_type: str = "auto") -> bool:
        """
        Adiciona contador para uma ROI.

        Args:
            roi: ROI para contar
            counter_type: Tipo do contador ('line', 'area', 'auto')

        Returns:
            True se adicionado com sucesso
        """
        if roi.name in self.counters:
            self.logger.warning(f"Contador para ROI '{roi.name}' já existe")
            return False

        # Determinar tipo automaticamente se necessário
        if counter_type == "auto":
            counter_type = "line" if roi.roi_type == ROIType.LINE else "area"

        # Criar contador apropriado
        try:
            if counter_type == "line":
                counter = LineCounter(roi, self.config)
            elif counter_type == "area":
                counter = AreaCounter(roi, self.config)
            else:
                self.logger.error(f"Tipo de contador desconhecido: {counter_type}")
                return False

            self.counters[roi.name] = counter
            self.logger.info(f"Contador {counter_type} adicionado para ROI '{roi.name}'")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao criar contador: {e}")
            return False

    def remove_counter(self, roi_name: str) -> bool:
        """
        Remove contador de uma ROI.

        Args:
            roi_name: Nome da ROI

        Returns:
            True se removido com sucesso
        """
        if roi_name not in self.counters:
            self.logger.warning(f"Contador para ROI '{roi_name}' não encontrado")
            return False

        del self.counters[roi_name]
        self.logger.info(f"Contador para ROI '{roi_name}' removido")
        return True

    def update_all(self, tracks: List[Track]) -> Dict[str, List[CountEvent]]:
        """
        Atualiza todos os contadores.

        Args:
            tracks: Lista de tracks ativos

        Returns:
            Dicionário de eventos por ROI
        """
        all_events = {}

        for roi_name, counter in self.counters.items():
            try:
                events = counter.update(tracks)
                if events:
                    all_events[roi_name] = events
            except Exception as e:
                self.logger.error(f"Erro ao atualizar contador '{roi_name}': {e}")

        return all_events

    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna estatísticas de todos os contadores.

        Returns:
            Dicionário de estatísticas por ROI
        """
        stats = {}
        for roi_name, counter in self.counters.items():
            stats[roi_name] = counter.get_statistics().to_dict()
        return stats

    def reset_all(self):
        """Reseta todos os contadores."""
        for counter in self.counters.values():
            counter.reset()
        self.logger.info("Todos os contadores foram resetados")


def create_counter_manager(config: Optional[Dict[str, Any]] = None) -> CounterManager:
    """
    Factory function para criar gerenciador de contadores.

    Args:
        config: Configurações específicas

    Returns:
        Instância do gerenciador
    """
    return CounterManager(config)
