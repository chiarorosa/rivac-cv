# Pipeline integrado atualizado com novos módulos

import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .counting import CounterManager, CountEvent, create_counter_manager
from .deteccao.base_detector import BaseDetector
from .deteccao.yolo_detector import YOLODetector
from .ingestao.base_source import BaseVideoSource
from .ingestao.video_source import VideoSource
from .roi import ROIManager, create_roi_manager
from .tracking import BaseTracker, Track, create_tracker
from .utils.config import get_config, load_config
from .utils.logger import LoggerMixin, get_logger


class DetectionPipeline(LoggerMixin):
    """
    Pipeline principal do sistema RIVAC-CV.
    Coordena todos os módulos: ingestão, detecção, tracking, ROI, contagem.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa pipeline de detecção.

        Args:
            config_path: Caminho para arquivo de configuração
        """
        # Carregar configuração
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = get_config()

        # Componentes principais
        self.video_source: Optional[BaseVideoSource] = None
        self.detector: Optional[BaseDetector] = None
        self.tracker: Optional[BaseTracker] = None
        self.roi_manager: Optional[ROIManager] = None
        self.counter_manager: Optional[CounterManager] = None

        # Estado do pipeline
        self.is_running = False
        self.frame_count = 0
        self.session_id = str(uuid.uuid4())[:8]
        self.start_time = None

        # Callbacks e estatísticas
        self.frame_callbacks: List[Callable] = []
        self.detection_callbacks: List[Callable] = []
        self.count_callbacks: List[Callable] = []

        # Estatísticas
        self.stats = {
            "total_frames": 0,
            "total_detections": 0,
            "average_fps": 0.0,
            "processing_time": 0.0,
            "total_counts": {},
            "frame_times": [],
        }

        self.logger.info(f"Pipeline inicializado - Sessão: {self.session_id}")

    def setup_source(self, source_path: str, **kwargs) -> bool:
        """
        Configura fonte de vídeo.

        Args:
            source_path: Caminho, URL ou ID da fonte
            **kwargs: Parâmetros adicionais para a fonte

        Returns:
            True se configurado com sucesso
        """
        try:
            # Mesclar configurações
            source_config = self.config.get("input", {})
            source_config.update(kwargs)

            # Criar fonte de vídeo
            self.video_source = VideoSource(source_path, source_config)

            if not self.video_source.is_opened():
                self.logger.error(f"Falha ao abrir fonte: {source_path}")
                return False

            # Obter informações da fonte
            width = self.video_source.get_property("width")
            height = self.video_source.get_property("height")
            fps = self.video_source.get_property("fps")

            self.logger.info(f"Fonte configurada: {width}x{height} @ {fps}fps")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao configurar fonte: {e}")
            return False

    def setup_detector(self, model_path: str, **kwargs) -> bool:
        """
        Configura detector de objetos.

        Args:
            model_path: Caminho para o modelo
            **kwargs: Parâmetros do detector

        Returns:
            True se configurado com sucesso
        """
        try:
            # Mesclar configurações
            detector_config = self.config.get("detection", {})
            detector_config.update(kwargs)

            # Criar detector YOLO
            self.detector = YOLODetector(model_path, detector_config)

            self.logger.info(f"Detector configurado: {model_path}")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao configurar detector: {e}")
            return False

    def setup_tracker(self, tracker_type: str = "simple", **kwargs) -> bool:
        """
        Configura sistema de tracking.

        Args:
            tracker_type: Tipo do tracker
            **kwargs: Parâmetros do tracker

        Returns:
            True se configurado com sucesso
        """
        try:
            # Mesclar configurações
            tracker_config = self.config.get("tracking", {})
            tracker_config.update(kwargs)

            # Criar tracker
            self.tracker = create_tracker(tracker_type, tracker_config)

            self.logger.info(f"Tracker configurado: {tracker_type}")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao configurar tracker: {e}")
            return False

    def setup_roi_manager(self, **kwargs) -> bool:
        """
        Configura gerenciador de ROI.

        Args:
            **kwargs: Parâmetros do ROI manager

        Returns:
            True se configurado com sucesso
        """
        try:
            # Mesclar configurações
            roi_config = self.config.get("roi", {})
            roi_config.update(kwargs)

            # Criar gerenciador
            self.roi_manager = create_roi_manager(roi_config)

            self.logger.info("ROI manager configurado")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao configurar ROI manager: {e}")
            return False

    def setup_counter_manager(self, **kwargs) -> bool:
        """
        Configura gerenciador de contadores.

        Args:
            **kwargs: Parâmetros do counter manager

        Returns:
            True se configurado com sucesso
        """
        try:
            # Mesclar configurações
            counting_config = self.config.get("counting", {})
            counting_config.update(kwargs)

            # Criar gerenciador
            self.counter_manager = create_counter_manager(counting_config)

            # Adicionar contadores para ROIs existentes
            if self.roi_manager:
                for roi in self.roi_manager.get_active_rois():
                    self.counter_manager.add_counter(roi)

            self.logger.info("Counter manager configurado")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao configurar counter manager: {e}")
            return False

    def setup_auto(self, source_path: str, model_path: str = "yolo11n.pt") -> bool:
        """
        Configuração automática do pipeline completo.

        Args:
            source_path: Fonte de vídeo
            model_path: Modelo para detecção

        Returns:
            True se configurado com sucesso
        """
        success = True

        # Configurar fonte
        if not self.setup_source(source_path):
            success = False

        # Configurar detector
        if not self.setup_detector(model_path):
            success = False

        # Configurar tracker (opcional)
        if self.config.get("tracking", {}).get("enabled", True):
            tracker_type = self.config.get("tracking", {}).get("tracker_type", "simple")
            if not self.setup_tracker(tracker_type):
                self.logger.warning("Falha ao configurar tracker, continuando sem tracking")

        # Configurar ROI manager (opcional)
        if self.config.get("roi", {}).get("enabled", True):
            if not self.setup_roi_manager():
                self.logger.warning("Falha ao configurar ROI manager")

        # Configurar counter manager (opcional)
        if self.config.get("counting", {}).get("enabled", True):
            if not self.setup_counter_manager():
                self.logger.warning("Falha ao configurar counter manager")

        return success

    def add_frame_callback(self, callback: Callable[[np.ndarray, List[Dict], int], None]):
        """Adiciona callback para cada frame processado."""
        self.frame_callbacks.append(callback)

    def add_detection_callback(self, callback: Callable[[List[Dict]], None]):
        """Adiciona callback para detecções."""
        self.detection_callbacks.append(callback)

    def add_count_callback(self, callback: Callable[[Dict[str, List[CountEvent]]], None]):
        """Adiciona callback para eventos de contagem."""
        self.count_callbacks.append(callback)

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict], List[Track], Dict[str, List[CountEvent]]]:
        """
        Processa um único frame.

        Args:
            frame: Frame de entrada

        Returns:
            Tupla com (frame_processado, detections, tracks, count_events)
        """
        frame_start_time = time.time()

        # 1. Detecção de objetos
        detections = []
        if self.detector:
            detections = self.detector.detect(frame)
            self.stats["total_detections"] += len(detections)

        # 2. Tracking
        tracks = []
        if self.tracker and detections:
            # Converter detecções para formato do tracker
            detection_dicts = [det.to_dict() for det in detections]
            tracks = self.tracker.update(detection_dicts)

        # 3. Análise de ROI
        roi_frame = frame.copy()
        if self.roi_manager:
            roi_frame = self.roi_manager.draw_all_rois(roi_frame)

        # 4. Contagem
        count_events = {}
        if self.counter_manager and tracks:
            count_events = self.counter_manager.update_all(tracks)

        # 5. Visualização
        annotated_frame = self._annotate_frame(roi_frame, detections, tracks)

        # 6. Callbacks
        self._call_frame_callbacks(annotated_frame, detections, self.frame_count)
        if detections:
            self._call_detection_callbacks(detections)
        if count_events:
            self._call_count_callbacks(count_events)

        # 7. Estatísticas
        frame_time = time.time() - frame_start_time
        self.stats["frame_times"].append(frame_time)

        # Manter apenas últimos 100 tempos para cálculo de FPS
        if len(self.stats["frame_times"]) > 100:
            self.stats["frame_times"].pop(0)

        return annotated_frame, detections, tracks, count_events

    def run(self, max_frames: Optional[int] = None, show_progress: bool = True) -> Dict[str, Any]:
        """
        Executa o pipeline completo.

        Args:
            max_frames: Número máximo de frames (None = todos)
            show_progress: Mostrar progresso

        Returns:
            Estatísticas finais
        """
        if not self.video_source or not self.detector:
            raise RuntimeError("Pipeline não configurado completamente")

        self.is_running = True
        self.start_time = time.time()

        try:
            with self.video_source as source:
                for frame in source:
                    if not self.is_running:
                        break

                    if max_frames and self.frame_count >= max_frames:
                        break

                    # Processar frame
                    annotated_frame, detections, tracks, count_events = self.process_frame(frame)

                    # Mostrar vídeo se configurado
                    if self.config.get("visualization", {}).get("show_video", False):
                        cv2.imshow("RIVAC-CV", annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    self.frame_count += 1

                    # Log de progresso
                    if show_progress and self.frame_count % 100 == 0:
                        current_fps = self.get_current_fps()
                        self.logger.info(f"Frame {self.frame_count} - FPS: {current_fps:.1f}")

        except KeyboardInterrupt:
            self.logger.info("Processamento interrompido pelo usuário")
        except Exception as e:
            self.logger.error(f"Erro durante processamento: {e}")
            raise
        finally:
            self.is_running = False
            cv2.destroyAllWindows()

        # Calcular estatísticas finais
        self._calculate_final_stats()

        return self.stats

    def stop(self):
        """Para a execução do pipeline."""
        self.is_running = False

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas atuais."""
        stats = self.stats.copy()
        stats["current_fps"] = self.get_current_fps()

        # Adicionar estatísticas de contagem
        if self.counter_manager:
            stats["counting"] = self.counter_manager.get_all_statistics()

        return stats

    def get_current_fps(self) -> float:
        """Calcula FPS atual baseado nos últimos frames."""
        if len(self.stats["frame_times"]) < 2:
            return 0.0

        avg_frame_time = sum(self.stats["frame_times"]) / len(self.stats["frame_times"])
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    def export_results(self, filepath: str, format: str = "csv") -> bool:
        """
        Exporta resultados do pipeline.

        Args:
            filepath: Caminho do arquivo
            format: Formato ('csv', 'json')

        Returns:
            True se exportado com sucesso
        """
        try:
            export_path = Path(filepath)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            # Coletar dados para exportação
            export_data = {
                "session_info": {
                    "session_id": self.session_id,
                    "start_time": self.start_time,
                    "end_time": time.time(),
                    "total_frames": self.frame_count,
                    "config": self.config,
                },
                "statistics": self.get_statistics(),
            }

            # Exportar conforme formato
            if format.lower() == "csv":
                return self._export_csv(export_path, export_data)
            elif format.lower() == "json":
                return self._export_json(export_path, export_data)
            else:
                self.logger.error(f"Formato de exportação não suportado: {format}")
                return False

        except Exception as e:
            self.logger.error(f"Erro ao exportar resultados: {e}")
            return False

    def _annotate_frame(self, frame: np.ndarray, detections: List[Dict], tracks: List[Track]) -> np.ndarray:
        """Adiciona anotações visuais no frame."""
        annotated = frame.copy()

        # Configurações de visualização
        viz_config = self.config.get("visualization", {})

        # Desenhar detecções
        if viz_config.get("show_detections", True):
            for det in detections:
                det_dict = det.to_dict() if hasattr(det, "to_dict") else det
                self._draw_detection(annotated, det_dict)

        # Desenhar tracks
        if viz_config.get("show_tracks", True) and tracks:
            for track in tracks:
                self._draw_track(annotated, track)

        # Adicionar informações de status
        if viz_config.get("show_info", True):
            self._draw_status_info(annotated)

        return annotated

    def _draw_detection(self, frame: np.ndarray, detection: Dict[str, Any]):
        """Desenha uma detecção no frame."""
        bbox = detection.get("bbox", [0, 0, 0, 0])
        confidence = detection.get("confidence", 0.0)
        class_name = detection.get("class_name", "object")

        x1, y1, x2, y2 = map(int, bbox)

        # Desenhar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Desenhar label
        label = f"{class_name}: {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def _draw_track(self, frame: np.ndarray, track: Track):
        """Desenha um track no frame."""
        center = track.center
        track_id = track.track_id

        # Desenhar centro do track
        cv2.circle(frame, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)

        # Desenhar ID do track
        cv2.putText(
            frame,
            f"ID:{track_id}",
            (int(center[0]) + 10, int(center[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    def _draw_status_info(self, frame: np.ndarray):
        """Desenha informações de status no frame."""
        info_text = [
            f"Frame: {self.frame_count}",
            f"FPS: {self.get_current_fps():.1f}",
            f"Detections: {self.stats['total_detections']}",
        ]

        # Adicionar informações de contagem
        if self.counter_manager:
            all_stats = self.counter_manager.get_all_statistics()
            for roi_name, stats in all_stats.items():
                info_text.append(f"{roi_name}: {stats['current_count']}")

        # Desenhar no frame
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _call_frame_callbacks(self, frame: np.ndarray, detections: List[Dict], frame_number: int):
        """Chama callbacks de frame."""
        for callback in self.frame_callbacks:
            try:
                callback(frame, detections, frame_number)
            except Exception as e:
                self.logger.error(f"Erro em callback de frame: {e}")

    def _call_detection_callbacks(self, detections: List[Dict]):
        """Chama callbacks de detecção."""
        for callback in self.detection_callbacks:
            try:
                callback(detections)
            except Exception as e:
                self.logger.error(f"Erro em callback de detecção: {e}")

    def _call_count_callbacks(self, count_events: Dict[str, List[CountEvent]]):
        """Chama callbacks de contagem."""
        for callback in self.count_callbacks:
            try:
                callback(count_events)
            except Exception as e:
                self.logger.error(f"Erro em callback de contagem: {e}")

    def _calculate_final_stats(self):
        """Calcula estatísticas finais."""
        end_time = time.time()

        self.stats["total_frames"] = self.frame_count
        self.stats["processing_time"] = end_time - self.start_time if self.start_time else 0

        if self.stats["processing_time"] > 0:
            self.stats["average_fps"] = self.frame_count / self.stats["processing_time"]

        self.logger.info(
            f"Processamento finalizado - {self.frame_count} frames em {self.stats['processing_time']:.2f}s"
        )
        self.logger.info(f"FPS médio: {self.stats['average_fps']:.2f}")

    def _export_csv(self, filepath: Path, data: Dict[str, Any]) -> bool:
        """Exporta dados em formato CSV."""
        import pandas as pd

        # Criar DataFrame com estatísticas de contagem
        rows = []
        counting_stats = data["statistics"].get("counting", {})

        for roi_name, stats in counting_stats.items():
            rows.append(
                {
                    "session_id": data["session_info"]["session_id"],
                    "roi_name": roi_name,
                    "total_in": stats["total_in"],
                    "total_out": stats["total_out"],
                    "current_count": stats["current_count"],
                    "net_count": stats["net_count"],
                    "total_crossings": stats["total_crossings"],
                    "rate_per_hour": stats["rate_per_hour"],
                }
            )

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
            self.logger.info(f"Dados exportados para {filepath}")
            return True
        else:
            self.logger.warning("Nenhum dado de contagem para exportar")
            return False

    def _export_json(self, filepath: Path, data: Dict[str, Any]) -> bool:
        """Exporta dados em formato JSON."""
        import json

        # Converter dados para JSON serializável
        json_data = self._make_json_serializable(data)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Dados exportados para {filepath}")
        return True

    def _make_json_serializable(self, obj):
        """Converte objeto para formato JSON serializável."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
