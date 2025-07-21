"""
Pipeline principal de detecção e tracking
Coordena todos os módulos do sistema
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .deteccao.base_detector import BaseDetector, Detection
from .deteccao.yolo_detector import YOLODetector
from .ingestao.base_source import BaseVideoSource
from .ingestao.video_source import VideoSource
from .utils.config import load_config
from .utils.logger import LoggerMixin, log_performance


class DetectionPipeline(LoggerMixin):
    """
    Pipeline principal que coordena todos os módulos do sistema.
    """

    def __init__(
        self,
        source: Optional[BaseVideoSource] = None,
        detector: Optional[BaseDetector] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Inicializa o pipeline de detecção.

        Args:
            source: Fonte de vídeo (opcional, pode ser configurada depois)
            detector: Detector de objetos (opcional, pode ser configurado depois)
            config: Configuração personalizada (opcional)
            **kwargs: Argumentos adicionais
        """
        # Carregar configuração
        self.config = config or load_config()

        # Componentes principais
        self.source = source
        self.detector = detector

        # Estado do pipeline
        self.is_running = False
        self.frame_count = 0
        self.start_time = None
        self.detections_history: List[List[Detection]] = []

        # Callbacks para eventos
        self.on_frame_callbacks: List[Callable] = []
        self.on_detection_callbacks: List[Callable] = []

        # Métricas
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0

        self.logger.info("Pipeline de detecção inicializado")

    def setup_source(self, source_config: str, **kwargs) -> bool:
        """
        Configura fonte de vídeo baseada na configuração.

        Args:
            source_config: Caminho, URL ou tipo de fonte
            **kwargs: Parâmetros adicionais para a fonte

        Returns:
            True se configurou com sucesso
        """
        try:
            # Mesclar configurações
            input_config = self.config.get("input", {})
            merged_kwargs = {**input_config, **kwargs}

            # Criar fonte de vídeo
            self.source = VideoSource(source_config, **merged_kwargs)

            self.logger.info(f"Fonte configurada: {source_config}")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao configurar fonte: {e}")
            return False

    def setup_detector(self, model_path: Optional[str] = None, **kwargs) -> bool:
        """
        Configura detector baseado na configuração.

        Args:
            model_path: Caminho para o modelo (opcional)
            **kwargs: Parâmetros adicionais para o detector

        Returns:
            True se configurou com sucesso
        """
        try:
            # Usar modelo da configuração se não especificado
            if model_path is None:
                model_path = self.config.get("detection", {}).get("model", "yolo11n.pt")

            # Mesclar configurações
            detection_config = self.config.get("detection", {})
            merged_kwargs = {**detection_config, **kwargs}

            # Criar detector (por enquanto apenas YOLO)
            self.detector = YOLODetector(model_path, **merged_kwargs)

            # Carregar modelo
            if not self.detector.load_model():
                self.logger.error("Falha ao carregar modelo")
                return False

            self.logger.info(f"Detector configurado: {model_path}")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao configurar detector: {e}")
            return False

    @log_performance
    def run(self, max_frames: Optional[int] = None, show_progress: bool = True) -> Dict[str, Any]:
        """
        Executa o pipeline de detecção.

        Args:
            max_frames: Número máximo de frames para processar (None = todos)
            show_progress: Mostrar progresso no console

        Returns:
            Dicionário com resultados e estatísticas
        """
        if not self._validate_setup():
            return {"error": "Pipeline não configurado corretamente"}

        # Inicializar
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0
        self.detections_history.clear()

        results = {
            "total_frames": 0,
            "total_detections": 0,
            "average_fps": 0.0,
            "execution_time": 0.0,
            "detections_per_frame": [],
            "error": None,
        }

        try:
            with self.source:
                if not self.source.is_opened:
                    raise RuntimeError("Não foi possível abrir fonte de vídeo")

                self.logger.info("Iniciando processamento...")

                # Loop principal
                while self.is_running:
                    # Verificar limite de frames
                    if max_frames and self.frame_count >= max_frames:
                        break

                    # Ler frame
                    success, frame = self.source.read()
                    if not success:
                        self.logger.info("Fim do vídeo ou erro na leitura")
                        break

                    # Processar frame
                    detections = self._process_frame(frame)

                    # Atualizar estatísticas
                    self.frame_count += 1
                    self.detections_history.append(detections)
                    results["detections_per_frame"].append(len(detections))

                    # Calcular FPS
                    self._update_fps()

                    # Mostrar progresso
                    if show_progress and self.frame_count % 30 == 0:
                        self._show_progress()

                    # Callbacks de frame
                    for callback in self.on_frame_callbacks:
                        try:
                            callback(frame, detections, self.frame_count)
                        except Exception as e:
                            self.logger.warning(f"Erro em callback de frame: {e}")

                # Finalizar
                execution_time = time.time() - self.start_time

                results.update(
                    {
                        "total_frames": self.frame_count,
                        "total_detections": sum(len(dets) for dets in self.detections_history),
                        "average_fps": self.frame_count / execution_time if execution_time > 0 else 0,
                        "execution_time": execution_time,
                    }
                )

                self.logger.info(f"Processamento concluído: {self.frame_count} frames em {execution_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Erro durante execução: {e}")
            results["error"] = str(e)

        finally:
            self.is_running = False

        return results

    def _validate_setup(self) -> bool:
        """
        Valida se o pipeline está configurado corretamente.

        Returns:
            True se válido
        """
        if self.source is None:
            self.logger.error("Fonte de vídeo não configurada")
            return False

        if self.detector is None:
            self.logger.error("Detector não configurado")
            return False

        if not self.detector.is_loaded:
            self.logger.error("Modelo do detector não carregado")
            return False

        return True

    def _process_frame(self, frame: np.ndarray) -> List[Detection]:
        """
        Processa um frame individual.

        Args:
            frame: Frame a ser processado

        Returns:
            Lista de detecções
        """
        # Executar detecção
        detections = self.detector.detect(frame)

        # Callbacks de detecção
        for callback in self.on_detection_callbacks:
            try:
                callback(detections, self.frame_count)
            except Exception as e:
                self.logger.warning(f"Erro em callback de detecção: {e}")

        return detections

    def _update_fps(self):
        """Atualiza contador de FPS."""
        self.fps_counter += 1

        if self.fps_counter % 30 == 0:  # Atualizar a cada 30 frames
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            self.current_fps = 30 / elapsed if elapsed > 0 else 0
            self.fps_start_time = current_time

    def _show_progress(self):
        """Mostra progresso no console."""
        if self.source and not self.source.is_live_stream():
            progress = self.source.get_progress()
            self.logger.info(
                f"Progresso: {progress*100:.1f}% - Frame {self.frame_count} - FPS: {self.current_fps:.1f}"
            )
        else:
            self.logger.info(f"Frame {self.frame_count} - FPS: {self.current_fps:.1f}")

    def add_frame_callback(self, callback: Callable):
        """
        Adiciona callback que é chamado a cada frame processado.

        Args:
            callback: Função que recebe (frame, detections, frame_number)
        """
        self.on_frame_callbacks.append(callback)

    def add_detection_callback(self, callback: Callable):
        """
        Adiciona callback que é chamado após cada detecção.

        Args:
            callback: Função que recebe (detections, frame_number)
        """
        self.on_detection_callbacks.append(callback)

    def stop(self):
        """Para a execução do pipeline."""
        self.is_running = False
        self.logger.info("Pipeline parado pelo usuário")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do processamento atual.

        Returns:
            Dicionário com estatísticas
        """
        if self.start_time is None:
            return {"status": "not_started"}

        current_time = time.time()
        elapsed = current_time - self.start_time

        total_detections = sum(len(dets) for dets in self.detections_history)

        return {
            "status": "running" if self.is_running else "stopped",
            "frames_processed": self.frame_count,
            "total_detections": total_detections,
            "average_detections_per_frame": total_detections / max(self.frame_count, 1),
            "current_fps": self.current_fps,
            "average_fps": self.frame_count / elapsed if elapsed > 0 else 0,
            "elapsed_time": elapsed,
            "source_progress": self.source.get_progress() if self.source else 0.0,
        }

    def export_results(self, output_path: str, format: str = "csv") -> bool:
        """
        Exporta resultados do processamento.

        Args:
            output_path: Caminho para salvar
            format: Formato ('csv', 'json')

        Returns:
            True se exportou com sucesso
        """
        try:
            import json

            import pandas as pd

            # Preparar dados
            all_detections = []

            for frame_idx, detections in enumerate(self.detections_history):
                for detection in detections:
                    detection_data = detection.to_dict()
                    detection_data["frame_number"] = frame_idx
                    all_detections.append(detection_data)

            if not all_detections:
                self.logger.warning("Nenhuma detecção para exportar")
                return False

            # Criar diretório se necessário
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Exportar baseado no formato
            if format.lower() == "csv":
                df = pd.DataFrame(all_detections)
                df.to_csv(output_path, index=False)

            elif format.lower() == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(all_detections, f, indent=2)

            else:
                self.logger.error(f"Formato não suportado: {format}")
                return False

            self.logger.info(f"Resultados exportados para: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao exportar resultados: {e}")
            return False
