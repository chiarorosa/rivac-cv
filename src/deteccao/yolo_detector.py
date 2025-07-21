"""
Detector YOLO usando Ultralytics
Suporta YOLOv8, YOLOv11 e modelos customizados
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

from .base_detector import BaseDetector, Detection


class YOLODetector(BaseDetector):
    """
    Detector usando modelos YOLO da Ultralytics.
    Suporta YOLOv8, YOLOv11 e modelos customizados.
    """

    def __init__(self, model_path: str, **kwargs):
        """
        Inicializa detector YOLO.

        Args:
            model_path: Caminho para o modelo (.pt) ou nome do modelo oficial
            **kwargs: Parâmetros adicionais:
                - confidence: Threshold de confiança (default: 0.3)
                - iou_threshold: Threshold de IoU para NMS (default: 0.5)
                - classes: Lista de classes a detectar (default: None = todas)
                - device: 'auto', 'cpu', 'cuda', 'mps' (default: 'auto')
                - imgsz: Tamanho da imagem para inferência (default: 640)
                - half: Usar precisão half (default: False)
                - verbose: Logs verbosos (default: False)
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics não está instalado. " "Instale com: pip install ultralytics")

        super().__init__(model_path, **kwargs)

        # Parâmetros específicos do YOLO
        self.imgsz = kwargs.get("imgsz", 640)
        self.half = kwargs.get("half", False)
        self.verbose = kwargs.get("verbose", False)

        # Determinar device
        self.device = self._setup_device(kwargs.get("device", "auto"))

        self.logger.info(f"YOLODetector configurado para device: {self.device}")

    def _setup_device(self, device: str) -> str:
        """
        Configura o device para execução.

        Args:
            device: Device especificado

        Returns:
            Device final a ser usado
        """
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"

        return device

    def load_model(self) -> bool:
        """
        Carrega o modelo YOLO.

        Returns:
            True se carregou com sucesso
        """
        try:
            self.logger.info(f"Carregando modelo YOLO: {self.model_path}")

            # Verificar se o arquivo existe (para modelos locais)
            if not self._is_official_model(self.model_path):
                if not Path(self.model_path).exists():
                    self.logger.error(f"Arquivo do modelo não encontrado: {self.model_path}")
                    return False

            # Carregar modelo
            self.model = YOLO(self.model_path)

            # Configurar device
            if hasattr(self.model, "to"):
                self.model.to(self.device)

            # Configurar precisão half se solicitado
            if self.half and self.device != "cpu":
                if hasattr(self.model.model, "half"):
                    self.model.model.half()

            self.is_loaded = True

            # Log informações do modelo
            model_info = self._get_model_details()
            self.logger.info(f"Modelo carregado com sucesso: {model_info}")

            return True

        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo YOLO: {e}")
            self.is_loaded = False
            return False

    def _is_official_model(self, model_path: str) -> bool:
        """
        Verifica se é um modelo oficial (será baixado automaticamente).

        Args:
            model_path: Caminho do modelo

        Returns:
            True se é modelo oficial
        """
        official_models = [
            "yolo11n.pt",
            "yolo11s.pt",
            "yolo11m.pt",
            "yolo11l.pt",
            "yolo11x.pt",
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt",
            "yolo11n-seg.pt",
            "yolo11s-seg.pt",
            "yolo11m-seg.pt",
            "yolo11l-seg.pt",
            "yolo11x-seg.pt",
            "yolov8n-seg.pt",
            "yolov8s-seg.pt",
            "yolov8m-seg.pt",
            "yolov8l-seg.pt",
            "yolov8x-seg.pt",
        ]

        return Path(model_path).name in official_models

    def _get_model_details(self) -> Dict[str, Any]:
        """
        Obtém detalhes do modelo carregado.

        Returns:
            Dicionário com informações do modelo
        """
        if not self.is_loaded or self.model is None:
            return {}

        try:
            details = {"model_name": Path(self.model_path).name, "device": self.device, "imgsz": self.imgsz}

            # Tentar obter informações adicionais
            if hasattr(self.model, "names"):
                details["num_classes"] = len(self.model.names)
                details["class_names"] = self.model.names

            if hasattr(self.model, "model"):
                model_info = str(self.model.model)
                if "parameters" in model_info:
                    # Extrair número de parâmetros se disponível
                    pass

            return details

        except Exception as e:
            self.logger.warning(f"Erro ao obter detalhes do modelo: {e}")
            return {"model_name": Path(self.model_path).name}

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """
        Executa detecção YOLO em um frame.

        Args:
            frame: Frame de vídeo (BGR)

        Returns:
            Lista de detecções
        """
        if not self.is_loaded or self.model is None:
            self.logger.error("Modelo não carregado")
            return []

        try:
            # Executar predição
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=self.classes,
                device=self.device,
                imgsz=self.imgsz,
                half=self.half,
                verbose=self.verbose,
            )

            # Converter resultados para formato padrão
            detections = []

            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes

                    # Extrair dados das boxes
                    xyxy = boxes.xyxy.cpu().numpy()  # Coordenadas x1,y1,x2,y2
                    conf = boxes.conf.cpu().numpy()  # Confiança
                    cls = boxes.cls.cpu().numpy().astype(int)  # Classes

                    # Criar detecções
                    for i in range(len(xyxy)):
                        detection = Detection(
                            bbox=(float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])),
                            confidence=float(conf[i]),
                            class_id=int(cls[i]),
                        )
                        detections.append(detection)

            return detections

        except Exception as e:
            self.logger.error(f"Erro durante predição YOLO: {e}")
            return []

    def get_class_names(self) -> Dict[int, str]:
        """
        Retorna nomes das classes do modelo.

        Returns:
            Dicionário {class_id: class_name}
        """
        if self.is_loaded and self.model is not None and hasattr(self.model, "names"):
            return {i: name for i, name in enumerate(self.model.names)}

        # Fallback para classes COCO padrão
        return super().get_class_names()

    def track(self, frame: np.ndarray, persist: bool = True) -> List[Detection]:
        """
        Executa detecção com tracking.

        Args:
            frame: Frame de vídeo
            persist: Manter tracks entre chamadas

        Returns:
            Lista de detecções com IDs de track
        """
        if not self.is_loaded or self.model is None:
            self.logger.error("Modelo não carregado")
            return []

        try:
            # Executar tracking
            results = self.model.track(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=self.classes,
                device=self.device,
                imgsz=self.imgsz,
                persist=persist,
                verbose=self.verbose,
            )

            # Converter resultados
            detections = []

            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes

                    # Extrair dados
                    xyxy = boxes.xyxy.cpu().numpy()
                    conf = boxes.conf.cpu().numpy()
                    cls = boxes.cls.cpu().numpy().astype(int)

                    # IDs de track (se disponíveis)
                    track_ids = None
                    if hasattr(boxes, "id") and boxes.id is not None:
                        track_ids = boxes.id.cpu().numpy().astype(int)

                    # Criar detecções
                    for i in range(len(xyxy)):
                        track_id = int(track_ids[i]) if track_ids is not None else None

                        detection = Detection(
                            bbox=(float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])),
                            confidence=float(conf[i]),
                            class_id=int(cls[i]),
                            track_id=track_id,
                        )
                        detections.append(detection)

            return detections

        except Exception as e:
            self.logger.error(f"Erro durante tracking YOLO: {e}")
            return []

    def export_model(self, format: str = "onnx", **kwargs) -> Optional[str]:
        """
        Exporta modelo para outros formatos.

        Args:
            format: Formato de exportação ('onnx', 'tensorrt', etc.)
            **kwargs: Argumentos adicionais para exportação

        Returns:
            Caminho do modelo exportado ou None se falhou
        """
        if not self.is_loaded or self.model is None:
            self.logger.error("Modelo não carregado")
            return None

        try:
            export_path = self.model.export(format=format, **kwargs)
            self.logger.info(f"Modelo exportado para: {export_path}")
            return export_path

        except Exception as e:
            self.logger.error(f"Erro ao exportar modelo: {e}")
            return None

    def benchmark(self) -> Dict[str, Any]:
        """
        Executa benchmark do modelo.

        Returns:
            Resultados do benchmark
        """
        if not self.is_loaded or self.model is None:
            self.logger.error("Modelo não carregado")
            return {}

        try:
            results = self.model.val()  # Executa validação/benchmark
            return results if results else {}

        except Exception as e:
            self.logger.error(f"Erro durante benchmark: {e}")
            return {}
