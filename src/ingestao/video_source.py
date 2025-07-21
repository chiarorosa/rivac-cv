"""
Fonte de vídeo usando OpenCV VideoCapture
Suporta arquivos, câmeras e streams RTSP
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from .base_source import BaseVideoSource


class VideoSource(BaseVideoSource):
    """
    Fonte de vídeo usando OpenCV VideoCapture.
    Suporta múltiplos tipos de entrada.
    """

    def __init__(self, source: Union[str, int, Path], **kwargs):
        """
        Inicializa fonte de vídeo.

        Args:
            source: Pode ser:
                - str: Caminho para arquivo, URL RTSP, ou 'camera' para webcam padrão
                - int: Índice da câmera (0, 1, 2, ...)
                - Path: Caminho para arquivo de vídeo
            **kwargs: Parâmetros adicionais:
                - resize_width: Largura para redimensionar
                - resize_height: Altura para redimensionar
                - frame_skip: Pular N frames (para acelerar processamento)
        """
        super().__init__()

        self.source = source
        self.resize_width = kwargs.get("resize_width")
        self.resize_height = kwargs.get("resize_height")
        self.frame_skip = kwargs.get("frame_skip", 1)
        self.skip_counter = 0

        self.cap: Optional[cv2.VideoCapture] = None

        # Processar diferentes tipos de source
        self._processed_source = self._process_source(source)

        self.logger.info(f"VideoSource criada para: {self._processed_source}")

    def _process_source(self, source: Union[str, int, Path]) -> Union[str, int]:
        """
        Processa o source para formato adequado ao OpenCV.

        Args:
            source: Source original

        Returns:
            Source processado
        """
        if isinstance(source, int):
            return source

        if isinstance(source, Path):
            return str(source)

        if isinstance(source, str):
            # Casos especiais
            if source.lower() in ["camera", "webcam", "cam"]:
                return 0  # Webcam padrão

            # Se é número como string
            if source.isdigit():
                return int(source)

            # URL ou caminho de arquivo
            return source

        raise ValueError(f"Tipo de source não suportado: {type(source)}")

    def open(self) -> bool:
        """
        Abre a fonte de vídeo.

        Returns:
            True se abriu com sucesso
        """
        try:
            self.cap = cv2.VideoCapture(self._processed_source)

            if not self.cap.isOpened():
                self.logger.error(f"Não foi possível abrir fonte: {self._processed_source}")
                return False

            # Obter propriedades
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Para streams ao vivo, total_frames pode ser 0 ou -1
            if self.total_frames <= 0:
                self.total_frames = -1

            self.is_opened = True
            self.frame_count = 0

            self.logger.info(f"Vídeo aberto: {self.width}x{self.height} @ {self.fps:.1f}fps")
            if self.total_frames > 0:
                self.logger.info(f"Total de frames: {self.total_frames}")
            else:
                self.logger.info("Stream ao vivo detectado")

            return True

        except Exception as e:
            self.logger.error(f"Erro ao abrir vídeo: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lê o próximo frame.

        Returns:
            Tuple (sucesso, frame)
        """
        if not self.is_opened or self.cap is None:
            return False, None

        # Pular frames se necessário
        frames_to_skip = self.frame_skip - 1
        for _ in range(frames_to_skip):
            ret, _ = self.cap.read()
            if not ret:
                return False, None
            self.frame_count += 1

        # Ler frame principal
        ret, frame = self.cap.read()

        if not ret:
            self.logger.debug("Fim do vídeo ou erro na leitura")
            return False, None

        self.frame_count += 1

        # Redimensionar se necessário
        if self.resize_width and self.resize_height:
            frame = cv2.resize(frame, (self.resize_width, self.resize_height))
        elif self.resize_width:
            # Manter aspect ratio
            aspect = frame.shape[0] / frame.shape[1]
            new_height = int(self.resize_width * aspect)
            frame = cv2.resize(frame, (self.resize_width, new_height))

        return True, frame

    def release(self) -> None:
        """Libera recursos da fonte de vídeo."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.is_opened = False
        self.logger.info("Recursos de vídeo liberados")

    def get_properties(self) -> Dict[str, Any]:
        """
        Retorna propriedades da fonte de vídeo.

        Returns:
            Dicionário com propriedades
        """
        properties = {
            "source": str(self.source),
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "current_frame": self.frame_count,
            "is_live": self.is_live_stream(),
            "is_opened": self.is_opened,
        }

        if self.resize_width or self.resize_height:
            properties["resize_width"] = self.resize_width
            properties["resize_height"] = self.resize_height

        if self.frame_skip > 1:
            properties["frame_skip"] = self.frame_skip

        return properties

    def set_position(self, frame_number: int) -> bool:
        """
        Define posição no vídeo.

        Args:
            frame_number: Número do frame

        Returns:
            True se conseguiu posicionar
        """
        if not self.is_opened or self.cap is None or self.is_live_stream():
            return False

        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.frame_count = frame_number
            return True
        except Exception as e:
            self.logger.error(f"Erro ao posicionar vídeo: {e}")
            return False

    def set_fps(self, fps: float) -> bool:
        """
        Tenta definir FPS da câmera (funciona apenas para algumas câmeras).

        Args:
            fps: FPS desejado

        Returns:
            True se conseguiu definir
        """
        if not self.is_opened or self.cap is None:
            return False

        try:
            result = self.cap.set(cv2.CAP_PROP_FPS, fps)
            if result:
                self.fps = fps
                self.logger.info(f"FPS definido para: {fps}")
            return result
        except Exception as e:
            self.logger.error(f"Erro ao definir FPS: {e}")
            return False

    def get_codec_info(self) -> Dict[str, Any]:
        """
        Retorna informações do codec (apenas para arquivos).

        Returns:
            Dicionário com informações do codec
        """
        if not self.is_opened or self.cap is None or self.is_live_stream():
            return {}

        try:
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

            return {
                "fourcc": fourcc,
                "codec": codec,
                "bitrate": self.cap.get(cv2.CAP_PROP_BITRATE) if hasattr(cv2, "CAP_PROP_BITRATE") else None,
            }
        except Exception as e:
            self.logger.error(f"Erro ao obter informações do codec: {e}")
            return {}
