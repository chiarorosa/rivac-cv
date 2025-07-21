"""
Classe base para fontes de vídeo
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..utils.logger import LoggerMixin


class BaseVideoSource(ABC, LoggerMixin):
    """
    Classe abstrata para fontes de vídeo.
    Define interface comum para diferentes tipos de entrada.
    """

    def __init__(self):
        self.is_opened = False
        self.frame_count = 0
        self.fps = 30.0
        self.width = 640
        self.height = 480
        self.total_frames = -1  # -1 para streams infinitos

    @abstractmethod
    def open(self) -> bool:
        """
        Abre a fonte de vídeo.

        Returns:
            True se abriu com sucesso, False caso contrário
        """
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lê o próximo frame.

        Returns:
            Tuple (sucesso, frame) onde:
            - sucesso: True se leu frame com sucesso
            - frame: Array numpy com o frame ou None se falhou
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Libera recursos da fonte de vídeo."""
        pass

    @abstractmethod
    def get_properties(self) -> Dict[str, Any]:
        """
        Retorna propriedades da fonte de vídeo.

        Returns:
            Dicionário com propriedades (fps, width, height, etc.)
        """
        pass

    def set_position(self, frame_number: int) -> bool:
        """
        Define posição no vídeo (para fontes que suportam).

        Args:
            frame_number: Número do frame para posicionar

        Returns:
            True se conseguiu posicionar, False caso contrário
        """
        self.logger.warning("set_position não implementado para esta fonte")
        return False

    def get_position(self) -> int:
        """
        Retorna posição atual no vídeo.

        Returns:
            Número do frame atual
        """
        return self.frame_count

    def is_live_stream(self) -> bool:
        """
        Verifica se é um stream ao vivo.

        Returns:
            True se é stream ao vivo, False se é arquivo
        """
        return self.total_frames == -1

    def get_progress(self) -> float:
        """
        Retorna progresso de reprodução (0.0 a 1.0).
        Para streams ao vivo, sempre retorna 0.0.

        Returns:
            Progresso de 0.0 a 1.0
        """
        if self.is_live_stream() or self.total_frames <= 0:
            return 0.0

        return min(self.frame_count / self.total_frames, 1.0)

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

    def __iter__(self):
        """Torna a fonte iterável."""
        return self

    def __next__(self):
        """Próximo frame na iteração."""
        success, frame = self.read()
        if not success:
            raise StopIteration
        return frame
