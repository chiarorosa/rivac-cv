"""
Sistema RIVAC-CV - Monitoramento por Visão Computacional
Core package para detecção e tracking de pessoas em ambientes de varejo.
"""

__version__ = "1.0.0"
__author__ = "RIVAC-CV Team"
__description__ = "Sistema modular de visão computacional para monitoramento de varejo"

from .pipeline import DetectionPipeline
from .utils.config import load_config
from .utils.logger import get_logger

__all__ = ["DetectionPipeline", "load_config", "get_logger"]
