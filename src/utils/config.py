"""
Utilitários para configuração do sistema RIVAC-CV
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Configuração global carregada
_global_config: Optional[Dict[str, Any]] = None


def get_config() -> Dict[str, Any]:
    """
    Obtém a configuração global do sistema.

    Returns:
        Dicionário com as configurações carregadas
    """
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Carrega configurações do arquivo YAML.

    Args:
        config_path: Caminho para o arquivo de configuração.
                    Se None, usa config/app_config.yaml

    Returns:
        Dicionário com as configurações
    """
    if config_path is None:
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "app_config.yaml"

    if not os.path.exists(config_path):
        return get_default_config()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Erro ao carregar configuração: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    Retorna configuração padrão do sistema.

    Returns:
        Dicionário com configurações padrão
    """
    return {
        "input": {"default_source": "camera", "frame_skip": 1, "resize_width": 640, "resize_height": 480},
        "detection": {
            "model": "yolo11n.pt",
            "confidence": 0.3,
            "iou_threshold": 0.5,
            "classes": [0],  # Apenas pessoas
            "device": "auto",  # auto, cpu, cuda
        },
        "tracking": {"enabled": True, "tracker": "bytetrack.yaml", "max_age": 30, "min_hits": 3},
        "roi": {"enabled": True, "interactive_setup": True, "save_regions": True, "default_regions": []},
        "output": {
            "save_video": True,
            "save_data": True,
            "export_format": ["csv", "json"],
            "output_dir": "data/exports",
            "fps": 30,
        },
        "database": {"type": "sqlite", "path": "data/rivac_cv.db", "create_tables": True},
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "logs/rivac_cv.log",
        },
    }
