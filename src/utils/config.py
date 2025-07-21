"""
Utilitários para configuração do sistema RIVAC-CV
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


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


def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Salva configurações em arquivo YAML.

    Args:
        config: Dicionário com as configurações
        config_path: Caminho para salvar o arquivo

    Returns:
        True se salvou com sucesso, False caso contrário
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        return True
    except Exception as e:
        print(f"Erro ao salvar configuração: {e}")
        return False


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Valida se a configuração tem os campos obrigatórios.

    Args:
        config: Configuração para validar

    Returns:
        True se válida, False caso contrário
    """
    required_sections = ["input", "detection", "tracking", "roi", "output"]

    for section in required_sections:
        if section not in config:
            print(f"Seção obrigatória '{section}' não encontrada na configuração")
            return False

    # Validações específicas
    if "model" not in config["detection"]:
        print("Campo 'model' obrigatório na seção 'detection'")
        return False

    if "confidence" not in config["detection"]:
        print("Campo 'confidence' obrigatório na seção 'detection'")
        return False

    return True


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mescla duas configurações, dando prioridade para override_config.

    Args:
        base_config: Configuração base
        override_config: Configuração que sobrescreve a base

    Returns:
        Configuração mesclada
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged
