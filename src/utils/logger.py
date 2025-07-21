"""
Sistema de logging para RIVAC-CV
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "rivac_cv", level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Configura e retorna um logger para o sistema.

    Args:
        name: Nome do logger
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Caminho para arquivo de log. Se None, não salva em arquivo

    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)

    # Se já foi configurado, retorna o existente
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # Formatador
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler para arquivo (se especificado)
    if log_file:
        try:
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Não foi possível configurar log em arquivo: {e}")

    return logger


def setup_logging_from_config(config: dict) -> logging.Logger:
    """
    Configura logging baseado na configuração do sistema.

    Args:
        config: Dicionário de configuração

    Returns:
        Logger configurado
    """
    logging_config = config.get("logging", {})

    level = logging_config.get("level", "INFO")
    log_file = logging_config.get("file")

    # Criar diretório de logs se especificado
    if log_file:
        project_root = Path(__file__).parent.parent.parent
        log_file = project_root / log_file

    return get_logger("rivac_cv", level, str(log_file) if log_file else None)


class LoggerMixin:
    """
    Mixin para adicionar capacidade de logging a qualquer classe.
    """

    @property
    def logger(self) -> logging.Logger:
        """Retorna logger para a classe atual."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(f"rivac_cv.{self.__class__.__name__}")
        return self._logger


def log_performance(func):
    """
    Decorator para medir e logar tempo de execução de funções.

    Args:
        func: Função a ser decorada

    Returns:
        Função decorada com logging de performance
    """
    import functools
    import time

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger("rivac_cv.performance")
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executada em {execution_time:.4f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} falhou após {execution_time:.4f}s: {e}")
            raise

    return wrapper


def log_function_call(func):
    """
    Decorator para logar chamadas de função com argumentos.

    Args:
        func: Função a ser decorada

    Returns:
        Função decorada com logging de chamadas
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(f"rivac_cv.{func.__module__}")

        # Log da chamada (sem valores sensíveis)
        args_repr = [repr(a)[:100] + "..." if len(repr(a)) > 100 else repr(a) for a in args]
        kwargs_repr = {k: repr(v)[:100] + "..." if len(repr(v)) > 100 else repr(v) for k, v in kwargs.items()}

        logger.debug(f"Chamando {func.__name__}(args={args_repr}, kwargs={kwargs_repr})")

        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} retornou: {type(result).__name__}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} falhou: {e}")
            raise

    return wrapper
