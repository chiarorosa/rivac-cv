#!/usr/bin/env python3
"""
Script principal para executar o sistema RIVAC-CV
"""
import argparse
import logging
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline import DetectionPipeline
from src.utils.config import load_config
from src.utils.logger import get_logger


def setup_logging(level: str = "INFO"):
    """Configura logging para execução standalone."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/app.log")],
    )


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="RIVAC-CV - Sistema de Monitoramento por Visão Computacional")

    # Argumentos de entrada
    parser.add_argument("--input", "-i", required=True, help="Fonte de entrada (arquivo, webcam, URL)")
    parser.add_argument("--output", "-o", help="Arquivo de saída para vídeo processado")
    parser.add_argument("--config", "-c", default="config/app_config.yaml", help="Arquivo de configuração")

    # Parâmetros de processamento
    parser.add_argument("--model", "-m", default="data/models/yolo11n.pt", help="Modelo YOLO a utilizar")
    parser.add_argument("--confidence", type=float, default=0.3, help="Threshold de confiança")
    parser.add_argument("--device", default="auto", help="Dispositivo (cpu, cuda, auto)")
    parser.add_argument("--max-frames", type=int, help="Número máximo de frames a processar")

    # Controle de execução
    parser.add_argument("--no-display", action="store_true", help="Não mostrar vídeo durante processamento")
    parser.add_argument("--save-detections", action="store_true", help="Salvar detecções em CSV")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Configurar logging
    setup_logging(args.log_level)
    logger = get_logger("main")

    logger.info("Iniciando RIVAC-CV")
    logger.info(f"Entrada: {args.input}")
    logger.info(f"Modelo: {args.model}")
    logger.info(f"Confiança: {args.confidence}")

    try:
        # Carregar configuração
        config = load_config(args.config)

        # Criar pipeline
        pipeline = DetectionPipeline()

        # Configurar fonte
        if not pipeline.setup_source(args.input):
            logger.error("Falha ao configurar fonte de vídeo")
            return 1

        # Configurar detector
        detector_config = {"confidence": args.confidence, "device": args.device, "classes": [0]}  # Apenas pessoas

        if not pipeline.setup_detector(args.model, **detector_config):
            logger.error("Falha ao configurar detector")
            return 1

        # Configurar saída se especificada
        if args.output:
            pipeline.config["output"]["save_video"] = True
            pipeline.config["output"]["video_path"] = args.output

        # Configurar display
        pipeline.config["visualization"]["show_video"] = not args.no_display

        # Executar processamento
        results = pipeline.run(max_frames=args.max_frames)

        # Mostrar resultados
        logger.info("=== RESULTADOS FINAIS ===")
        logger.info(f"Frames processados: {results.get('total_frames', 0)}")
        logger.info(f"Total de detecções: {results.get('total_detections', 0)}")
        logger.info(f"FPS médio: {results.get('average_fps', 0):.2f}")
        logger.info(f"Tempo de processamento: {results.get('processing_time', 0):.2f}s")

        # Salvar detecções se solicitado
        if args.save_detections:
            export_path = f"exports/detections_{pipeline.session_id}.csv"
            if pipeline.export_results(export_path, format="csv"):
                logger.info(f"Detecções salvas em: {export_path}")

        logger.info("Processamento concluído com sucesso")
        return 0

    except KeyboardInterrupt:
        logger.info("Processamento interrompido pelo usuário")
        return 0
    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
