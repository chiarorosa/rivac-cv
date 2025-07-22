"""
Interface principal do sistema usando Streamlit
Dashboard interativo para monitoramento em tempo real
"""

import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Configurar página
st.set_page_config(
    page_title="RIVAC-CV - Monitoramento por Visão Computacional",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Imports do sistema
try:
    from src.deteccao.yolo_detector import YOLODetector
    from src.ingestao.video_source import VideoSource
    from src.pipeline import DetectionPipeline
    from src.utils.config import load_config, save_config
    from src.utils.logger import get_logger
except ImportError as e:
    st.error(f"Erro ao importar módulos do sistema: {e}")
    st.stop()

# Configurar logger
logger = get_logger("streamlit_app")

# Configurar estilos CSS
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #17becf);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .status-running {
        color: #00ff00;
        font-weight: bold;
    }
    .status-stopped {
        color: #ff0000;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)


def get_available_videos():
    """Retorna lista de vídeos disponíveis em data/videos/."""
    videos_dir = Path("data/videos")
    if not videos_dir.exists():
        return []
    
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".m4v", ".flv", ".wmv"]
    available_videos = []
    
    for video_file in videos_dir.iterdir():
        if video_file.is_file() and video_file.suffix.lower() in video_extensions:
            # Obter informações do arquivo
            file_size = video_file.stat().st_size / (1024 * 1024)  # MB
            
            # Tentar obter informações do vídeo usando OpenCV
            video_info = {"duration": "N/A", "fps": "N/A", "resolution": "N/A"}
            try:
                cap = cv2.VideoCapture(str(video_file))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    if fps > 0:
                        duration = frame_count / fps
                        video_info["duration"] = f"{duration:.1f}s"
                        video_info["fps"] = f"{fps:.1f}"
                        video_info["resolution"] = f"{width}x{height}"
                cap.release()
            except:
                pass  # Se não conseguir obter informações, usa valores padrão
            
            available_videos.append({
                "name": video_file.name,
                "path": str(video_file),
                "size_mb": round(file_size, 1),
                **video_info
            })
    
    return sorted(available_videos, key=lambda x: x["name"])


def initialize_session_state():
    """Inicializa estado da sessão."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "config" not in st.session_state:
        st.session_state.config = load_config()
    if "detection_results" not in st.session_state:
        st.session_state.detection_results = []
    if "current_frame" not in st.session_state:
        st.session_state.current_frame = None
    if "statistics" not in st.session_state:
        st.session_state.statistics = {}


def render_header():
    """Renderiza cabeçalho da aplicação."""
    st.markdown(
        """
    <div class="main-header">
        <h1>🛒 RIVAC-CV - Sistema de Monitoramento por Visão Computacional</h1>
        <p>Análise inteligente de fluxo de pessoas em ambientes de varejo</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Renderiza barra lateral com configurações."""
    st.sidebar.title("⚙️ Configurações")

    # Seção de entrada de vídeo
    st.sidebar.header("📹 Fonte de Vídeo")

    source_type = st.sidebar.selectbox("Tipo de fonte:", ["Webcam", "Arquivo de vídeo", "URL/Stream", "Câmera IP"])

    source_path = None

    if source_type == "Webcam":
        camera_id = st.sidebar.number_input("ID da câmera:", min_value=0, max_value=10, value=0)
        source_path = camera_id

    elif source_type == "Arquivo de vídeo":
        # Obter vídeos disponíveis
        available_videos = get_available_videos()
        
        if available_videos:
            st.sidebar.subheader("📋 Vídeos Disponíveis")
            
            # Criar opções para selectbox
            video_options = ["🆕 Fazer upload de novo arquivo..."]
            for video in available_videos:
                video_options.append(f"📽️ {video['name']} ({video['size_mb']} MB)")
            
            selected_option = st.sidebar.selectbox("Escolher vídeo:", video_options)
            
            if selected_option.startswith("📽️"):
                # Usuário selecionou um vídeo existente
                video_name = selected_option.split("📽️ ")[1].split(" (")[0]
                selected_video = next(v for v in available_videos if v["name"] == video_name)
                source_path = selected_video["path"]
                
                # Mostrar informações detalhadas do vídeo selecionado
                st.sidebar.success(f"✅ Vídeo selecionado: {video_name}")
                
                # Criar um expander com informações detalhadas
                with st.sidebar.expander("ℹ️ Informações do Vídeo"):
                    st.write(f"**📁 Arquivo:** {selected_video['name']}")
                    st.write(f"**📏 Tamanho:** {selected_video['size_mb']} MB")
                    st.write(f"**⏱️ Duração:** {selected_video['duration']}")
                    st.write(f"**🎬 FPS:** {selected_video['fps']}")
                    st.write(f"**📺 Resolução:** {selected_video['resolution']}")
                
            else:
                # Usuário quer fazer upload
                uploaded_file = st.sidebar.file_uploader(
                    "Fazer upload de arquivo:", 
                    type=["mp4", "avi", "mov", "mkv"],
                    help="Formatos suportados: MP4, AVI, MOV, MKV"
                )
                if uploaded_file:
                    # Salvar arquivo temporário
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        source_path = tmp_file.name
                    st.sidebar.success(f"✅ Upload concluído: {uploaded_file.name}")
        else:
            # Nenhum vídeo disponível, apenas upload
            st.sidebar.info("ℹ️ Nenhum vídeo encontrado em data/videos/")
            uploaded_file = st.sidebar.file_uploader(
                "Fazer upload de arquivo:", 
                type=["mp4", "avi", "mov", "mkv"],
                help="Formatos suportados: MP4, AVI, MOV, MKV"
            )
            if uploaded_file:
                # Salvar arquivo temporário
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    source_path = tmp_file.name
                st.sidebar.success(f"✅ Upload concluído: {uploaded_file.name}")

    elif source_type == "URL/Stream":
        source_path = st.sidebar.text_input("URL do stream:")

    elif source_type == "Câmera IP":
        ip = st.sidebar.text_input("IP da câmera:", value="192.168.1.100")
        port = st.sidebar.number_input("Porta:", value=8080)
        username = st.sidebar.text_input("Usuário:")
        password = st.sidebar.text_input("Senha:", type="password")

        if ip and username and password:
            source_path = f"rtsp://{username}:{password}@{ip}:{port}/stream"

    # Configurações de detecção
    st.sidebar.header("🔍 Detecção")

    model_options = {
        "YOLOv11n (Rápido)": "data/models/yolo11n.pt",
        "YOLOv11s (Balanceado)": "data/models/yolo11s.pt",
        "YOLOv11m (Preciso)": "data/models/yolo11m.pt",
        "YOLOv11l (Muito Preciso)": "data/models/yolo11l.pt",
    }

    selected_model = st.sidebar.selectbox("Modelo:", list(model_options.keys()))
    model_path = model_options[selected_model]

    confidence = st.sidebar.slider("Confiança mínima:", min_value=0.1, max_value=1.0, value=0.3, step=0.05)

    iou_threshold = st.sidebar.slider("Threshold IoU:", min_value=0.1, max_value=1.0, value=0.5, step=0.05)

    # Configurações de visualização
    st.sidebar.header("👁️ Visualização")

    show_bboxes = st.sidebar.checkbox("Mostrar bounding boxes", value=True)
    show_confidence = st.sidebar.checkbox("Mostrar confiança", value=True)
    show_fps = st.sidebar.checkbox("Mostrar FPS", value=True)

    return {
        "source_path": source_path,
        "model_path": model_path,
        "confidence": confidence,
        "iou_threshold": iou_threshold,
        "show_bboxes": show_bboxes,
        "show_confidence": show_confidence,
        "show_fps": show_fps,
    }


def setup_pipeline(config: Dict[str, Any]) -> bool:
    """Configura o pipeline de detecção."""
    try:
        if not config["source_path"]:
            st.error("⚠️ Por favor, configure uma fonte de vídeo.")
            return False

        # Criar pipeline
        pipeline = DetectionPipeline()

        # Configurar fonte
        if not pipeline.setup_source(config["source_path"]):
            st.error("❌ Erro ao configurar fonte de vídeo.")
            return False

        # Configurar detector
        detector_config = {
            "confidence": config["confidence"],
            "iou_threshold": config["iou_threshold"],
            "classes": [0],  # Apenas pessoas
            "device": "auto",
        }

        if not pipeline.setup_detector(config["model_path"], **detector_config):
            st.error("❌ Erro ao configurar detector.")
            return False

        st.session_state.pipeline = pipeline
        return True

    except Exception as e:
        st.error(f"❌ Erro ao configurar pipeline: {e}")
        logger.error(f"Erro na configuração: {e}")
        return False


def process_frame_callback(frame, detections, frame_number):
    """Callback chamado a cada frame processado."""
    # Atualizar frame atual
    st.session_state.current_frame = frame.copy()

    # Adicionar detecções aos resultados
    frame_results = {
        "frame_number": frame_number,
        "timestamp": time.time(),
        "detections_count": len(detections),
        "detections": [det.to_dict() for det in detections],
    }
    st.session_state.detection_results.append(frame_results)

    # Manter apenas últimos 100 frames para economizar memória
    if len(st.session_state.detection_results) > 100:
        st.session_state.detection_results.pop(0)


def draw_detections(frame: np.ndarray, detections: list, config: Dict[str, Any]) -> np.ndarray:
    """Desenha detecções no frame."""
    if not config.get("show_bboxes", True):
        return frame

    annotated_frame = frame.copy()

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])
        confidence = detection["confidence"]
        class_name = detection.get("class_name", "object")

        # Desenhar bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Desenhar label
        if config.get("show_confidence", True):
            label = f"{class_name}: {confidence:.2f}"
        else:
            label = class_name

        # Calcular posição do texto
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return annotated_frame


def render_main_content():
    """Renderiza conteúdo principal."""
    config = render_sidebar()

    # Área principal dividida em colunas
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📺 Monitor de Vídeo")

        # Container para o vídeo
        video_container = st.empty()

        # Controles
        control_col1, control_col2, control_col3 = st.columns(3)

        with control_col1:
            if st.button("▶️ Iniciar", type="primary"):
                if setup_pipeline(config):
                    st.session_state.is_running = True

                    # Adicionar callback
                    st.session_state.pipeline.add_frame_callback(process_frame_callback)

                    # Executar pipeline em thread separada
                    def run_pipeline():
                        try:
                            results = st.session_state.pipeline.run(max_frames=None, show_progress=False)
                            st.session_state.statistics = results
                        except Exception as e:
                            logger.error(f"Erro na execução: {e}")
                        finally:
                            st.session_state.is_running = False

                    thread = threading.Thread(target=run_pipeline, daemon=True)
                    thread.start()

                    st.success("✅ Processamento iniciado!")

        with control_col2:
            if st.button("⏸️ Parar"):
                if st.session_state.pipeline:
                    st.session_state.pipeline.stop()
                st.session_state.is_running = False
                st.info("⏸️ Processamento parado.")

        with control_col3:
            if st.button("💾 Exportar Dados"):
                if st.session_state.detection_results:
                    # Preparar dados para download
                    df = pd.DataFrame()
                    for frame_result in st.session_state.detection_results:
                        for detection in frame_result["detections"]:
                            row = detection.copy()
                            row["frame_number"] = frame_result["frame_number"]
                            row["timestamp"] = frame_result["timestamp"]
                            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"detections_{int(time.time())}.csv",
                        mime="text/csv",
                    )

        # Mostrar frame atual
        if st.session_state.current_frame is not None:
            # Obter últimas detecções
            latest_detections = []
            if st.session_state.detection_results:
                latest_detections = st.session_state.detection_results[-1]["detections"]

            # Desenhar detecções
            annotated_frame = draw_detections(st.session_state.current_frame, latest_detections, config)

            # Mostrar no Streamlit
            video_container.image(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True
            )

    with col2:
        st.header("📊 Estatísticas")

        # Status do sistema
        if st.session_state.is_running:
            st.markdown('<p class="status-running">🟢 EXECUTANDO</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-stopped">🔴 PARADO</p>', unsafe_allow_html=True)

        # Métricas em tempo real
        if st.session_state.detection_results:
            latest_frame = st.session_state.detection_results[-1]

            st.metric("Pessoas Detectadas", latest_frame["detections_count"])
            st.metric("Frame Atual", latest_frame["frame_number"])

            if st.session_state.pipeline:
                stats = st.session_state.pipeline.get_statistics()
                st.metric("FPS Médio", f"{stats.get('average_fps', 0):.1f}")
                st.metric("Total de Detecções", stats.get("total_detections", 0))

        # Gráfico de detecções por frame
        if len(st.session_state.detection_results) > 1:
            st.subheader("📈 Detecções por Frame")

            df_chart = pd.DataFrame(
                [
                    {"frame": r["frame_number"], "detections": r["detections_count"]}
                    for r in st.session_state.detection_results[-50:]  # Últimos 50 frames
                ]
            )

            st.line_chart(df_chart.set_index("frame"))

        # Histórico de detecções
        st.subheader("📋 Histórico Recente")
        if st.session_state.detection_results:
            for result in st.session_state.detection_results[-10:]:  # Últimos 10
                timestamp = time.strftime("%H:%M:%S", time.localtime(result["timestamp"]))
                st.write(f"⏰ {timestamp} - Frame {result['frame_number']}: {result['detections_count']} pessoas")


def main():
    """Função principal da aplicação."""
    initialize_session_state()
    render_header()
    render_main_content()

    # Auto-refresh para atualizar interface
    if st.session_state.is_running:
        time.sleep(0.1)
        st.rerun()


if __name__ == "__main__":
    main()
