# Sistema de Monitoramento por Visão Computacional no Varejo (RIVAC-CV)

Sistema modular e extensível para contagem e análise de fluxo de pessoas em ambientes de varejo usando visão computacional com modelos YOLO e OpenCV.

## 🚀 Características

- **Detecção Multi-modelo**: Suporte para YOLOv8, YOLOv11 e modelos customizados
- **Tracking Avançado**: Rastreamento de objetos com ByteTrack e BoTSORT
- **ROI Dinâmico**: Definição interativa de regiões de interesse
- **Interface Web**: Dashboard interativo com Streamlit
- **Análise Temporal**: Métricas e relatórios de fluxo por período
- **Exportação Flexível**: Dados em CSV, JSON e relatórios visuais

## 📁 Arquitetura do Sistema

```
rivac-cv/
├── src/
│   ├── ingestao/          # Captura de vídeo (arquivos, câmeras, streams)
│   ├── deteccao/          # Modelos de detecção (YOLO, etc.)
│   ├── tracking/          # Rastreamento de objetos
│   ├── roi/               # Gerenciamento de regiões de interesse
│   ├── contagem/          # Algoritmos de contagem
│   ├── visualizacao/      # Interfaces e overlays
│   ├── exportacao/        # Exportação de dados e relatórios
│   ├── persistencia/      # Banco de dados e logs
│   └── utils/             # Utilitários e configurações
├── config/
│   ├── models.yaml        # Configurações de modelos
│   ├── trackers.yaml      # Configurações de tracking
│   └── app_config.yaml    # Configurações da aplicação
├── data/
│   ├── videos/            # Vídeos de teste
│   ├── models/            # Modelos pré-treinados
│   └── exports/           # Dados exportados
├── tests/                 # Testes unitários
├── docs/                  # Documentação técnica
├── streamlit_app.py       # Interface web principal
├── main.py                # CLI principal
├── .uv-commands           # Comandos UV para RIVAC-CV
└── pyproject.toml         # Configuração de dependências
```

### Módulos Principais

#### 🎥 Ingestão (`src/ingestao/`)

- **VideoSource**: Captura de vídeo de arquivos, câmeras ou streams RTSP
- **FrameProcessor** _ToDo_: Pré-processamento e redimensionamento de frames
- **StreamManager** _ToDo_: Gerenciamento de múltiplas fontes simultâneas

#### 🎯 Detecção (`src/deteccao/`)

- **YOLODetector**: Interface para modelos YOLO (v8, v11)
- **CustomDetector** _ToDo_: Suporte para modelos personalizados
- **DetectionFilter** _ToDo_: Filtragem por confiança e classes

#### 🔍 Tracking (`src/tracking/`) _ToDo_

- **ByteTracker**: Tracking rápido e eficiente
- **BoTSORTTracker**: Tracking com re-identificação
- **TrackManager**: Gerenciamento de trajetórias

#### 📍 ROI (`src/roi/`) _ToDo_

- **ROIManager**: Definição e gerenciamento de regiões
- **InteractiveROI**: Interface para desenho de ROIs
- **GeometryUtils**: Utilitários geométricos

#### 📊 Contagem (`src/contagem/`) _ToDo_

- **LineCounter**: Contagem por linha de passagem
- **AreaCounter**: Contagem por permanência em área
- **FlowAnalyzer**: Análise de fluxo direcional

## 🛠️ Instalação

### Pré-requisitos

- Python 3.10+
- UV Package Manager
- CUDA (opcional, para GPU)

### Setup

```bash
# Clone o repositório
git clone https://github.com/chiarorosa/rivac-cv.git
cd rivac-cv

# Instale UV (se ainda não tiver)
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Instale todas as dependências
uv sync --all-extras
```

## 🎯 Uso

### Interface Web (Streamlit)

```bash
uv run streamlit run streamlit_app.py --server.headless true --server.port 8501
```

Acesse `http://localhost:8501` no seu navegador.

### Linha de Comando

```bash
# Processamento básico
uv run python main.py --input data/videos/sample_1.mp4

# Com exportação de detecções (CSV)
uv run python main.py --input data/videos/sample_1.mp4 --save-detections

# Com exportação JSON
uv run python main.py --input data/videos/sample_1.mp4 --save-detections --export-format json

# Webcam em tempo real
uv run python main.py --input 0

# Com modelo específico e confiança personalizada
uv run python main.py --input data/videos/sample_1.mp4 --model data/models/yolo11m.pt --confidence 0.5
```

### API Python

```python
from src.pipeline import DetectionPipeline
from src.ingestao.video_source import VideoSource
from src.deteccao.yolo_detector import YOLODetector

# Configurar pipeline
pipeline = DetectionPipeline(
    source=VideoSource("video.mp4"),
    detector=YOLODetector("data/models/yolo11n.pt"),
    tracker_config="bytetrack.yaml"
)

# Executar detecção e contagem
results = pipeline.run()

# Exportar resultados
pipeline.export_results("output.csv")
```

## 📊 Modelos Suportados

### Detecção

- **YOLOv11**: `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`
- **YOLOv8**: `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`
- **Modelos Customizados**: Treinados pelo usuário

### Tracking

- **ByteTrack**: Tracking rápido e eficiente
- **BoTSORT**: Tracking com re-identificação
- **DeepSORT**: Tracking clássico (em desenvolvimento)

## 🔧 Configuração

### Arquivo Principal (`config/app_config.yaml`)

```yaml
# Configurações de entrada
input:
  default_source: "camera" # camera, file, rtsp
  frame_skip: 1 # Processar a cada N frames
  resize_width: 640 # Largura para processamento

# Configurações de detecção
detection:
  model: "data/models/yolo11n.pt"
  confidence: 0.3
  iou_threshold: 0.5
  classes: [0] # Apenas pessoas (classe 0 no COCO)

# Configurações de tracking
tracking:
  enabled: true
  tracker: "bytetrack.yaml"
  max_age: 30 # Frames para manter track perdido

# Configurações de ROI
roi:
  enabled: true
  interactive_setup: true
  save_regions: true

# Configurações de saída
output:
  save_video: true
  save_data: true
  export_format: ["csv", "json"]
  fps: 30
```

## 🎨 Interface Web

A interface Streamlit oferece:

- **Upload de Vídeo**: Arraste e solte arquivos ou conecte câmeras
- **Configuração Visual**: Ajuste parâmetros em tempo real
- **ROI Interativo**: Desenhe regiões de interesse no vídeo
- **Visualização Live**: Veja resultados em tempo real
- **Dashboard de Métricas**: Gráficos e estatísticas de fluxo
- **Exportação**: Download de dados e relatórios

## 📈 Métricas e Análises

### Métricas Básicas

- Contagem total por período
- Densidade de pessoas por região
- Tempo de permanência médio
- Fluxo de entrada/saída

### Análises Avançadas

- Heatmaps de movimentação
- Padrões temporais (horário/dia da semana)
- Análise de trajetórias
- Detecção de aglomerações

## 🧪 Testes _ToDo_

```bash
# Executar todos os testes
uv run pytest

# Testes com cobertura
uv run pytest --cov=src --cov-report=html

# Testes específicos
uv run pytest tests/test_detection.py -v
```

## 📚 Documentação

- [Arquitetura de Software](docs/software_architecture.md) - Detalhes da arquitetura
- [Guia do UV](docs/uv_guide.md) - Gerenciador de pacotes UV
- [CI/CD com UV](docs/ci_cd_uv_examples.md) - Exemplos para CI/CD
- [Interpretação de Detecções](docs/detection_analysis_guide.md) - Como analisar exportações

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

### Estrutura para Novos Detectores

```python
from src.deteccao.base_detector import BaseDetector

class MeuDetector(BaseDetector):
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)

    def load_model(self, model_path: str):
        # Implementar carregamento do modelo
        pass

    def predict(self, frame: np.ndarray) -> List[Detection]:
        # Implementar predição
        pass
```

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## ✨ Reconhecimentos

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Modelos de detecção
- [OpenCV](https://opencv.org/) - Processamento de vídeo
- [Streamlit](https://streamlit.io/) - Interface web
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - Algoritmo de tracking

## 📞 Suporte

- 🐛 Issues: [GitHub Issues](https://github.com/chiarorosa/rivac-cv/issues)
- 📖 Documentação: [Repositório](https://github.com/chiarorosa/rivac-cv)

---

**RIVAC-CV** - Sistema de Monitoramento Inteligente para Varejo 🛒👥📊
