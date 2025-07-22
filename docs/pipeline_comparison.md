# Comparação: Pipeline.py vs Pipeline_v2.py

## Resumo Executivo

O sistema RIVAC-CV possui duas implementações de pipeline distintas:

- **`pipeline.py`**: Versão básica focada em detecção simples de objetos
- **`pipeline_v2.py`**: Versão avançada com sistema completo de visão computacional

O **`pipeline_v2.py`** representa uma **evolução significativa** do pipeline original, oferecendo funcionalidades completas para monitoramento de varejo, incluindo tracking, ROI, contagem e visualização avançada.

---

## Comparação Detalhada

### 1. Funcionalidades Principais

| Funcionalidade                 | `pipeline.py`                | `pipeline_v2.py`                                  |
| ------------------------------ | ---------------------------- | ------------------------------------------------- |
| **Detecção de Objetos**        | ✅ Básica (YOLO)             | ✅ Avançada (YOLO)                                |
| **Tracking de Objetos**        | ❌ Não implementado          | ✅ Múltiplos algoritmos (Simple, YOLO, ByteTrack) |
| **ROI (Regiões de Interesse)** | ❌ Não implementado          | ✅ Sistema completo de gerenciamento              |
| **Contagem de Objetos**        | ❌ Não implementado          | ✅ Contagem por ROI com estatísticas              |
| **Visualização em Tempo Real** | ❌ Limitada                  | ✅ Completa com anotações visuais                 |
| **Callbacks Especializados**   | ✅ Básicos (frame, detecção) | ✅ Avançados (frame, detecção, contagem)          |
| **Exportação de Dados**        | ✅ CSV/JSON básico           | ✅ CSV/JSON com pandas + estatísticas             |

### 2. Arquitetura do Sistema

#### Pipeline.py - Arquitetura Simples

```
Input → Detection → Export
```

**Fluxo básico:**

1. Configura fonte de vídeo
2. Configura detector YOLO
3. Processa frames (apenas detecção)
4. Exporta resultados básicos

#### Pipeline_v2.py - Arquitetura Modular Completa

```
Input → Detection → Tracking → ROI Analysis → Counting → Visualization → Export
```

**Fluxo avançado:**

1. Configura fonte de vídeo
2. Configura detector YOLO
3. Configura sistema de tracking
4. Configura gerenciador de ROI
5. Configura sistema de contagem
6. Processa frames com pipeline completo
7. Exporta dados especializados

### 3. Dependências e Imports

#### Pipeline.py - Dependências Básicas

```python
import csv, json, time, uuid
from datetime import datetime
from pathlib import Path
import numpy as np

# Módulos internos
from .deteccao.base_detector import BaseDetector, Detection
from .deteccao.yolo_detector import YOLODetector
from .ingestao.base_source import BaseVideoSource
from .ingestao.video_source import VideoSource
from .utils.config import load_config
from .utils.logger import LoggerMixin, log_performance
```

#### Pipeline_v2.py - Dependências Avançadas

```python
import time, uuid, cv2
from datetime import datetime
from pathlib import Path
import numpy as np

# Módulos especializados
from .counting import CounterManager, CountEvent, create_counter_manager
from .deteccao.base_detector import BaseDetector
from .deteccao.yolo_detector import YOLODetector
from .ingestao.base_source import BaseVideoSource
from .ingestao.video_source import VideoSource
from .roi import ROIManager, create_roi_manager
from .tracking import BaseTracker, Track, create_tracker
from .utils.config import get_config, load_config
from .utils.logger import LoggerMixin, get_logger

# Bibliotecas adicionais
import pandas as pd  # Para exportação avançada
```

### 4. Métodos de Configuração

#### Pipeline.py - Configuração Básica (2 métodos)

- `setup_source(source_config, **kwargs)` - Configura fonte de vídeo
- `setup_detector(model_path, **kwargs)` - Configura detector YOLO

#### Pipeline_v2.py - Configuração Completa (6 métodos)

- `setup_source(source_path, **kwargs)` - Configura fonte de vídeo
- `setup_detector(model_path, **kwargs)` - Configura detector YOLO
- `setup_tracker(tracker_type, **kwargs)` - Configura sistema de tracking
- `setup_roi_manager(**kwargs)` - Configura gerenciador de ROI
- `setup_counter_manager(**kwargs)` - Configura sistema de contagem
- `setup_auto(source_path, model_path)` - Configuração automática completa

### 5. Processamento de Frames

#### Pipeline.py - Processamento Simples

```python
def _process_frame(self, frame: np.ndarray) -> List[Detection]:
    """Processa um único frame - apenas detecção."""
    detections = self.detector.detect(frame)
    return detections
```

#### Pipeline_v2.py - Processamento Avançado

```python
def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict], List[Track], Dict]:
    """Processa um único frame com pipeline completo."""
    # 1. Detecção de objetos
    detections = self.detector.detect(frame)

    # 2. Tracking de objetos
    tracks = self.tracker.update(detections) if self.tracker else []

    # 3. Análise de ROI
    roi_frame = self.roi_manager.draw_all_rois(frame) if self.roi_manager else frame

    # 4. Contagem de objetos
    count_events = self.counter_manager.update_all(tracks) if self.counter_manager else {}

    # 5. Anotações visuais
    annotated_frame = self._annotate_frame(roi_frame, detections, tracks)

    # 6. Callbacks especializados
    self._call_callbacks(annotated_frame, detections, count_events)

    return annotated_frame, detections, tracks, count_events
```

### 6. Capacidades de Visualização

#### Pipeline.py - Visualização Limitada

- ❌ Sem visualização integrada
- ❌ Sem anotações visuais
- ❌ Sem display em tempo real

#### Pipeline_v2.py - Visualização Completa

- ✅ Visualização em tempo real com OpenCV
- ✅ Desenho de bounding boxes com labels
- ✅ Exibição de track IDs
- ✅ Desenho de ROIs no frame
- ✅ Informações de status (FPS, contadores)
- ✅ Sistema de cores e estilos configurável

**Métodos de visualização:**

- `_annotate_frame()` - Adiciona todas as anotações
- `_draw_detection()` - Desenha bounding boxes
- `_draw_track()` - Desenha tracks com IDs
- `_draw_status_info()` - Exibe estatísticas no frame

### 7. Sistema de Estatísticas

#### Pipeline.py - Estatísticas Básicas

```python
stats = {
    "total_frames": 0,
    "total_detections": 0,
    "average_fps": 0.0,
    "execution_time": 0.0,
    "detections_per_frame": [],
    "error": None
}
```

#### Pipeline_v2.py - Estatísticas Avançadas

```python
stats = {
    "total_frames": 0,
    "total_detections": 0,
    "average_fps": 0.0,
    "processing_time": 0.0,
    "total_counts": {},           # Contagens por ROI
    "frame_times": [],            # Performance detalhada
    "current_fps": 0.0,           # FPS em tempo real
    "counting": {                 # Estatísticas de contagem
        "roi_name": {
            "total_in": 0,
            "total_out": 0,
            "current_count": 0,
            "net_count": 0,
            "total_crossings": 0,
            "rate_per_hour": 0.0
        }
    }
}
```

### 8. Sistema de Callbacks

#### Pipeline.py - Callbacks Básicos

```python
# 2 tipos de callbacks
def add_frame_callback(callback):
    """Callback: (frame, detections, frame_number)"""

def add_detection_callback(callback):
    """Callback: (detections, frame_number)"""
```

#### Pipeline_v2.py - Callbacks Especializados

```python
# 3 tipos de callbacks especializados
def add_frame_callback(callback: Callable[[np.ndarray, List[Dict], int], None]):
    """Callback para cada frame processado"""

def add_detection_callback(callback: Callable[[List[Dict]], None]):
    """Callback para detecções"""

def add_count_callback(callback: Callable[[Dict[str, List[CountEvent]]], None]):
    """Callback para eventos de contagem"""
```

### 9. Capacidades de Exportação

#### Pipeline.py - Exportação Básica

```python
def export_results(output_path: str, format: str = "csv") -> bool:
    """Exporta apenas detecções básicas"""
    # CSV: frame_number, class_id, confidence, bbox
    # JSON: detections simples
```

#### Pipeline_v2.py - Exportação Avançada

```python
def export_results(filepath: str, format: str = "csv") -> bool:
    """Exporta dados completos do sistema"""
    # CSV com pandas: estatísticas de contagem por ROI
    # JSON estruturado: dados de sessão + estatísticas completas
```

**Dados exportados no v2:**

- Informações de sessão
- Estatísticas de contagem por ROI
- Métricas de performance
- Dados de tracking
- Configurações utilizadas

### 10. Configuração Automática

#### Pipeline.py

- ❌ Configuração manual obrigatória
- ❌ Setup passo-a-passo necessário

#### Pipeline_v2.py

- ✅ Método `setup_auto()` disponível
- ✅ Configuração inteligente baseada em config.yaml
- ✅ Ativação automática de módulos opcionais

```python
def setup_auto(self, source_path: str, model_path: str = "yolo11n.pt") -> bool:
    """Configuração automática completa do pipeline"""
    # Configura todos os módulos automaticamente
    # Baseado nas configurações em config.yaml
```

---

## Status de Uso no Projeto

### Atualmente Ativo

**`pipeline.py`** está sendo usado como pipeline principal:

- `src/__init__.py` → `from .pipeline import DetectionPipeline`
- `main.py` → importa e usa pipeline básico
- `streamlit_app.py` → interface web usa versão básica
- `README.md` → exemplos com pipeline básico

### Disponível mas Não Utilizado

**`pipeline_v2.py`** está implementado mas não está sendo usado pelo sistema (ainda demanda muitos testes durante a Pesquisa)

---

## Casos de Uso Recomendados

### Use Pipeline.py quando:

- **Detecção simples**: Apenas precisa identificar objetos
- **Performance crítica**: Máxima velocidade de processamento
- **Recursos limitados**: Ambiente com pouca memória/CPU
- **Prototipagem rápida**: Desenvolvimento inicial
- **Integração simples**: API básica sem complexidade

### Use Pipeline_v2.py quando:

- **Monitoramento de varejo**: Contagem de pessoas em lojas
- **Análise de tráfego**: Tracking de pessoas/veículos
- **ROIs específicas**: Monitoramento de áreas determinadas
- **Relatórios detalhados**: Estatísticas e métricas avançadas
- **Visualização rica**: Interface com anotações visuais
- **Sistema de produção**: Aplicação comercial completa

---

## Características Técnicas

### Performance

- **Pipeline.py**: ~30% mais rápido (menos processamento)
- **Pipeline_v2.py**: Mais robusto (processamento completo)

### Memória

- **Pipeline.py**: Menor uso de memória
- **Pipeline_v2.py**: Maior uso devido a tracking e histórico

### Complexidade

- **Pipeline.py**: ~400 linhas, arquitetura simples
- **Pipeline_v2.py**: ~600 linhas, arquitetura modular

### Manutenibilidade

- **Pipeline.py**: Código direto, fácil de modificar
- **Pipeline_v2.py**: Modular, extensível, design patterns

---

## Considerações de Implementação

### Pipeline.py

```python
# Uso típico
pipeline = DetectionPipeline()
pipeline.setup_source("video.mp4")
pipeline.setup_detector("yolo11n.pt")
results = pipeline.run()
```

### Pipeline_v2.py

```python
# Uso típico
pipeline = DetectionPipeline()
# Configuração automática
pipeline.setup_auto("video.mp4", "yolo11n.pt")
# OU configuração manual
pipeline.setup_source("video.mp4")
pipeline.setup_detector("yolo11n.pt")
pipeline.setup_tracker("simple")
pipeline.setup_roi_manager()
pipeline.setup_counter_manager()

results = pipeline.run()
```

---

_[AINDA EM DESENVOLVIMENTO E PESQUISA]_ O **`pipeline_v2.py`** representa a **evolução natural** do sistema RIVAC-CV, oferecendo funcionalidades completas de um sistema de visão computacional moderno para monitoramento de varejo. Enquanto o `pipeline.py` serve bem para casos simples, o v2 é mais adequado para aplicações de produção que requerem tracking, contagem e análise de ROI.
