# Modelos YOLO

Este diretório contém os modelos YOLO utilizados pelo sistema RIVAC-CV.

## Modelos Disponíveis

### YOLOv11 (Recomendado)

- `yolo11n.pt` - Nano (mais rápido)
- `yolo11s.pt` - Small (balanceado)
- `yolo11m.pt` - Medium (preciso)
- `yolo11l.pt` - Large (muito preciso)
- `yolo11x.pt` - Extra Large (máxima precisão)

### YOLOv8 (Compatível)

- `yolov8n.pt` - Nano
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large

## Download Automático

Os modelos oficiais são baixados automaticamente na primeira execução e salvos neste diretório.

### Download Manual

Caso deseje baixar manualmente:

```bash
# YOLOv11n (recomendado para início)
curl -L https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -o data/models/yolo11n.pt

# Outros modelos
curl -L https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt -o data/models/yolo11s.pt
curl -L https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt -o data/models/yolo11m.pt
```

## Modelos Customizados

Para usar modelos customizados:

1. Coloque o arquivo `.pt` neste diretório
2. Use o caminho completo no sistema:
   ```bash
   uv run python main.py --input video.mp4 --model data/models/meu_modelo.pt
   ```

## Estrutura de Classes

Todos os modelos oficiais usam o dataset COCO com 80 classes:

- 0: person (pessoa) - **Classe principal para retail**
- 1: bicycle
- 2: car
- 3: motorcycle
- ...

Para contagem de pessoas em retail, o sistema foca na classe 0 (person).

## Performance

| Modelo   | Tamanho | Velocidade | Precisão   | Uso Recomendado    |
| -------- | ------- | ---------- | ---------- | ------------------ |
| YOLOv11n | 5.3MB   | ⚡⚡⚡⚡⚡ | ⭐⭐⭐     | Tempo real, webcam |
| YOLOv11s | 19.5MB  | ⚡⚡⚡⚡   | ⭐⭐⭐⭐   | Balanceado         |
| YOLOv11m | 43.2MB  | ⚡⚡⚡     | ⭐⭐⭐⭐⭐ | Alta precisão      |
| YOLOv11l | 86.9MB  | ⚡⚡       | ⭐⭐⭐⭐⭐ | Análise offline    |
| YOLOv11x | 194.7MB | ⚡         | ⭐⭐⭐⭐⭐ | Máxima precisão    |

## Exemplo de Uso

```python
from src.deteccao.yolo_detector import YOLODetector

# Modelo rápido para tempo real
detector = YOLODetector("data/models/yolo11n.pt")

# Modelo preciso para análise
detector = YOLODetector("data/models/yolo11m.pt")
```

⚠️ **Nota**: Arquivos .pt são grandes e não versionados no Git.
