# Modelos Pré-treinados

Este diretório contém os modelos de machine learning para o RIVAC-CV.

## Modelos YOLO Suportados:

### YOLOv11 (Recomendado)

- `yolo11n.pt` - Nano (mais rápido)
- `yolo11s.pt` - Small (equilibrado)
- `yolo11m.pt` - Medium (boa precisão)
- `yolo11l.pt` - Large (alta precisão)
- `yolo11x.pt` - Extra Large (máxima precisão)

### YOLOv8

- `yolov8n.pt` - Nano
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large

## Download dos Modelos

Os modelos são baixados automaticamente pelo Ultralytics na primeira execução.

Para download manual:

```bash
# YOLOv11
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

# YOLOv8
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
```

⚠️ **Nota**: Arquivos .pt são grandes e não versionados no Git.
