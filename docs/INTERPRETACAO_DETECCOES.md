# Interpretação de Exportações de Detecções

Este documento explica como interpretar e analisar os dados exportados pelo sistema RIVAC-CV.

## 📊 Visão Geral das Exportações

O sistema RIVAC-CV exporta detecções em dois formatos principais:

- **CSV**: Para análise em planilhas e ferramentas de BI
- **JSON**: Para integração com APIs e análise programática

Cada exportação contém:

- **Metadados da sessão**: Informações sobre o processamento
- **Detecções frame-by-frame**: Dados detalhados de cada objeto detectado
- **Coordenadas precisas**: Bounding boxes e posições centrais
- **Métricas de confiança**: Scores de detecção para filtragem

## 🔍 Estrutura dos Dados

### Formato CSV

```csv
frame_number,session_id,timestamp,class_id,class_name,confidence,x1,y1,x2,y2
0,ebc0d9ad,2025-07-21T21:36:14.356268,0,0,0.8675,163.37,110.61,222.34,242.93
```

#### Campos Detalhados:

| Campo          | Tipo         | Descrição                               | Exemplo                      |
| -------------- | ------------ | --------------------------------------- | ---------------------------- |
| `frame_number` | int          | Número sequencial do frame (0-indexed)  | `0, 1, 2, ...`               |
| `session_id`   | string       | Identificador único da sessão (8 chars) | `ebc0d9ad`                   |
| `timestamp`    | ISO datetime | Data/hora da exportação                 | `2025-07-21T21:36:14.356268` |
| `class_id`     | int          | ID da classe COCO detectada             | `0` (person)                 |
| `class_name`   | string       | Nome da classe detectada                | `person, car, bicycle`       |
| `confidence`   | float        | Confiança da detecção (0.0-1.0)         | `0.8675`                     |
| `x1, y1`       | float        | Coordenadas do canto superior esquerdo  | `163.37, 110.61`             |
| `x2, y2`       | float        | Coordenadas do canto inferior direito   | `222.34, 242.93`             |

### Formato JSON

```json
{
  "session_info": {
    "session_id": "ebc0d9ad",
    "export_timestamp": "2025-07-21T21:36:14.356268",
    "total_frames": 192,
    "total_detections": 684
  },
  "detections": [
    {
      "bbox": [163.37, 110.61, 222.34, 242.93],
      "confidence": 0.8675,
      "class_id": 0,
      "class_name": "person",
      "track_id": null,
      "center_x": 192.86,
      "center_y": 176.77,
      "width": 58.97,
      "height": 132.32,
      "frame_number": 0,
      "session_id": "ebc0d9ad",
      "timestamp": "2025-07-21T21:36:14.356268"
    }
  ]
}
```

#### Campos Adicionais no JSON:

| Campo      | Descrição                     | Cálculo           |
| ---------- | ----------------------------- | ----------------- |
| `center_x` | Centro horizontal da detecção | `(x1 + x2) / 2`   |
| `center_y` | Centro vertical da detecção   | `(y1 + y2) / 2`   |
| `width`    | Largura da bounding box       | `x2 - x1`         |
| `height`   | Altura da bounding box        | `y2 - y1`         |
| `track_id` | ID de rastreamento (futuro)   | `null` atualmente |

## 📈 Análises Possíveis

### 1. Análise Temporal

**Contagem de pessoas por frame:**

```python
import pandas as pd
import matplotlib.pyplot as plt

# Carregar dados
df = pd.read_csv('detections_session_id.csv')

# Contar detecções por frame
frame_counts = df.groupby('frame_number').size()

# Plotar evolução temporal
plt.figure(figsize=(12, 6))
plt.plot(frame_counts.index, frame_counts.values)
plt.title('Evolução do Número de Pessoas Detectadas')
plt.xlabel('Frame')
plt.ylabel('Número de Pessoas')
plt.grid(True)
plt.show()
```

### 2. Análise de Distribuição Espacial

**Mapa de calor das posições:**

```python
import seaborn as sns
import numpy as np

# Extrair coordenadas centrais
df['center_x'] = (df['x1'] + df['x2']) / 2
df['center_y'] = (df['y1'] + df['y2']) / 2

# Criar mapa de calor
plt.figure(figsize=(10, 8))
plt.hexbin(df['center_x'], df['center_y'], gridsize=30, cmap='YlOrRd')
plt.colorbar(label='Densidade de Detecções')
plt.title('Mapa de Calor - Posições das Pessoas')
plt.xlabel('Posição X (pixels)')
plt.ylabel('Posição Y (pixels)')
plt.gca().invert_yaxis()  # Inverter Y para coordenadas de imagem
plt.show()
```

### 3. Análise de Confiança

**Distribuição das confianças:**

```python
# Histograma de confianças
plt.figure(figsize=(10, 6))
plt.hist(df['confidence'], bins=50, alpha=0.7, edgecolor='black')
plt.title('Distribuição dos Scores de Confiança')
plt.xlabel('Confiança')
plt.ylabel('Frequência')
plt.axvline(df['confidence'].mean(), color='red', linestyle='--',
           label=f'Média: {df["confidence"].mean():.3f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Estatísticas
print(f"Confiança média: {df['confidence'].mean():.3f}")
print(f"Confiança mediana: {df['confidence'].median():.3f}")
print(f"Desvio padrão: {df['confidence'].std():.3f}")
```

### 4. Análise de Tamanhos

**Distribuição de tamanhos das detecções:**

```python
# Calcular dimensões
df['width'] = df['x2'] - df['x1']
df['height'] = df['y2'] - df['y1']
df['area'] = df['width'] * df['height']

# Gráfico de dispersão
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['width'], df['height'],
                     c=df['confidence'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Confiança')
plt.title('Distribuição de Tamanhos das Detecções')
plt.xlabel('Largura (pixels)')
plt.ylabel('Altura (pixels)')
plt.grid(True, alpha=0.3)
plt.show()
```

## 🎯 Casos de Uso Específicos para Retail

### 1. Contagem de Pessoas por Período

```python
# Assuming 24 FPS, calcular tempo em segundos
df['time_seconds'] = df['frame_number'] / 24.0
df['time_minutes'] = df['time_seconds'] / 60.0

# Agrupar por minutos
minute_counts = df.groupby(df['time_minutes'].astype(int)).size()

plt.figure(figsize=(12, 6))
plt.bar(minute_counts.index, minute_counts.values)
plt.title('Fluxo de Pessoas por Minuto')
plt.xlabel('Tempo (minutos)')
plt.ylabel('Número de Detecções')
plt.show()
```

### 2. Análise de Áreas de Interesse

```python
# Definir zonas (exemplo: entrada da loja)
def classify_zone(x, y):
    if x < 300:  # Lado esquerdo
        return 'Entrada'
    elif x > 700:  # Lado direito
        return 'Saída'
    else:
        return 'Centro'

# Aplicar classificação
df['zone'] = df.apply(lambda row: classify_zone(
    (row['x1'] + row['x2'])/2, (row['y1'] + row['y2'])/2), axis=1)

# Contar por zona
zone_counts = df['zone'].value_counts()
print("Distribuição por zona:")
print(zone_counts)

# Gráfico de pizza
plt.figure(figsize=(8, 8))
plt.pie(zone_counts.values, labels=zone_counts.index, autopct='%1.1f%%')
plt.title('Distribuição de Pessoas por Zona')
plt.show()
```

### 3. Detecção de Picos de Movimento

```python
# Calcular densidade por frame
frame_density = df.groupby('frame_number').size()

# Identificar picos (acima de 2 desvios padrão)
threshold = frame_density.mean() + 2 * frame_density.std()
peaks = frame_density[frame_density > threshold]

print(f"Picos de movimento detectados em {len(peaks)} frames:")
for frame, count in peaks.items():
    time_sec = frame / 24.0
    print(f"  Frame {frame} ({time_sec:.1f}s): {count} pessoas")
```

## 📊 Métricas de Performance para Retail

### 1. Taxa de Ocupação

```python
# Calcular ocupação média por minuto
df['minute'] = (df['frame_number'] / 24 / 60).astype(int)
occupancy_by_minute = df.groupby('minute')['frame_number'].nunique()

plt.figure(figsize=(12, 6))
plt.plot(occupancy_by_minute.index, occupancy_by_minute.values, marker='o')
plt.title('Taxa de Ocupação por Minuto')
plt.xlabel('Tempo (minutos)')
plt.ylabel('Frames com Detecções')
plt.grid(True)
plt.show()
```

### 2. Tempo de Permanência Estimado

```python
# Estimar permanência baseado em detecções contínuas
def estimate_dwell_time(df, confidence_threshold=0.5):
    # Filtrar por confiança
    high_conf = df[df['confidence'] > confidence_threshold]

    # Calcular duração do vídeo
    total_frames = df['frame_number'].max()
    duration_seconds = total_frames / 24.0

    # Estimar tempo médio de permanência
    avg_detections_per_frame = len(high_conf) / total_frames

    return {
        'total_duration': duration_seconds,
        'avg_people_per_frame': avg_detections_per_frame,
        'estimated_visits': high_conf.groupby('frame_number').size().mean()
    }

stats = estimate_dwell_time(df)
print("Estatísticas de permanência:")
for key, value in stats.items():
    print(f"  {key}: {value:.2f}")
```

## 🔧 Scripts de Análise Prontos

### Relatório Automático

```python
def generate_detection_report(csv_path, output_path='detection_report.html'):
    """Gera relatório HTML automático das detecções"""

    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # Carregar dados
    df = pd.read_csv(csv_path)

    # Calcular métricas
    total_detections = len(df)
    total_frames = df['frame_number'].nunique()
    avg_confidence = df['confidence'].mean()
    duration = df['frame_number'].max() / 24.0

    # Gerar gráficos
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Detecções por frame
    frame_counts = df.groupby('frame_number').size()
    ax1.plot(frame_counts.index, frame_counts.values)
    ax1.set_title('Detecções por Frame')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Contagem')

    # Distribuição de confiança
    ax2.hist(df['confidence'], bins=30, alpha=0.7)
    ax2.set_title('Distribuição de Confiança')
    ax2.set_xlabel('Confiança')
    ax2.set_ylabel('Frequência')

    # Posições (scatter)
    center_x = (df['x1'] + df['x2']) / 2
    center_y = (df['y1'] + df['y2']) / 2
    ax3.scatter(center_x, center_y, alpha=0.5, s=10)
    ax3.set_title('Distribuição Espacial')
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    ax3.invert_yaxis()

    # Timeline
    df['time_sec'] = df['frame_number'] / 24.0
    timeline = df.groupby(df['time_sec'].astype(int)).size()
    ax4.bar(timeline.index, timeline.values)
    ax4.set_title('Atividade por Segundo')
    ax4.set_xlabel('Tempo (s)')
    ax4.set_ylabel('Detecções')

    plt.tight_layout()
    plt.savefig(output_path.replace('.html', '.png'), dpi=300, bbox_inches='tight')

    # Salvar relatório
    report = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Relatório de Detecções</title></head>
    <body>
        <h1>Relatório de Detecções - {df['session_id'].iloc[0]}</h1>
        <h2>Resumo Executivo</h2>
        <ul>
            <li>Total de detecções: {total_detections}</li>
            <li>Frames processados: {total_frames}</li>
            <li>Duração do vídeo: {duration:.1f} segundos</li>
            <li>Confiança média: {avg_confidence:.3f}</li>
            <li>Detecções por frame: {total_detections/total_frames:.2f}</li>
        </ul>
        <img src="{output_path.replace('.html', '.png')}" alt="Gráficos de Análise">
    </body>
    </html>
    """

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Relatório salvo em: {output_path}")

# Usar o script
generate_detection_report('data/exports/detections_session_id.csv')
```

## 🚀 Integração com Ferramentas

### Power BI

1. Importar CSV diretamente
2. Criar medidas DAX para métricas customizadas
3. Dashboards em tempo real

### Tableau

1. Conectar fonte de dados CSV/JSON
2. Criar calculated fields para análises
3. Visualizações interativas

### Excel

1. Abrir CSV diretamente
2. Tabelas dinâmicas para agregações
3. Gráficos automáticos

### Python + Jupyter

1. Análises exploratórias
2. Machine learning sobre padrões
3. Relatórios automatizados

## 📝 Considerações Importantes

### Precisão dos Dados

- **Confiança**: Valores baixos podem indicar falsos positivos
- **Coordenadas**: Relativas ao frame original (1280x720 nos samples)
- **Sobreposição**: Múltiplas detecções podem ser da mesma pessoa

### Limitações

- **Oclusão**: Pessoas parcialmente visíveis podem não ser detectadas
- **Distância**: Pessoas muito pequenas podem ter baixa confiança
- **Iluminação**: Condições ruins afetam a detecção

### Boas Práticas

- **Filtrar por confiança**: Use threshold >= 0.5 para análises críticas
- **Validar manualmente**: Sempre revisar amostras dos dados
- **Considerar contexto**: Interpretar dados no contexto do ambiente filmado

---

Para dúvidas ou sugestões de análises adicionais, consulte a documentação técnica do sistema RIVAC-CV.
