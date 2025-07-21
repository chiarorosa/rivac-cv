# **Documento de Arquitetura de Software**

## **Projeto: Sistema de Monitoramento por Visão Computacional no Varejo**

### **1\. Objetivo**

Estabelecer uma arquitetura de software modular, extensível e didática para o desenvolvimento de sistemas de visão computacional voltados à contagem e análise de fluxo de pessoas em ambientes de varejo, priorizando facilidade de colaboração, experimentação com modelos de detecção, robustez e documentação.

---

### **2\. Visão Geral da Arquitetura**

O sistema está organizado em módulos independentes, favorecendo testes, manutenção e integração incremental. O pipeline principal abrange:

* Ingestão de vídeo

* Detecção de pessoas (com possibilidade de múltiplos modelos)

* Tracking (rastreamento entre frames)

* Gerenciamento de ROI (regiões de interesse)

* Contagem e extração de métricas

* Visualização e exportação de resultados

* Persistência e logging

---

### **3\. Diagrama Modular (Texto)**

\[Ingestão de Vídeo\]  
        ↓  
\[Pipeline de Processamento\]  
   ├─\> \[Detecção de Pessoas (hooks para modelos)\]  
   ├─\> \[Tracking (opcional/expansível)\]  
   ├─\> \[ROI & Contagem\]  
        ↓  
\[Exportação & Visualização\]  
        ↓  
\[Persistência de Dados & Logs\]

---

### **4\. Detalhamento dos Módulos**

#### **4.1 Ingestão de Dados de Vídeo**

* Suporta arquivos locais, câmeras IP/USB, RTSP/streams.

* Interface única para seleção da fonte.

* Encapsula dependências de leitura (OpenCV, etc).

#### **4.2 Detecção de Pessoas (Módulo Extensível)**

* **Interface Abstrata:** Define métodos essenciais (`load_model`, `predict`).

* **Implementações Específicas:** Cada modelo (YOLOv6, YOLOv8, outros futuros) implementa a interface.

* **Configuração Dinâmica:** Usuário define o modelo a usar em um arquivo `config.yaml` ou similar.

* **Plug & Play:** Facilita experimentação e substituição de modelos sem afetar o pipeline principal.

#### **4.3 Tracking (Rastreamento)**

* Módulo independente, plugável no pipeline.

* Possíveis técnicas: DeepSORT, filtros de Kalman, outros (OpenCV multi-object tracking).

* Inicialmente opcional, mas preparado para integração incremental.

#### **4.4 ROI e Contagem**

* Permite definição manual/automática de regiões de interesse nos frames.

* Algoritmo de contagem vinculado à ROI.

* Suporte para múltiplas ROIs, contagem total e por região.

* Interface gráfica básica para seleção/edição de ROIs.

#### **4.5 Exportação e Visualização**

* Visualização ao vivo dos resultados: boxes, ROIs, contagem, overlays.

* Exportação de métricas em CSV, JSON, ou outros formatos (compatível BI).

* Geração de relatórios de eventos, logs de detecção, estatísticas.

#### **4.6 Persistência Local**

* Logs, métricas e parâmetros salvos em banco leve (SQLite) ou arquivos simples (CSV, JSON).

* Histórico de execuções, configurações experimentais, métricas para análise posterior.

#### **4.7 Documentação e Testes**

* README para cada módulo, explicando uso, dependências e exemplos.

* Testes unitários para componentes-chave (ingestão, detecção, tracking).

* Scripts de setup e ambiente reprodutível (`requirements.txt`, `Dockerfile`).

---

### **5\. Estrutura de Diretórios Sugerida**

/src/  
  /ingestao/  
  /deteccao/  
    base\_detector.py  
    yolov6\_detector.py  
    yolov8\_detector.py  
    ...  
  /tracking/  
  /roi/  
  /visualizacao/  
  /exportacao/  
  /persistencia/  
  /utils/  
  /tests/  
  /docs/  
config.yaml  
main.py  
requirements.txt  
Dockerfile  
README.md

---

### **6\. Tecnologias Fundamentais**

* **Linguagem:** Python 3.10+

* **CV/ML:** PyTorch, OpenCV

* **Interface:** Streamlit/Gradio para protótipos rápidos (ou CLI no início)

* **Persistência:** SQLite (via SQLAlchemy) ou CSV/JSON puro

* **Exportação:** pandas, json, csv

* **Testes:** pytest

* **DevOps:** Docker, GitHub Actions

* **Documentação:** README, docstrings padrão Google/NumPy, preferencialmente com MkDocs ou Sphinx

---

### **7\. Padrões de Colaboração e Expansão**

* Cada módulo deve ter README próprio, exemplos de entrada/saída, e instruções de execução.

* Novo pesquisador pode criar uma branch e propor um novo detector plugando sua implementação à interface padrão.

* As issues devem ser usadas para bugs, sugestões e planejamento incremental.

* O onboarding de novos membros é facilitado por scripts de setup automático (`requirements.txt`, `Dockerfile`) e documentação central.

---

### **8\. Exemplos de Hooks para Detecção**

**base\_detector.py:**

class BaseDetector:  
    def load\_model(self): raise NotImplementedError  
    def predict(self, frame): raise NotImplementedError

**yolov6\_detector.py:**

from base\_detector import BaseDetector  
class YOLOv6Detector(BaseDetector):  
    def load\_model(self): ...  
    def predict(self, frame): ...

**No pipeline principal:**

def get\_detector(config):  
    if config\['model'\] \== 'yolov6':  
        return YOLOv6Detector()  
    elif config\['model'\] \== 'yolov8':  
        return YOLOv8Detector()  
    \# ...