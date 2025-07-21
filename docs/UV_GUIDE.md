# RIVAC-CV - Guia de Instalação e U```bash

# Clone o repositório

git clone https://github.com/chiarorosa/rivac-cv.git
cd rivac-cvom UV

## 🚀 Instalação do UV

O UV é um gerenciador de pacotes Python extremamente rápido, escrito em Rust. Ele substitui ferramentas como pip, pip-tools, pipx e poetry com performance 10-100x mais rápida.

### Instalação do UV

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Via pip (se já tiver Python)
pip install uv

# Via homebrew (macOS)
brew install uv

# Via cargo (se tiver Rust)
cargo install --git https://github.com/astral-sh/uv uv
```

### Verificar Instalação

```bash
uv --version
```

## 📦 Configuração do Projeto RIVAC-CV

### 1. Clonagem e Configuração Inicial

```bash
# Clonar repositório
git clone https://github.com/chiarorosa/rivac-cv.git
cd rivac-cv

# Verificar se o UV detecta o projeto
uv sync --dry-run
```

### 2. Configuração do Ambiente Python

```bash
# Instalar Python específico (recomendado: 3.11+)
uv python install 3.11

# Criar ambiente virtual com Python específico
uv venv --python 3.11
```

### 3. Instalação de Dependências

```bash
# Instalar apenas dependências principais
uv sync

# Instalar com dependências de desenvolvimento
uv sync --group dev

# Instalar com funcionalidades avançadas de CV
uv sync --extra advanced-cv

# Instalar tudo (recomendado para desenvolvimento)
uv sync --all-extras
```

### 4. Ativação do Ambiente

```bash
# Ativar ambiente virtual (opcional, uv run faz automaticamente)
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows

# Verificar instalação
uv run python --version
uv run pip list
```

## 🔧 Comandos Essenciais do UV

### Gerenciamento de Dependências

```bash
# Adicionar nova dependência
uv add opencv-python

# Adicionar dependência de desenvolvimento
uv add --group dev pytest

# Adicionar dependência opcional
uv add --optional advanced-cv scikit-image

# Remover dependência
uv remove opencv-python

# Atualizar dependências
uv sync --upgrade

# Bloquear versões específicas
uv lock
```

### Execução de Comandos

```bash
# Executar script principal
uv run python main.py --input 0

# Executar interface Streamlit
uv run streamlit run streamlit_app.py

# Executar testes
uv run pytest

# Executar formatação de código
uv run black src/

# Executar linting
uv run ruff check src/
```

### Gerenciamento de Python

```bash
# Listar versões Python disponíveis
uv python list

# Instalar versão específica
uv python install 3.12

# Encontrar Python por requisito
uv python find ">=3.11"

# Usar Python específico para o projeto
uv python pin 3.11
```

## 📁 Estrutura de Configuração

### pyproject.toml

Arquivo principal de configuração do projeto:

```toml
[project]
name = "rivac-cv"
version = "0.1.0"
requires-python = ">=3.8"
dependencies = [
    "torch>=2.0.0",
    "ultralytics>=8.0.0",
    # ... outras dependências
]

[project.optional-dependencies]
dev = ["pytest", "black", "flake8"]
advanced-cv = ["opencv-contrib-python", "scikit-image"]

[tool.uv]
managed = true
package = false  # Projeto tipo aplicação
```

### uv.toml

Configurações globais do UV:

```toml
[pip]
link-mode = "symlink"
strict = true

[python]
downloads = "automatic"
preference = "managed"
```

## 🎯 Workflows de Desenvolvimento

### Setup Inicial do Desenvolvedor

```bash
# 1. Clonar e entrar no projeto
git clone https://github.com/chiarorosa/rivac-cv.git
cd rivac-cv

# 2. Instalar UV se necessário
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Sincronizar ambiente completo
uv sync --all-extras

# 4. Verificar instalação
uv run python -c "import cv2, torch, ultralytics; print('✅ Tudo funcionando!')"

# 5. Executar testes
uv run pytest
```

### Desenvolvimento Diário

```bash
# Atualizar dependências
uv sync

# Executar aplicação
uv run streamlit run streamlit_app.py

# Testes durante desenvolvimento
uv run pytest tests/ -v

# Formatação e linting
uv run black src/
uv run ruff check src/
```

### Adicionando Nova Funcionalidade

```bash
# Adicionar nova dependência
uv add scikit-learn

# Atualizar lock file
uv lock

# Verificar se funciona
uv run python -c "import sklearn; print('✅ sklearn instalado')"

# Commitar mudanças
git add pyproject.toml uv.lock
git commit -m "feat: adicionar scikit-learn para ML"
```

## 🚀 Performance e Otimizações

### Cache do UV

```bash
# Verificar cache
uv cache info

# Limpar cache se necessário
uv cache clean

# Cache por projeto (configurado em uv.toml)
cache-dir = ".uv-cache"
```

### Links Simbólicos vs Cópia

```bash
# Configurar link mode (mais rápido)
uv pip install --link-mode symlink

# Para projetos que modificam arquivos instalados
uv pip install --link-mode copy
```

### Resolução de Dependências

```bash
# Resolução mais rápida (usar cache)
uv sync --frozen

# Resolução completa (verificar mudanças)
uv sync --upgrade

# Resolução offline (usar apenas cache)
uv sync --offline
```

## 🔧 Troubleshooting

### Problemas Comuns

#### 1. UV não encontrado

```bash
# Verificar PATH
echo $PATH

# Reinstalar UV
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # ou ~/.zshrc
```

#### 2. Erro de dependências conflitantes

```bash
# Resolver dependências do zero
uv lock --upgrade

# Verificar árvore de dependências
uv tree

# Resolver conflitos específicos
uv sync --resolution=lowest-direct
```

#### 3. Problemas com CUDA/PyTorch

```bash
# Instalar PyTorch para CPU
uv add torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Para CUDA específica
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 4. Erro de build de pacotes

```bash
# Instalar ferramentas de build
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# macOS
xcode-select --install

# Windows
# Instalar Visual Studio Build Tools
```

### Logs e Debug

```bash
# Modo verbose
uv sync -v

# Debug completo
uv sync -vv

# Logs de resolução
uv lock --verbose
```

## 📊 Comparação de Performance

| Operação           | pip | UV   | Melhoria         |
| ------------------ | --- | ---- | ---------------- |
| Instalação inicial | 45s | 3s   | 15x mais rápido  |
| Reinstalação       | 30s | 0.5s | 60x mais rápido  |
| Resolução          | 20s | 0.2s | 100x mais rápido |
| Lock file          | 15s | 0.1s | 150x mais rápido |

## 🌟 Recursos Avançados

### Scripts Personalizados

```toml
[project.scripts]
rivac-cv = "main:main"
rivac-streamlit = "streamlit_app:main"
rivac-test = "pytest:main"
```

```bash
# Executar scripts configurados
uv run rivac-cv --input video.mp4
uv run rivac-streamlit
uv run rivac-test
```

### Workspaces (para projetos multi-pacote)

```toml
[tool.uv.workspace]
members = [
    "packages/core",
    "packages/ui",
    "packages/api"
]
```

### Índices Personalizados

```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cpu"

[tool.uv.sources]
torch = { index = "pytorch" }
```

## 📚 Recursos Adicionais

- [Documentação oficial do UV](https://docs.astral.sh/uv/)
- [Guia de migração pip → UV](https://docs.astral.sh/uv/guides/migration/)
- [GitHub do projeto UV](https://github.com/astral-sh/uv)
- [Discussões da comunidade](https://github.com/astral-sh/uv/discussions)

---

🚀 **Com UV, o RIVAC-CV fica muito mais rápido para instalar e desenvolver!**
