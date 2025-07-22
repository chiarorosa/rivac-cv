# Documentação RIVAC-CV

Este diretório contém toda a documentação técnica do projeto RIVAC-CV.

## 📚 Documentos Disponíveis

### 🏗️ Arquitetura e Design

- **[Arquitetura de Software](software_architecture.md)** - Documentação completa da arquitetura do sistema
- **[Comparação de Pipelines](pipeline_comparison.md)** - Diferenças entre pipeline.py e pipeline_v2.py

### 📊 Análise de Dados

- **[Interpretação de Detecções](detection_analysis_guide.md)** - Como interpretar e analisar exportações de detecções

### ⚡ UV Package Manager

- **[Guia do UV](uv_guide.md)** - Manual completo do UV Package Manager
- **[CI/CD com UV](ci_cd_uv_examples.md)** - Exemplos de integração contínua com UV

## 🎯 Navegação Rápida

### Para Desenvolvedores

1. **Primeiro acesso**: Comece com o [README principal](../README.md)
2. **Entender a arquitetura**: Leia [Arquitetura de Software](software_architecture.md)
3. **Escolher pipeline**: Consulte [Comparação de Pipelines](pipeline_comparison.md)
4. **Setup do ambiente**: Use o [Guia do UV](uv_guide.md)

### Para Analistas de Dados

1. **Exportar detecções**: Use `--save-detections` no CLI
2. **Interpretar dados**: Consulte [Interpretação de Detecções](detection_analysis_guide.md)
3. **Análises avançadas**: Scripts prontos no documento de interpretação

### Para DevOps/CI-CD

1. **Integração contínua**: Consulte [CI/CD com UV](ci_cd_uv_examples.md)

## 🔗 Links Úteis

- **Repositório**: [https://github.com/chiarorosa/rivac-cv](https://github.com/chiarorosa/rivac-cv)
- **Issues**: [https://github.com/chiarorosa/rivac-cv/issues](https://github.com/chiarorosa/rivac-cv/issues)
- **UV Package Manager**: [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)

## 📝 Contribuindo com a Documentação

Para adicionar ou atualizar documentação:

1. Crie arquivos Markdown neste diretório (`docs/`)
2. Atualize este índice (`README.md`) com links para novos documentos
3. Mantenha a estrutura e formatação consistente
4. Use emojis para melhor organização visual

### Padrões de Documentação

- **Nomes de arquivo**: Use snake_case ou espaços para legibilidade
- **Estrutura**: Sempre inclua índice para documentos longos
- **Links**: Use caminhos relativos quando possível
- **Exemplos**: Inclua exemplos de código quando relevante

---

**Documentação mantida em**: `docs/`  
**Última atualização**: Julho 2025
