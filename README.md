# RepairAgent com Modelos Locais e Grid Search

Este README descreve a adaptacao do RepairAgent para uso com modelos locais e a execucao de benchmarks via Grid Search. O foco do Grid Search e comparar o comportamento de diferentes modelos mantendo o restante da configuracao estavel.

Para a descricao original do projeto, arquitetura do agente e resultados publicados, consulte o [README original](README_original.md) e o artigo [RepairAgent: An Autonomous, LLM-Based Agent for Program Repair](https://arxiv.org/abs/2403.17134).

## Projeto Base

RepairAgent e um agente autonomo para reparo automatico de bugs em projetos Java. A estrutura original combina:

1. selecao de um bug do Defects4J;
2. entendimento do erro e coleta de contexto;
3. geracao de patch;
4. execucao de testes;
5. iteracao ate encontrar um patch plausivel ou atingir o limite configurado.

Esta versao preserva essa estrutura e muda principalmente o provedor/modelo usado pelo agente, permitindo executar modelos locais via Ollama.

O ponto de entrada do agente fica em:

```bash
cd repair_agent
python3 repairagent.py
```

Execucao direta com modelo local:

```bash
python3 repairagent.py run --bugs "Chart 1" --model ollama:gpt-oss:20b
```

## Modelos Locais

Modelos locais sao usados por meio do prefixo `ollama:` no nome do modelo. Antes de executar o RepairAgent ou o Grid Search, confirme que o Ollama esta ativo e que o modelo desejado esta disponivel:

```bash
ollama list
ollama pull gpt-oss:20b
```

Depois disso, informe o modelo ao RepairAgent:

```bash
python3 repairagent.py run --bugs "Chart 1" --model ollama:gpt-oss:20b
```

No Grid Search, os modelos ficam apenas dentro da chave `grid_search.models` de `repair_agent/config.yaml`. Essa lista e a variavel principal do benchmark: cada item representa um modelo a ser comparado contra o mesmo conjunto de bugs e parametros.

Parametros uteis para modelos locais:

| Parametro | Uso |
|---|---|
| `grid_search.models` | Lista de modelos avaliados no benchmark. |
| `grid_search.ollama.num_ctx` | Janela de contexto usada nas chamadas ao Ollama. |
| `grid_search.ollama.repeat_detect` | Controle de deteccao de repeticao, quando suportado. |
| `grid_search.ollama.stream_timeout_s` | Timeout para resposta do modelo local. |

## Como Configurar o Grid Search

O Grid Search e configurado em `repair_agent/config.yaml`, na chave `grid_search`.

Exemplo minimo:

```yaml
grid_search:
  models:
    - "ollama:gpt-oss:20b"

  overwrite: false

  run:
    all_bugs: false
    bugs_ids:
      - "Chart 1"
      - "Chart 2"
    max_cycles: 15
    extra_args: []

  ollama:
    num_ctx: 16384
    repeat_detect: false
    stream_timeout_s: 180

  env: {}
  output_dir: "grid_results_final"
```

Parametros principais:

| Parametro | Obrigatorio | Descricao |
|---|---:|---|
| `models` | Sim | Modelos locais avaliados. Use o formato `ollama:<nome-do-modelo>`. |
| `overwrite` | Nao | Quando `false`, resultados ja existentes sao preservados. |
| `run.all_bugs` | Sim | Quando `true`, usa `bugs_file` ou todos os bugs disponiveis; quando `false`, usa `bugs_ids`. |
| `run.bugs_file` | Nao | Arquivo com um bug por linha, no formato `Projeto Indice`. |
| `run.bugs_ids` | Nao | Lista explicita de bugs, usada quando `all_bugs: false`. |
| `run.max_cycles` | Nao | Limite de ciclos por bug. Use `0` para sem limite. |
| `run.extra_args` | Nao | Argumentos extras repassados para `repairagent.py run`. |
| `ollama.num_ctx` | Nao | Tamanho da janela de contexto para modelos Ollama. |
| `ollama.repeat_detect` | Nao | Habilita ou desabilita deteccao de repeticao. |
| `ollama.stream_timeout_s` | Nao | Tempo maximo para o stream do modelo responder. |
| `env` | Nao | Variaveis de ambiente adicionais para cada execucao. |
| `output_dir` | Nao | Pasta onde logs e sumarios do Grid Search serao gravados. |

Para executar:

```bash
cd repair_agent
python3 grid_search.py --config config.yaml
```

## Resultados

Cada combinacao `modelo x bug` gera uma pasta dentro de `output_dir`:

```text
grid_results_final/
  model_summary.csv
  model_summary.json
  summary.json
  ollama_gpt-oss_20b/
    Chart_1/
      run.log
      status.json
      experiment_44/
```

Arquivos principais:

| Arquivo | Conteudo |
|---|---|
| `run.log` | Saida completa da execucao daquele bug. |
| `status.json` | Resultado final da combinacao. |
| `summary.json` | Resumo da invocacao atual do Grid Search. |
| `model_summary.csv` | Tabela consolidada por modelo. |
| `model_summary.json` | A mesma consolidacao em JSON. |

Campos principais de `status.json`:

| Campo | Significado |
|---|---|
| `completed` | O processo terminou com codigo de saida `0`. |
| `success` | O RepairAgent encontrou patch plausivel ou testes zerados. |
| `interrupted` | A execucao foi interrompida. |
| `returncode` | Codigo de saida do processo. |
| `note` | Marcador usado para classificar o resultado. |
| `updated_at` | Horario em UTC da ultima atualizacao. |

## Checklist Rapido

Antes de rodar um benchmark:

1. entre em `repair_agent`;
2. confirme que o Ollama esta ativo;
3. confirme que os modelos existem em `ollama list`;
4. revise `config.yaml`;
5. deixe apenas os modelos que deseja comparar em `grid_search.models`;
6. defina `overwrite: false` para preservar resultados existentes;
7. rode `python3 grid_search.py --config config.yaml`;
8. acompanhe `model_summary.csv` e os `run.log`.
