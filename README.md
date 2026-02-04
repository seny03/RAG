# CodeRAG Evaluation Project

Репозиторий с экспериментами по воспроизведению результатов статьи **CodeRAG** (EMNLP 2025): *"Finding Relevant and Necessary Knowledge for Retrieval-Augmented Repository-Level Code Completion"*.

Оригинальная имплементация: https://github.com/KDEGroup/CodeRAG

## О чем проект?

CodeRAG — это подход к дополнению кода на уровне репозитория с использованием RAG (Retrieval-Augmented Generation). Вместо того чтобы использовать только текущий файл, система извлекает релевантные фрагменты кода из всего репозитория для улучшения качества генерации.

Я воспроизвел результаты статьи и дополнительно протестировал подход retrieval на задаче локализации багов (SWE-bench). В репозитории:

1. Скрипты для запуска полного pipeline CodeRAG на бенчмарке CCEval
2. Эксперименты по локализации багов на SWE-bench Lite с использованием методов retrieval из CodeRAG
3. Сравнение sparse (TF-IDF), dense (CodeT5p) и гибридного retrieval

## Установка

### Требования

- Python 3.10+
- GPU с поддержкой CUDA (тестировал на A100-80GB и L4-24GB)
- Менеджер пакетов [uv](https://docs.astral.sh/uv/)

### Как развернуть

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# Клонировать репозиторий
git clone https://github.com/seny03/RAG.git
cd RAG

# Установить зависимости CodeRAG
cd CodeRAG
uv sync
source .venv/bin/activate
cd ..
```

## Эксперименты и результаты

### 1. Бенчмарк CCEval (кросс-файловое дополнение кода)

Запустил полный pipeline CodeRAG на CCEval с 128 сэмплами из 26 Python репозиториев.

**Конфигурация:**
- Модель: `Qwen/Qwen2.5-Coder-7B-Instruct` (через vLLM)
- Retrieval: Sparse (TF-IDF) + Dense (CodeT5p-110m-embedding)
- Reranker: Локальный reranker на основе likelihood

**Результаты:**

| Метрика | Наш результат | Статья (CodeRAG full) | Разница |
|---------|---------------|----------------------|---------|
| Edit Similarity (ES) | 59.1% | 67.8% | -8.7% |
| Exact Match (EM) | 19.5% | 24.1% | -4.6% |
| Identifier EM | 30.5% | - | - |
| Identifier F1 | 55.4% | - | - |

**Почему результаты отличаются?**

Разница ожидаема:
- Отключил dataflow retrieval (нет предпостроенного графа зависимостей) — в статье показано падение ~3-4% без него
- Использовал только 26 репозиториев вместо полного CCEval (многие репозитории стали приватными или удалены)

**Этапы pipeline:**
1. Build Query — CodeT5p-220m
2. Retrieve — Sparse + Dense retrieval
3. Rerank — Локальный reranker
4. Build Prompt — Объединение контекста
5. Inference — Qwen2.5-Coder-7B через vLLM
6. Evaluation — Вычисление метрик

### 2. Локализация багов на SWE-bench

Применил методы retrieval из CodeRAG к задаче локализации багов на SWE-bench Lite (300 сэмплов, 12 проектов).

**Задача:** По описанию бага найти релевантные файлы кода, которые нужно модифицировать.

**Сравниваемые методы:**
- **Sparse**: TF-IDF retrieval (baseline)
- **Dense**: CodeT5p-110m-embedding
- **Hybrid**: Sparse + Dense вместе

**Результаты:**

| Метрика | Sparse | Dense | Hybrid | Лучший |
|---------|--------|-------|--------|--------|
| Hit@1 | 6.3% | 10.3% | **11.0%** | +73.7% |
| Hit@5 | 27.0% | 37.3% | **39.7%** | +46.9% |
| Hit@10 | 34.3% | 46.7% | **49.0%** | +42.7% |
| Hit@20 | 47.0% | 54.0% | **58.0%** | +23.4% |

**Выводы:**
- Dense retrieval значительно превосходит sparse (TF-IDF) по всем метрикам
- Гибридный подход показывает лучшие результаты — сочетает лексическое сопоставление с семантическим пониманием
- Hit@20 = 58% означает, что для более половины багов мы находим нужный файл в топ-20 результатов

Это говорит о том, что retrieval в стиле CodeRAG может быть полезен для инструментов локализации багов.

## Как запустить

### Оценка на CCEval

```bash
# 1. Настроить пути и модель в CodeRAG/config/config.toml
# 2. Запустить vLLM сервер
./start_vllm.sh

# 3. Запустить pipeline (в другом терминале)
./run_pipeline.sh all
# Или отдельные шаги:
./run_pipeline.sh query
./run_pipeline.sh retrieve
./run_pipeline.sh rerank
./run_pipeline.sh prompt
./run_pipeline.sh inference
./run_pipeline.sh eval
```

### Локализация багов на SWE-bench

```bash
# Скачать датасет SWE-bench Lite
python -c "from datasets import load_dataset; ds = load_dataset('princeton-nlp/SWE-bench_Lite', split='test'); ds.to_parquet('swebench/swebench_lite_test.parquet')"

# Запустить sparse retrieval (baseline)
cd CodeRAG && source .venv/bin/activate && cd ..
python swebench_localization/localize_bug.py

# Запустить dense retrieval
python swebench_localization/localize_bug_dense.py

# Сравнить результаты
python swebench_localization/compare_results.py
```
