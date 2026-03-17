# 🤖 Dota 2 RAG Agent

Практический пет-проект: построение RAG-системы (Retrieval-Augmented Generation) для ответов на вопросы о Dota 2 — от простого семантического поиска до агента с маршрутизацией на LangGraph.

Проект создавался итерационно, и в репозитории намеренно сохранены **обе версии системы**, чтобы можно было сравнить архитектурные подходы.

---

## Эволюция системы

### v1 — Монолитный RAG (`rag_pipeline.ipynb`)

```
Запрос → [BGE-M3 embed] → [ChromaDB: поиск по всем 2200+ документам] → [BGE Reranker] → [Llama 3.2-3B]
```

Простой линейный пайплайн. Поиск идёт по **всем коллекциям сразу** без фильтрации. Нет понятия «маршрутизации» — каждый запрос обрабатывается одинаково.

**Проблема:** запрос «способности Anti-Mage» ищет среди всех 2200+ документов (герои + предметы + патчи всех версий), хотя нужны только ~10 документов конкретного героя.

### v2 — Агент с маршрутизацией (`langgraph_node_agent.ipynb`)

```
Запрос → [router: Llama извлекает entity_name, entity_type] → [ChromaDB: фильтр where по метаданным]
       → [BGE Reranker] → [Llama 3.2-3B генерирует ответ]
```

Добавлен **узел-маршрутизатор**: Llama анализирует запрос и строит план — какие коллекции искать и с каким фильтром метаданных. Тот же запрос об Anti-Mage теперь работает с фильтром `{"hero_name": "Anti-Mage"}` → достаётся ровно ~10 нужных документов.

Пайплайн реализован как **LangGraph-граф** с явным состоянием (`AgentState`), что делает каждый шаг изолированным и легко расширяемым.

| | v1 `rag_pipeline` | v2 `langgraph_node_agent` |
|---|---|---|
| Поиск | По всем коллекциям | Целевой: по конкретному герою/предмету |
| Фильтрация | Нет | `where` по метаданным (hero_name, entity_type) |
| Маршрутизация | Нет | Llama извлекает сущность из запроса |
| Архитектура | Функции + прямой вызов | LangGraph StateGraph |
| Скорость ответа | ~30–60 сек | ~30–60 сек |

### v3 — Tools-агент (в планах)

Следующий шаг: вместо жёсткой схемы `router → search → rerank → generate` дать Llama набор **инструментов** (search_heroes, search_items, search_patches) и позволить ей самостоятельно решать, какие инструменты вызвать и сколько раз. Это позволит агенту, например, сначала найти героя, потом уточнить по патчам — без прописанного вручную маршрута.

---

## Требования к железу

> ⚠️ Проект рассчитан на **NVIDIA GPU с ≥ 8 ГБ VRAM**.

Три модели одновременно занимают ~9 ГБ VRAM:

| Модель | VRAM |
|---|---|
| BGE-M3 (эмбеддинги) | ~2.3 ГБ |
| BGE Reranker v2-m3 | ~0.6 ГБ |
| Llama 3.2-3B-Instruct (bfloat16) | ~6 ГБ |

Проверено на **RTX 5080 (15.9 ГБ)**. На GPU с 8 ГБ должно работать, но может потребоваться выгрузить BGE-M3 перед загрузкой Llama.

---

## Быстрый старт

### 1. Клонируй репозиторий

```bash
git clone https://github.com/your-username/dota2-rag-agent.git
cd dota2-rag-agent
```

### 2. Создай виртуальное окружение

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 3. Установи зависимости

```bash
pip install -r requirements.txt
```

**Для PyTorch с CUDA** — установи отдельно по инструкции с [pytorch.org](https://pytorch.org/get-started/locally/):
```bash
# Пример для CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 4. Настрой HuggingFace токен

Llama 3.2-3B-Instruct — закрытая модель, требует принятия лицензии на [huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).

```bash
cp .env.example .env
# Отредактируй .env и вставь свой токен:
# HF_TOKEN=hf_...
```

Если модель уже скачана локально в `models/llama3.2-3b/` — токен не нужен.

### 5. Запусти ноутбук

Открой `rag_pipeline.ipynb` или `langgraph_node_agent.ipynb` и выполняй ячейки **сверху вниз**.

При первом запуске автоматически скачаются:
- BGE-M3 (~2.3 ГБ) → `models/bge-m3/`
- BGE Reranker v2-m3 (~570 МБ) → `models/bge-reranker-v2-m3/`
- Llama 3.2-3B-Instruct (~6 ГБ в bfloat16) — только если нет в `models/llama3.2-3b/`

---

## Структура репозитория

```
├── rag_pipeline.ipynb           # v1: монолитный RAG-пайплайн
├── langgraph_node_agent.ipynb   # v2: агент с маршрутизацией (LangGraph)
│
├── data/
│   ├── clean/                   # готовые JSONL-корпусы для индексации
│   │   ├── heroes_corpus.jsonl      # ~1268 документов: способности, таланты, лор
│   │   ├── items_corpus.jsonl       # ~400+ документов: описания предметов
│   │   └── patches_corpus.jsonl     # история патчей 7.40–7.40c
│   └── examples/                # примеры сырых данных (для ознакомления)
│       ├── hero_antimage.json           # структура hero JSON из Valve API
│       ├── item_blades_of_attack.json   # структура item JSON
│       └── patch_7.40c_excerpt.json     # структура patch JSON
│
├── requirements.txt
├── .env.example                 # шаблон переменных окружения
└── README.md
```

---

## Данные

### Готовые корпусы (`data/clean/`)

Файлы JSONL: одна строка = один документ. Каждый документ содержит поля `text` (текст для эмбеддинга) и `metadata` (поля для фильтрации в ChromaDB).

**Схема метаданных:**

| Коллекция ChromaDB | Поля метаданных |
|---|---|
| `dota2_heroes` | `hero_name`, `hero_id`, `language`; способности: + `ability_id`, `ability_name`; таланты: + `talents_count`; лор: + `chunk_id` |
| `dota2_items` | `item_name`, `item_id`, `slug`, `category`, `cost`, `neutral_tier`, `language` |
| `dota2_patches` | `patch_version`, `entity_type` (`"hero"` / `"item"` / `"general"`), `language`; hero-записи: + `hero_name`, `hero_id` |

### Сырые данные

Получены из **Valve OpenDota API** — официального дата-фида Dota 2. Примеры структуры JSON-файлов до предобработки — в `data/examples/`.

Предобработка (парсинг, нормализация, чанкинг) намеренно не включена в репозиторий — корпусы в `data/clean/` уже готовы к использованию.

---

## Модели

| Модель | Назначение | Ссылка |
|---|---|---|
| `BAAI/bge-m3` | Эмбеддинги (индексация и поиск) | [HuggingFace](https://huggingface.co/BAAI/bge-m3) |
| `BAAI/bge-reranker-v2-m3` | Cross-encoder реранкинг | [HuggingFace](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| `meta-llama/Llama-3.2-3B-Instruct` | Маршрутизатор + генерация ответов | [HuggingFace](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) *(требует принятия лицензии)* |

---

## Лицензия

Код проекта: MIT. Данные Dota 2 принадлежат Valve Corporation.
