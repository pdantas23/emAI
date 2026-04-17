# emAI — Project Architecture Graph

**Version:** v0.1
**Generated:** 2026-04-16
**Source of truth:** approved scope (chat) + on-disk file inventory of `/Users/philip/Desktop/Philip/emAI/`

---

## How to read this graph

- **Nodes** = folders (boxes) and files (dots).
- **Edges** come in two flavors:
  - *Containment* (dashed, no arrow) — folder contains file.
  - *Dependency* (solid arrow `A → B`) — module A depends on / imports / consumes B.
- **Colors** encode implementation status:
  - 🟢 **GREEN** `#2ecc71` — Finalizado e validado (no disco, com conteúdo).
  - 🟡 **YELLOW** `#f1c40f` — Em desenvolvimento / draft.
  - ⚪ **GRAY** `#bdc3c7` — Planejado, ainda não criado.

Open `graph.html` in any browser — interactive, no server needed.

---

## Stats snapshot

| Métrica                  | Valor       |
|--------------------------|-------------|
| Pastas (módulos)         | 16          |
| Arquivos planejados      | 43          |
| 🟢 Finalizados           | **16**      |
| 🟡 Em desenvolvimento    | 0           |
| ⚪ Planejados            | 27          |
| Containment edges        | 43          |
| Dependency edges         | 47          |
| **Conclusão geral**      | **~37%**    |

---

## Status por arquivo

### 🟢 Finalizados (16)
- `pyproject.toml`
- `.env.example`
- `.gitignore`
- `README.md`
- `config/__init__.py`
- `config/settings.py` ← **núcleo de configuração com Pydantic Settings**
- `src/__init__.py`
- `src/core/__init__.py`
- `src/email_client/__init__.py`
- `src/filters/__init__.py`
- `src/ai/__init__.py`
- `src/messaging/__init__.py`
- `src/storage/__init__.py`
- `src/utils/__init__.py`
- `src/utils/logger.py` ← **loguru wrapper, console+file rotativo**
- `tests/__init__.py`

### 🟡 Em desenvolvimento (0)
_Nenhum no momento._

### ⚪ Planejados (27)
- **Core (2):** `src/main.py`, `src/core/orchestrator.py`, `src/core/exceptions.py`
- **Email (4):** `base.py`, `gmail_imap.py`, `outlook_imap.py`, `parser.py`
- **Filters (3):** `spam_filter.py`, `relevance_filter.py`, `rules.py`
- **AI (3):** `llm_client.py`, `summarizer.py`, `classifier.py`
- **Messaging (4):** `base.py`, `whatsapp_twilio.py`, `whatsapp_evolution.py`, `formatter.py`
- **Storage (2):** `state.py`, `models.py`
- **Utils (1):** `retry.py`
- **Config externo (2):** `filter_rules.yaml`, `senders_whitelist.yaml`
- **Prompts (3):** `summarize_executive.md`, `classify_priority.md`, `extract_actions.md`
- **Outros (2):** `.env`, `docker-compose.yml`

---

## God nodes (alto fan-in / fan-out)

São os módulos com maior centralidade no grafo — quem mexer aqui afeta tudo.

| Nó                          | Fan-in | Fan-out | Porquê é crítico                                 |
|-----------------------------|--------|---------|--------------------------------------------------|
| `config/settings.py` 🟢     | **11** | 0       | Toda I/O config passa por aqui (DI invertida)    |
| `src/utils/logger.py` 🟢    | **11** | 0       | Todo logging estruturado depende dele            |
| `src/core/orchestrator.py` ⚪| 0      | **10**  | Coração do pipeline (fetch→filter→AI→send)       |
| `src/email_client/base.py` ⚪| 3      | 0       | Interface ABC — todo provider IMAP implementa    |
| `src/messaging/base.py` ⚪  | 3      | 0       | Interface ABC — todo provider WhatsApp implementa|
| `src/ai/llm_client.py` ⚪   | 2      | 0       | Wrapper unificado Anthropic/OpenAI               |

**Observação:** Os 2 god nodes finalizados (`settings.py`, `logger.py`) são exatamente os que **deviam** ser feitos primeiro — base estável para os 41 arquivos que dependem deles.

---

## Surprising connections (cross-module)

Conexões que cruzam fronteiras de módulo e merecem atenção arquitetural:

1. **`src/filters/relevance_filter.py` → `config/senders_whitelist.yaml`**
   Filtros leem dados de configuração externa em YAML, não código. Permite o cliente editar a whitelist sem deploy.

2. **`src/filters/rules.py` → `config/filter_rules.yaml`**
   Mesma ideia: regras heurísticas como dado, não código → menos churn no Python.

3. **`src/ai/{summarizer,classifier}.py` → `prompts/*.md`**
   Prompts ficam **versionados como Markdown**, fora do código Python. Facilita iteração de prompt engineering sem PR de código.

4. **`src/core/orchestrator.py` é o único que toca todos os 6 módulos de domínio.**
   Validação do princípio de Dependency Inversion: orchestrator depende de **interfaces** (`base.py`), não de implementações concretas.

---

## Próximos passos imediatos (para mover GRAY → GREEN)

Sequência sugerida — segue o pipeline de dados (ingest → process → deliver):

| # | Arquivo                              | Por quê agora                                              |
|---|--------------------------------------|------------------------------------------------------------|
| 1 | `src/email_client/base.py` ⚪→🟡    | Interface ABC habilita Gmail+Outlook lado-a-lado           |
| 2 | `src/email_client/parser.py` ⚪→🟡  | HTML→texto + extração de metadados (zero deps externas)    |
| 3 | `src/email_client/gmail_imap.py` ⚪→🟡 | MVP de ingestão real com `imap-tools`, só unread        |
| 4 | `tests/integration/test_imap.py`     | Validar com email mock (.eml)                              |
| 5 | `src/ai/llm_client.py`               | Wrapper para Anthropic/OpenAI                              |
| 6 | `src/ai/summarizer.py` + prompt      | Primeira chamada à LLM                                     |

**Esta entrega cobre os passos 1–3** (módulo `email_client/` completo), conforme alinhado.

---

## Honesty / audit trail

- ✅ **Containment edges:** EXTRACTED — derivados diretamente da estrutura de pastas no escopo.
- ✅ **Status GREEN:** EXTRACTED — verificado por `find` no diretório real.
- ⚠️ **Dependency edges:** INFERRED — derivados do escopo aprovado e dos princípios SOLID descritos. Confidence ~0.9 (são intenções de design, não código existente). Serão validadas/atualizadas conforme cada arquivo for de fato criado.
- ⚠️ **`.env`:** marcado como GRAY porque não deve ser commitado. Esperado que o usuário crie a partir de `.env.example` no setup local — não é "atrasado", é por design.

Edges marcadas como INFERRED serão promovidas a EXTRACTED quando o `import` correspondente existir no código.

---

## Arquivos gerados nesta entrega

```
docs/graph/
├── graph.html        ← visualização interativa (vis-network, abre no browser)
├── graph.json        ← dados estruturados (fonte de verdade do grafo)
└── GRAPH_REPORT.md   ← este documento
```

Para regerar/atualizar o grafo após cada novo módulo, basta editar `graph.json` (mudar `status` de GRAY → GREEN nos nós correspondentes) e recarregar `graph.html`.
