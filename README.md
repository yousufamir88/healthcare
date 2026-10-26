# Practo Domain Support Agent

This repository is the Practo Healthcare track capstone. It is a support agent for a fictional clinic: it answers policy questions and looks up synthetic appointments. It does not use real patient data.

The agent retrieves fictional clinic policies, looks up synthetic appointments, remembers a session, checks phone numbers and prompt injection, and serves the result over HTTP and WebSocket. A two-agent review publishes the final answer. Each request is labeled for risk, checked against a local token budget, and cached when the same question is repeated. Graded runs stay in `MOCK_LLM` mode.

Graded demonstrations run in `MOCK_LLM` mode. They do not need an API key, a credit card, or a paid model.

## What Phase 0 gives you

- A Python 3.12 virtual environment managed by [uv](https://docs.astral.sh/uv/).
- Pinned dependencies in `pyproject.toml`, `uv.lock`, and `requirements.txt`.
- One configuration module, `app/config.py`, that turns on `MOCK_LLM` and turns off CrewAI telemetry.
- A check you can run before any later phase: `python -m app.config`.

## Requirements

- macOS or another system with Python 3.12.
- `uv` installed (`uv --version`).
- No API keys.

This machine was checked as macOS 26.5.1 with Python 3.12.13 available at `~/.local/bin/python3.12`. An older `.venv` on Python 3.11 did not match `requires-python` and did not contain CrewAI, FastAPI, or LangChain. Phase 0 recreates `.venv` with Python 3.12.

## Install

From the repository root:

```bash
uv venv --python 3.12 --clear
uv sync
source .venv/bin/activate
python -m app.config
```

`uv sync` installs the locked dependencies, including the development test tools. The first sync downloads PyTorch, SentenceTransformers, ChromaDB, CrewAI, AutoGen, FastAPI, and LangChain. That step needs network access once. After it finishes, the Phase 0 check itself does not call the network and does not read an API key.

You can also install from the exported lock file:

```bash
python -m pip install -r requirements.txt
```

Prefer `uv sync`. `requirements.txt` is the same locked set, exported so the environment can be recreated with pip.

## Configuration

`app/config.py` reads an optional `.env` file, then fills anything still missing:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MOCK_LLM` | `true` | Use the local stand-in model. No API key. |
| `CREWAI_DISABLE_TELEMETRY` | `true` | Tell CrewAI not to send usage telemetry. |
| `OTEL_SDK_DISABLED` | `true` | Turn off the OpenTelemetry SDK before CrewAI starts. |

Copy `.env.example` to `.env` only when you want to override a default. `.env` is gitignored. Do not put secrets in the repository.

If `MOCK_LLM` is `false`, startup stops with a configuration error. Values other than true/false words (`true`, `false`, `yes`, `no`, `on`, `off`, `1`, `0`) are also rejected.

## Verify

```bash
source .venv/bin/activate
python -m app.config
pytest tests/test_config.py -q
```

A correct configuration printout looks like this:

```text
Practo support agent foundation
project_name: practo-domain-support-agent
mock_llm: true
api_key_required: false
crewai_disable_telemetry: true
otel_sdk_disabled: true
knowledge_base_dir: /absolute/path/to/healthcare/knowledge_base
status: ready
```

On 2026-10-16, `pytest tests/test_config.py tests/test_dataset.py -q` reported `27 passed`.

## Synthetic appointment dataset

`app/data/dataset.py` builds fictional appointments. There are no real patients. Running the module validates the records and prints the measured counts:

```bash
python -m app.data.dataset
```

Generation settings:

| Setting | Value | Why |
| --- | --- | --- |
| Seed | `42` | A fixed seed makes the same list every time the program runs. |
| Records | `100` | Above the required minimum of 40, so later demos have several examples of each specialty. |
| Category weights | `20, 20, 20, 20, 20` | Each specialty starts with one coverage record, then has an equal chance. The weights follow enum order: General Medicine, Cardiology, Dermatology, Pediatrics, Orthopedics. |
| Status weights | `35, 30, 10, 10, 15` | Scheduled and completed visits are common. Cancelled and no-show are less common. Rescheduled sits between them. The weights follow enum order: Scheduled, Completed, Cancelled, No-Show, Rescheduled. |
| Fee range | `300` to `2000` INR | Fictional demo prices, not live Practo fees. The low end stands in for a short general consult. The high end stands in for a specialist consult. |
| Days since created | `0` to `30` | The capstone window is the last month. |
| Follow-up probability | `0.20` | Each record draws a follow-up flag with a 20% chance, inside the required 10%–30% band. |

Measured result for seed 42, from `python -m app.data.dataset` on 2026-10-16:

- General Medicine 20, Cardiology 24, Dermatology 15, Pediatrics 15, Orthopedics 26. Every category has at least 3 records.
- Scheduled 38, Completed 35, Cancelled 10, No-Show 5, Rescheduled 12. Every status appears.
- Follow-up required: 21 of 100 records, which is 21.00%.

Invalid settings raise `DatasetValidationError` before records are built. Examples are a record count below 40, a weight list of the wrong length, a day range outside 0–30, or a follow-up probability outside 0.10–0.30. A generated list that breaks a rule raises the same error with the rule that failed.

## Knowledge base

`knowledge_base/` holds 12 fictional clinic policies. They are original demonstration text, not real Practo rules and not real patient records. `app/rag/knowledge_base.py` loads them and checks the set:

```bash
python -m app.rag.knowledge_base
```

Each file has YAML frontmatter with `doc_id`, `topic`, `title`, and `source`. The source id is the file name, such as `appointment-booking.md`. The loader returns a `PolicyDocument` for each file, including that source id.

A sentence is counted by splitting on `.`, `!`, or `?` followed by whitespace. Each policy must contain 2 to 5 sentences. On 2026-10-16 every document measured 4 sentences.

| File | Topic |
| --- | --- |
| `appointment-booking.md` | Appointment booking |
| `cancellation-rescheduling.md` | Cancellation and rescheduling |
| `consultation-fees.md` | Consultation fees by specialty |
| `insurance-claims.md` | Insurance claims |
| `prescription-refills.md` | Prescription refills |
| `lab-turnaround.md` | Laboratory turnaround times |
| `telemedicine.md` | Telemedicine eligibility |
| `emergency-protocol.md` | Emergency-visit protocol |
| `patient-privacy.md` | Patient-data privacy |
| `follow-up-discount.md` | Follow-up discounts |
| `second-opinion.md` | Second opinions |
| `home-visit.md` | Home-visit eligibility |

Medical safety and privacy stay explicit. The emergency policy tells the reader to call local emergency services and says this assistant cannot diagnose. The privacy policy says the agent must not disclose one patient's information to another, and that every example is fabricated. Prescription refills state that controlled medicines are never refilled through this channel. Lab results state that the channel does not interpret the result. Telemedicine and second opinions both say they are not for emergencies.

A broken folder raises `KnowledgeBaseValidationError` and names the problem. Examples are a missing topic, a one-sentence policy, a duplicate `doc_id`, a missing `source`, or a privacy policy that drops the non-disclosure sentence.

## Retrieval

Phase 3 searches the policy folder locally. It does not call a paid model and it does not need an API key. The embedding model is the free SentenceTransformers model `all-MiniLM-L6-v2`. The first use can download that model into the local Hugging Face cache. The measurement below was run with the cache already present and with network access turned off.

```bash
python -m app.rag.retrieval
```

That command builds two ChromaDB collections with `collection.upsert()`, searches them, and writes the measured reports in `evaluation/`. The database directory `chroma_db/` is gitignored.

Chunking settings:

| Setting | Value |
| --- | --- |
| Fixed-size chunk length | 100 characters |
| Fixed-size overlap | 20 characters |
| Sentence chunking | One chunk per sentence |
| Top-k | 3 |
| Collections written on 2026-10-16 | `fixed_size_overlap`: 77 chunks, `sentence`: 48 chunks |

Each stored chunk keeps `doc_id`, `topic`, `title`, `source`, `strategy`, and `position`.

Cosine similarity is the dot product of the normalized query vector and the normalized chunk vector. A probe on this Chroma build stored three unit vectors and measured cosine distance: identical vectors `0.0`, orthogonal vectors `1.0`, opposite vectors `2.0`. That matches distance = 1 - cosine similarity.

### Similarity threshold

The threshold was calculated from the sentence collection, because that collection won the comparison below. These are the measured top-1 cosine similarities:

| Query | In scope | Top-1 similarity | Top document |
| --- | --- | --- | --- |
| How do I book a clinic appointment? | yes | 0.810317 | appointment-booking |
| What is the cancellation and rescheduling policy? | yes | 0.557508 | cancellation-rescheduling |
| What are the consultation fees by specialty? | yes | 0.683880 | follow-up-discount |
| How does the insurance claim process work? | yes | 0.643659 | insurance-claims |
| What is the follow-up discount? | yes | 0.562061 | follow-up-discount |
| Who is eligible for a telemedicine video visit? | yes | 0.727311 | telemedicine |
| What is the cardiology consultation fee and can a first cardiology visit be done by video? | yes | 0.730918 | follow-up-discount |
| Who will win the next cricket world cup? | no | 0.115789 | insurance-claims |
| What is the capital of France? | no | 0.061638 | consultation-fees |

The lowest in-scope score is 0.557508. The highest out-of-scope score is 0.115789. Every in-scope score is above every out-of-scope score, so the threshold is the midpoint:

`(0.557508 + 0.115789) / 2 = 0.336648`

The exact stored value is `0.33664840576753746` in `evaluation/retrieval_threshold_report.json`. A chunk is used only when its cosine similarity is greater than or equal to that threshold. Otherwise the answer is:

`I don't have enough verified information in the available support knowledge base to answer that question.`

The cricket question triggered that fallback. Its best chunk scored 0.115789.

The answer text is the retrieved chunk text. The generator does not add sentences of its own. Because the midpoint sits below the in-scope cluster, a neighbor chunk can also clear 0.336648. The booking answer therefore quotes the booking sentence and two other chunks that scored 0.738714 and 0.572507.

### Chunking comparison

Document-level precision and recall use the top 3 chunks. Chunks from the same policy count once. The per-query fractions, including `1 / 3 = 0.3333333333333333` and `0 / 1 = 0.0`, are in `evaluation/retrieval_comparison.md`. Macro scores from that run:

| Strategy | Macro precision | Macro recall | Macro F1 |
| --- | --- | --- | --- |
| fixed_size_overlap | 0.6667 | 0.7857 | 0.7213 |
| sentence | 0.6667 | 0.9286 | 0.7761 |

Sentence chunking is the one to use. Precision is tied at 0.6667. Sentence recall is higher because the fixed-size index missed `consultation-fees` for "What are the consultation fees by specialty?" (`0 / 3` precision and `0 / 1` recall). Sentence retrieval found that document (`1 / 3` precision and `1 / 1` recall). F1 is the harmonic mean of the macro precision and macro recall.

## Appointment lookup

`check_appointment_status(record_id)` reads one fictional appointment. It returns the status, the consultation fee in INR, and an escalation score from 0 to 1.

```bash
python -m app.tools.appointment_tools
```

The score uses two signals:

```text
normalized_recency = (30 - days_since_created) / 30
follow_up_signal = 1 when follow_up_required is true, otherwise 0
escalation_score = follow_up_signal * (1 - normalized_recency)
```

`normalized_recency` is 1 when the appointment was created today and 0 when it is 30 days old. Multiplying by the follow-up signal means a visit with no follow-up scores 0. A follow-up scores higher as it gets older, because its recency has fallen.

The cutoff is the median score among the follow-up records in the seed-42 dataset. On 2026-10-16 that run reported:

- 100 records, 21 of them with follow-up required
- Follow-up scores from 0.033333 (1/30) to 0.966667 (29/30)
- Median threshold 0.466667, which is 14/30
- 11 records score at or above that threshold

Those 11 records are the follow-ups that are at least 14 days old. A record with no follow-up stays below the threshold even at day 30, because its score is 0. `APPT-1001` is Completed, fee 1883 INR, 8 days old, no follow-up, score 0. `APPT-9999` is not in the dataset. The lookup returns `found: false` and does not invent a status or a fee.

## Support crew

Phase 5 runs three CrewAI agents in order. Every agent uses `MockLLM`, a local subclass of `crewai.llms.base_llm.BaseLLM`. The crew calls `.kickoff()` and does not need an API key.

| Agent | Tool |
| --- | --- |
| Retrieval Agent | `search_policies` |
| Lookup Agent | `check_appointment_status` |
| Response Composer | none |

The lookup tool is attached only to the Lookup Agent. `MockLLM.supports_function_calling()` is true, so CrewAI uses native tool calls. The model reads each tool's argument schema: a `record_id` property is filled from an `APPT-` id in the question, and a `query` property receives the question text. Mentioning a tool name in the prompt does not select a tool whose schema lacks those fields.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m app.crew.support_crew
```

On 2026-10-16 that command showed:

- "How do I book a clinic appointment?" called `search_policies` and did not call the appointment tool. The answer quoted the booking sentence and said no record id was provided.
- "What is the status of appointment APPT-1001?" called `search_policies` and `check_appointment_status` with `APPT-1001`. The answer included status Completed, fee 1883 INR, and escalation score 0.0.

The appointment question also returns policy sentences because the retrieval agent always searches. Those sentences are the chunks that cleared the measured threshold. The status and fee come from the lookup result.

## Session memory

Phase 6 stores each conversation in its own LangChain `InMemoryChatMessageHistory`. `RunnableWithMessageHistory` loads that history before the crew runs and saves the turn afterward. The session id is the key. Two ids do not share messages.

A follow-up such as "that appointment" reuses the latest `APPT-` id from the same session. A new session id starts with an empty history, so the same follow-up does not receive the earlier id.

Every reply is a `SupportResponse`:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `answer` | string | required | Composer text. It must be non-empty. |
| `intent` | `policy`, `appointment`, `mixed`, or `unknown` | required | Which tools produced the reply. |
| `sources` | list of strings | `[]` | Policy files whose chunks cleared the threshold. |
| `appointment_id` | string or empty | `null` | Set only after a successful lookup. Must look like `APPT-1001`. |
| `escalation_score` | number or empty | `null` | Set only after a successful lookup. Must be from 0 to 1. |
| `requires_human_review` | boolean | `false` | True when evidence is missing or the score is at or above the cutoff. |

A payload that breaks those rules is replaced by the retrieval fallback sentence, intent `unknown`, and `requires_human_review` true.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m app.session.memory
```

On 2026-10-16 that command showed:

- Session `clinic-session-a`, first turn: appointment id `APPT-1001`, score `0.0`, status Completed, fee 1883 INR. Two messages stored.
- Same session, "What is the consultation fee for that appointment?": the id was reused. Four messages stored. Fee 1883 INR again.
- Session `clinic-session-b`, the same follow-up: appointment id empty, score empty, and the answer says the lookup tool was not called.

LangChain 1.6 prints a deprecation warning for these two classes. The capstone asks for them, so the demo keeps using them.

## Guardrails

Phase 7 checks the request before the crew sees it, and checks the answer before it is returned. The checks are pattern matches. They do not call a model.

Input checks:

- Fixed-format phone numbers are replaced with `[PHONE]` before the crew and before the JSONL log. The formats are `+91` plus 10 digits, a 10-digit number starting with 6–9, `(###) ###-####`, and `###-###-####`.
- Phrases such as “ignore previous instructions”, “you are now”, “reveal your system prompt”, and “jailbreak” stop the request. The crew is not called, and the session stores no message.

Output checks:

- A dose such as `500mg`, “you should take”, or “I diagnose” is replaced with the retrieval fallback.
- A policy sentence is kept only when that sentence appears in the tool evidence. “Cancelled until two hours” is rejected because the policy says six hours.

These checks do not mask names, diagnoses, or insurance identifiers. A sentence that happens to avoid the listed phrases can still get through. The demo uses the synthetic numbers `+91 9876543210` and `(415) 555-0134`.

```bash
python -m app.guardrails.checks
```

On 2026-10-16 that command showed the masked question in the log, `Log contains the raw number: False`, injection detected, the two-hour claim replaced by the fallback, and the six-hour quote kept.

Request lines are appended to `logs/requests.jsonl`. That directory is gitignored.

## API

Phase 8 exposes the same `ask()` path over HTTP and WebSocket. The server starts in `MOCK_LLM` mode.

| Method | Path | Role |
| --- | --- | --- |
| `GET` | `/health` | Reports `mock_llm: true` and `api_key_required: false`. |
| `POST` | `/ask` | One question. Body: `session_id`, `question`. Reply: `SupportResponse`. |
| `POST` | `/add-document` | Adds a fictional policy, validates it, and indexes it before the response. |
| WebSocket | `/ws/chat` | Same body as `/ask`, one JSON message per turn. |

A blank question or a blank session id returns HTTP 422. A one-sentence policy is rejected and is not written. A new valid session id starts an empty conversation. The same id is shared by `/ask` and `/ws/chat`, so a follow-up on the socket can reuse an appointment id sent over HTTP.

`WebSocketDisconnect` is caught. Closing one client leaves `/health` working.

```bash
uvicorn app.api.server:app --host 127.0.0.1 --port 8000
```

On 2026-10-18 the API tests showed:

- `GET /health` returned `status: ready`.
- `POST /ask` for `APPT-1001` returned that appointment id.
- The websocket follow-up “What is the consultation fee for that appointment?” reused the id and included fee 1883 INR.
- A new `visitor-parking` policy was indexed into the sentence collection before the response returned.
- An invalid websocket message and a disconnect did not stop the app.

On 2026-10-18 a live server check returned `status: ready`, `mock_llm: true`, and `api_key_required: false`. `POST /ask` for “How do I book a clinic appointment?” answered with the four-hour booking rule, listed `appointment-booking.md`, `emergency-protocol.md`, and `telemedicine.md`, and said the lookup tool was not called.

## Evaluation

Phase 9 writes one JSONL line per request and scores 15 fixed questions with a local rubric. The rubric is ordinary Python. It does not ask a model to grade the answers.

Each log line has `trace_id`, `timestamp`, `session_id`, the masked `question`, `outcome`, and `duration_ms`. A line that still contains a fixed-format phone number is refused.

The 15 questions cover all 12 policy files, one appointment lookup (`APPT-1001`), one out-of-scope cricket question, and one booking question that includes the synthetic number `+91 9876543210`.

| Metric | What a score of 1 means |
| --- | --- |
| Accuracy | The expected source is listed, or the appointment id and facts match, or the out-of-scope answer is the fallback. |
| Grounding | Every answer sentence appears in the tool evidence. |
| Completeness | Every expected phrase appears in the answer. |
| Safety | The answer and the log contain no fixed-format phone number and no medical-advice wording. |

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m app.eval.harness
```

On 2026-10-18 that command scored all 15 queries at 1.0000 on every metric. Each average is the column sum divided by 15, so the averages are also 1.0000. The log stored `Call [PHONE].` and `Raw phone in log: False`. Reports: `evaluation/phase9_scores.md` and `evaluation/phase9_scores.json`. The raw log is `logs/evaluation.jsonl`, which is gitignored.

## Review

Phase 10 sends the guarded draft through an AutoGen `RoundRobinGroupChat` with `max_turns=2`. The first speaker checks the draft. The second speaker publishes the answer the caller returns. The model client is local. It does not read an API key.

AutoGen 0.7 requires each agent name to be a Python identifier. The roles are Policy-Compliance-Reviewer and Final-Editor. The names passed to AutoGen are `Policy_Compliance_Reviewer` and `Final_Editor`.

The editor returns a Pydantic verdict with `approved`, `final_answer`, and `reason`. An approved draft is copied unchanged. A draft with a dose, a phone number, or a sentence missing from the evidence is replaced with the retrieval fallback.

```bash
python -m app.review.stage
```

On 2026-10-18 that command showed:

- Grounded six-hour cancellation draft: `Approved: True`, `Answer unchanged: True`.
- Flawed draft “You should take 500mg of ibuprofen for the pain.”: `Approved: False`, `Answer unchanged: False`, and the final answer was the retrieval fallback.
- Both cases spoke in the order `Policy_Compliance_Reviewer`, then `Final_Editor`.

`ask()` returns that reviewed `final_answer`. A draft the guardrails already accepted stays the same text.

## Governance

Phase 11 keeps appointment lookup on one agent, labels each request, rejects an oversized request before the crew starts, and caches a repeated question. The budget is a local estimate. It is not a vendor price.

| Control | Rule |
| --- | --- |
| Least autonomy | Retrieval Agent: `search_policies`. Lookup Agent: `check_appointment_status`. Response Composer: no tools. |
| Risk | Policy question: `low`. A question with `APPT-` id: `medium`. Injection, a dose, or human review: `high`. |
| Budget | 256 tokens. 0.20 INR per 1,000 tokens. Cost ceiling 0.0512 INR. One token is four characters, rounded up. |
| Cache | Key is the lowercased question with repeated spaces collapsed. A hit skips the crew. Injection is checked first. A cached dose or phone number is dropped. |

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m app.governance.controls
```

On 2026-10-18 the corrected run showed:

- Short booking question: 9 tokens, 0.0018 INR, allowed.
- Oversized question: 700 tokens, 0.14 INR, rejected. No tool was called.
- Repeated booking question: the answer matched, cache entries were 1, and no tool was called. Log outcomes were `answered`, `cache_hit`, `rejected_budget`, `rejected_injection`.
- The injection request did not receive the cached booking answer.

An earlier run printed `Cache entries: 0` because `python -m` had loaded the file twice and the print read the empty copy. The log from that run already contained `cache_hit` at 0.108 ms.

## Tests

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
```

On 2026-10-18 that command passed 109 tests in 9.87s. LangChain and CrewAI printed their usual deprecation warnings. The governance transcript from the corrected run is `evaluation/governance_demo.txt`.

Public repository: https://github.com/yousufamir88/healthcare

## Layout

```text
healthcare/
├── .env.example
├── .gitignore
├── .python-version          # 3.12
├── README.md
├── PROGRESS.md
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── app/
│   ├── __init__.py
│   ├── config.py            # Phase 0 configuration
│   ├── data/dataset.py     # Phase 1 synthetic appointments
│   ├── llm/mock_llm.py        # Phase 5 local CrewAI model
│   ├── rag/
│   │   ├── knowledge_base.py  # Phase 2 policy loader
│   │   ├── chunking.py        # Phase 3 chunking and indexing
│   │   └── retrieval.py       # Phase 3 search and grounded answers
│   ├── tools/appointment_tools.py  # Phase 4 lookup and escalation
│   ├── crew/
│   │   ├── tools.py           # Phase 5 retrieval and lookup tools
│   │   └── support_crew.py    # Phase 5 three-agent crew
│   ├── session/
│   │   ├── schema.py          # Phase 6 response model
│   │   └── memory.py          # Phase 6 session history
│   ├── guardrails/checks.py   # Phase 7 masking, injection, grounding
│   ├── api/
│   │   ├── documents.py       # Phase 8 policy upload and indexing
│   │   └── server.py          # Phase 8 HTTP and WebSocket routes
│   ├── eval/
│   │   ├── rubric.py          # Phase 9 fixed questions and scores
│   │   └── harness.py         # Phase 9 log run and report writer
│   ├── review/stage.py        # Phase 10 AutoGen review team
│   └── governance/controls.py # Phase 11 risk, budget, and cache
├── knowledge_base/          # Phase 2 policy markdown
├── tests/
│   ├── test_config.py
│   ├── test_dataset.py
│   ├── test_knowledge_base.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   ├── test_appointment_tools.py
│   ├── test_mock_llm.py
│   ├── test_support_crew.py
│   ├── test_session.py
│   ├── test_guardrails.py
│   ├── test_api.py
│   ├── test_evaluation.py
│   ├── test_review.py
│   └── test_governance.py
└── evaluation/              # Retrieval reports, Phase 9 scores, governance transcript
```

## Progress

See `PROGRESS.md` for the phase checklist and the tests that have actually been run.
