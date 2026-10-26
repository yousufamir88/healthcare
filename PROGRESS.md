# Project progress

Phases 0 through 12 are complete. Public repository: https://github.com/yousufamir88/healthcare

## Checklist

- [x] Phase 0 — Environment and project foundation (approved)
- [x] Phase 1 — Synthetic appointment dataset (approved)
- [x] Phase 2 — Knowledge base (approved)
- [x] Phase 3 — RAG and vector retrieval (approved)
- [x] Phase 4 — Appointment lookup tool (approved)
- [x] Phase 5 — CrewAI multi-agent system (approved)
- [x] Phase 6 — Session memory and structured output (approved)
- [x] Phase 7 — Input and output guardrails (approved)
- [x] Phase 8 — FastAPI deployment (approved)
- [x] Phase 9 — Structured logging and evaluation (approved)
- [x] Phase 10 — AutoGen review stage (approved)
- [x] Phase 11 — AI governance, budget, and cache (approved)
- [x] Phase 12 — Integration, testing, and GitHub submission (approved). 109 tests passed. Public repo: https://github.com/yousufamir88/healthcare

## Phase 0 notes

- Python on PATH at the start of this phase: 3.11.15.
- Python required by `pyproject.toml` and `.python-version`: 3.12.
- Python 3.12.13 is installed at `~/.local/bin/python3.12`.
- The old `.venv` was Python 3.11 and did not have CrewAI, FastAPI, or LangChain installed.
- Operating system: macOS 26.5.1 (Darwin arm64).

## Tests actually executed

| Date | Command | Result |
| --- | --- | --- |
| 2026-10-16 | `.venv/bin/python -m app.config` | Exit 0. Printed `status: ready`, `mock_llm: true`, `api_key_required: false`. |
| 2026-10-16 | `.venv/bin/python -m pytest tests/test_config.py tests/test_dataset.py -q` | 12 passed in 0.29s. This was the Phase 0 run. |
| 2026-10-16 | `.venv/bin/python -m app.data.dataset` | Exit 0. 100 records, seed 42, follow-up 21/100 (21.00%). Category and status counts are in the README. |
| 2026-10-16 | `.venv/bin/python -m pytest tests/test_dataset.py tests/test_config.py -q` | 27 passed in 0.31s. |
| 2026-10-16 | User rerun: `python -m app.data.dataset` | Same report: 100 records, seed 42, follow-up 21/100 (21.00%). |
| 2026-10-16 | User rerun: `pytest tests/test_dataset.py tests/test_config.py -q` | 27 passed in 0.34s, then 27 passed in 0.29s. |
| 2026-10-16 | `.venv/bin/python -m app.rag.knowledge_base` | Exit 0. 12 documents, each with 4 sentences and a source file name. |
| 2026-10-16 | `.venv/bin/python -m pytest tests/test_knowledge_base.py tests/test_dataset.py tests/test_config.py -q` | 40 passed in 0.33s. |
| 2026-10-16 | User rerun: `pytest tests/test_knowledge_base.py tests/test_dataset.py tests/test_config.py -q` | 40 passed in 0.33s. |
| 2026-10-16 | User rerun: `python -m app.rag.knowledge_base` | Exit 0. 12 documents, each with 4 sentences and a matching source file name. |
| 2026-10-16 | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest tests/test_chunking.py tests/test_retrieval.py -q` | 11 passed in 53.87s. The integration test built a temporary Chroma index. |
| 2026-10-16 | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m app.rag.retrieval` | Exit 0. Collections: fixed_size_overlap 77, sentence 48. Threshold 0.336648. Sentence macro F1 0.7761. Cricket query triggered the fallback. |
| 2026-10-16 | User rerun, offline: `python -m app.rag.retrieval` | Same transcript. Threshold 0.336648. Collections 77 and 48. Cricket fallback true. |
| 2026-10-16 | User rerun: `pytest tests/test_chunking.py tests/test_retrieval.py -q` | 11 passed in 6.67s. |
| 2026-10-16 | User rerun without offline flags: `python -m app.rag.retrieval` | Same scores and threshold. Hugging Face printed an unauthenticated-request warning. No API token is required. |
| 2026-10-16 | `.venv/bin/python -m app.tools.appointment_tools` | Exit 0. Follow-up scores 0.033333 to 0.966667. Median threshold 0.466667. 11 records at or above it. `APPT-9999` returned found false. |
| 2026-10-16 | `.venv/bin/python -m pytest tests/test_appointment_tools.py -q` | 8 passed in 0.02s. |
| 2026-10-16 | User rerun: `python -m app.tools.appointment_tools` | Same report. Threshold 0.466667. 11 records at or above it. `APPT-1001` score 0. `APPT-9999` found false. |
| 2026-10-16 | User rerun: `pytest tests/test_appointment_tools.py -q` | 8 passed in 0.01s. |
| 2026-10-16 | `.venv/bin/python -m pytest -q -m "not integration"` | 58 passed, 1 deselected in 0.42s. The deselected test is the Phase 3 model test. |
| 2026-10-16 | This session: `.venv/bin/python -m app.config` | Exit 0. Printed `status: ready`, `mock_llm: true`, `api_key_required: false`. |
| 2026-10-16 | This session: `.venv/bin/python -m pytest tests/test_config.py -q` | 8 passed in 0.28s. |
| 2026-10-16 | This session, after Phase 0 approval: `.venv/bin/python -m app.data.dataset` | Exit 0. 100 records, seed 42, follow-up 21/100 (21.00%). |
| 2026-10-16 | This session: `.venv/bin/python -m pytest tests/test_dataset.py -q` | 19 passed in 0.02s. |
| 2026-10-16 | User rerun: `python -m app.data.dataset` | Exit 0. Same report: 100 records, seed 42, follow-up 21/100 (21.00%). Phase 1 approved. |
| 2026-10-16 | This session, after Phase 1 approval: `.venv/bin/python -m app.rag.knowledge_base` | Exit 0. 12 documents, each with 4 sentences and a source file name. |
| 2026-10-16 | This session: `.venv/bin/python -m pytest tests/test_knowledge_base.py -q` | 13 passed in 0.04s. |
| 2026-10-16 | User rerun: `python -m app.rag.knowledge_base` | Exit 0. 12 documents, each with 4 sentences and a matching source file name. Phase 2 approved. |
| 2026-10-16 | This session, offline: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m app.rag.retrieval` | Exit 0. Collections: fixed_size_overlap 77, sentence 48. Threshold 0.336648. Sentence macro F1 0.7761. Cricket query fallback true. |
| 2026-10-16 | This session, offline: `.venv/bin/python -m pytest tests/test_chunking.py tests/test_retrieval.py -q` | 11 passed in 6.58s. |
| 2026-10-16 | User rerun, offline: `python -m app.rag.retrieval` | Exit 0. Same transcript. Threshold 0.336648. Collections 77 and 48. Cricket fallback true. Phase 3 approved. |
| 2026-10-16 | This session, after Phase 3 approval: `.venv/bin/python -m app.tools.appointment_tools` | Exit 0. Follow-up scores 0.033333 to 0.966667. Median threshold 0.466667. 11 records at or above it. `APPT-9999` found false. |
| 2026-10-16 | This session: `.venv/bin/python -m pytest tests/test_appointment_tools.py -q` | 8 passed in 0.01s. |
| 2026-10-16 | User rerun: `python -m app.tools.appointment_tools` then `pytest tests/test_appointment_tools.py -q` | Exit 0. Threshold 0.466667. 11 records at or above it. `APPT-1001` Completed, fee 1883, score 0. `APPT-9999` found false. 8 passed in 0.01s. Phase 4 approved. |
| 2026-10-16 | This session, offline: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m app.crew.support_crew` | Exit 0. Policy question called `search_policies` only. Appointment question called `search_policies` and `check_appointment_status` for `APPT-1001`. Final answer included Completed, 1883 INR, and score 0.0. |
| 2026-10-16 | This session, offline: `.venv/bin/python -m pytest tests/test_mock_llm.py tests/test_support_crew.py -q` | 9 passed in 8.69s. CrewAI printed deprecation warnings from its own internals. |
| 2026-10-16 | Phase 5 approved by the user. | No new command. |
| 2026-10-16 | This session, offline: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m app.session.memory` | Exit 0. Session A reused `APPT-1001` on the second turn (4 messages, fee 1883, score 0.0). Session B had no appointment id and did not call lookup. |
| 2026-10-16 | This session, offline: `.venv/bin/python -m pytest tests/test_session.py -q` | 8 passed in 7.81s. LangChain printed deprecation warnings for the required history classes. |
| 2026-10-16 | Phase 6 approved by the user. | No new command. |
| 2026-10-16 | This session: `.venv/bin/python -m app.guardrails.checks` | Exit 0. Log stored `[PHONE]` and did not store `9876543210`. Injection detected. Two-hour claim replaced by the fallback. Six-hour quote kept. |
| 2026-10-16 | This session, offline: `.venv/bin/python -m pytest tests/test_guardrails.py tests/test_session.py -q` | 16 passed in 10.84s. The phone reached the crew only as `[PHONE]`. The earlier session test still passed. |
| 2026-10-18 | User rerun: `python -m app.guardrails.checks` then `pytest tests/test_guardrails.py tests/test_session.py -q` | Exit 0. Mask, injection, invented-claim, and quoted-policy checks matched. 16 passed. Phase 7 approved. |
| 2026-10-18 | This session, offline: `.venv/bin/python -m pytest tests/test_api.py -q` | 7 passed in 10.91s. Health, invalid inputs, websocket disconnect, shared session follow-up, and document indexing passed. |
| 2026-10-18 | User curl: `GET /health` then `POST /ask` booking question | Exit 0. Health `status: ready`, `mock_llm: true`, `api_key_required: false`. Answer included “four hours”, sources `appointment-booking.md`, `emergency-protocol.md`, `telemedicine.md`, intent `policy`, no appointment id, `requires_human_review: false`. Phase 8 approved. |
| 2026-10-18 | This session, offline: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m app.eval.harness` | Exit 0. 15 queries. Averages: accuracy 1.0000, grounding 1.0000, completeness 1.0000, safety 1.0000. `Raw phone in log: False`. |
| 2026-10-18 | This session, offline: `.venv/bin/python -m pytest tests/test_evaluation.py tests/test_guardrails.py -q -m "not integration"` | 12 passed, 1 deselected in 2.00s. The deselected test is the guardrail integration test. |
| 2026-10-18 | User rerun, offline: `python -m app.eval.harness` | Exit 0. All 15 queries scored 1.0000. Averages 1.0000. `Raw phone in log: False`. Phase 9 approved. |
| 2026-10-18 | This session: `.venv/bin/python -m app.review.stage` | Exit 0. Grounded draft approved and unchanged. Flawed `500mg` draft replaced with the fallback. Speakers: `Policy_Compliance_Reviewer`, then `Final_Editor`. `max_turns: 2`. |
| 2026-10-18 | This session, offline: `.venv/bin/python -m pytest tests/test_review.py tests/test_session.py -q` | 13 passed in 10.29s. The session follow-up still returned `APPT-1001` and fee 1883. |
| 2026-10-18 | User rerun: `python -m app.review.stage` | Exit 0. Grounded draft approved and unchanged. Flawed draft replaced with the fallback. Phase 10 approved. |
| 2026-10-18 | User rerun, offline: `python -m app.governance.controls` | Exit 0. Lookup agent alone has the appointment tool. Risks: low, medium, high, high. Short question 9 tokens, 0.0018 INR, allowed. Oversized question 700 tokens, 0.14 INR, rejected, no tools called. Repeated booking answer matched. Injection did not reuse the booking answer. The printed cache count was 0. The log from that same run recorded `cache_hit` in 0.108 ms. |
| 2026-10-18 | This session, offline: `.venv/bin/python -m pytest tests/test_governance.py tests/test_guardrails.py tests/test_evaluation.py -q -m "not integration"` | 20 passed, 1 deselected in 2.32s. |
| 2026-10-18 | This session, offline: `.venv/bin/python -m app.governance.controls` | Exit 0. Same budget and risk figures. Cache entries: 1. Log outcomes: `answered`, `cache_hit`, `rejected_budget`, `rejected_injection`. Phase 11 approved. |
| 2026-10-18 | This session, offline: `.venv/bin/python -m pytest -q` | 109 passed in 9.87s. |
| 2026-10-26 | User: first commit `c02bd68`, remote `origin` set to `https://github.com/yousufamir88/healthcare.git`, branch `main` up to date. | Public repo confirmed: visibility PUBLIC, default branch `main`. Phase 12 GitHub submission complete. |

The Phase 1 file hash of the generated records matches the earlier generator: `6226914ad06138130f05acc646bbcdf74feabdce9f00f1c6297d3c3e981e41b2`. The generation path was kept. Validation and reporting were added around it.

## Known issues

- LangChain 1.6 warns that `InMemoryChatMessageHistory` and `RunnableWithMessageHistory` are deprecated. Phase 6 still uses them because the capstone names those classes.
- Phone masking covers four fixed formats only. Names, diagnoses, and insurance identifiers are not masked. Injection detection matches listed phrases, not every possible override.
- An appointment-status question still runs the policy search, so the answer can include policy sentences along with the lookup facts.
- The retrieval reports in `evaluation/` were rewritten by `python -m app.rag.retrieval` on 2026-10-16. Older numbers from a different document set are not used.
- Phase 9 scores are a local phrase rubric. They are not a human rating and not a model judge. A perfect score means every expected phrase and source check passed on this run.
- AutoGen 0.7 rejects hyphens in agent names. The review roles are Policy-Compliance-Reviewer and Final-Editor. The agent names passed to AutoGen are `Policy_Compliance_Reviewer` and `Final_Editor`.
- Running `python -m app.governance.controls` loads that file twice. The cache `ask()` uses is `app.governance.controls`. The demo prints that cache. An earlier print read the empty copy and showed 0 entries while the log already said `cache_hit`.
- GitHub submission uses account `yousufamir88`. The remote is `origin` → `https://github.com/yousufamir88/healthcare.git`.
