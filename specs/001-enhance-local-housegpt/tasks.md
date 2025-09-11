# Tasks: HouseGPT Personality Injector with RAG + Rule-Based Style Filter

**Input**: Design documents from `/specs/001-enhance-local-housegpt/`
**Prerequisites**: plan.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → ✅ Found: Python 3.11+, transformers, torch, pgvector, pytest
   → ✅ Extract: Single modular project structure
2. Load optional design documents:
   → ✅ data-model.md: 5 entities (UserInput, HouseQuote, ModelResponse, ConversationLog, Configuration)
   → ✅ contracts/: 6 contract files → 6 contract test tasks
   → ✅ research.md: Technology decisions → setup tasks
3. Generate tasks by category:
   → Setup: Python project, dependencies, PostgreSQL, models
   → Tests: 6 contract tests, 4 integration tests
   → Core: 5 models, 6 services, CLI interface
   → Integration: Database setup, model loading, pipeline
   → Polish: unit tests, performance, documentation
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001-T040)
6. Generate dependency graph and parallel execution examples
7. Validate task completeness:
   → ✅ All contracts have tests
   → ✅ All entities have models
   → ✅ All components implemented
8. Return: SUCCESS (40 tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
**Single project structure** (from plan.md):
- `src/models/` - Data classes and entities
- `src/services/` - Core business logic modules
- `src/cli/` - Command-line interface
- `src/lib/` - Shared utilities
- `tests/contract/` - API contract tests
- `tests/integration/` - Component interaction tests
- `tests/unit/` - Individual module tests

## Phase 3.1: Setup

- [x] **T001** Create Python project structure with src/, tests/, data/, docs/ directories ✅
- [x] **T002** Initialize Python project with requirements.txt (transformers, torch, sentence-transformers, pgvector, pyaudio, whisper, TTS, pytest, psycopg2) ✅
- [x] **T003** [P] Configure development tools: black formatting, flake8 linting, pre-commit hooks in pyproject.toml ✅
- [x] **T004** [P] Set up PostgreSQL database with pgvector extension and create housegpt database ✅
- [x] **T005** [P] Create environment configuration with .env.example and config loading in src/lib/config.py ✅

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests (Components)
- [x] **T006** [P] Contract test for WakeWordDetector class in tests/contract/test_wake_word_contract.py ✅
- [x] **T007** [P] Contract test for RAGStore class in tests/contract/test_rag_store_contract.py ✅
- [x] **T008** [P] Contract test for HouseModel class in tests/contract/test_house_model_contract.py ✅
- [x] **T009** [P] Contract test for StyleFilter class in tests/contract/test_style_filter_contract.py ✅
- [x] **T010** [P] Contract test for VoiceOutput class in tests/contract/test_voice_output_contract.py ✅
- [x] **T011** [P] Contract test for ConversationLogger class in tests/contract/test_logger_contract.py ✅

### Integration Tests (User Scenarios)
- [x] **T012** [P] Integration test for complete wake word → response → voice pipeline in tests/integration/test_full_conversation.py ✅
- [x] **T013** [P] Integration test for RAG quote retrieval and relevance in tests/integration/test_rag_pipeline.py ✅
- [x] **T014** [P] Integration test for style filter sarcasm transformation in tests/integration/test_style_processing.py ✅
- [x] **T015** [P] Integration test for error handling and fallback modes in tests/integration/test_error_recovery.py ✅

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Data Models
- [x] **T016** [P] UserInput data model in src/models/user_input.py with validation rules ✅
- [x] **T017** [P] HouseQuote data model in src/models/house_quote.py with embedding vector support ✅
- [x] **T018** [P] HouseResponse data model in src/models/house_response.py with processing stages ✅
- [x] **T019** [P] ConversationEntry data model in src/models/conversation_entry.py with JSON serialization ✅
- [x] **T020** [P] Configuration data model in src/models/configuration.py with environment loading ✅

### Service Modules
- [x] **T021** [P] WakeWordDetector service in src/services/wake_word.py with Whisper integration ✅
- [x] **T022** [P] RAGStore service in src/services/rag_store.py with pgvector operations ✅
- [x] **T023** [P] HouseModel service in src/services/house_model.py with LoRA loading from housegpt-lora-large/ ✅
- [x] **T024** [P] StyleFilter service in src/services/style_filter.py with sarcasm rules and cranky openers ✅
- [x] **T025** [P] VoiceOutput service in src/services/voice_output.py with XTTS synthesis ✅
- [x] **T026** [P] ConversationLogger service in src/services/conversation_logger.py with JSONL storage ✅

### Shared Utilities
- [x] **T027** [P] Audio utilities in src/lib/audio_utils.py for real-time processing ✅
- [x] **T028** [P] Text utilities in src/lib/text_utils.py for preprocessing and validation ✅
- [x] **T029** Database utilities in src/lib/db_utils.py for PostgreSQL connection management ✅

### CLI Interface
- [x] **T030** Main CLI application in src/cli/main.py with argument parsing and mode selection ✅
- [x] **T031** Interactive mode handler in src/cli/interactive.py for wake word conversations ✅
- [x] **T032** Text-only mode handler in src/cli/text_mode.py for direct input/output ✅

## Phase 3.4: Integration

- [x] **T033** Database schema initialization in scripts/setup_nosql_database.py with NoSQL document storage ✅
- [x] **T034** House quotes data loading in scripts/load_quotes.py with NoSQL document storage ✅
- [ ] **T035** Model downloading and validation in scripts/download_models.py
- [ ] **T036** Application startup sequence with model preloading and health checks
- [ ] **T037** End-to-end conversation pipeline integration with error handling

## Phase 3.5: Polish

- [ ] **T038** [P] Unit tests for all utility functions in tests/unit/test_utils.py
- [ ] **T039** [P] Performance testing and optimization with <5s response time validation
- [ ] **T040** [P] Documentation updates: README.md, API docs, and quickstart validation

## Dependencies

### Critical Path
1. **Setup (T001-T005)** → Everything else
2. **Tests (T006-T015)** → Implementation (T016-T037)
3. **Models (T016-T020)** → Services (T021-T026)
4. **Services (T021-T026)** → CLI (T030-T032)
5. **Integration (T033-T037)** → Polish (T038-T040)

### Specific Dependencies
- T022 (RAGStore) requires T004 (PostgreSQL setup)
- T023 (HouseModel) requires T002 (dependencies installed)
- T034 (Load quotes) requires T022 (RAGStore) and T033 (Database schema)
- T036 (Startup) requires T023 (HouseModel) and T025 (VoiceOutput)
- T037 (Pipeline) requires T021-T026 (All services)

## Parallel Execution Examples

### Setup Phase (can run together):
```bash
# T003, T004, T005 in parallel:
Task: "Configure development tools: black formatting, flake8 linting, pre-commit hooks in pyproject.toml"
Task: "Set up PostgreSQL database with pgvector extension and create housegpt database"  
Task: "Create environment configuration with .env.example and config loading in src/lib/config.py"
```

### Contract Tests Phase (following TDD - tests MUST fail first):
```bash
# T006-T011 in parallel: ✅ COMPLETED
✅ T006: "Contract test for WakeWordDetector class in tests/contract/test_wake_word_contract.py"
✅ T007: "Contract test for RAGStore class in tests/contract/test_rag_store_contract.py"
✅ T008: "Contract test for HouseModel class in tests/contract/test_house_model_contract.py"
✅ T009: "Contract test for StyleFilter class in tests/contract/test_style_filter_contract.py"
✅ T010: "Contract test for VoiceOutput class in tests/contract/test_voice_output_contract.py"
✅ T011: "Contract test for ConversationLogger class in tests/contract/test_conversation_logger_contract.py"
```

**Status**: All contract tests created and failing as expected in TDD. ModuleNotFoundError confirms test-first approach working correctly.

### Integration Tests Phase (can run together):
```bash
# T012-T015 in parallel:
Task: "Integration test for complete wake word → response → voice pipeline in tests/integration/test_full_conversation.py"
Task: "Integration test for RAG quote retrieval and relevance in tests/integration/test_rag_pipeline.py"
Task: "Integration test for style filter sarcasm transformation in tests/integration/test_style_processing.py"
Task: "Integration test for error handling and fallback modes in tests/integration/test_error_recovery.py"
```

### Models Phase (can run together):
```bash
# T016-T020 in parallel:
Task: "UserInput data model in src/models/user_input.py with validation rules"
Task: "HouseQuote data model in src/models/house_quote.py with embedding vector support"  
Task: "ModelResponse data model in src/models/model_response.py with processing stages"
Task: "ConversationLog data model in src/models/conversation_log.py with JSON serialization"
Task: "Configuration data model in src/models/configuration.py with environment loading"
```

### Services Phase (can run together):
```bash
# T021-T026 in parallel:
Task: "WakeWordDetector service in src/services/wake_word.py with Whisper integration"
Task: "RAGStore service in src/services/rag_store.py with pgvector operations"
Task: "HouseModel service in src/services/house_model.py with LoRA loading from housegpt-lora-large/"
Task: "StyleFilter service in src/services/style_filter.py with sarcasm rules and cranky openers"
Task: "VoiceOutput service in src/services/voice_output.py with XTTS synthesis"
Task: "ConversationLogger service in src/services/logger.py with PostgreSQL storage"
```

## Notes
- **TDD Enforcement**: All contract and integration tests (T006-T015) MUST fail before implementing services (T021-T026)
- **File Independence**: [P] tasks modify different files and can run simultaneously
- **LoRA Integration**: T023 uses existing model weights in housegpt-lora-large/ directory
- **Real Dependencies**: Integration tests use actual PostgreSQL, audio devices, and model files
- **Performance Targets**: <2s wake word detection, <5s total response time (validated in T039)

## Task Generation Rules Applied

1. **From Contracts**: 6 contract files → 6 contract test tasks (T006-T011) [P]
2. **From Data Model**: 5 entities → 5 model creation tasks (T016-T020) [P]  
3. **From User Stories**: Quickstart scenarios → 4 integration tests (T012-T015) [P]
4. **From Research**: Technology decisions → setup and utility tasks
5. **TDD Ordering**: All tests (T006-T015) before implementation (T016-T037)
6. **Dependency Ordering**: Models → Services → CLI → Integration → Polish

## Validation Checklist ✅

- [x] All 6 contracts have corresponding contract tests (T006-T011)
- [x] All 5 entities have model creation tasks (T016-T020)
- [x] All tests (T006-T015) come before implementation (T016-T037)
- [x] Parallel tasks ([P]) modify different files and are independent
- [x] Each task specifies exact file path for implementation
- [x] No [P] task conflicts with another [P] task on same file
- [x] TDD workflow enforced with failing tests before implementation
- [x] Integration tests cover all major user scenarios from quickstart.md
