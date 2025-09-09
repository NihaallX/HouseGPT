# Implementation Plan: HouseGPT Personality Injector with RAG + Rule-Based Style Filter

**Branch**: `001-enhance-local-housegpt` | **Date**: 2025-09-09 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-enhance-local-housegpt/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → ✅ Feature spec loaded and processed
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → ✅ Project Type: AI/ML application with local inference
   → ✅ Structure Decision: Single project with modular components
3. Evaluate Constitution Check section below
   → ✅ Simple modular structure (6 core modules)
   → ✅ No complex patterns, direct component interaction
   → ✅ Update Progress Tracking: Initial Constitution Check
4. Execute Phase 0 → research.md
   → Research required for: LoRA loading, RAG implementation, voice synthesis
5. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file
6. Re-evaluate Constitution Check section
   → Constitution compliance maintained throughout design
   → Update Progress Tracking: Post-Design Constitution Check
7. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
8. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Primary requirement: Create a local HouseGPT assistant that activates on "House?" wake word, retrieves relevant quotes via RAG, and responds with sarcastic personality injection using pretrained LoRA model and XTTS voice synthesis.

Technical approach: Modular Python application with separate components for wake word detection, RAG retrieval, model inference, style filtering, and voice output, all running locally without external APIs.

## Technical Context
**Language/Version**: Python 3.11+  
**Primary Dependencies**: transformers, torch, sentence-transformers, pgvector, pyaudio, whisper, XTTS  
**Storage**: pgvector for embeddings, local file system for model weights and logs  
**Testing**: pytest with real component integration  
**Target Platform**: Local desktop (Windows/Linux/Mac)  
**Project Type**: single - AI/ML application with modular components  
**Performance Goals**: <2s wake word detection, <5s total response time, real-time voice processing  
**Constraints**: Local-only execution, no internet required after setup, <4GB RAM usage  
**Scale/Scope**: Personal assistant, ~1000 House quotes, conversation logging, extensible design

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:
- Projects: 1 (single AI assistant application) ✅
- Using framework directly? Yes - transformers, torch, pgvector directly ✅
- Single data model? Yes - conversation flow with simple entities ✅
- Avoiding patterns? Yes - direct module interaction, no Repository/UoW ✅

**Architecture**:
- EVERY feature as library? Yes - each component (wake_word, rag_store, etc.) as importable modules ✅
- Libraries listed: 
  - wake_word: Audio capture and "House?" detection
  - rag_store: Embedding generation and vector search
  - house_model: LoRA model loading and inference
  - style_filter: Sarcasm injection and cranky openers
  - voice_output: XTTS text-to-speech
  - logger: Conversation persistence
- CLI per library: Main CLI with --wake-word, --process-text, --voice-only modes ✅
- Library docs: llms.txt format planned for each module ✅

**Testing (NON-NEGOTIABLE)**:
- RED-GREEN-Refactor cycle enforced? Yes - tests written before implementation ✅
- Git commits show tests before implementation? Yes - TDD workflow ✅
- Order: Contract→Integration→E2E→Unit strictly followed? Yes ✅
- Real dependencies used? Yes - actual model files, real audio devices ✅
- Integration tests for: Module interactions, voice pipeline, RAG retrieval ✅
- FORBIDDEN: Implementation before test, skipping RED phase ✅

**Observability**:
- Structured logging included? Yes - JSON logs with timestamps, inputs, outputs ✅
- Frontend logs → backend? N/A - single application ✅
- Error context sufficient? Yes - detailed error handling with fallbacks ✅

**Versioning**:
- Version number assigned? 1.0.0 ✅
- BUILD increments on every change? Yes - semantic versioning ✅
- Breaking changes handled? Yes - backward compatibility for conversation logs ✅

## Project Structure

### Documentation (this feature)
```
specs/001-enhance-local-housegpt/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Single project structure (AI/ML application)
src/
├── models/              # Data classes for conversation, quotes, responses
├── services/            # Core logic modules
│   ├── wake_word.py
│   ├── rag_store.py
│   ├── house_model.py
│   ├── style_filter.py
│   ├── voice_output.py
│   └── logger.py
├── cli/                 # Command-line interface
│   └── main.py
└── lib/                 # Shared utilities
    ├── audio_utils.py
    ├── text_utils.py
    └── config.py

tests/
├── contract/            # API contract tests
├── integration/         # Component interaction tests
└── unit/                # Individual module tests

data/                    # Local data storage
├── quotes/              # House M.D. transcript files
├── models/              # LoRA weights (existing housegpt-lora-large/)
├── embeddings/          # Vector database
└── logs/                # Conversation logs
```

**Structure Decision**: Single project - AI/ML application with modular components for local execution

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - LoRA model loading with transformers library best practices
   - pgvector setup and embedding generation strategies
   - XTTS integration for local voice synthesis
   - Wake word detection with pyaudio and whisper
   - House M.D. quote extraction and preprocessing
   - Real-time audio processing optimization

2. **Generate and dispatch research agents**:
   ```
   Task: "Research LoRA model loading with transformers for local inference"
   Task: "Find best practices for pgvector embedding storage and retrieval"
   Task: "Research XTTS local text-to-speech implementation"
   Task: "Find patterns for real-time wake word detection"
   Task: "Research House M.D. transcript sources and parsing approaches"
   Task: "Find optimization techniques for real-time audio processing"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all technical implementation decisions documented

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - UserInput: text_input, voice_input, wake_word_detected, timestamp
   - HouseQuote: quote_text, embedding_vector, episode_info
   - ModelResponse: raw_output, filtered_output, voice_data
   - ConversationLog: user_message, house_response, timestamp, log_id
   - Configuration: model_path, voice_settings, wake_word_sensitivity

2. **Generate API contracts** from functional requirements:
   - Wake word detection interface
   - RAG retrieval interface  
   - Model inference interface
   - Style filtering interface
   - Voice output interface
   - Logging interface
   - Output contract definitions to `/contracts/`

3. **Generate contract tests** from contracts:
   - Test each module interface independently
   - Assert input/output schemas match specifications
   - Tests must fail initially (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Complete conversation flow integration test
   - Wake word detection accuracy test
   - RAG relevance scoring test
   - Style filter sarcasm validation test
   - Voice output quality test

5. **Update agent file incrementally** (O(1) operation):
   - Run `/scripts/update-agent-context.sh [claude|gemini|copilot]` for your AI assistant
   - Add: Python AI/ML stack, local inference, voice processing
   - Update: HouseGPT personality injection context
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Each module contract → contract test task [P]
- Each data entity → model creation task [P]
- Each user story → integration test task
- Implementation tasks to make tests pass
- Setup tasks for data preparation and model installation

**Ordering Strategy**:
- TDD order: Tests before implementation
- Dependency order: Data models → Core services → Integration → CLI
- Mark [P] for parallel execution (independent modules)
- Data preparation tasks first (quote processing, model setup)

**Estimated Output**: 35-40 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*No constitutional violations - simple modular structure maintained*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research planned (/plan command)
- [x] Phase 1: Design planned (/plan command)
- [x] Phase 2: Task planning approach defined (/plan command)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS (planned)
- [x] All technical context defined
- [x] No complexity deviations needed

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
