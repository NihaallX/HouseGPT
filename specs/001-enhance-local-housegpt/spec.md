# Feature Specification: HouseGPT Personality Injector with RAG + Rule-Based Style Filter

**Feature Branch**: `001-enhance-local-housegpt`  
**Created**: 2025-09-09  
**Status**: Draft  
**Input**: User description: "Enhance local HouseGPT LoRA model with RAG-based House quotes and rule-based sarcasm filters. Pretrained model folder already placed in the root directory."

## Execution Flow (main)
```
1. Load Pretrained Model
   → Load base model + LoRA weights from root directory
   → Initialize inference pipeline

2. Prepare Knowledge Base (RAG)
   → Parse House M.D. transcripts
   → Extract only Gregory House's lines
   → Generate embeddings (sentence-level vectors)
   → Store in pgvector (or local vector DB)

3. Wake Word Detection
   → Listen continuously for "House?"
   → Ignore all other inputs

4. Input Handling
   → Once wake word is detected:
      - Capture user's voice or text
      - Convert voice to text (if spoken)

5. Retrieval
   → Query vector DB for relevant House quotes
   → Select top N most relevant (semantic similarity)

6. Response Generation
   → Pass (user_input + retrieved_quotes) to LoRA model
   → Generate raw output

7. Style Injection
   → Apply cranky openers: randomly prepend "Whattt?", "Yahh?", or "Give me Vicodin?"
   → Post-process for sarcasm:
      - Replace neutral/polite words with sarcastic equivalents
      - Ensure brevity (avoid long explanations)
      - Add wit/dark humor if missing

8. Voice Output
   → Send final response to XTTS for House-like voice synthesis

9. Logging
   → Save {timestamp, user_input, final_response} into conversation log

10. Error Handling
   → If RAG fails: fall back to LoRA-only output
   → If TTS fails: return text response only
   → If LoRA missing: error log and exit safely
```

---

## User Scenarios & Testing

### Primary User Story
As a user, I want to say "House?" and get a cranky, sarcastic reply in Dr. House's voice, so I feel like I'm talking to Gregory House himself.

### Acceptance Scenarios
1. **Given** the system is idle, **When** I say "House?", **Then** the system must wake up and respond with a cranky opener.
2. **Given** the system has House quotes stored, **When** I ask a question, **Then** it must pull in a relevant quote to guide the response.
3. **Given** the model outputs a plain response, **When** it passes through the sarcasm filter, **Then** the final output must be sarcastic and witty.
4. **Given** the final response is generated, **When** TTS is available, **Then** the system must reply in a House-like voice.

### Edge Cases
- If wake word not detected → system ignores input.
- If pgvector returns no quotes → fallback to model-only response.
- If response is too long → auto-truncate and make witty.
- If XTTS fails → return text response only.
- If model weights are missing/corrupt → log error and exit.

## Requirements

### Functional Requirements
- **FR-001**: System MUST load LoRA-pretrained model from root directory.
- **FR-002**: System MUST activate only on the wake word "House?".
- **FR-003**: System MUST prepend responses with cranky openers (Whattt?, Yahh?, Give me Vicodin?).
- **FR-004**: System MUST retrieve relevant House quotes via pgvector.
- **FR-005**: System MUST inject sarcasm filters (replace polite words, enforce wit).
- **FR-006**: System MUST generate both text and voice replies.
- **FR-007**: System MUST log all interactions with timestamp, input, and output.
- **FR-008**: System SHOULD allow adding new datasets (extra quotes).
- **FR-009**: System SHOULD enforce short, witty responses instead of generic lectures.
- **FR-010**: System MUST fail gracefully without crashing (fallbacks in place).

### Key Entities
- **User Input**: Attributes: text_input, voice_input, wake_word_detected, timestamp
- **HouseGPT Model**: Attributes: base_model, lora_weights, inference_pipeline
- **RAG Context**: Attributes: quote_text, embedding_vector, retrieval_score
- **Filtered Response**: Attributes: raw_model_output, sarcasm_injected_output, final_voice_output
- **Conversation Log**: Attributes: user_message, final_house_response, timestamp, log_id

---

## Review & Acceptance Checklist

### Content Quality
- [x] Offline/local solution (no APIs required)
- [x] Clear execution flow from input → response
- [x] User value focused (House-like sarcasm)
- [x] All sections fully explained

### Requirement Completeness
- [x] All mandatory requirements defined
- [x] Testable acceptance criteria included
- [x] Edge cases covered
- [x] Entities mapped properly
- [x] Scope clearly bounded

---

## Execution Status

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities resolved (local-only setup)
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed

---

## ⚡ Notes for Smart Coding Agent

Must use modular structure:
- `wake_word.py` → listens for "House?"
- `rag_store.py` → handles pgvector store + retrieval
- `house_model.py` → loads LoRA model from root and runs inference
- `style_filter.py` → sarcasm injection + cranky openers
- `voice_output.py` → XTTS handling
- `logger.py` → saves conversations

Must write clean, extensible code with hooks for:
- Adding new datasets (quotes, memes, dialogues).
- Expanding wake words (e.g., "Oi House?").
- Hot-swapping LoRA weights.

Must enforce House's tone: sarcastic, cranky, witty, emotionally sharp, never generic.
