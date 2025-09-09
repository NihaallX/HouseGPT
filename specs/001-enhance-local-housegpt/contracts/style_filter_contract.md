# Style Filter Contract

**Component**: `style_filter.py`  
**Purpose**: Sarcasm injection and House personality enhancement  
**Version**: 1.0.0

## Interface Definition

### Class: `StyleFilter`

#### Constructor
```python
def __init__(self, config: Configuration) -> None
```
**Parameters**:
- `config`: Application configuration object
**Raises**:
- `ConfigurationError`: If sarcasm rules file not found

#### Methods

##### `apply_house_style(raw_text: str, context_quotes: List[HouseQuote]) -> str`
Transforms raw model output into characteristic House style.

**Parameters**:
- `raw_text`: Original model-generated response
- `context_quotes`: Quotes used for context (influences style intensity)
**Returns**:
- Filtered text with House personality enhancements
**Preconditions**:
- raw_text must not be empty
- Must be valid UTF-8 text
**Postconditions**:
- Text enhanced with sarcasm and wit
- Cranky opener prepended if appropriate
- Length constraints respected
**Raises**:
- `FilterError`: If text processing fails
- `ValidationError`: If input parameters invalid

##### `add_cranky_opener(text: str) -> str`
Prepends random cranky opener to response.

**Parameters**:
- `text`: Response text to modify
**Returns**:
- Text with opener prepended
**Opener Options**: ["Whattt?", "Yahh?", "Give me Vicodin?"]
**Selection**: Random with configurable weights

##### `inject_sarcasm(text: str, intensity: int = 3) -> str`
Applies sarcasm transformations to text.

**Parameters**:
- `text`: Input text to transform
- `intensity`: Sarcasm level 1-5 (1=subtle, 5=maximum)
**Returns**:
- Text with sarcastic modifications applied
**Transformations**:
- Polite word replacement
- Rhetorical question injection
- Understatement → exaggeration conversion
- Professional → condescending tone shift

##### `ensure_brevity(text: str, max_length: int = 150) -> str`
Enforces response length constraints with House-style truncation.

**Parameters**:
- `text`: Input text to potentially truncate
- `max_length`: Maximum character count
**Returns**:
- Truncated text ending with witty conclusion
**Behavior**:
- If under limit: return unchanged
- If over limit: intelligent truncation with snappy ending

##### `get_filter_stats() -> Dict[str, Any]`
Returns filtering performance metrics.

**Returns**:
```python
{
    "total_filtered": int,
    "avg_processing_time_ms": float,
    "sarcasm_applications": int,
    "opener_frequencies": Dict[str, int],
    "length_truncations": int
}
```

## Sarcasm Transformation Rules

### Polite Word Replacements
```python
POLITE_TO_SARCASTIC = {
    "please": "oh please",
    "thank you": "how generous",
    "sorry": "my bad",
    "excuse me": "pardon my existence",
    "certainly": "obviously",
    "definitely": "absolutely positively",
    "understand": "comprehend with your tiny brain",
    "help": "assist your helplessness",
    "problem": "adorable little issue",
    "question": "burning inquiry"
}
```

### Rhetorical Question Injection
- Pattern: Statement → "Are you seriously asking me about X?"
- Trigger: Medical terms, obvious symptoms, basic concepts
- Frequency: 30% of responses containing medical content

### Professional → Condescending Shifts
```python
PROFESSIONAL_TO_CONDESCENDING = {
    "I recommend": "You should probably",
    "The diagnosis is": "Even you could figure out it's",
    "Consider": "Try not to hurt yourself while",
    "It appears": "Shockingly",
    "Based on": "Since you clearly don't know",
    "Treatment involves": "The cure for your ignorance is"
}
```

### Wit Injection Patterns
- Add dismissive qualifiers: "obviously", "clearly", "surprisingly"
- Inject medical superiority: "as any doctor would know"
- Include recreational references: "unlike my Vicodin"
- Apply dramatic flair: "shocking revelation", "earth-shattering discovery"

## Cranky Opener System

### Opener Selection
```python
CRANKY_OPENERS = {
    "Whattt?": {"weight": 0.4, "context": "surprise/confusion"},
    "Yahh?": {"weight": 0.3, "context": "impatience/dismissal"},
    "Give me Vicodin?": {"weight": 0.3, "context": "exasperation/pain"}
}
```

### Context-Aware Selection
- Medical questions → "Whattt?" (feigned surprise)
- Obvious symptoms → "Yahh?" (impatience)
- Complex cases → "Give me Vicodin?" (dramatic flair)
- Random fallback for general queries

### Opener Application Rules
- Always capitalize and punctuate properly
- Add brief pause after opener (space or line break)
- Ensure opener matches response tone
- Skip opener if response already starts with exclamation

## Length Management

### Truncation Strategy
1. **Target Length**: 150 characters maximum
2. **Smart Truncation**: Cut at sentence boundaries when possible
3. **Witty Endings**: Append House-style conclusions when truncated
4. **Preserve Impact**: Maintain key sarcastic elements

### Truncation Endings
```python
WITTY_ENDINGS = [
    "...but you wouldn't understand.",
    "...obviously.",
    "...shocking, I know.",
    "...next!",
    "...my work here is done.",
    "...you're welcome."
]
```

## Error Handling

### Exception Hierarchy
```python
class StyleFilterError(Exception): pass
class FilterError(StyleFilterError): pass
class ValidationError(StyleFilterError): pass
class ConfigurationError(StyleFilterError): pass
```

### Recovery Strategies
- **Transformation failure**: Return original text with opener only
- **Rule file corruption**: Use built-in fallback rules
- **Memory overflow**: Apply minimal filtering (opener + truncation)
- **Encoding issues**: Clean text and retry with ASCII subset

## Performance Contract

- **Processing Speed**: < 10ms for typical response (150 chars)
- **Memory Usage**: < 10MB for rule storage and processing
- **Transformation Accuracy**: 95% successful rule applications
- **Quality Preservation**: No grammatical errors introduced
- **Character Consistency**: House personality maintained across all responses

## Configuration Parameters

```python
{
    "sarcasm_intensity": 3,              # 1-5 scale
    "opener_frequency": 0.8,             # 0.0-1.0 probability
    "max_response_length": 150,          # Character limit
    "truncation_preference": "sentence", # "sentence" or "word"
    "wit_injection_rate": 0.6,          # 0.0-1.0 probability
    "politeness_replacement": True,      # Enable polite→sarcastic
    "medical_condescension": True,       # Enable medical superiority
    "vicodin_references": 0.2           # 0.0-1.0 probability
}
```

## Testing Contract

### Unit Tests Required
- Sarcasm rule application accuracy
- Opener selection and application
- Length truncation with quality preservation
- Character consistency validation
- Edge cases (empty text, special characters, non-English)

### Integration Test Scenarios
- End-to-end filtering pipeline with model outputs
- Performance testing with various text lengths
- Quality assessment with human evaluation
- Style consistency across different input types
- Error handling with malformed inputs

### Quality Assurance Metrics
- **Sarcasm Score**: Measure applied transformations per response
- **Character Consistency**: Validate House personality traits
- **Readability**: Ensure filtered text remains coherent
- **Length Compliance**: Verify all outputs meet length constraints
- **Error Rate**: Track processing failures and recoveries

### Reference Test Cases
Include 50 sample raw model outputs with expected filtered results for regression testing and quality validation.
