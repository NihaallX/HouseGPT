"""
Text-Only Mode Handler for HouseGPT CLI

Provides simple, direct text input/output functionality:
- Single query processing
- Batch processing
- Simple response formatting
- No conversation state management
"""

import sys
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.models.user_input import UserInput
from src.models.context import ConversationContext
from src.services.house_model import HouseModel
from src.services.style_filter import StyleFilter


class TextOnlyProcessor:
    """Handles text-only interactions with minimal state."""
    
    def __init__(self, house_model: HouseModel, style_filter: StyleFilter):
        """Initialize text processor."""
        self.house_model = house_model
        self.style_filter = style_filter
        self.query_count = 0
    
    def process_single_query(self, query: str, verbose: bool = False) -> Dict[str, Any]:
        """Process a single query and return response."""
        self.query_count += 1
        
        try:
            # Create user input
            input_obj = UserInput(
                text=query,
                source="text_cli",
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "query_number": self.query_count,
                    "mode": "single_query"
                }
            )
            
            # Create minimal conversation context
            context = ConversationContext(
                user_input=input_obj,
                conversation_history=[],  # No history in text-only mode
                system_context={
                    "mode": "text_only",
                    "query_count": self.query_count
                }
            )
            
            # Generate response
            house_response = self.house_model.generate_response(context)
            
            # Apply style filtering
            filtered_response = self.style_filter.apply_filter(house_response)
            
            result = {
                "query": query,
                "response": filtered_response.text,
                "confidence": filtered_response.confidence_score,
                "query_number": self.query_count,
                "timestamp": datetime.now().isoformat()
            }
            
            if verbose:
                result.update({
                    "raw_response": house_response.text,
                    "processing_metadata": {
                        "model_confidence": house_response.confidence_score,
                        "filter_applied": True,
                        "source": input_obj.source
                    }
                })
            
            return result
            
        except Exception as e:
            return {
                "query": query,
                "response": f"🚨 House had a diagnostic error: {e}",
                "confidence": 0.0,
                "error": str(e),
                "query_number": self.query_count,
                "timestamp": datetime.now().isoformat()
            }
    
    def process_batch_queries(self, queries: List[str], verbose: bool = False) -> List[Dict[str, Any]]:
        """Process multiple queries in batch."""
        results = []
        
        print(f"🏥 Processing {len(queries)} queries...")
        
        for i, query in enumerate(queries, 1):
            if verbose:
                print(f"  Processing query {i}/{len(queries)}: {query[:50]}...")
            
            result = self.process_single_query(query, verbose)
            results.append(result)
        
        return results
    
    def format_simple_response(self, result: Dict[str, Any]) -> str:
        """Format response for simple output."""
        confidence_emoji = self._get_confidence_emoji(result.get('confidence', 0))
        return f"{confidence_emoji} {result['response']}"
    
    def format_detailed_response(self, result: Dict[str, Any]) -> str:
        """Format response with detailed information."""
        confidence = result.get('confidence', 0)
        confidence_emoji = self._get_confidence_emoji(confidence)
        
        output = [
            f"Query #{result.get('query_number', '?')}: {result['query']}",
            f"Response {confidence_emoji}: {result['response']}",
            f"Confidence: {confidence:.2f}",
            f"Timestamp: {result.get('timestamp', 'Unknown')}"
        ]
        
        if 'error' in result:
            output.append(f"Error: {result['error']}")
        
        if result.get('processing_metadata'):
            metadata = result['processing_metadata']
            output.extend([
                f"Raw Response: {metadata.get('raw_response', 'N/A')[:100]}...",
                f"Model Confidence: {metadata.get('model_confidence', 'N/A')}"
            ])
        
        return "\n".join(output)
    
    def _get_confidence_emoji(self, confidence: float) -> str:
        """Get emoji based on confidence level."""
        if confidence >= 0.9:
            return "🎯"  # Very high confidence
        elif confidence >= 0.7:
            return "✅"  # High confidence
        elif confidence >= 0.5:
            return "🤔"  # Medium confidence
        elif confidence >= 0.3:
            return "⚠️"   # Low confidence
        else:
            return "❓"  # Very low confidence


class BatchProcessor:
    """Handles batch processing of multiple queries."""
    
    def __init__(self, text_processor: TextOnlyProcessor):
        """Initialize batch processor."""
        self.processor = text_processor
    
    def process_file(self, file_path: str, output_format: str = "simple") -> str:
        """Process queries from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
            
            if not queries:
                return "❌ No queries found in file."
            
            print(f"📂 Processing {len(queries)} queries from {file_path}")
            
            results = self.processor.process_batch_queries(queries, verbose=True)
            
            return self._format_batch_results(results, output_format)
            
        except FileNotFoundError:
            return f"❌ File not found: {file_path}"
        except Exception as e:
            return f"❌ Error processing file: {e}"
    
    def process_stdin(self, output_format: str = "simple") -> str:
        """Process queries from standard input."""
        try:
            print("📥 Enter queries (one per line). Press Ctrl+D (Unix) or Ctrl+Z (Windows) when done:")
            
            queries = []
            while True:
                try:
                    line = input().strip()
                    if line:
                        queries.append(line)
                except EOFError:
                    break
            
            if not queries:
                return "❌ No queries provided."
            
            results = self.processor.process_batch_queries(queries, verbose=True)
            
            return self._format_batch_results(results, output_format)
            
        except KeyboardInterrupt:
            return "\n❌ Batch processing interrupted."
    
    def _format_batch_results(self, results: List[Dict[str, Any]], format_type: str) -> str:
        """Format batch processing results."""
        if format_type == "json":
            import json
            return json.dumps(results, indent=2)
        
        elif format_type == "csv":
            return self._format_as_csv(results)
        
        elif format_type == "detailed":
            output = [f"🏥 House processed {len(results)} queries:", "=" * 50]
            for i, result in enumerate(results, 1):
                output.extend([
                    f"\nQuery {i}:",
                    self.processor.format_detailed_response(result),
                    "-" * 30
                ])
            return "\n".join(output)
        
        else:  # simple format
            output = [f"🏥 House's responses to {len(results)} queries:", "=" * 40]
            for i, result in enumerate(results, 1):
                confidence_emoji = self.processor._get_confidence_emoji(result.get('confidence', 0))
                output.append(f"{i}. {confidence_emoji} {result['response']}")
            return "\n".join(output)
    
    def _format_as_csv(self, results: List[Dict[str, Any]]) -> str:
        """Format results as CSV."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(['Query Number', 'Query', 'Response', 'Confidence', 'Timestamp', 'Error'])
        
        # Data
        for result in results:
            writer.writerow([
                result.get('query_number', ''),
                result.get('query', ''),
                result.get('response', ''),
                result.get('confidence', ''),
                result.get('timestamp', ''),
                result.get('error', '')
            ])
        
        return output.getvalue()


class SimpleFormatter:
    """Simple response formatting utilities."""
    
    @staticmethod
    def format_for_terminal(text: str, width: int = 80) -> str:
        """Format text for terminal display."""
        import textwrap
        
        # Wrap long lines
        wrapped_lines = []
        for line in text.split('\n'):
            if len(line) <= width:
                wrapped_lines.append(line)
            else:
                wrapped_lines.extend(textwrap.wrap(line, width=width))
        
        return '\n'.join(wrapped_lines)
    
    @staticmethod
    def add_house_styling(text: str) -> str:
        """Add House-style formatting to text."""
        # Add some visual flair
        styled_text = text
        
        # Highlight medical terms
        medical_terms = ['diagnosis', 'symptoms', 'treatment', 'patient', 'medicine']
        for term in medical_terms:
            if term in styled_text.lower():
                styled_text = styled_text.replace(term, f"*{term}*")
        
        return styled_text
    
    @staticmethod
    def create_response_box(response: str, confidence: float) -> str:
        """Create a boxed response display."""
        confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
        confidence_emoji = "🎯" if confidence >= 0.7 else "🤔" if confidence >= 0.4 else "⚠️"
        
        box = f"""
┌─ 🏥 Dr. House ─────────────────────────────────────────────────────────┐
│                                                                          │
│  {response[:68]:<68}  │
│  {'...' if len(response) > 68 else '':<68}  │
│                                                                          │
│  Confidence: {confidence_emoji} [{confidence_bar}] {confidence:.2f}                        │
└──────────────────────────────────────────────────────────────────────────┘
        """
        return box.strip()


def create_text_mode_cli():
    """Create a simple text-only CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="HouseGPT Text-Only Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli.text_mode "What's wrong with me?"
  python -m src.cli.text_mode --file queries.txt
  python -m src.cli.text_mode --stdin --format json
        """
    )
    
    parser.add_argument('query', nargs='?', help='Single query to process')
    parser.add_argument('--file', '-f', help='Process queries from file')
    parser.add_argument('--stdin', action='store_true', help='Process queries from stdin')
    parser.add_argument('--format', choices=['simple', 'detailed', 'json', 'csv'], 
                       default='simple', help='Output format')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--config', '-c', help='Configuration file path')
    
    return parser


if __name__ == "__main__":
    # Simple test/demo mode
    parser = create_text_mode_cli()
    args = parser.parse_args()
    
    print("🏥 HouseGPT Text-Only Mode Demo")
    print("Note: Use main CLI for full functionality")
    print("-" * 40)
    
    if args.query:
        print(f"Query: {args.query}")
        print("Response: [Demo mode - integrate with main CLI for real responses]")
    else:
        print("No query provided. Use --help for usage information.")
