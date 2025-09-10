"""
Interactive Mode Handler for HouseGPT CLI

Provides enhanced interactive conversation features including:
- Real-time conversation flow
- Context-aware responses
- Conversation management
- Advanced user commands
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from src.models.user_input import UserInput
from src.models.context import ConversationContext
from src.models.conversation_entry import ConversationEntry
from src.services.house_model import HouseModel
from src.services.style_filter import StyleFilter


class InteractiveSession:
    """Manages an interactive conversation session with House."""
    
    def __init__(self, house_model: HouseModel, style_filter: StyleFilter):
        """Initialize interactive session."""
        self.house_model = house_model
        self.style_filter = style_filter
        self.conversation_history: List[ConversationEntry] = []
        self.session_stats = {
            'start_time': datetime.now(),
            'total_exchanges': 0,
            'avg_confidence': 0.0,
            'topics_discussed': set()
        }
    
    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and generate House response."""
        try:
            # Create structured user input
            input_obj = UserInput(
                text=user_input,
                source="interactive_cli",
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "session_id": id(self),
                    "exchange_number": self.session_stats['total_exchanges'] + 1
                }
            )
            
            # Create conversation context
            context = ConversationContext(
                user_input=input_obj,
                conversation_history=self.conversation_history,
                system_context={
                    "mode": "interactive",
                    "session_stats": self.session_stats,
                    "last_topic": self._get_last_topic()
                }
            )
            
            # Generate House response
            house_response = self.house_model.generate_response(context)
            
            # Apply style filtering
            filtered_response = self.style_filter.apply_filter(house_response)
            
            # Create conversation entry
            entry = ConversationEntry(
                user_input=user_input,
                house_response=filtered_response.text,
                timestamp=datetime.now().isoformat(),
                metadata={
                    "confidence": filtered_response.confidence_score,
                    "processing_time": 0.0,  # Could add timing
                    "filter_applied": True,
                    "raw_response": house_response.text
                }
            )
            
            # Update session
            self._update_session(entry, filtered_response)
            
            return {
                "response": filtered_response.text,
                "confidence": filtered_response.confidence_score,
                "entry": entry,
                "stats": self._get_session_summary()
            }
            
        except Exception as e:
            error_response = f"🚨 House encountered a diagnostic error: {e}"
            return {
                "response": error_response,
                "confidence": 0.0,
                "entry": None,
                "error": str(e)
            }
    
    def _update_session(self, entry: ConversationEntry, response):
        """Update session statistics and history."""
        self.conversation_history.append(entry)
        self.session_stats['total_exchanges'] += 1
        
        # Update confidence average
        confidences = [float(e.metadata.get('confidence', 0)) 
                      for e in self.conversation_history 
                      if e.metadata and 'confidence' in e.metadata]
        
        if confidences:
            self.session_stats['avg_confidence'] = sum(confidences) / len(confidences)
        
        # Extract topic keywords (simple implementation)
        words = entry.user_input.lower().split()
        medical_keywords = {'pain', 'hurt', 'sick', 'fever', 'headache', 'nausea', 
                          'dizzy', 'tired', 'cough', 'rash', 'symptoms', 'doctor',
                          'medicine', 'treatment', 'diagnosis', 'patient'}
        
        for word in words:
            if word in medical_keywords:
                self.session_stats['topics_discussed'].add(word)
    
    def _get_last_topic(self) -> Optional[str]:
        """Get the last discussed topic."""
        if not self.conversation_history:
            return None
            
        last_entry = self.conversation_history[-1]
        # Simple topic extraction from last user input
        words = last_entry.user_input.lower().split()
        medical_keywords = {'pain', 'hurt', 'sick', 'fever', 'headache', 'nausea'}
        
        for word in words:
            if word in medical_keywords:
                return word
        return None
    
    def _get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        duration = datetime.now() - self.session_stats['start_time']
        
        return {
            "exchanges": self.session_stats['total_exchanges'],
            "duration_minutes": duration.total_seconds() / 60,
            "avg_confidence": round(self.session_stats['avg_confidence'], 2),
            "topics_count": len(self.session_stats['topics_discussed']),
            "topics": list(self.session_stats['topics_discussed'])
        }
    
    def get_conversation_summary(self) -> str:
        """Generate a formatted conversation summary."""
        if not self.conversation_history:
            return "📝 No conversation yet."
        
        summary = self._get_session_summary()
        
        output = [
            "\n📊 Session Summary",
            "="*40,
            f"💬 Total exchanges: {summary['exchanges']}",
            f"⏱️  Duration: {summary['duration_minutes']:.1f} minutes",
            f"🎯 Average confidence: {summary['avg_confidence']:.2f}",
            f"🏷️  Topics discussed: {', '.join(summary['topics']) or 'None detected'}"
        ]
        
        if summary['exchanges'] > 0:
            output.extend([
                "\n🔍 Recent Conversation:",
                "-" * 30
            ])
            
            # Show last 3 exchanges
            recent_entries = self.conversation_history[-3:]
            for i, entry in enumerate(recent_entries, 1):
                confidence = entry.metadata.get('confidence', 0)
                output.extend([
                    f"{i}. You: {entry.user_input}",
                    f"   House: {entry.house_response[:80]}...",
                    f"   (Confidence: {confidence:.2f})",
                    ""
                ])
        
        return "\n".join(output)
    
    def export_conversation(self, format_type: str = "text") -> str:
        """Export conversation in specified format."""
        if format_type == "text":
            return self._export_as_text()
        elif format_type == "json":
            return self._export_as_json()
        else:
            return "❌ Unsupported export format. Use 'text' or 'json'."
    
    def _export_as_text(self) -> str:
        """Export conversation as plain text."""
        if not self.conversation_history:
            return "No conversation to export."
        
        output = [
            f"HouseGPT Conversation Export",
            f"Generated: {datetime.now().isoformat()}",
            f"Total Exchanges: {len(self.conversation_history)}",
            "="*50,
            ""
        ]
        
        for i, entry in enumerate(self.conversation_history, 1):
            timestamp = entry.timestamp if hasattr(entry, 'timestamp') else 'Unknown'
            confidence = entry.metadata.get('confidence', 0) if entry.metadata else 0
            
            output.extend([
                f"Exchange {i} ({timestamp})",
                f"You: {entry.user_input}",
                f"House: {entry.house_response}",
                f"Confidence: {confidence:.2f}",
                "-" * 30,
                ""
            ])
        
        return "\n".join(output)
    
    def _export_as_json(self) -> str:
        """Export conversation as JSON."""
        import json
        
        export_data = {
            "export_metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_exchanges": len(self.conversation_history),
                "session_stats": self._get_session_summary()
            },
            "conversation": []
        }
        
        for entry in self.conversation_history:
            export_data["conversation"].append({
                "user_input": entry.user_input,
                "house_response": entry.house_response,
                "timestamp": getattr(entry, 'timestamp', None),
                "metadata": entry.metadata or {}
            })
        
        return json.dumps(export_data, indent=2)
    
    def clear_conversation(self):
        """Clear conversation history and reset session."""
        self.conversation_history.clear()
        self.session_stats = {
            'start_time': datetime.now(),
            'total_exchanges': 0,
            'avg_confidence': 0.0,
            'topics_discussed': set()
        }


class AdvancedCommands:
    """Handles advanced commands in interactive mode."""
    
    def __init__(self, session: InteractiveSession):
        self.session = session
        self.commands = {
            'help': self.show_help,
            'stats': self.show_stats,
            'summary': self.show_summary,
            'export': self.export_conversation,
            'clear': self.clear_conversation,
            'history': self.show_history,
            'confidence': self.show_confidence_info
        }
    
    def process_command(self, command: str) -> str:
        """Process an advanced command."""
        parts = command.strip().split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd in self.commands:
            try:
                return self.commands[cmd](args)
            except Exception as e:
                return f"❌ Command error: {e}"
        else:
            return f"❓ Unknown command: {cmd}. Type 'help' for available commands."
    
    def show_help(self, args: List[str]) -> str:
        """Show available commands."""
        help_text = """
🔧 Advanced Commands:
  help         - Show this help message
  stats        - Show session statistics
  summary      - Show conversation summary
  history      - Show conversation history
  confidence   - Show confidence information
  export [fmt] - Export conversation (text/json)
  clear        - Clear conversation history
  
💡 Basic Commands:
  quit/exit/bye - Exit HouseGPT
  
🏥 House is ready for your medical mysteries!
        """
        return help_text.strip()
    
    def show_stats(self, args: List[str]) -> str:
        """Show session statistics."""
        return self.session.get_conversation_summary()
    
    def show_summary(self, args: List[str]) -> str:
        """Show conversation summary."""
        return self.session.get_conversation_summary()
    
    def show_history(self, args: List[str]) -> str:
        """Show conversation history."""
        if not self.session.conversation_history:
            return "📝 No conversation history yet."
        
        # Limit history display
        limit = 10
        if args and args[0].isdigit():
            limit = min(int(args[0]), 50)  # Max 50 entries
        
        entries = self.session.conversation_history[-limit:]
        
        output = [f"\n📝 Last {len(entries)} Conversation Entries:", "-" * 40]
        
        for i, entry in enumerate(entries, 1):
            confidence = entry.metadata.get('confidence', 0) if entry.metadata else 0
            output.extend([
                f"{i}. You: {entry.user_input}",
                f"   House: {entry.house_response[:100]}...",
                f"   (Confidence: {confidence:.2f})",
                ""
            ])
        
        return "\n".join(output)
    
    def show_confidence_info(self, args: List[str]) -> str:
        """Show confidence score information."""
        if not self.session.conversation_history:
            return "📊 No confidence data yet."
        
        confidences = []
        for entry in self.session.conversation_history:
            if entry.metadata and 'confidence' in entry.metadata:
                confidences.append(float(entry.metadata['confidence']))
        
        if not confidences:
            return "📊 No confidence scores available."
        
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        max_conf = max(confidences)
        
        # Confidence interpretation
        if avg_conf >= 0.8:
            interpretation = "🎯 Excellent - House is very confident"
        elif avg_conf >= 0.6:
            interpretation = "👍 Good - House is reasonably confident"
        elif avg_conf >= 0.4:
            interpretation = "🤔 Fair - House has some uncertainty"
        else:
            interpretation = "⚠️  Low - House is struggling with these cases"
        
        return f"""
📊 Confidence Analysis:
  Average: {avg_conf:.2f}
  Range: {min_conf:.2f} - {max_conf:.2f}
  Interpretation: {interpretation}
  Total responses: {len(confidences)}
        """.strip()
    
    def export_conversation(self, args: List[str]) -> str:
        """Export conversation."""
        format_type = args[0] if args else "text"
        
        if format_type not in ["text", "json"]:
            return "❌ Export format must be 'text' or 'json'"
        
        try:
            content = self.session.export_conversation(format_type)
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"housegpt_conversation_{timestamp}.{format_type}"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return f"✅ Conversation exported to: {filename}"
            
        except Exception as e:
            return f"❌ Export failed: {e}"
    
    def clear_conversation(self, args: List[str]) -> str:
        """Clear conversation history."""
        self.session.clear_conversation()
        return "🧹 Conversation history cleared. Fresh start with House!"
