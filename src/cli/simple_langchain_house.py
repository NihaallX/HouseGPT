"""
Simple LangChain-powered Gregory House CLI

A simplified version that works directly with our HouseModel for better conversation management.
"""

from typing import List, Optional
import logging
import random
import sys

from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage

from src.services.house_model import HouseModel
from src.services.style_filter import StyleFilter
from src.models.context import ConversationContext


class SimpleHouseConversation:
    """Simple conversation manager with memory for Gregory House."""
    
    def __init__(self, house_model: HouseModel, style_filter: StyleFilter):
        """Initialize conversation manager."""
        self.house_model = house_model
        self.style_filter = style_filter
        self.logger = logging.getLogger(__name__)
        
        # Setup conversation memory
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Remember last 10 exchanges
            return_messages=True
        )
        
        # Fallback responses for when model fails
        self.fallback_responses = [
            "Everybody lies. What's your point?",
            "Fascinating. And by fascinating, I mean mind-numbingly boring.",
            "Right, because that makes perfect sense... if you're an idiot.",
            "Are you trying to waste my time, or does it come naturally?",
            "That's either genius or completely insane. I'm leaning toward insane.",
            "People are idiots. This is not news.",
            "Love is just chemicals. Very addictive chemicals.",
            "You can't always get what you want, but you usually get what you deserve.",
            "And the award for most predictable human behavior goes to...",
            "Wow, never saw that coming. Oh wait, yes I did."
        ]
    
    def generate_response(self, user_input: str) -> str:
        """Generate a response using HouseModel with conversation memory."""
        try:
            # Get conversation history from memory
            history_messages = self.memory.chat_memory.messages
            conversation_history = []
            
            # Convert LangChain messages to our format
            for i in range(0, len(history_messages), 2):
                if i + 1 < len(history_messages):
                    human_msg = history_messages[i]
                    ai_msg = history_messages[i + 1]
                    if isinstance(human_msg, HumanMessage) and isinstance(ai_msg, AIMessage):
                        conversation_history.append(f"Human: {human_msg.content}")
                        conversation_history.append(f"House: {ai_msg.content}")
            
            # Create context for HouseModel
            context = ConversationContext(
                user_query=user_input,
                conversation_history=conversation_history,
                relevant_quotes=[]  # Could add House quotes here
            )
            
            # Generate response using HouseModel
            house_response = self.house_model.generate_response(context)
            
            # Apply style filtering
            filter_result = self.style_filter.validate_response(house_response)
            
            if not filter_result.is_valid:
                corrected_response = self.style_filter.apply_corrections(house_response)
                response_text = corrected_response.text
            else:
                response_text = house_response.text
            
            # Handle empty or too short responses
            if not response_text or len(response_text.strip()) < 3:
                response_text = random.choice(self.fallback_responses)
            
            # Clean up response
            response_text = response_text.strip()
            
            # Save to memory
            self.memory.save_context(
                {"input": user_input},
                {"output": response_text}
            )
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            response_text = random.choice(self.fallback_responses)
            
            # Still save to memory even if there was an error
            try:
                self.memory.save_context(
                    {"input": user_input},
                    {"output": response_text}
                )
            except:
                pass
            
            return response_text
    
    def get_conversation_history(self) -> List[str]:
        """Get formatted conversation history."""
        history = []
        messages = self.memory.chat_memory.messages
        
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                human_msg = messages[i]
                ai_msg = messages[i + 1]
                if isinstance(human_msg, HumanMessage) and isinstance(ai_msg, AIMessage):
                    history.append(f"👤 You: {human_msg.content}")
                    history.append(f"🏥 House: {ai_msg.content}")
        
        return history
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        print("🧠 Memory cleared. House has forgotten everything... typical.")


class SimpleLangChainHouseCLI:
    """Simple CLI interface for Gregory House with LangChain memory."""
    
    def __init__(self, house_model: HouseModel, style_filter: StyleFilter):
        """Initialize the CLI."""
        self.conversation = SimpleHouseConversation(house_model, style_filter)
        self.logger = logging.getLogger(__name__)
    
    def single_query(self, query: str) -> str:
        """Process a single query and return response."""
        return self.conversation.generate_response(query)
    
    def interactive_mode(self):
        """Run interactive conversation mode."""
        print("=" * 60)
        print("🏥 HouseGPT with Simple LangChain - Enhanced Conversation")
        print("=" * 60)
        print("Dr. Gregory House is ready for conversation!")
        print("Commands: 'quit'/'exit'/'bye', 'history', 'clear'")
        print("-" * 60)
        
        while True:
            try:
                # Get user input
                user_input = input("👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n🏥 House: 'Everybody lies... except about leaving. See ya.'")
                    break
                elif user_input.lower() == 'history':
                    self._show_history()
                    continue
                elif user_input.lower() == 'clear':
                    self.conversation.clear_memory()
                    continue
                
                # Generate and display response
                response = self.conversation.generate_response(user_input)
                print(f"\n🏥 House: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n🏥 House: 'Interrupted by user. How... predictable.'")
                break
            except Exception as e:
                self.logger.error(f"CLI error: {e}")
                print(f"\n🏥 House: Something went wrong. Everybody lies, including computers.\n")
    
    def _show_history(self):
        """Display conversation history."""
        history = self.conversation.get_conversation_history()
        
        if not history:
            print("🧠 No conversation history yet. House's memory is clean... for once.")
            return
        
        print("\n📚 Conversation History:")
        print("-" * 40)
        for entry in history:
            print(entry)
        print("-" * 40)
        print(f"💾 Total exchanges: {len(history) // 2}")
        print()


def create_simple_langchain_house_cli(house_model: HouseModel, style_filter: StyleFilter) -> SimpleLangChainHouseCLI:
    """Factory function to create the simple LangChain House CLI."""
    return SimpleLangChainHouseCLI(house_model, style_filter)


if __name__ == "__main__":
    # For testing
    from src.config.house_config import HouseConfig
    
    config = HouseConfig()
    house_model = HouseModel(config)
    house_model.load_model()
    style_filter = StyleFilter()
    
    cli = create_simple_langchain_house_cli(house_model, style_filter)
    
    if len(sys.argv) > 1:
        # Single query mode
        query = " ".join(sys.argv[1:])
        response = cli.single_query(query)
        print(f"🏥 House: {response}")
    else:
        # Interactive mode
        cli.interactive_mode()
