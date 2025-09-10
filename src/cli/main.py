#!/usr/bin/env python3
"""
HouseGPT CLI Application

Main command-line interface for interacting with HouseGPT.
Provides text-based interaction with House M.D. personality AI.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from src.lib.config import get_config, reload_config
from src.services.house_model import HouseModel
from src.services.style_filter import StyleFilter
from src.models.user_input import UserInput
from src.models.context import ConversationContext


class HouseGPTCLI:
    """Main CLI application for HouseGPT."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize CLI with configuration."""
        if config_path:
            self.config = reload_config(config_path)
        else:
            self.config = get_config()
        self.house_model = None
        self.style_filter = None
        self.conversation_history = []
        
    def initialize_services(self):
        """Initialize HouseModel and StyleFilter services."""
        print("🏥 Initializing HouseGPT services...")
        
        try:
            # Initialize HouseModel
            print("  📚 Loading House personality model...")
            self.house_model = HouseModel(self.config)
            
            # Initialize StyleFilter
            print("  🎭 Loading style filter...")
            self.style_filter = StyleFilter(self.config)
            
            print("✅ HouseGPT ready! Dr. House is in the house.")
            
        except Exception as e:
            print(f"❌ Failed to initialize services: {e}")
            sys.exit(1)
    
    def get_house_response(self, user_input: str) -> str:
        """Get a House-style response to user input."""
        try:
            # Create user input
            input_obj = UserInput(
                text=user_input,
                timestamp=datetime.now(),
                input_mode="text",
                session_id="cli_session"
            )
            
            # Create conversation context
            context = ConversationContext(
                user_query=user_input,
                conversation_history=self.conversation_history
            )
            
            # Generate response using HouseModel
            print("🤔 House is thinking...")
            house_response = self.house_model.generate_response(context)
            
            # Apply style filtering
            print("🎭 Applying House personality filter...")
            filtered_response = self.style_filter.validate_response(house_response)
            
            # If there are violations, apply corrections
            if not filtered_response.is_valid:
                print("🔧 Applying corrections...")
                corrected_response = self.style_filter.apply_corrections(house_response)
                final_text = corrected_response.text
            else:
                final_text = house_response.text
            
            # Update conversation history
            from src.models.conversation_entry import ConversationEntry
            entry = ConversationEntry(
                timestamp=datetime.now(),
                user_query=user_input,
                house_response=final_text,
                confidence_score=getattr(house_response, 'confidence_score', 0.8)
            )
            self.conversation_history.append(entry)
            
            return final_text
            
        except Exception as e:
            return f"🚨 House had a medical emergency: {e}"
    
    def run_interactive_mode(self):
        """Run interactive conversation mode."""
        print("\n" + "="*60)
        print("🏥 HouseGPT Interactive Mode")
        print("="*60)
        print("Dr. House is ready to solve your medical mysteries!")
        print("Type 'quit', 'exit', or 'bye' to leave.")
        print("Type 'history' to see conversation history.")
        print("Type 'clear' to clear conversation history.")
        print("-"*60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n🏥 Dr. House: 'Everybody lies... except about leaving. Goodbye!'")
                    break
                    
                elif user_input.lower() == 'history':
                    self.show_conversation_history()
                    continue
                    
                elif user_input.lower() == 'clear':
                    self.conversation_history.clear()
                    print("🧹 Conversation history cleared.")
                    continue
                    
                elif not user_input:
                    print("🤨 House: 'Silent treatment? How... diagnostically useless.'")
                    continue
                
                # Get House response
                response = self.get_house_response(user_input)
                print(f"\n🏥 House: {response}")
                
            except KeyboardInterrupt:
                print("\n\n🏥 House: 'Interrupted? How... rude. But effective.'")
                break
            except EOFError:
                print("\n\n🏥 House: 'End of file? End of conversation.'")
                break
    
    def run_single_query(self, query: str):
        """Run a single query and return the response."""
        print(f"\n👤 Query: {query}")
        response = self.get_house_response(query)
        print(f"\n🏥 House: {response}")
        return response
    
    def show_conversation_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            print("📝 No conversation history yet.")
            return
            
        print("\n📝 Conversation History:")
        print("-"*40)
        for i, entry in enumerate(self.conversation_history, 1):
            print(f"{i}. You: {entry.user_query}")
            print(f"   House: {entry.house_response[:100]}...")
            if hasattr(entry, 'confidence_score') and entry.confidence_score:
                confidence = entry.confidence_score
                print(f"   (Confidence: {confidence:.2f})")
            print()


def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="HouseGPT - Dr. House's AI Personality CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  housegpt                           # Interactive mode
  housegpt -q "What's wrong with me?" # Single query
  housegpt --config custom.yml       # Custom config file
  housegpt --interactive             # Explicitly interactive mode
  housegpt --langchain               # Use LangChain for better conversations
        """
    )
    
    parser.add_argument(
        '-q', '--query',
        type=str,
        help='Single query to ask House (non-interactive mode)'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode (default if no query provided)'
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--langchain',
        action='store_true',
        help='Use LangChain for enhanced conversation management'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='HouseGPT CLI v1.0.0'
    )
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Initialize CLI application
    cli = HouseGPTCLI(config_path=args.config)
    cli.initialize_services()
    
    # Check if LangChain mode is requested
    if args.langchain:
        try:
            from src.cli.langchain_house import create_langchain_house_cli
            
            print("🧠 Using LangChain for enhanced conversation management!")
            langchain_cli = create_langchain_house_cli(cli.house_model, cli.style_filter)
            
            if args.query:
                # Single query with LangChain
                response = langchain_cli.single_query(args.query)
                print(f"\n👤 Query: {args.query}")
                print(f"\n🏥 House: {response}")
            else:
                # Interactive mode with LangChain
                langchain_cli.interactive_mode()
            
            return
            
        except ImportError as e:
            print(f"❌ LangChain not available: {e}")
            print("📦 Install with: pip install langchain langchain-community")
            print("🔄 Falling back to standard mode...")
    
    # Standard mode (original implementation)
    if args.query:
        # Single query mode
        cli.run_single_query(args.query)
    else:
        # Interactive mode (default)
        cli.run_interactive_mode()


if __name__ == "__main__":
    main()
