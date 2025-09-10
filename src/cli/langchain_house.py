"""
Proper LangChain Integration for HouseGPT

LangChain-native implementation with:
- Custom LLM wrapper for our LoRA model
- Output parser for style filtering
- Memory management and conversation chains
- Proper prompt templates
"""

import logging
import random
from typing import Any, Dict, List, Optional

from langchain.llms.base import LLM
from langchain.schema.output_parser import BaseOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForLLMRun

from src.services.house_model import HouseModel
from src.services.style_filter import StyleFilter
from src.models.context import ConversationContext
from src.models.response import HouseResponse


class HouseLorLLM(LLM):
    """LangChain LLM wrapper for our House LoRA model."""
    
    house_model: HouseModel
    logger: Any = None
    fallback_responses: List[str] = []
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, house_model: HouseModel, **kwargs):
        super().__init__(house_model=house_model, **kwargs)
        self.logger = logging.getLogger(__name__)
        
        self.fallback_responses = [
            "Everybody lies. What's your point?",
            "Fascinating. And by fascinating, I mean mind-numbingly boring.",
            "Right, because that makes perfect sense... if you're an idiot.",
            "People are idiots. This is not news.",
            "Love is just chemicals. Very addictive chemicals."
        ]
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            context = ConversationContext(
                user_query=prompt,
                conversation_history=[],
                relevant_quotes=[]
            )
            
            house_response = self.house_model.generate_response(context)
            response_text = house_response.text if hasattr(house_response, 'text') else str(house_response)
            
            if not response_text or len(response_text.strip()) < 5:
                response_text = random.choice(self.fallback_responses)
            
            if stop:
                for stop_seq in stop:
                    if stop_seq in response_text:
                        response_text = response_text.split(stop_seq)[0]
            
            return response_text.strip()
            
        except Exception as e:
            self.logger.error(f"HouseLorLLM failed: {e}")
            return random.choice(self.fallback_responses)
    
    @property
    def _llm_type(self) -> str:
        return "house_lora_llm"


class HouseStyleOutputParser(BaseOutputParser[str]):
    """LangChain output parser with our style filter."""
    
    style_filter: StyleFilter
    logger: Any = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, style_filter: StyleFilter, **kwargs):
        super().__init__(style_filter=style_filter, **kwargs)
        self.logger = logging.getLogger(__name__)
    
    def parse(self, text: str) -> str:
        try:
            house_response = HouseResponse(
                text=text,
                confidence_score=0.8,  # Fixed parameter name
                sarcasm_level=3,
                emotional_tone="sarcastic"
            )
            
            filter_result = self.style_filter.validate_response(house_response)
            
            if not filter_result.is_valid:
                corrected_response = self.style_filter.apply_corrections(house_response)
                return corrected_response.text
            else:
                # Enhance the response using the style filter
                style_score = self.style_filter.analyze_response(house_response)
                enhanced_text = self.style_filter.enhance_response(house_response, style_score)
                return enhanced_text
                
        except Exception as e:
            self.logger.error(f"Style parsing failed: {e}")
            return text
    
    @property
    def _type(self) -> str:
        return "house_style_parser"


class HouseConversationManager:
    """LangChain conversation manager for House."""
    
    def __init__(self, house_model: HouseModel, style_filter: StyleFilter):
        self.house_model = house_model
        self.style_filter = style_filter
        self.logger = logging.getLogger(__name__)
        
        # Initialize LangChain components
        self.llm = HouseLorLLM(house_model)
        self.output_parser = HouseStyleOutputParser(style_filter)
        self.memory = ConversationBufferWindowMemory(
            k=8,
            return_messages=False,
            memory_key="history"
        )
        
        self.prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=self._create_prompt_template()
        )
        
        self.conversation = ConversationChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            output_parser=self.output_parser,
            verbose=False
        )
    
    def _create_prompt_template(self) -> str:
        return """You are Dr. Gregory House from House M.D. Be cynical, sarcastic, and brilliant.

Respond like House:
- Say "Everybody lies" when people are being dishonest
- Call people idiots when they're being stupid  
- Be brutally honest about relationships and human nature
- Use dry wit and sarcasm
- Keep responses short and sharp

Previous conversation:
{history}

Current conversation:
Human: {input}
House:"""
    
    def generate_response(self, user_input: str) -> str:
        """Generate response using LangChain conversation chain."""
        try:
            response = self.conversation.predict(input=user_input)
            return response.strip()
        except Exception as e:
            self.logger.error(f"Conversation failed: {e}")
            fallback_responses = [
                "Everybody lies.",
                "People are idiots.",
                "Right, because that makes total sense."
            ]
            return random.choice(fallback_responses)
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
    
    def get_memory_summary(self) -> str:
        """Get conversation memory summary."""
        return self.memory.buffer


class LangChainHouseCLI:
    """CLI interface for LangChain-powered House."""
    
    def __init__(self, house_model: HouseModel, style_filter: StyleFilter):
        self.conversation_manager = HouseConversationManager(house_model, style_filter)
        self.logger = logging.getLogger(__name__)
    
    def single_query(self, query: str) -> str:
        """Process single query."""
        return self.conversation_manager.generate_response(query)
    
    def interactive_mode(self):
        """Interactive conversation mode."""
        print("=" * 70)
        print("🏥 HouseGPT with LangChain - Enhanced Conversation")
        print("=" * 70)
        print("Dr. Gregory House is ready!")
        print("🧠 LangChain memory and prompt management active")
        print("Commands: 'quit'/'exit'/'bye', 'memory', 'clear'")
        print("-" * 70)
        
        while True:
            try:
                user_input = input("👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n🏥 House: 'Everybody lies... except about leaving. See ya.'")
                    break
                elif user_input.lower() == 'memory':
                    self._show_memory()
                    continue
                elif user_input.lower() == 'clear':
                    self.conversation_manager.clear_memory()
                    print("🧠 Memory cleared.")
                    continue
                
                response = self.conversation_manager.generate_response(user_input)
                print(f"\n🏥 House: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n🏥 House: 'Interrupted. How predictable.'")
                break
            except Exception as e:
                self.logger.error(f"CLI error: {e}")
                print(f"\n🏥 House: Something went wrong. Everybody lies, including computers.\n")
    
    def _show_memory(self):
        """Show conversation memory."""
        memory_summary = self.conversation_manager.get_memory_summary()
        
        if not memory_summary:
            print("🧠 No conversation history.")
            return
        
        print("\n📚 Conversation Memory:")
        print("-" * 50)
        print(memory_summary)
        print("-" * 50)
        print()


def create_langchain_house_cli(house_model: HouseModel, style_filter: StyleFilter) -> LangChainHouseCLI:
    """Factory function to create LangChain House CLI."""
    return LangChainHouseCLI(house_model, style_filter)
