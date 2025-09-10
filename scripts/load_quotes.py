#!/usr/bin/env python3
"""House quotes data loading script for HouseGPT.

Loads House MD quotes into the database and generates embeddings
for similarity search and RAG functionality.
"""

import argparse
import sys
import os
import json
import csv
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import asyncio

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

from src.lib.db_utils import DatabaseManager, DatabaseConfig
from src.models.configuration import HouseGPTConfig
from src.models.house_quote import HouseQuote


# Sample House MD quotes for initial database population
SAMPLE_HOUSE_QUOTES = [
    {
        "quote_text": "Everybody lies.",
        "context": "House's most famous saying about human nature",
        "episode": "Pilot",
        "season": 1,
        "character": "House",
        "tags": ["philosophy", "human nature", "cynicism"]
    },
    {
        "quote_text": "People don't change. They just become more of who they really are.",
        "context": "House explaining human psychology",
        "episode": "Three Stories",
        "season": 1,
        "character": "House",
        "tags": ["psychology", "human nature", "personality"]
    },
    {
        "quote_text": "I'm not a nice person. I'm a doctor.",
        "context": "House defending his bedside manner",
        "episode": "Detox",
        "season": 1,
        "character": "House",
        "tags": ["medicine", "personality", "professional"]
    },
    {
        "quote_text": "It's a basic truth of the human condition that everybody lies. The only variable is about what.",
        "context": "House explaining why he doesn't trust patients",
        "episode": "Love Hurts",
        "season": 1,
        "character": "House",
        "tags": ["philosophy", "truth", "human condition"]
    },
    {
        "quote_text": "Reality is almost always wrong.",
        "context": "House challenging conventional thinking",
        "episode": "Acceptance",
        "season": 2,
        "character": "House",
        "tags": ["philosophy", "reality", "thinking"]
    },
    {
        "quote_text": "If you talk to God you're religious. If God talks to you, you're psychotic.",
        "context": "House discussing religion and mental health",
        "episode": "House vs. God",
        "season": 2,
        "character": "House",
        "tags": ["religion", "psychology", "mental health"]
    },
    {
        "quote_text": "People like talking about people. Makes them feel superior. Makes them feel in control.",
        "context": "House on gossip and human psychology",
        "episode": "Heavy",
        "season": 1,
        "character": "House",
        "tags": ["psychology", "human behavior", "control"]
    },
    {
        "quote_text": "There's an evolutionary imperative why we give a damn about our family and friends. And there's an evolutionary imperative why we don't give a damn about anybody else.",
        "context": "House explaining human social behavior",
        "episode": "One Day, One Room",
        "season": 3,
        "character": "House",
        "tags": ["evolution", "family", "social behavior"]
    },
    {
        "quote_text": "I take risks. Sometimes patients die. But not taking risks causes more patients to die.",
        "context": "House defending his medical approach",
        "episode": "Acceptance",
        "season": 2,
        "character": "House",
        "tags": ["medicine", "risk", "decision making"]
    },
    {
        "quote_text": "If somebody's not dead, you can always get more information.",
        "context": "House on the importance of thorough diagnosis",
        "episode": "Pilot",
        "season": 1,
        "character": "House",
        "tags": ["medicine", "diagnosis", "information"]
    },
    {
        "quote_text": "You can't always get what you want. But if you try sometimes, you just might find you get what you need.",
        "context": "House quoting the Rolling Stones to make a point",
        "episode": "Need to Know",
        "season": 2,
        "character": "House",
        "tags": ["philosophy", "expectations", "needs"]
    },
    {
        "quote_text": "Truth begins in lies.",
        "context": "House explaining his diagnostic philosophy",
        "episode": "Maternity",
        "season": 1,
        "character": "House",
        "tags": ["truth", "lies", "diagnosis"]
    },
    {
        "quote_text": "Being miserable doesn't mean you're not depressed. Depressed people want to be miserable.",
        "context": "House discussing depression with a patient",
        "episode": "Heavy",
        "season": 1,
        "character": "House",
        "tags": ["depression", "mental health", "psychology"]
    },
    {
        "quote_text": "Occam's razor. The simplest explanation is almost always somebody screwed up.",
        "context": "House's version of Occam's razor",
        "episode": "Razor",
        "season": 7,
        "character": "House",
        "tags": ["logic", "simplicity", "mistakes"]
    },
    {
        "quote_text": "I solve problems. That's what I do. That's who I am.",
        "context": "House defining his identity",
        "episode": "No Reason",
        "season": 2,
        "character": "House",
        "tags": ["identity", "problem solving", "self definition"]
    },
    {
        "quote_text": "You want to know how two chemicals interact, do you ask them? No, they're going to lie through their lying little chemical teeth.",
        "context": "House explaining why he doesn't trust patient histories",
        "episode": "Pilot",
        "season": 1,
        "character": "House",
        "tags": ["science", "truth", "methodology"]
    },
    {
        "quote_text": "Humanity is overrated.",
        "context": "House's cynical view of human compassion",
        "episode": "Love Hurts",
        "season": 1,
        "character": "House",
        "tags": ["cynicism", "humanity", "compassion"]
    },
    {
        "quote_text": "If you could reason with religious people, there would be no religious people.",
        "context": "House on religion and rational thinking",
        "episode": "House vs. God",
        "season": 2,
        "character": "House",
        "tags": ["religion", "reason", "logic"]
    },
    {
        "quote_text": "The nature of life is not permanence, but flux.",
        "context": "House philosophizing about change",
        "episode": "Acceptance",
        "season": 2,
        "character": "House",
        "tags": ["philosophy", "change", "life"]
    },
    {
        "quote_text": "I'm not insulting you. I'm describing you.",
        "context": "House defending his blunt observations",
        "episode": "The Socratic Method",
        "season": 1,
        "character": "House",
        "tags": ["honesty", "observation", "directness"]
    }
]

# Additional quotes for different contexts
MEDICAL_QUOTES = [
    {
        "quote_text": "It's not lupus. It's never lupus.",
        "context": "House's running joke about lupus diagnoses",
        "episode": "Various",
        "season": 0,
        "character": "House",
        "tags": ["medical", "lupus", "diagnosis", "humor"]
    },
    {
        "quote_text": "Symptoms never lie.",
        "context": "House trusting physical evidence over patient reports",
        "episode": "DNR",
        "season": 1,
        "character": "House",
        "tags": ["symptoms", "evidence", "diagnosis"]
    },
    {
        "quote_text": "Differential diagnosis: what else could it be?",
        "context": "House teaching diagnostic methodology",
        "episode": "Various",
        "season": 0,
        "character": "House",
        "tags": ["differential", "diagnosis", "methodology"]
    }
]

SARCASTIC_QUOTES = [
    {
        "quote_text": "I'm sorry, did I hurt your feelings? Good. The truth often does that.",
        "context": "House being characteristically blunt",
        "episode": "Various",
        "season": 0,
        "character": "House",
        "tags": ["sarcasm", "truth", "feelings"]
    },
    {
        "quote_text": "Idiocy is not a medical condition.",
        "context": "House commenting on human stupidity",
        "episode": "Various",
        "season": 0,
        "character": "House",
        "tags": ["intelligence", "stupidity", "medical"]
    },
    {
        "quote_text": "Brilliant. Do you also know that water is wet?",
        "context": "House mocking obvious statements",
        "episode": "Various",
        "season": 0,
        "character": "House",
        "tags": ["sarcasm", "obvious", "intelligence"]
    }
]


class QuoteLoader:
    """Loads House quotes into database with embeddings."""
    
    def __init__(self, config: DatabaseConfig, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize quote loader.
        
        Args:
            config: Database configuration
            embedding_model: Name of embedding model to use
        """
        self.config = config
        self.embedding_model_name = embedding_model
        self.logger = logging.getLogger(__name__)
        
        # Initialize database manager
        self.db_manager = DatabaseManager(config)
        
        # Initialize embedding model
        self.embedding_model = None
        if HAS_EMBEDDINGS:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                self.logger.info(f"Loaded embedding model: {embedding_model}")
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {e}")
        else:
            self.logger.warning("Sentence transformers not available - embeddings will be skipped")
    
    def load_quotes_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load quotes from JSON or CSV file.
        
        Args:
            file_path: Path to quotes file
            
        Returns:
            List of quote dictionaries
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Quotes file not found: {file_path}")
        
        quotes = []
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    quotes = data
                elif isinstance(data, dict) and 'quotes' in data:
                    quotes = data['quotes']
                else:
                    raise ValueError("Invalid JSON format - expected list or dict with 'quotes' key")
        
        elif file_path.suffix.lower() == '.csv':
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                quotes = list(reader)
                
                # Convert season to int if present
                for quote in quotes:
                    if 'season' in quote and quote['season']:
                        try:
                            quote['season'] = int(quote['season'])
                        except ValueError:
                            quote['season'] = None
                    
                    # Convert tags from string to list if needed
                    if 'tags' in quote and isinstance(quote['tags'], str):
                        quote['tags'] = [tag.strip() for tag in quote['tags'].split(',')]
        
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        self.logger.info(f"Loaded {len(quotes)} quotes from {file_path}")
        return quotes
    
    def validate_quote(self, quote: Dict[str, Any]) -> bool:
        """Validate quote data structure.
        
        Args:
            quote: Quote dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['quote_text']
        
        for field in required_fields:
            if field not in quote or not quote[field]:
                self.logger.warning(f"Quote missing required field '{field}': {quote}")
                return False
        
        # Set defaults for optional fields
        quote.setdefault('context', '')
        quote.setdefault('episode', '')
        quote.setdefault('season', None)
        quote.setdefault('character', 'House')
        quote.setdefault('tags', [])
        
        # Ensure tags is a list
        if isinstance(quote['tags'], str):
            quote['tags'] = [tag.strip() for tag in quote['tags'].split(',')]
        
        return True
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if embedding generation fails
        """
        if not self.embedding_model:
            return None
        
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Failed to generate embedding for text: {text[:50]}... Error: {e}")
            return None
    
    def insert_quote(self, quote: Dict[str, Any]) -> Optional[int]:
        """Insert single quote into database.
        
        Args:
            quote: Quote dictionary
            
        Returns:
            Quote ID if successful, None otherwise
        """
        if not self.validate_quote(quote):
            return None
        
        try:
            # Insert quote
            insert_query = """
            INSERT INTO house_quotes (quote_text, context, episode, season, character, tags)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """
            
            params = (
                quote['quote_text'],
                quote['context'],
                quote['episode'],
                quote['season'],
                quote['character'],
                quote['tags']
            )
            
            result = self.db_manager.execute_query(insert_query, params)
            
            if result and len(result) > 0:
                quote_id = result[0]['id']
                self.logger.debug(f"Inserted quote {quote_id}: {quote['quote_text'][:50]}...")
                return quote_id
            else:
                self.logger.error(f"Failed to insert quote: {quote['quote_text'][:50]}...")
                return None
                
        except Exception as e:
            self.logger.error(f"Error inserting quote: {e}")
            return None
    
    def insert_embedding(self, quote_id: int, embedding: List[float]) -> bool:
        """Insert embedding for quote.
        
        Args:
            quote_id: Quote ID
            embedding: Embedding vector
            
        Returns:
            True if successful, False otherwise
        """
        try:
            insert_query = """
            INSERT INTO house_quotes_embeddings (quote_id, embedding, embedding_model)
            VALUES (%s, %s, %s)
            """
            
            # Convert embedding to proper format for pgvector
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            
            params = (quote_id, embedding_str, self.embedding_model_name)
            
            result = self.db_manager.execute_query(insert_query, params, fetch=False)
            return result is not None
            
        except Exception as e:
            self.logger.error(f"Error inserting embedding for quote {quote_id}: {e}")
            return False
    
    def load_quotes_batch(self, quotes: List[Dict[str, Any]], batch_size: int = 50) -> Tuple[int, int]:
        """Load quotes in batches with embeddings.
        
        Args:
            quotes: List of quote dictionaries
            batch_size: Number of quotes to process per batch
            
        Returns:
            Tuple of (successful_inserts, failed_inserts)
        """
        successful = 0
        failed = 0
        
        for i in range(0, len(quotes), batch_size):
            batch = quotes[i:i + batch_size]
            self.logger.info(f"Processing batch {i // batch_size + 1}/{(len(quotes) - 1) // batch_size + 1}")
            
            for quote in batch:
                # Insert quote
                quote_id = self.insert_quote(quote)
                
                if quote_id:
                    # Generate and insert embedding
                    embedding = self.generate_embedding(quote['quote_text'])
                    
                    if embedding:
                        if self.insert_embedding(quote_id, embedding):
                            successful += 1
                        else:
                            failed += 1
                            self.logger.warning(f"Failed to insert embedding for quote {quote_id}")
                    else:
                        successful += 1  # Quote inserted, but no embedding
                        self.logger.warning(f"No embedding generated for quote {quote_id}")
                else:
                    failed += 1
            
            # Small delay between batches
            time.sleep(0.1)
        
        return successful, failed
    
    def load_sample_quotes(self) -> Tuple[int, int]:
        """Load sample House quotes into database.
        
        Returns:
            Tuple of (successful_inserts, failed_inserts)
        """
        all_quotes = SAMPLE_HOUSE_QUOTES + MEDICAL_QUOTES + SARCASTIC_QUOTES
        
        self.logger.info(f"Loading {len(all_quotes)} sample quotes...")
        return self.load_quotes_batch(all_quotes)
    
    def clear_existing_quotes(self) -> bool:
        """Clear all existing quotes from database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Clearing existing quotes...")
            
            # Delete embeddings first (foreign key constraint)
            self.db_manager.execute_query("DELETE FROM house_quotes_embeddings", fetch=False)
            
            # Delete quotes
            result = self.db_manager.execute_query("DELETE FROM house_quotes", fetch=False)
            
            if result is not None:
                self.logger.info("✅ Existing quotes cleared")
                return True
            else:
                self.logger.error("Failed to clear existing quotes")
                return False
                
        except Exception as e:
            self.logger.error(f"Error clearing quotes: {e}")
            return False
    
    def get_quote_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded quotes.
        
        Returns:
            Dictionary with quote statistics
        """
        try:
            stats = {}
            
            # Total quotes
            result = self.db_manager.execute_query("SELECT COUNT(*) as count FROM house_quotes")
            stats['total_quotes'] = result[0]['count'] if result else 0
            
            # Quotes with embeddings
            result = self.db_manager.execute_query("SELECT COUNT(*) as count FROM house_quotes_embeddings")
            stats['quotes_with_embeddings'] = result[0]['count'] if result else 0
            
            # Quotes by character
            result = self.db_manager.execute_query(
                "SELECT character, COUNT(*) as count FROM house_quotes GROUP BY character ORDER BY count DESC"
            )
            stats['by_character'] = {row['character']: row['count'] for row in result} if result else {}
            
            # Quotes by season
            result = self.db_manager.execute_query(
                "SELECT season, COUNT(*) as count FROM house_quotes WHERE season IS NOT NULL GROUP BY season ORDER BY season"
            )
            stats['by_season'] = {row['season']: row['count'] for row in result} if result else {}
            
            # Most common tags
            result = self.db_manager.execute_query(
                "SELECT unnest(tags) as tag, COUNT(*) as count FROM house_quotes GROUP BY tag ORDER BY count DESC LIMIT 10"
            )
            stats['top_tags'] = {row['tag']: row['count'] for row in result} if result else {}
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting quote statistics: {e}")
            return {}


def main():
    """Main function for quote loading script."""
    parser = argparse.ArgumentParser(description="HouseGPT Quote Loading")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--quotes-file",
        type=str,
        help="Path to quotes JSON/CSV file"
    )
    parser.add_argument(
        "--sample-quotes",
        action="store_true",
        help="Load sample quotes instead of file"
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear existing quotes before loading"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show quote statistics"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    logger = logging.getLogger(__name__)
    
    if not HAS_EMBEDDINGS:
        logger.warning("Sentence transformers not available. Install with: pip install sentence-transformers")
    
    # Get database configuration
    if args.config:
        config = HouseGPTConfig.from_env(args.config)
        db_config = config.database
    else:
        # Use environment variables or defaults
        db_config = DatabaseConfig()
    
    # Initialize quote loader
    loader = QuoteLoader(db_config, args.embedding_model)
    
    if args.stats_only:
        # Just show statistics
        stats = loader.get_quote_stats()
        print("\n📊 Quote Statistics:")
        print(f"  Total quotes: {stats.get('total_quotes', 0)}")
        print(f"  Quotes with embeddings: {stats.get('quotes_with_embeddings', 0)}")
        
        if stats.get('by_character'):
            print(f"  By character: {stats['by_character']}")
        
        if stats.get('top_tags'):
            print(f"  Top tags: {dict(list(stats['top_tags'].items())[:5])}")
        
        return 0
    
    # Clear existing quotes if requested
    if args.clear_existing:
        if not loader.clear_existing_quotes():
            return 1
    
    # Load quotes
    if args.sample_quotes:
        logger.info("🏥 Loading sample House quotes...")
        successful, failed = loader.load_sample_quotes()
    elif args.quotes_file:
        logger.info(f"📚 Loading quotes from file: {args.quotes_file}")
        try:
            quotes = loader.load_quotes_from_file(args.quotes_file)
            successful, failed = loader.load_quotes_batch(quotes, args.batch_size)
        except Exception as e:
            logger.error(f"Failed to load quotes from file: {e}")
            return 1
    else:
        logger.error("Either --sample-quotes or --quotes-file must be specified")
        return 1
    
    # Show results
    logger.info(f"✅ Quote loading completed:")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    
    # Show final statistics
    stats = loader.get_quote_stats()
    logger.info(f"📊 Database now contains {stats.get('total_quotes', 0)} quotes")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
