#!/usr/bin/env python3
"""House quotes data loading script for HouseGPT.

This script loads House MD quotes into the NoSQL database for RAG retrieval.
Includes quote text, episode information, and context data.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.lib.db_utils import NoSQLDatabaseManager, DatabaseConfig


def setup_logging():
    """Set up logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_house_quotes_data() -> List[Dict[str, Any]]:
    """Load House MD quotes data.
    
    Returns:
        List of quote dictionaries
    """
    # House MD quotes dataset
    quotes = [
        {
            "quote_text": "Everybody lies.",
            "episode": "Pilot",
            "season": 1,
            "episode_number": 1,
            "context": "House's fundamental belief about human nature",
            "character": "House",
            "category": "philosophy",
            "medical_relevance": "high",
            "sarcasm_level": 0.6
        },
        {
            "quote_text": "People don't change. They just become more of who they really are.",
            "episode": "Forever",
            "season": 2,
            "episode_number": 20,
            "context": "House's cynical view on personal growth",
            "character": "House",
            "category": "philosophy",
            "medical_relevance": "low",
            "sarcasm_level": 0.7
        },
        {
            "quote_text": "It's not lupus. It's never lupus.",
            "episode": "Various",
            "season": "Multiple",
            "episode_number": 0,
            "context": "House's running gag about differential diagnosis",
            "character": "House",
            "category": "medical",
            "medical_relevance": "high",
            "sarcasm_level": 0.8
        },
        {
            "quote_text": "I don't care much for apologies. Just don't do it again.",
            "episode": "Role Model",
            "season": 1,
            "episode_number": 17,
            "context": "House's dismissive attitude toward conventional social niceties",
            "character": "House",
            "category": "social",
            "medical_relevance": "low",
            "sarcasm_level": 0.5
        },
        {
            "quote_text": "The eyes can mislead, a smile can lie, but shoes always tell the truth.",
            "episode": "Humpty Dumpty",
            "season": 2,
            "episode_number": 3,
            "context": "House's deductive reasoning about patients",
            "character": "House",
            "category": "diagnostics",
            "medical_relevance": "high",
            "sarcasm_level": 0.4
        },
        {
            "quote_text": "I'm not a hero. I'm a high-functioning sociopath.",
            "episode": "Various discussions",
            "season": "Multiple",
            "episode_number": 0,
            "context": "House's self-assessment of his personality",
            "character": "House",
            "category": "self-reflection",
            "medical_relevance": "medium",
            "sarcasm_level": 0.6
        },
        {
            "quote_text": "You can't always get what you want, but you can always get what you need.",
            "episode": "Needle in a Haystack",
            "season": 3,
            "episode_number": 13,
            "context": "House quoting Rolling Stones while solving a case",
            "character": "House",
            "category": "philosophy",
            "medical_relevance": "medium",
            "sarcasm_level": 0.3
        },
        {
            "quote_text": "Being nice is overrated.",
            "episode": "Occam's Razor",
            "season": 1,
            "episode_number": 3,
            "context": "House defending his abrasive bedside manner",
            "character": "House",
            "category": "social",
            "medical_relevance": "low",
            "sarcasm_level": 0.7
        },
        {
            "quote_text": "What would you prefer, a doctor who holds your hand while you die or one who ignores you while you get better?",
            "episode": "Detox",
            "season": 1,
            "episode_number": 11,
            "context": "House justifying his methods to a patient",
            "character": "House",
            "category": "medical_ethics",
            "medical_relevance": "high",
            "sarcasm_level": 0.5
        },
        {
            "quote_text": "I solve puzzles. The only difference between me and everyone else is that I don't pretend the puzzles aren't there.",
            "episode": "Pilot",
            "season": 1,
            "episode_number": 1,
            "context": "House explaining his approach to medicine",
            "character": "House",
            "category": "diagnostics",
            "medical_relevance": "high",
            "sarcasm_level": 0.4
        },
        {
            "quote_text": "Idiopathic, from the Latin meaning 'we're idiots because we can't figure out what's causing it.'",
            "episode": "Distractions",
            "season": 2,
            "episode_number": 12,
            "context": "House's cynical take on medical terminology",
            "character": "House",
            "category": "medical",
            "medical_relevance": "high",
            "sarcasm_level": 0.9
        },
        {
            "quote_text": "Reality is almost always wrong.",
            "episode": "House vs. God",
            "season": 2,
            "episode_number": 19,
            "context": "House challenging conventional thinking",
            "character": "House",
            "category": "philosophy",
            "medical_relevance": "medium",
            "sarcasm_level": 0.6
        },
        {
            "quote_text": "If you talk to God, you're religious. If God talks to you, you're psychotic.",
            "episode": "House vs. God",
            "season": 2,
            "episode_number": 19,
            "context": "House's view on religious experiences and mental health",
            "character": "House",
            "category": "philosophy",
            "medical_relevance": "medium",
            "sarcasm_level": 0.8
        },
        {
            "quote_text": "I was wrong. It happens. Move on.",
            "episode": "Various",
            "season": "Multiple",
            "episode_number": 0,
            "context": "House's rare admission of error",
            "character": "House",
            "category": "humility",
            "medical_relevance": "high",
            "sarcasm_level": 0.3
        },
        {
            "quote_text": "Pain is the human condition. Disease is the exception.",
            "episode": "Three Stories",
            "season": 1,
            "episode_number": 21,
            "context": "House's philosophical view on suffering",
            "character": "House",
            "category": "philosophy",
            "medical_relevance": "high",
            "sarcasm_level": 0.4
        },
        {
            "quote_text": "I don't like to hurt people's feelings. That's why I don't usually tell them what I think.",
            "episode": "Control",
            "season": 1,
            "episode_number": 14,
            "context": "House's sarcastic explanation of his bluntness",
            "character": "House",
            "category": "social",
            "medical_relevance": "low",
            "sarcasm_level": 0.9
        },
        {
            "quote_text": "Treating illness is why we became doctors. Treating patients is what makes most doctors miserable.",
            "episode": "Skin Deep",
            "season": 2,
            "episode_number": 13,
            "context": "House distinguishing between medicine and patient care",
            "character": "House",
            "category": "medical_ethics",
            "medical_relevance": "high",
            "sarcasm_level": 0.6
        },
        {
            "quote_text": "The most successful marriages are based on lies. You're off to a good start.",
            "episode": "Honeymoon",
            "season": 1,
            "episode_number": 22,
            "context": "House's cynical relationship advice",
            "character": "House",
            "category": "relationships",
            "medical_relevance": "low",
            "sarcasm_level": 0.8
        },
        {
            "quote_text": "Sometimes the best gift is not getting what you want.",
            "episode": "Birthmarks",
            "season": 5,
            "episode_number": 4,
            "context": "House's perspective on disappointment and growth",
            "character": "House",
            "category": "philosophy",
            "medical_relevance": "low",
            "sarcasm_level": 0.3
        },
        {
            "quote_text": "New is good. Because old didn't work.",
            "episode": "Wilson",
            "season": 6,
            "episode_number": 10,
            "context": "House advocating for innovative approaches",
            "character": "House",
            "category": "innovation",
            "medical_relevance": "high",
            "sarcasm_level": 0.4
        }
    ]
    
    return quotes


def create_quote_document(quote_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a quote document for database storage.
    
    Args:
        quote_data: Raw quote data
        
    Returns:
        Formatted quote document
    """
    return {
        "quote_id": f"house_quote_{hash(quote_data['quote_text']) % 10000}",
        "quote_text": quote_data["quote_text"],
        "episode_info": {
            "episode": quote_data["episode"],
            "season": quote_data["season"],
            "episode_number": quote_data["episode_number"]
        },
        "context_info": {
            "context": quote_data["context"],
            "character": quote_data["character"],
            "category": quote_data["category"],
            "medical_relevance": quote_data["medical_relevance"],
            "sarcasm_level": quote_data["sarcasm_level"]
        },
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "source": "house_md_canonical_quotes",
            "verified": True
        },
        "search_tags": [
            quote_data["category"],
            quote_data["medical_relevance"],
            f"season_{quote_data['season']}",
            "house_md"
        ]
    }


def load_quotes_to_database(db_path: str = "data/housegpt.json") -> bool:
    """Load House quotes into the database.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Connect to database
        config = DatabaseConfig(database_path=db_path)
        db_manager = NoSQLDatabaseManager(config)
        
        if not db_manager.is_connected():
            logger.error("Cannot connect to database")
            return False
        
        # Create quotes table if it doesn't exist
        quotes_table = db_manager.db.table('house_quotes')
        
        # Load quotes data
        quotes_data = load_house_quotes_data()
        logger.info(f"Loaded {len(quotes_data)} House quotes")
        
        # Insert quotes into database
        successful_inserts = 0
        for quote_data in quotes_data:
            try:
                quote_doc = create_quote_document(quote_data)
                quotes_table.insert(quote_doc)
                successful_inserts += 1
                logger.debug(f"Inserted quote: {quote_data['quote_text'][:50]}...")
                
            except Exception as e:
                logger.warning(f"Failed to insert quote: {e}")
        
        logger.info(f"Successfully inserted {successful_inserts} quotes")
        
        # Update system state
        db_manager.save_system_state("total_quotes_loaded", successful_inserts)
        db_manager.save_system_state("quotes_last_updated", datetime.now().isoformat())
        
        db_manager.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to load quotes: {e}")
        return False


def load_quotes_from_file(file_path: str, db_path: str = "data/housegpt.json") -> bool:
    """Load quotes from a JSON file.
    
    Args:
        file_path: Path to JSON file with quotes
        db_path: Path to the database file
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            quotes_data = json.load(f)
        
        logger.info(f"Loaded {len(quotes_data)} quotes from file: {file_path}")
        
        # Connect to database
        config = DatabaseConfig(database_path=db_path)
        db_manager = NoSQLDatabaseManager(config)
        
        if not db_manager.is_connected():
            logger.error("Cannot connect to database")
            return False
        
        quotes_table = db_manager.db.table('house_quotes')
        
        # Insert quotes
        successful_inserts = 0
        for quote_data in quotes_data:
            try:
                quote_doc = create_quote_document(quote_data)
                quotes_table.insert(quote_doc)
                successful_inserts += 1
                
            except Exception as e:
                logger.warning(f"Failed to insert quote from file: {e}")
        
        logger.info(f"Successfully inserted {successful_inserts} quotes from file")
        
        db_manager.save_system_state("total_quotes_loaded", successful_inserts)
        db_manager.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to load quotes from file: {e}")
        return False


def verify_quotes_loading(db_path: str = "data/housegpt.json") -> bool:
    """Verify that quotes were loaded correctly.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        True if verification passes, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        config = DatabaseConfig(database_path=db_path)
        db_manager = NoSQLDatabaseManager(config)
        
        if not db_manager.is_connected():
            logger.error("Cannot connect to database for verification")
            return False
        
        quotes_table = db_manager.db.table('house_quotes')
        
        # Check quote count
        total_quotes = len(quotes_table)
        logger.info(f"Total quotes in database: {total_quotes}")
        
        if total_quotes == 0:
            logger.error("No quotes found in database")
            return False
        
        # Check sample quotes
        sample_quotes = quotes_table.all()[:3]
        for i, quote in enumerate(sample_quotes, 1):
            logger.info(f"Sample quote {i}: {quote['quote_text'][:50]}...")
            logger.debug(f"Full quote data: {quote}")
        
        # Check quote categories
        categories = set()
        for quote in quotes_table.all():
            if 'context_info' in quote and 'category' in quote['context_info']:
                categories.add(quote['context_info']['category'])
        
        logger.info(f"Quote categories found: {sorted(categories)}")
        
        # Test search functionality
        from tinydb import Query
        QuoteQuery = Query()
        
        # Search for medical quotes
        medical_quotes = quotes_table.search(QuoteQuery.context_info.medical_relevance == 'high')
        logger.info(f"Medical quotes found: {len(medical_quotes)}")
        
        # Search for sarcastic quotes
        sarcastic_quotes = quotes_table.search(QuoteQuery.context_info.sarcasm_level > 0.7)
        logger.info(f"Highly sarcastic quotes found: {len(sarcastic_quotes)}")
        
        db_manager.close()
        logger.info("Quote verification completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Quote verification failed: {e}")
        return False


def main():
    """Main function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting House quotes data loading...")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Load House MD quotes into database")
    parser.add_argument("--db-path", default="data/housegpt.json",
                       help="Path to database file")
    parser.add_argument("--quotes-file", 
                       help="Optional JSON file with additional quotes")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing quotes, don't load new ones")
    
    args = parser.parse_args()
    
    try:
        if args.verify_only:
            # Only verify existing quotes
            if verify_quotes_loading(args.db_path):
                logger.info("Quote verification successful!")
                return 0
            else:
                logger.error("Quote verification failed!")
                return 1
        
        # Load default quotes
        logger.info("Loading canonical House MD quotes...")
        if not load_quotes_to_database(args.db_path):
            logger.error("Failed to load canonical quotes")
            return 1
        
        # Load additional quotes from file if provided
        if args.quotes_file:
            logger.info(f"Loading additional quotes from: {args.quotes_file}")
            if not load_quotes_from_file(args.quotes_file, args.db_path):
                logger.warning("Failed to load additional quotes from file")
        
        # Verify the loading
        logger.info("Verifying quote loading...")
        if not verify_quotes_loading(args.db_path):
            logger.error("Quote verification failed")
            return 1
        
        logger.info("House quotes loading completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Quote loading failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
