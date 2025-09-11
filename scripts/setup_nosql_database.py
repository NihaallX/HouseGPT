#!/usr/bin/env python3
"""Database setup script for HouseGPT NoSQL storage.

This script initializes the TinyDB database structure and creates
necessary data directories for HouseGPT document storage.
"""

import os
import sys
from pathlib import Path
import json
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.lib.db_utils import NoSQLDatabaseManager, DatabaseConfig


def setup_logging():
    """Set up logging for the setup script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def create_data_directories(base_path: str = "data") -> None:
    """Create necessary data directories.
    
    Args:
        base_path: Base directory for data storage
    """
    logger = logging.getLogger(__name__)
    
    directories = [
        base_path,
        f"{base_path}/conversations",
        f"{base_path}/embeddings", 
        f"{base_path}/models",
        f"{base_path}/audio",
        f"{base_path}/logs",
        f"{base_path}/backups"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def initialize_database(db_path: str = "data/housegpt.json") -> bool:
    """Initialize the TinyDB database.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Create database configuration
        config = DatabaseConfig(
            database_path=db_path,
            backup_enabled=True,
            pretty_print=True
        )
        
        # Initialize database manager
        logger.info(f"Initializing database at: {db_path}")
        db_manager = NoSQLDatabaseManager(config)
        
        if not db_manager.is_connected():
            logger.error("Failed to connect to database")
            return False
        
        # Insert initial system state
        initial_states = {
            "app_version": "1.0.0",
            "setup_date": datetime.now().isoformat(),
            "database_version": "1.0",
            "last_cleanup": None,
            "total_conversations": 0
        }
        
        for key, value in initial_states.items():
            if not db_manager.save_system_state(key, value):
                logger.warning(f"Failed to set initial state: {key}")
            else:
                logger.info(f"Set initial system state: {key} = {value}")
        
        # Get database stats
        stats = db_manager.get_stats()
        logger.info(f"Database initialized successfully")
        logger.info(f"Database stats: {stats}")
        
        db_manager.close()
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


def create_sample_data(db_path: str = "data/housegpt.json") -> None:
    """Create sample conversation data for testing.
    
    Args:
        db_path: Path to the database file
    """
    logger = logging.getLogger(__name__)
    
    try:
        config = DatabaseConfig(database_path=db_path)
        db_manager = NoSQLDatabaseManager(config)
        
        if not db_manager.is_connected():
            logger.error("Cannot connect to database for sample data")
            return
        
        # Sample conversations
        sample_conversations = [
            {
                "user_input": "Hi House, how are you?",
                "house_response": "Peachy. Just had my morning Vicodin smoothie. What's your fascinating medical mystery?",
                "session_id": "sample_session_1"
            },
            {
                "user_input": "I have a headache.",
                "house_response": "Brilliant observation. It's either a brain tumor or you're not drinking enough coffee. My money's on the coffee.",
                "session_id": "sample_session_1"
            },
            {
                "user_input": "What should I do about my cough?",
                "house_response": "Stop coughing. Alternatively, try not having whatever's making you cough. Revolutionary, I know.",
                "session_id": "sample_session_2"
            }
        ]
        
        for i, conversation in enumerate(sample_conversations, 1):
            from src.lib.db_utils import create_conversation_record
            
            record = create_conversation_record(
                conversation_id=f"sample_{i}",
                user_input=conversation["user_input"],
                house_response=conversation["house_response"],
                session_id=conversation["session_id"],
                user_id="sample_user",
                model_version="housegpt-lora-large",
                response_time_ms=1500
            )
            
            if db_manager.save_conversation(record):
                logger.info(f"Added sample conversation {i}")
            else:
                logger.warning(f"Failed to add sample conversation {i}")
        
        # Sample user preferences
        sample_preferences = {
            "voice_enabled": True,
            "sarcasm_level": 0.8,
            "max_response_length": 200,
            "preferred_topics": ["medicine", "puzzles", "sarcasm"]
        }
        
        for key, value in sample_preferences.items():
            if db_manager.save_user_preference("sample_user", key, value):
                logger.info(f"Added sample preference: {key} = {value}")
        
        db_manager.close()
        logger.info("Sample data created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create sample data: {e}")


def verify_setup(db_path: str = "data/housegpt.json") -> bool:
    """Verify the database setup is working correctly.
    
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
            logger.error("Database connection verification failed")
            return False
        
        # Test basic operations
        logger.info("Testing database operations...")
        
        # Test system state
        test_value = "verification_test"
        if not db_manager.save_system_state("test_key", test_value):
            logger.error("System state save test failed")
            return False
        
        retrieved_value = db_manager.get_system_state("test_key")
        if retrieved_value != test_value:
            logger.error("System state retrieve test failed")
            return False
        
        # Test conversation history
        history = db_manager.get_conversation_history(limit=10)
        logger.info(f"Retrieved {len(history)} conversation records")
        
        # Test user preferences
        preferences = db_manager.get_user_preferences("sample_user")
        logger.info(f"Retrieved {len(preferences)} user preferences")
        
        # Get final stats
        stats = db_manager.get_stats()
        logger.info(f"Final database stats: {stats}")
        
        db_manager.close()
        logger.info("Database verification completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        return False


def main():
    """Main setup function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting HouseGPT NoSQL database setup...")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Set up HouseGPT NoSQL database")
    parser.add_argument("--db-path", default="data/housegpt.json", 
                       help="Path to database file")
    parser.add_argument("--no-sample-data", action="store_true",
                       help="Skip creating sample data")
    parser.add_argument("--data-dir", default="data",
                       help="Base directory for data storage")
    
    args = parser.parse_args()
    
    try:
        # Step 1: Create data directories
        logger.info("Creating data directories...")
        create_data_directories(args.data_dir)
        
        # Step 2: Initialize database
        logger.info("Initializing database...")
        if not initialize_database(args.db_path):
            logger.error("Database initialization failed")
            return 1
        
        # Step 3: Create sample data (unless skipped)
        if not args.no_sample_data:
            logger.info("Creating sample data...")
            create_sample_data(args.db_path)
        
        # Step 4: Verify setup
        logger.info("Verifying setup...")
        if not verify_setup(args.db_path):
            logger.error("Setup verification failed")
            return 1
        
        logger.info("HouseGPT NoSQL database setup completed successfully!")
        logger.info(f"Database file: {Path(args.db_path).absolute()}")
        logger.info("You can now run HouseGPT with the configured database.")
        
        return 0
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
