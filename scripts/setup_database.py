#!/usr/bin/env python3
"""Database schema initialization script for HouseGPT.

Sets up PostgreSQL database with required tables for:
- Conversation logging
- User preferences  
- House quotes and embeddings
- System state management
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional, List, Dict
import argparse
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.lib.db_utils import DatabaseManager, DatabaseConfig
    from src.models.configuration import HouseGPTConfig, load_config
    HAS_DATABASE_SUPPORT = True
except ImportError as e:
    HAS_DATABASE_SUPPORT = False
    print(f"Database utilities not available: {e}")

# SQL for creating database schema
SCHEMA_SQL = {
    "conversations": """
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            conversation_id VARCHAR(255) NOT NULL,
            user_input TEXT NOT NULL,
            house_response TEXT NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            session_id VARCHAR(255),
            user_id VARCHAR(255),
            model_version VARCHAR(100),
            response_time_ms INTEGER,
            confidence_score FLOAT,
            style_score FLOAT,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_conversations_conversation_id ON conversations(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp);
        CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at);
    """,
    
    "user_preferences": """
        CREATE TABLE IF NOT EXISTS user_preferences (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            preference_key VARCHAR(255) NOT NULL,
            preference_value JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, preference_key)
        );
        
        CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id);
        CREATE INDEX IF NOT EXISTS idx_user_preferences_key ON user_preferences(preference_key);
    """,
    
    "house_quotes": """
        CREATE TABLE IF NOT EXISTS house_quotes (
            id SERIAL PRIMARY KEY,
            quote_text TEXT NOT NULL,
            episode_season INTEGER,
            episode_number INTEGER,
            episode_title VARCHAR(255),
            character_speaker VARCHAR(100) DEFAULT 'House',
            context_description TEXT,
            medical_relevance BOOLEAN DEFAULT FALSE,
            emotional_tone VARCHAR(50),
            quote_category VARCHAR(100),
            source_accuracy VARCHAR(20) DEFAULT 'verified',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_house_quotes_season_episode ON house_quotes(episode_season, episode_number);
        CREATE INDEX IF NOT EXISTS idx_house_quotes_speaker ON house_quotes(character_speaker);
        CREATE INDEX IF NOT EXISTS idx_house_quotes_category ON house_quotes(quote_category);
        CREATE INDEX IF NOT EXISTS idx_house_quotes_medical ON house_quotes(medical_relevance);
        CREATE INDEX IF NOT EXISTS idx_house_quotes_text_search ON house_quotes USING gin(to_tsvector('english', quote_text));
    """,
    
    "house_quotes_embeddings": """
        CREATE TABLE IF NOT EXISTS house_quotes_embeddings (
            id SERIAL PRIMARY KEY,
            quote_id INTEGER NOT NULL REFERENCES house_quotes(id) ON DELETE CASCADE,
            embedding_model VARCHAR(255) NOT NULL,
            embedding_vector VECTOR(384), -- Default dimension for all-MiniLM-L6-v2
            embedding_metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(quote_id, embedding_model)
        );
        
        CREATE INDEX IF NOT EXISTS idx_house_quotes_embeddings_quote_id ON house_quotes_embeddings(quote_id);
        CREATE INDEX IF NOT EXISTS idx_house_quotes_embeddings_model ON house_quotes_embeddings(embedding_model);
        -- Vector similarity index will be created after pgvector is available
    """,
    
    "system_state": """
        CREATE TABLE IF NOT EXISTS system_state (
            id SERIAL PRIMARY KEY,
            state_key VARCHAR(255) UNIQUE NOT NULL,
            state_value JSONB NOT NULL,
            description TEXT,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_by VARCHAR(255)
        );
        
        CREATE INDEX IF NOT EXISTS idx_system_state_key ON system_state(state_key);
        CREATE INDEX IF NOT EXISTS idx_system_state_updated ON system_state(updated_at);
    """,
    
    "conversation_sessions": """
        CREATE TABLE IF NOT EXISTS conversation_sessions (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(255) UNIQUE NOT NULL,
            user_id VARCHAR(255),
            session_type VARCHAR(50) DEFAULT 'interactive', -- interactive, text_only, voice
            started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP WITH TIME ZONE,
            total_exchanges INTEGER DEFAULT 0,
            avg_response_time_ms FLOAT,
            session_metadata JSONB,
            is_active BOOLEAN DEFAULT TRUE
        );
        
        CREATE INDEX IF NOT EXISTS idx_conversation_sessions_session_id ON conversation_sessions(session_id);
        CREATE INDEX IF NOT EXISTS idx_conversation_sessions_user_id ON conversation_sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_conversation_sessions_started ON conversation_sessions(started_at);
        CREATE INDEX IF NOT EXISTS idx_conversation_sessions_active ON conversation_sessions(is_active);
    """,
    
    "model_performance": """
        CREATE TABLE IF NOT EXISTS model_performance (
            id SERIAL PRIMARY KEY,
            model_version VARCHAR(100) NOT NULL,
            metric_name VARCHAR(100) NOT NULL,
            metric_value FLOAT NOT NULL,
            metric_metadata JSONB,
            measurement_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            environment VARCHAR(50) DEFAULT 'production'
        );
        
        CREATE INDEX IF NOT EXISTS idx_model_performance_version ON model_performance(model_version);
        CREATE INDEX IF NOT EXISTS idx_model_performance_metric ON model_performance(metric_name);
        CREATE INDEX IF NOT EXISTS idx_model_performance_timestamp ON model_performance(measurement_timestamp);
    """
}

# Functions for schema initialization
def check_dependencies() -> bool:
    """Check if all required dependencies are available."""
    if not HAS_DATABASE_SUPPORT:
        print("❌ Database support not available. Please install psycopg2-binary.")
        return False
    
    try:
        import psycopg2
        return True
    except ImportError:
        print("❌ psycopg2 not available. Please install psycopg2-binary.")
        return False


def check_pgvector_extension(db_manager: DatabaseManager) -> bool:
    """Check if pgvector extension is available."""
    try:
        result = db_manager.execute_query("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        return result is not None and len(result) > 0
    except Exception as e:
        logging.warning(f"Could not check pgvector extension: {e}")
        return False


def create_pgvector_extension(db_manager: DatabaseManager) -> bool:
    """Create pgvector extension if possible."""
    try:
        db_manager.execute_query("CREATE EXTENSION IF NOT EXISTS vector", fetch=False)
        print("✅ Created pgvector extension")
        return True
    except Exception as e:
        print(f"⚠️  Could not create pgvector extension: {e}")
        print("   Vector similarity search will not be available.")
        return False


def create_vector_indexes(db_manager: DatabaseManager) -> bool:
    """Create vector similarity indexes."""
    try:
        # Create HNSW index for vector similarity search
        vector_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_house_quotes_embeddings_vector_hnsw 
        ON house_quotes_embeddings 
        USING hnsw (embedding_vector vector_cosine_ops) 
        WITH (m = 16, ef_construction = 64);
        """
        
        db_manager.execute_query(vector_index_sql, fetch=False)
        print("✅ Created vector similarity indexes")
        return True
    except Exception as e:
        print(f"⚠️  Could not create vector indexes: {e}")
        return False


def initialize_system_state(db_manager: DatabaseManager) -> bool:
    """Initialize default system state values."""
    default_states = [
        {
            "state_key": "schema_version",
            "state_value": {"version": "1.0.0", "created": datetime.now().isoformat()},
            "description": "Database schema version"
        },
        {
            "state_key": "model_status",
            "state_value": {"status": "uninitialized", "last_check": None},
            "description": "AI model loading status"
        },
        {
            "state_key": "quotes_loaded",
            "state_value": {"loaded": False, "count": 0, "last_update": None},
            "description": "House quotes loading status"
        },
        {
            "state_key": "embeddings_generated",
            "state_value": {"generated": False, "model": None, "count": 0},
            "description": "Quote embeddings generation status"
        }
    ]
    
    try:
        for state in default_states:
            db_manager.execute_query(
                """
                INSERT INTO system_state (state_key, state_value, description, updated_by)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (state_key) 
                DO UPDATE SET 
                    updated_at = CURRENT_TIMESTAMP,
                    updated_by = EXCLUDED.updated_by
                """,
                (state["state_key"], state["state_value"], state["description"], "setup_database.py"),
                fetch=False
            )
        
        print("✅ Initialized system state")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize system state: {e}")
        return False


def create_database_schema(config: Optional[HouseGPTConfig] = None, 
                          force: bool = False) -> bool:
    """Create complete database schema.
    
    Args:
        config: HouseGPT configuration
        force: Force recreation of existing tables
        
    Returns:
        True if successful, False otherwise
    """
    if not check_dependencies():
        return False
    
    # Load configuration
    if config is None:
        try:
            config = load_config()
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            return False
    
    # Create database manager
    db_config = DatabaseConfig(
        host=config.database.host,
        port=config.database.port,
        database=config.database.database,
        username=config.database.username,
        password=config.database.password
    )
    
    db_manager = DatabaseManager(db_config)
    
    if db_manager.state.value != "connected":
        print(f"❌ Failed to connect to database: {db_manager._last_error}")
        return False
    
    print(f"✅ Connected to database: {config.database.host}:{config.database.port}/{config.database.database}")
    
    # Check and create pgvector extension
    has_pgvector = check_pgvector_extension(db_manager)
    if not has_pgvector:
        has_pgvector = create_pgvector_extension(db_manager)
    else:
        print("✅ pgvector extension already available")
    
    # Create tables
    success_count = 0
    total_tables = len(SCHEMA_SQL)
    
    for table_name, table_sql in SCHEMA_SQL.items():
        try:
            print(f"📋 Creating table: {table_name}")
            
            if force:
                # Drop table if force is enabled
                drop_sql = f"DROP TABLE IF EXISTS {table_name} CASCADE"
                db_manager.execute_query(drop_sql, fetch=False)
                print(f"   Dropped existing table: {table_name}")
            
            # Create table
            db_manager.execute_query(table_sql, fetch=False)
            print(f"✅ Created table: {table_name}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Failed to create table {table_name}: {e}")
    
    # Create vector indexes if pgvector is available
    if has_pgvector:
        create_vector_indexes(db_manager)
    
    # Initialize system state
    initialize_system_state(db_manager)
    
    # Print summary
    print(f"\n📊 Schema creation summary:")
    print(f"   Tables created: {success_count}/{total_tables}")
    print(f"   pgvector available: {has_pgvector}")
    print(f"   Vector indexes: {has_pgvector}")
    
    if success_count == total_tables:
        print("🎉 Database schema initialization completed successfully!")
        return True
    else:
        print("⚠️  Database schema initialization completed with errors.")
        return False


def main():
    """Main function for database setup script."""
    parser = argparse.ArgumentParser(description="Initialize HouseGPT database schema")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--force", "-f", action="store_true", 
                       help="Force recreation of existing tables")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("🏥 HouseGPT Database Schema Initialization")
    print("=" * 50)
    
    # Load configuration
    config = None
    if args.config:
        try:
            config = load_config(args.config)
            print(f"✅ Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            sys.exit(1)
    
    # Create schema
    success = create_database_schema(config, force=args.force)
    
    if success:
        print("\n🎉 Database initialization completed successfully!")
        print("   You can now run:")
        print("   - python scripts/load_quotes.py (to load House quotes)")
        print("   - python scripts/download_models.py (to download AI models)")
        sys.exit(0)
    else:
        print("\n❌ Database initialization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
