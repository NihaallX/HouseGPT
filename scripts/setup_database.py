#!/usr/bin/env python3
"""Database schema initialization script for HouseGPT.

Sets up PostgreSQL database with required tables, indexes, and extensions
for HouseGPT conversation logging, user preferences, and vector storage.
"""

import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    from psycopg2 import sql
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

from src.lib.db_utils import DatabaseManager, DatabaseConfig
from src.models.configuration import HouseGPTConfig


# SQL for database creation
CREATE_DATABASE_SQL = """
CREATE DATABASE {database_name}
    WITH ENCODING 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TEMPLATE = template0;
"""

# SQL for extensions
CREATE_EXTENSIONS_SQL = [
    "CREATE EXTENSION IF NOT EXISTS vector;",
    "CREATE EXTENSION IF NOT EXISTS pg_trgm;",
    "CREATE EXTENSION IF NOT EXISTS btree_gin;",
    "CREATE EXTENSION IF NOT EXISTS uuid-ossp;"
]

# SQL for table creation
CREATE_TABLES_SQL = {
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
        metadata JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """,
    
    "user_preferences": """
    CREATE TABLE IF NOT EXISTS user_preferences (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        preference_key VARCHAR(255) NOT NULL,
        preference_value JSONB NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT unique_user_preference UNIQUE(user_id, preference_key)
    );
    """,
    
    "system_state": """
    CREATE TABLE IF NOT EXISTS system_state (
        id SERIAL PRIMARY KEY,
        state_key VARCHAR(255) UNIQUE NOT NULL,
        state_value JSONB NOT NULL,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """,
    
    "house_quotes": """
    CREATE TABLE IF NOT EXISTS house_quotes (
        id SERIAL PRIMARY KEY,
        quote_text TEXT NOT NULL,
        context TEXT,
        episode VARCHAR(100),
        season INTEGER,
        character VARCHAR(100) DEFAULT 'House',
        tags TEXT[],
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """,
    
    "house_quotes_embeddings": """
    CREATE TABLE IF NOT EXISTS house_quotes_embeddings (
        id SERIAL PRIMARY KEY,
        quote_id INTEGER REFERENCES house_quotes(id) ON DELETE CASCADE,
        embedding vector(384),
        embedding_model VARCHAR(255) NOT NULL DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """,
    
    "conversation_sessions": """
    CREATE TABLE IF NOT EXISTS conversation_sessions (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(255) UNIQUE NOT NULL,
        user_id VARCHAR(255),
        start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        end_time TIMESTAMP WITH TIME ZONE,
        interaction_count INTEGER DEFAULT 0,
        total_response_time_ms INTEGER DEFAULT 0,
        session_metadata JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """,
    
    "model_performance": """
    CREATE TABLE IF NOT EXISTS model_performance (
        id SERIAL PRIMARY KEY,
        model_version VARCHAR(100) NOT NULL,
        query_hash VARCHAR(64) NOT NULL,
        response_time_ms INTEGER NOT NULL,
        memory_usage_mb FLOAT,
        gpu_usage_percent FLOAT,
        style_score FLOAT,
        user_rating INTEGER CHECK (user_rating >= 1 AND user_rating <= 5),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """
}

# SQL for indexes
CREATE_INDEXES_SQL = [
    # Conversations table indexes
    "CREATE INDEX IF NOT EXISTS idx_conversations_conversation_id ON conversations(conversation_id);",
    "CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_conversations_metadata ON conversations USING gin(metadata);",
    
    # User preferences indexes
    "CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_user_preferences_key ON user_preferences(preference_key);",
    
    # House quotes indexes
    "CREATE INDEX IF NOT EXISTS idx_house_quotes_episode ON house_quotes(episode);",
    "CREATE INDEX IF NOT EXISTS idx_house_quotes_season ON house_quotes(season);",
    "CREATE INDEX IF NOT EXISTS idx_house_quotes_character ON house_quotes(character);",
    "CREATE INDEX IF NOT EXISTS idx_house_quotes_tags ON house_quotes USING gin(tags);",
    "CREATE INDEX IF NOT EXISTS idx_house_quotes_text_search ON house_quotes USING gin(to_tsvector('english', quote_text));",
    
    # House quotes embeddings indexes
    "CREATE INDEX IF NOT EXISTS idx_house_quotes_embeddings_quote_id ON house_quotes_embeddings(quote_id);",
    "CREATE INDEX IF NOT EXISTS idx_house_quotes_embeddings_vector ON house_quotes_embeddings USING ivfflat(embedding vector_cosine_ops) WITH (lists = 100);",
    
    # Session indexes
    "CREATE INDEX IF NOT EXISTS idx_conversation_sessions_session_id ON conversation_sessions(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_conversation_sessions_user_id ON conversation_sessions(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_conversation_sessions_start_time ON conversation_sessions(start_time);",
    
    # Performance indexes
    "CREATE INDEX IF NOT EXISTS idx_model_performance_model_version ON model_performance(model_version);",
    "CREATE INDEX IF NOT EXISTS idx_model_performance_created_at ON model_performance(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_model_performance_response_time ON model_performance(response_time_ms);"
]

# SQL for functions and triggers
CREATE_FUNCTIONS_SQL = [
    """
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = CURRENT_TIMESTAMP;
        RETURN NEW;
    END;
    $$ language 'plpgsql';
    """,
    
    """
    CREATE OR REPLACE FUNCTION calculate_similarity_score(query_embedding vector(384), quote_embedding vector(384))
    RETURNS FLOAT AS $$
    BEGIN
        RETURN 1 - (query_embedding <=> quote_embedding);
    END;
    $$ language 'plpgsql';
    """,
    
    """
    CREATE OR REPLACE FUNCTION search_house_quotes(query_text TEXT, similarity_threshold FLOAT DEFAULT 0.7, max_results INTEGER DEFAULT 5)
    RETURNS TABLE(
        quote_id INTEGER,
        quote_text TEXT,
        context TEXT,
        episode VARCHAR,
        similarity_score FLOAT
    ) AS $$
    BEGIN
        RETURN QUERY
        SELECT 
            hq.id,
            hq.quote_text,
            hq.context,
            hq.episode,
            (1 - (get_query_embedding(query_text) <=> hqe.embedding)) AS sim_score
        FROM house_quotes hq
        JOIN house_quotes_embeddings hqe ON hq.id = hqe.quote_id
        WHERE (1 - (get_query_embedding(query_text) <=> hqe.embedding)) >= similarity_threshold
        ORDER BY sim_score DESC
        LIMIT max_results;
    END;
    $$ language 'plpgsql';
    """
]

# SQL for triggers
CREATE_TRIGGERS_SQL = [
    """
    DROP TRIGGER IF EXISTS update_user_preferences_updated_at ON user_preferences;
    CREATE TRIGGER update_user_preferences_updated_at
        BEFORE UPDATE ON user_preferences
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """,
    
    """
    DROP TRIGGER IF EXISTS update_system_state_updated_at ON system_state;
    CREATE TRIGGER update_system_state_updated_at
        BEFORE UPDATE ON system_state
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """
]

# Initial data for system state
INITIAL_DATA_SQL = [
    """
    INSERT INTO system_state (state_key, state_value) 
    VALUES ('database_version', '{"version": "1.0.0", "created_at": "2025-09-11"}')
    ON CONFLICT (state_key) DO NOTHING;
    """,
    
    """
    INSERT INTO system_state (state_key, state_value)
    VALUES ('model_info', '{"default_model": "google/flan-t5-large", "adapter_path": "housegpt-lora-large"}')
    ON CONFLICT (state_key) DO NOTHING;
    """
]


class DatabaseSetup:
    """Database setup and initialization manager."""
    
    def __init__(self, config: DatabaseConfig):
        """Initialize database setup.
        
        Args:
            config: Database configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def create_database(self) -> bool:
        """Create the HouseGPT database if it doesn't exist.
        
        Returns:
            True if successful, False otherwise
        """
        if not HAS_PSYCOPG2:
            self.logger.error("psycopg2 not available")
            return False
        
        try:
            # Connect to PostgreSQL server (not specific database)
            conn_params = {
                "host": self.config.host,
                "port": self.config.port,
                "user": self.config.username,
                "password": self.config.password,
                "dbname": "postgres"  # Connect to default database
            }
            
            self.logger.info(f"Connecting to PostgreSQL server at {self.config.host}:{self.config.port}")
            
            with psycopg2.connect(**conn_params) as conn:
                conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                
                with conn.cursor() as cursor:
                    # Check if database exists
                    cursor.execute(
                        "SELECT 1 FROM pg_database WHERE datname = %s",
                        (self.config.database,)
                    )
                    
                    if cursor.fetchone():
                        self.logger.info(f"Database '{self.config.database}' already exists")
                        return True
                    
                    # Create database
                    self.logger.info(f"Creating database '{self.config.database}'")
                    cursor.execute(
                        sql.SQL(CREATE_DATABASE_SQL).format(
                            database_name=sql.Identifier(self.config.database)
                        )
                    )
                    
                    self.logger.info(f"✅ Database '{self.config.database}' created successfully")
                    return True
                    
        except Exception as e:
            self.logger.error(f"Failed to create database: {e}")
            return False
    
    def setup_extensions(self) -> bool:
        """Set up required PostgreSQL extensions.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            db_manager = DatabaseManager(self.config)
            
            self.logger.info("Setting up PostgreSQL extensions...")
            
            for extension_sql in CREATE_EXTENSIONS_SQL:
                result = db_manager.execute_query(extension_sql, fetch=False)
                if result is None:
                    self.logger.warning(f"Failed to execute: {extension_sql}")
            
            self.logger.info("✅ Extensions setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup extensions: {e}")
            return False
    
    def create_tables(self) -> bool:
        """Create all required tables.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            db_manager = DatabaseManager(self.config)
            
            self.logger.info("Creating database tables...")
            
            for table_name, table_sql in CREATE_TABLES_SQL.items():
                self.logger.info(f"Creating table: {table_name}")
                result = db_manager.execute_query(table_sql, fetch=False)
                if result is None:
                    self.logger.warning(f"Failed to create table: {table_name}")
            
            self.logger.info("✅ Tables created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create tables: {e}")
            return False
    
    def create_indexes(self) -> bool:
        """Create database indexes for performance.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            db_manager = DatabaseManager(self.config)
            
            self.logger.info("Creating database indexes...")
            
            for index_sql in CREATE_INDEXES_SQL:
                result = db_manager.execute_query(index_sql, fetch=False)
                if result is None:
                    self.logger.warning(f"Failed to create index: {index_sql[:50]}...")
            
            self.logger.info("✅ Indexes created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create indexes: {e}")
            return False
    
    def create_functions(self) -> bool:
        """Create database functions and stored procedures.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            db_manager = DatabaseManager(self.config)
            
            self.logger.info("Creating database functions...")
            
            for function_sql in CREATE_FUNCTIONS_SQL:
                result = db_manager.execute_query(function_sql, fetch=False)
                if result is None:
                    self.logger.warning(f"Failed to create function: {function_sql[:50]}...")
            
            self.logger.info("✅ Functions created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create functions: {e}")
            return False
    
    def create_triggers(self) -> bool:
        """Create database triggers.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            db_manager = DatabaseManager(self.config)
            
            self.logger.info("Creating database triggers...")
            
            for trigger_sql in CREATE_TRIGGERS_SQL:
                result = db_manager.execute_query(trigger_sql, fetch=False)
                if result is None:
                    self.logger.warning(f"Failed to create trigger: {trigger_sql[:50]}...")
            
            self.logger.info("✅ Triggers created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create triggers: {e}")
            return False
    
    def insert_initial_data(self) -> bool:
        """Insert initial system data.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            db_manager = DatabaseManager(self.config)
            
            self.logger.info("Inserting initial data...")
            
            for data_sql in INITIAL_DATA_SQL:
                result = db_manager.execute_query(data_sql, fetch=False)
                if result is None:
                    self.logger.warning(f"Failed to insert data: {data_sql[:50]}...")
            
            self.logger.info("✅ Initial data inserted successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to insert initial data: {e}")
            return False
    
    def setup_complete_database(self) -> bool:
        """Run complete database setup process.
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("🏥 Starting HouseGPT database setup...")
        
        steps = [
            ("Creating database", self.create_database),
            ("Setting up extensions", self.setup_extensions),
            ("Creating tables", self.create_tables),
            ("Creating indexes", self.create_indexes),
            ("Creating functions", self.create_functions),
            ("Creating triggers", self.create_triggers),
            ("Inserting initial data", self.insert_initial_data)
        ]
        
        for step_name, step_function in steps:
            self.logger.info(f"Step: {step_name}")
            if not step_function():
                self.logger.error(f"Failed at step: {step_name}")
                return False
        
        self.logger.info("🎉 Database setup completed successfully!")
        return True
    
    def verify_setup(self) -> bool:
        """Verify database setup is correct.
        
        Returns:
            True if verification successful, False otherwise
        """
        try:
            db_manager = DatabaseManager(self.config)
            
            self.logger.info("Verifying database setup...")
            
            # Check tables exist
            tables_to_check = list(CREATE_TABLES_SQL.keys())
            
            for table in tables_to_check:
                result = db_manager.execute_query(
                    "SELECT 1 FROM information_schema.tables WHERE table_name = %s",
                    (table,)
                )
                
                if not result:
                    self.logger.error(f"Table '{table}' not found")
                    return False
            
            # Check extensions
            result = db_manager.execute_query(
                "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm', 'btree_gin', 'uuid-ossp')"
            )
            
            if not result or len(result) < 2:  # At least vector and pg_trgm should be there
                self.logger.error("Required extensions not found")
                return False
            
            self.logger.info("✅ Database verification passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Database verification failed: {e}")
            return False


def main():
    """Main function for database setup script."""
    parser = argparse.ArgumentParser(description="HouseGPT Database Setup")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Database host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5432,
        help="Database port (default: 5432)"
    )
    parser.add_argument(
        "--database",
        type=str,
        default="housegpt",
        help="Database name (default: housegpt)"
    )
    parser.add_argument(
        "--username",
        type=str,
        default="postgres",
        help="Database username (default: postgres)"
    )
    parser.add_argument(
        "--password",
        type=str,
        help="Database password (will prompt if not provided)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing setup, don't create anything"
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
    
    if not HAS_PSYCOPG2:
        logger.error("psycopg2 not available. Please install: pip install psycopg2-binary")
        return 1
    
    # Get database configuration
    if args.config:
        config = HouseGPTConfig.from_env(args.config)
        db_config = config.database
    else:
        # Get password if not provided
        password = args.password
        if not password:
            import getpass
            password = getpass.getpass("Database password: ")
        
        db_config = DatabaseConfig(
            host=args.host,
            port=args.port,
            database=args.database,
            username=args.username,
            password=password
        )
    
    # Initialize setup
    setup = DatabaseSetup(db_config)
    
    if args.verify_only:
        success = setup.verify_setup()
    else:
        success = setup.setup_complete_database()
        
        if success:
            success = setup.verify_setup()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
