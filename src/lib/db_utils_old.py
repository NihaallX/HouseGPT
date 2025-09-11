"""Database utilities for HouseGPT NoSQL document storage.

Provides document storage, query execution, and data persistence utilities
for conversation logging, user preferences, and system state using TinyDB.
"""

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, List, Any, Generator, Tuple
from enum import Enum
from pathlib import Path
import json

try:
    from tinydb import TinyDB, Query, where
    from tinydb.storages import JSONStorage
    from tinydb.middlewares import CachingMiddleware
    HAS_TINYDB = True
except ImportError:
    HAS_TINYDB = False
    TinyDB = None
    Query = None
    where = None


class ConnectionState(Enum):
    """Database connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = "localhost"
    port: int = 5432
    database: str = "housegpt"
    username: str = "postgres"
    password: str = ""
    min_connections: int = 2
    max_connections: int = 10
    connection_timeout: int = 30
    query_timeout: int = 60
    ssl_mode: str = "prefer"
    application_name: str = "HouseGPT"


@dataclass
class ConversationRecord:
    """Conversation record for database storage."""
    conversation_id: str
    user_input: str
    house_response: str
    timestamp: datetime
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    model_version: Optional[str] = None
    response_time_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class UserPreference:
    """User preference record."""
    user_id: str
    preference_key: str
    preference_value: Any
    created_at: datetime
    updated_at: datetime


class DatabaseManager:
    """PostgreSQL database connection and query manager."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """Initialize database manager.
        
        Args:
            config: Database configuration, defaults from environment
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or self._load_config_from_env()
        self.connection_pool = None  # Optional[pool.ThreadedConnectionPool]
        self.state = ConnectionState.DISCONNECTED
        self._last_error: Optional[str] = None
        
        if not HAS_PSYCOPG2:
            self.logger.warning("psycopg2 not available - database features disabled")
            return
        
        # Initialize connection pool
        self._create_connection_pool()
    
    def _load_config_from_env(self) -> DatabaseConfig:
        """Load database configuration from environment variables."""
        return DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "housegpt"),
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            min_connections=int(os.getenv("DB_MIN_CONNECTIONS", "2")),
            max_connections=int(os.getenv("DB_MAX_CONNECTIONS", "10")),
            connection_timeout=int(os.getenv("DB_CONNECTION_TIMEOUT", "30")),
            query_timeout=int(os.getenv("DB_QUERY_TIMEOUT", "60")),
            ssl_mode=os.getenv("DB_SSL_MODE", "prefer"),
            application_name=os.getenv("DB_APP_NAME", "HouseGPT")
        )
    
    def _create_connection_pool(self) -> None:
        """Create database connection pool."""
        if not HAS_PSYCOPG2:
            return
        
        try:
            self.state = ConnectionState.CONNECTING
            self.logger.info(f"Creating connection pool to {self.config.host}:{self.config.port}")
            
            # Build connection string
            connection_params = {
                "host": self.config.host,
                "port": self.config.port,
                "database": self.config.database,
                "user": self.config.username,
                "password": self.config.password,
                "sslmode": self.config.ssl_mode,
                "application_name": self.config.application_name,
                "connect_timeout": self.config.connection_timeout
            }
            
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.config.min_connections,
                maxconn=self.config.max_connections,
                **connection_params
            )
            
            # Test connection
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
            
            self.state = ConnectionState.CONNECTED
            self.logger.info("Database connection pool created successfully")
            
        except Exception as e:
            self.state = ConnectionState.ERROR
            self._last_error = str(e)
            self.logger.error(f"Failed to create connection pool: {e}")
    
    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """Get database connection from pool.
        
        Yields:
            Database connection
            
        Raises:
            RuntimeError: If connection pool not available
        """
        if not HAS_PSYCOPG2 or not self.connection_pool:
            raise RuntimeError("Database connection not available")
        
        connection = None
        try:
            connection = self.connection_pool.getconn()
            yield connection
        finally:
            if connection:
                self.connection_pool.putconn(connection)
    
    def execute_query(self, query: str, params: Optional[Tuple] = None, 
                     fetch: bool = True) -> Optional[List[Dict[str, Any]]]:
        """Execute database query.
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch: Whether to fetch results
            
        Returns:
            Query results if fetch=True, None otherwise
        """
        if not HAS_PSYCOPG2:
            self.logger.warning("Cannot execute query - psycopg2 not available")
            return None
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, params)
                    
                    if fetch:
                        results = cursor.fetchall()
                        return [dict(row) for row in results]
                    else:
                        conn.commit()
                        return None
                        
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            self.logger.debug(f"Query: {query}, Params: {params}")
            return None
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> bool:
        """Execute query with multiple parameter sets.
        
        Args:
            query: SQL query string
            params_list: List of parameter tuples
            
        Returns:
            True if successful, False otherwise
        """
        if not HAS_PSYCOPG2:
            self.logger.warning("Cannot execute query - psycopg2 not available")
            return False
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.executemany(query, params_list)
                    conn.commit()
                    return True
                    
        except Exception as e:
            self.logger.error(f"Batch query execution failed: {e}")
            return False
    
    def create_tables(self) -> bool:
        """Create required database tables.
        
        Returns:
            True if successful, False otherwise
        """
        if not HAS_PSYCOPG2:
            self.logger.warning("Cannot create tables - psycopg2 not available")
            return False
        
        # SQL for creating tables
        create_conversations_table = """
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
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            INDEX(conversation_id),
            INDEX(session_id),
            INDEX(user_id),
            INDEX(timestamp)
        );
        """
        
        create_user_preferences_table = """
        CREATE TABLE IF NOT EXISTS user_preferences (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            preference_key VARCHAR(255) NOT NULL,
            preference_value JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, preference_key)
        );
        """
        
        create_system_state_table = """
        CREATE TABLE IF NOT EXISTS system_state (
            id SERIAL PRIMARY KEY,
            state_key VARCHAR(255) UNIQUE NOT NULL,
            state_value JSONB NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            # Execute table creation queries
            for table_sql in [create_conversations_table, create_user_preferences_table, create_system_state_table]:
                self.execute_query(table_sql, fetch=False)
            
            self.logger.info("Database tables created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create tables: {e}")
            return False
    
    def save_conversation(self, record: ConversationRecord) -> bool:
        """Save conversation record to database.
        
        Args:
            record: Conversation record to save
            
        Returns:
            True if successful, False otherwise
        """
        query = """
        INSERT INTO conversations 
        (conversation_id, user_input, house_response, timestamp, session_id, 
         user_id, model_version, response_time_ms, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        params = (
            record.conversation_id,
            record.user_input,
            record.house_response,
            record.timestamp,
            record.session_id,
            record.user_id,
            record.model_version,
            record.response_time_ms,
            Json(record.metadata) if record.metadata else None
        )
        
        result = self.execute_query(query, params, fetch=False)
        return result is not None
    
    def get_conversation_history(self, session_id: Optional[str] = None,
                               user_id: Optional[str] = None,
                               limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history from database.
        
        Args:
            session_id: Filter by session ID
            user_id: Filter by user ID
            limit: Maximum number of records
            
        Returns:
            List of conversation records
        """
        query = "SELECT * FROM conversations WHERE 1=1"
        params = []
        
        if session_id:
            query += " AND session_id = %s"
            params.append(session_id)
        
        if user_id:
            query += " AND user_id = %s"
            params.append(user_id)
        
        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)
        
        result = self.execute_query(query, tuple(params))
        return result or []
    
    def save_user_preference(self, user_id: str, key: str, value: Any) -> bool:
        """Save user preference to database.
        
        Args:
            user_id: User identifier
            key: Preference key
            value: Preference value
            
        Returns:
            True if successful, False otherwise
        """
        query = """
        INSERT INTO user_preferences (user_id, preference_key, preference_value, updated_at)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (user_id, preference_key)
        DO UPDATE SET preference_value = EXCLUDED.preference_value, 
                     updated_at = CURRENT_TIMESTAMP
        """
        
        params = (user_id, key, Json(value))
        result = self.execute_query(query, params, fetch=False)
        return result is not None
    
    def get_user_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """Get user preference from database.
        
        Args:
            user_id: User identifier
            key: Preference key
            default: Default value if not found
            
        Returns:
            Preference value or default
        """
        query = """
        SELECT preference_value FROM user_preferences 
        WHERE user_id = %s AND preference_key = %s
        """
        
        result = self.execute_query(query, (user_id, key))
        if result and len(result) > 0:
            return result[0]['preference_value']
        return default
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get all user preferences from database.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary of preferences
        """
        query = """
        SELECT preference_key, preference_value FROM user_preferences 
        WHERE user_id = %s
        """
        
        result = self.execute_query(query, (user_id,))
        if result:
            return {row['preference_key']: row['preference_value'] for row in result}
        return {}
    
    def save_system_state(self, key: str, value: Any) -> bool:
        """Save system state to database.
        
        Args:
            key: State key
            value: State value
            
        Returns:
            True if successful, False otherwise
        """
        query = """
        INSERT INTO system_state (state_key, state_value, updated_at)
        VALUES (%s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (state_key)
        DO UPDATE SET state_value = EXCLUDED.state_value, 
                     updated_at = CURRENT_TIMESTAMP
        """
        
        params = (key, Json(value))
        result = self.execute_query(query, params, fetch=False)
        return result is not None
    
    def get_system_state(self, key: str, default: Any = None) -> Any:
        """Get system state from database.
        
        Args:
            key: State key
            default: Default value if not found
            
        Returns:
            State value or default
        """
        query = "SELECT state_value FROM system_state WHERE state_key = %s"
        
        result = self.execute_query(query, (key,))
        if result and len(result) > 0:
            return result[0]['state_value']
        return default
    
    def cleanup_old_conversations(self, days_old: int = 30) -> int:
        """Clean up old conversation records.
        
        Args:
            days_old: Delete conversations older than this many days
            
        Returns:
            Number of records deleted
        """
        query = """
        DELETE FROM conversations 
        WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '%s days'
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (days_old,))
                    deleted_count = cursor.rowcount
                    conn.commit()
                    self.logger.info(f"Cleaned up {deleted_count} old conversation records")
                    return deleted_count
        except Exception as e:
            self.logger.error(f"Failed to cleanup old conversations: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        stats = {
            "connection_state": self.state.value,
            "last_error": self._last_error,
            "pool_available": False
        }
        
        if not HAS_PSYCOPG2:
            return stats
        
        if self.connection_pool:
            stats["pool_available"] = True
            stats["pool_size"] = self.connection_pool.minconn
            stats["pool_max"] = self.connection_pool.maxconn
        
        # Get table stats
        try:
            conversations_count = self.execute_query("SELECT COUNT(*) as count FROM conversations")
            if conversations_count:
                stats["conversation_count"] = conversations_count[0]["count"]
            
            preferences_count = self.execute_query("SELECT COUNT(*) as count FROM user_preferences")
            if preferences_count:
                stats["preference_count"] = preferences_count[0]["count"]
                
        except Exception as e:
            stats["stats_error"] = str(e)
        
        return stats
    
    def close(self) -> None:
        """Close database connection pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            self.connection_pool = None
            self.state = ConnectionState.DISCONNECTED
            self.logger.info("Database connection pool closed")


# Utility functions for easier database operations
def get_database_manager(config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """Get database manager instance.
    
    Args:
        config: Optional database configuration
        
    Returns:
        DatabaseManager instance
    """
    return DatabaseManager(config)


def create_conversation_record(conversation_id: str, user_input: str, 
                             house_response: str, session_id: Optional[str] = None,
                             **kwargs) -> ConversationRecord:
    """Create conversation record for database storage.
    
    Args:
        conversation_id: Unique conversation identifier
        user_input: User's input text
        house_response: House's response text
        session_id: Optional session identifier
        **kwargs: Additional fields for the record
        
    Returns:
        ConversationRecord instance
    """
    return ConversationRecord(
        conversation_id=conversation_id,
        user_input=user_input,
        house_response=house_response,
        timestamp=datetime.now(),
        session_id=session_id,
        **kwargs
    )


if __name__ == "__main__":
    # Test database utilities
    config = DatabaseConfig(database="housegpt_test")
    db = DatabaseManager(config)
    
    print(f"Database state: {db.state}")
    print(f"Database stats: {db.get_stats()}")
    
    # Test conversation record creation
    record = create_conversation_record(
        conversation_id="test-123",
        user_input="What's wrong with me?",
        house_response="Probably everything. More specifically?",
        session_id="session-456"
    )
    
    print(f"Created record: {record}")
