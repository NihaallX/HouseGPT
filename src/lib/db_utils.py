"""Database utilities for HouseGPT NoSQL document storage.

Provides document storage, query execution, and data persistence utilities
for conversation logging, user preferences, and system state using TinyDB.
"""

import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any, Union
from enum import Enum
from pathlib import Path
import json
import uuid

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
    JSONStorage = None
    CachingMiddleware = None


class DatabaseState(Enum):
    """Database connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    database_path: str = "data/housegpt.json"
    backup_enabled: bool = True
    backup_interval: int = 3600  # seconds
    max_backups: int = 5
    cache_size: int = 100
    auto_sync: bool = True
    pretty_print: bool = True


@dataclass
class ConversationRecord:
    """Conversation record for database storage."""
    conversation_id: str
    user_input: str
    house_response: str
    timestamp: str  # ISO format datetime string
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    model_version: Optional[str] = None
    response_time_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create(cls, conversation_id: str, user_input: str, house_response: str, 
               session_id: Optional[str] = None, **kwargs) -> 'ConversationRecord':
        """Create a new conversation record with current timestamp."""
        return cls(
            conversation_id=conversation_id,
            user_input=user_input,
            house_response=house_response,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            **kwargs
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationRecord':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class UserPreference:
    """User preference record."""
    user_id: str
    preference_key: str
    preference_value: Any
    created_at: str
    updated_at: str
    
    @classmethod
    def create(cls, user_id: str, key: str, value: Any) -> 'UserPreference':
        """Create a new user preference with current timestamp."""
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            user_id=user_id,
            preference_key=key,
            preference_value=value,
            created_at=now,
            updated_at=now
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)


@dataclass
class SystemState:
    """System state record."""
    state_key: str
    state_value: Any
    updated_at: str
    
    @classmethod
    def create(cls, key: str, value: Any) -> 'SystemState':
        """Create a new system state with current timestamp."""
        return cls(
            state_key=key,
            state_value=value,
            updated_at=datetime.now(timezone.utc).isoformat()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)


class NoSQLDatabaseManager:
    """TinyDB document database manager."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """Initialize database manager.
        
        Args:
            config: Database configuration, defaults from environment
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or self._load_config_from_env()
        self.db = None  # Optional[TinyDB]
        self.state = DatabaseState.DISCONNECTED
        self._last_error: Optional[str] = None
        
        if not HAS_TINYDB:
            self.logger.warning("TinyDB not available - database features disabled")
            return
        
        # Initialize database connection
        self._connect()
    
    def _load_config_from_env(self) -> DatabaseConfig:
        """Load database configuration from environment variables."""
        return DatabaseConfig(
            database_path=os.getenv("DB_PATH", "data/housegpt.json"),
            backup_enabled=os.getenv("DB_BACKUP_ENABLED", "true").lower() == "true",
            backup_interval=int(os.getenv("DB_BACKUP_INTERVAL", "3600")),
            max_backups=int(os.getenv("DB_MAX_BACKUPS", "5")),
            cache_size=int(os.getenv("DB_CACHE_SIZE", "100")),
            auto_sync=os.getenv("DB_AUTO_SYNC", "true").lower() == "true",
            pretty_print=os.getenv("DB_PRETTY_PRINT", "true").lower() == "true"
        )
    
    def _connect(self) -> None:
        """Connect to database."""
        if not HAS_TINYDB:
            return
        
        try:
            self.state = DatabaseState.CONNECTING
            self.logger.info(f"Connecting to database: {self.config.database_path}")
            
            # Ensure database directory exists
            db_path = Path(self.config.database_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to database directly without middleware for now
            self.db = TinyDB(
                str(db_path),
                sort_keys=True,
                indent=2 if self.config.pretty_print else None
            )
            
            self.state = DatabaseState.CONNECTED
            self.logger.info("Database connected successfully")
            
            # Initialize tables
            self._initialize_tables()
            
        except Exception as e:
            self.state = DatabaseState.ERROR
            self._last_error = str(e)
            self.logger.error(f"Failed to connect to database: {e}")
    
    def _initialize_tables(self) -> None:
        """Initialize database tables (collections)."""
        self.logger.debug(f"Initializing tables, db object: {self.db}")
        self.logger.debug(f"db is None: {self.db is None}")
        
        if self.db is None:
            self.logger.error("Database object is None")
            return
        
        try:
            # Get table references (creates them if they don't exist)
            self.conversations = self.db.table('conversations')
            self.user_preferences = self.db.table('user_preferences')
            self.system_state = self.db.table('system_state')
            
            self.logger.info("Database tables initialized successfully")
            self.logger.debug(f"Tables created: conversations, user_preferences, system_state")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tables: {e}")
            raise
    
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self.state == DatabaseState.CONNECTED and self.db is not None
    
    def save_conversation(self, record: ConversationRecord) -> bool:
        """Save conversation record to database.
        
        Args:
            record: Conversation record to save
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            self.logger.error("Database not connected")
            return False
        
        try:
            doc_id = self.conversations.insert(record.to_dict())
            self.logger.debug(f"Saved conversation record with ID: {doc_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save conversation: {e}")
            return False
    
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
        if not self.is_connected():
            self.logger.error("Database not connected")
            return []
        
        try:
            ConversationQuery = Query()
            query_conditions = []
            
            if session_id:
                query_conditions.append(ConversationQuery.session_id == session_id)
            
            if user_id:
                query_conditions.append(ConversationQuery.user_id == user_id)
            
            if query_conditions:
                # Combine conditions with AND
                combined_query = query_conditions[0]
                for condition in query_conditions[1:]:
                    combined_query = combined_query & condition
                
                results = self.conversations.search(combined_query)
            else:
                results = self.conversations.all()
            
            # Sort by timestamp (most recent first) and limit
            sorted_results = sorted(results, key=lambda x: x.get('timestamp', ''), reverse=True)
            return sorted_results[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get conversation history: {e}")
            return []
    
    def save_user_preference(self, user_id: str, key: str, value: Any) -> bool:
        """Save user preference to database.
        
        Args:
            user_id: User identifier
            key: Preference key
            value: Preference value
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            self.logger.error("Database not connected")
            return False
        
        try:
            UserQuery = Query()
            
            # Check if preference already exists
            existing = self.user_preferences.search(
                (UserQuery.user_id == user_id) & (UserQuery.preference_key == key)
            )
            
            if existing:
                # Update existing preference
                self.user_preferences.update(
                    {
                        'preference_value': value,
                        'updated_at': datetime.now(timezone.utc).isoformat()
                    },
                    (UserQuery.user_id == user_id) & (UserQuery.preference_key == key)
                )
            else:
                # Create new preference
                preference = UserPreference.create(user_id, key, value)
                self.user_preferences.insert(preference.to_dict())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save user preference: {e}")
            return False
    
    def get_user_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """Get user preference from database.
        
        Args:
            user_id: User identifier
            key: Preference key
            default: Default value if not found
            
        Returns:
            Preference value or default
        """
        if not self.is_connected():
            self.logger.error("Database not connected")
            return default
        
        try:
            UserQuery = Query()
            result = self.user_preferences.search(
                (UserQuery.user_id == user_id) & (UserQuery.preference_key == key)
            )
            
            if result:
                return result[0]['preference_value']
            return default
            
        except Exception as e:
            self.logger.error(f"Failed to get user preference: {e}")
            return default
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get all user preferences from database.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary of preferences
        """
        if not self.is_connected():
            self.logger.error("Database not connected")
            return {}
        
        try:
            UserQuery = Query()
            results = self.user_preferences.search(UserQuery.user_id == user_id)
            
            preferences = {}
            for result in results:
                preferences[result['preference_key']] = result['preference_value']
            
            return preferences
            
        except Exception as e:
            self.logger.error(f"Failed to get user preferences: {e}")
            return {}
    
    def save_system_state(self, key: str, value: Any) -> bool:
        """Save system state to database.
        
        Args:
            key: State key
            value: State value
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            self.logger.error("Database not connected")
            return False
        
        try:
            StateQuery = Query()
            
            # Check if state already exists
            existing = self.system_state.search(StateQuery.state_key == key)
            
            if existing:
                # Update existing state
                self.system_state.update(
                    {
                        'state_value': value,
                        'updated_at': datetime.now(timezone.utc).isoformat()
                    },
                    StateQuery.state_key == key
                )
            else:
                # Create new state
                state = SystemState.create(key, value)
                self.system_state.insert(state.to_dict())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save system state: {e}")
            return False
    
    def get_system_state(self, key: str, default: Any = None) -> Any:
        """Get system state from database.
        
        Args:
            key: State key
            default: Default value if not found
            
        Returns:
            State value or default
        """
        if not self.is_connected():
            self.logger.error("Database not connected")
            return default
        
        try:
            StateQuery = Query()
            result = self.system_state.search(StateQuery.state_key == key)
            
            if result:
                return result[0]['state_value']
            return default
            
        except Exception as e:
            self.logger.error(f"Failed to get system state: {e}")
            return default
    
    def cleanup_old_conversations(self, days_old: int = 30) -> int:
        """Clean up old conversation records.
        
        Args:
            days_old: Delete conversations older than this many days
            
        Returns:
            Number of records deleted
        """
        if not self.is_connected():
            self.logger.error("Database not connected")
            return 0
        
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
            cutoff_str = cutoff_date.isoformat()
            
            ConversationQuery = Query()
            deleted_docs = self.conversations.remove(ConversationQuery.timestamp < cutoff_str)
            
            self.logger.info(f"Cleaned up {len(deleted_docs)} old conversation records")
            return len(deleted_docs)
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old conversations: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        stats = {
            "database_state": self.state.value,
            "last_error": self._last_error,
            "database_available": HAS_TINYDB,
            "database_path": self.config.database_path
        }
        
        if not self.is_connected():
            return stats
        
        try:
            stats["conversation_count"] = len(self.conversations)
            stats["preference_count"] = len(self.user_preferences)
            stats["system_state_count"] = len(self.system_state)
            
            # Database file size
            db_path = Path(self.config.database_path)
            if db_path.exists():
                stats["database_size_bytes"] = db_path.stat().st_size
                
        except Exception as e:
            stats["stats_error"] = str(e)
        
        return stats
    
    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """Create database backup.
        
        Args:
            backup_path: Optional custom backup path
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            self.logger.error("Database not connected")
            return False
        
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{self.config.database_path}.backup_{timestamp}"
            
            # Copy database file
            import shutil
            shutil.copy2(self.config.database_path, backup_path)
            
            self.logger.info(f"Database backed up to: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to backup database: {e}")
            return False
    
    def close(self) -> None:
        """Close database connection."""
        if self.db:
            self.db.close()
            self.db = None
            self.state = DatabaseState.DISCONNECTED
            self.logger.info("Database connection closed")


# Utility functions for easier database operations
def get_database_manager(config: Optional[DatabaseConfig] = None) -> NoSQLDatabaseManager:
    """Get database manager instance.
    
    Args:
        config: Optional database configuration
        
    Returns:
        NoSQLDatabaseManager instance
    """
    return NoSQLDatabaseManager(config)


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
    return ConversationRecord.create(
        conversation_id=conversation_id,
        user_input=user_input,
        house_response=house_response,
        session_id=session_id,
        **kwargs
    )


if __name__ == "__main__":
    # Test database utilities
    config = DatabaseConfig(database_path="test_housegpt.json")
    db = NoSQLDatabaseManager(config)
    
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
    
    # Test saving and retrieving
    if db.save_conversation(record):
        print("Conversation saved successfully")
        
        history = db.get_conversation_history(session_id="session-456")
        print(f"Retrieved history: {len(history)} records")
    
    db.close()
