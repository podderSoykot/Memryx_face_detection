#!/usr/bin/env python3
"""
Database setup script for the Multi-Camera Face Recognition System.
This script initializes the SQLite database and creates necessary tables.
"""

import os
import sys
import sqlite3
from config import DATABASE_CONFIG

def create_database():
    """Create the SQLite database file if it doesn't exist."""
    db_config = DATABASE_CONFIG['sqlite']
    db_path = db_config['database_path']
    
    try:
        # Ensure the directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            print(f"Created directory: {db_dir}")
        
        # Create database file (will create if it doesn't exist)
        connection = sqlite3.connect(db_path)
        connection.close()
        
        print(f"SQLite database initialized: {db_path}")
        
    except Exception as e:
        print(f"Error creating database: {e}")
        sys.exit(1)

def create_tables():
    """Create necessary tables for the face recognition system."""
    db_config = DATABASE_CONFIG['sqlite']
    db_path = db_config['database_path']
    
    # SQL statements to create tables (SQLite syntax)
    create_tables_sql = {
        'face_detections': """
            CREATE TABLE IF NOT EXISTS face_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_uuid TEXT NOT NULL,
                camera_id INTEGER NOT NULL,
                camera_name TEXT,
                location TEXT,
                detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_width INTEGER,
                bbox_height INTEGER,
                confidence REAL
            );
        """,
        
        'face_embeddings': """
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_uuid TEXT UNIQUE NOT NULL,
                embedding BLOB NOT NULL,
                embedding_model TEXT,
                created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """,
        
        'face_demographics': """
            CREATE TABLE IF NOT EXISTS face_demographics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_uuid TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                gender_confidence REAL,
                analysis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """,
        
        'system_logs': """
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_level TEXT,
                message TEXT,
                module TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
    }
    
    try:
        # Connect to the SQLite database
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        
        # Create tables
        for table_name, sql in create_tables_sql.items():
            try:
                cursor.execute(sql)
                print(f"Table '{table_name}' created successfully!")
            except Exception as e:
                print(f"Error creating table '{table_name}': {e}")
        
        # Create indexes for better performance
        indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_face_detections_uuid ON face_detections(face_uuid);",
            "CREATE INDEX IF NOT EXISTS idx_face_detections_time ON face_detections(detection_time DESC);",
            "CREATE INDEX IF NOT EXISTS idx_face_detections_camera ON face_detections(camera_id, detection_time DESC);",
            "CREATE INDEX IF NOT EXISTS idx_face_embeddings_uuid ON face_embeddings(face_uuid);",
            "CREATE INDEX IF NOT EXISTS idx_face_demographics_uuid ON face_demographics(face_uuid);",
            "CREATE INDEX IF NOT EXISTS idx_face_demographics_time ON face_demographics(analysis_time DESC);",
            "CREATE INDEX IF NOT EXISTS idx_system_logs_time ON system_logs(timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(log_level);",
        ]
        
        for index_sql in indexes_sql:
            try:
                cursor.execute(index_sql)
                print(f"Index created successfully!")
            except Exception as e:
                print(f"Error creating index: {e}")
        
        # Commit changes
        connection.commit()
        cursor.close()
        connection.close()
        
        print("All tables and indexes created successfully!")
        
    except Exception as e:
        print(f"Error creating tables: {e}")
        sys.exit(1)

def test_connection():
    """Test the database connection."""
    db_config = DATABASE_CONFIG['sqlite']
    db_path = db_config['database_path']
    
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        
        # Test basic query
        cursor.execute("SELECT sqlite_version();")
        version = cursor.fetchone()
        print(f"Database connection successful! SQLite version: {version[0]}")
        
        # Test table creation
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables found: {[table[0] for table in tables]}")
        
        cursor.close()
        connection.close()
        
        return True
        
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

def get_database_info():
    """Get information about the database."""
    db_config = DATABASE_CONFIG['sqlite']
    db_path = db_config['database_path']
    
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        
        # Get table information
        cursor.execute("""
            SELECT name, sql FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name;
        """)
        tables = cursor.fetchall()
        
        print("\nDatabase Schema:")
        print("-" * 50)
        for table_name, table_sql in tables:
            print(f"Table: {table_name}")
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  Rows: {count}")
        
        # Get database file size
        if os.path.exists(db_path):
            size = os.path.getsize(db_path)
            print(f"\nDatabase file size: {size} bytes ({size/1024:.2f} KB)")
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"Error getting database info: {e}")

def clear_database():
    """Clear all data from the database (for development/testing)."""
    db_config = DATABASE_CONFIG['sqlite']
    db_path = db_config['database_path']
    
    response = input("⚠️  This will delete ALL data in the database. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = cursor.fetchall()
        
        # Clear each table
        for table in tables:
            table_name = table[0]
            cursor.execute(f"DELETE FROM {table_name}")
            print(f"Cleared table: {table_name}")
        
        connection.commit()
        cursor.close()
        connection.close()
        
        print("✅ Database cleared successfully!")
        
    except Exception as e:
        print(f"Error clearing database: {e}")

def main():
    """Main function to set up the database."""
    print("=" * 60)
    print("Multi-Camera Face Recognition System - SQLite Database Setup")
    print("=" * 60)
    
    # Display configuration
    db_config = DATABASE_CONFIG['sqlite']
    print(f"Database Path: {db_config['database_path']}")
    print(f"Database Directory: {os.path.dirname(db_config['database_path']) or 'Current directory'}")
    print()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--clear":
            clear_database()
            return
        elif sys.argv[1] == "--info":
            get_database_info()
            return
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python setup_database.py          # Setup database")
            print("  python setup_database.py --clear  # Clear all data")
            print("  python setup_database.py --info   # Show database info")
            print("  python setup_database.py --help   # Show this help")
            return
    
    # Create database
    print("1. Creating database...")
    create_database()
    
    # Create tables
    print("\n2. Creating tables...")
    create_tables()
    
    # Test connection
    print("\n3. Testing connection...")
    if test_connection():
        print("\n✅ SQLite database setup completed successfully!")
        print(f"Database file: {db_config['database_path']}")
        print("\nYou can now run the face recognition system:")
        print("  python app.py")
        print("\nOther commands:")
        print("  python setup_database.py --info   # View database information")
        print("  python setup_database.py --clear  # Clear all data")
    else:
        print("\n❌ Database setup failed!")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()

# EOF 