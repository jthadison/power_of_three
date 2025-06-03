#!/usr/bin/env python3
"""
Power of 3 Trading System - Database Setup Script
=================================================

This script helps you set up the PostgreSQL database for tracking
trades and performance metrics in your Power of 3 trading system.

Usage:
    python database_setup.py --create-db      # Create database and tables
    python database_setup.py --test          # Test connection
    python database_setup.py --sample-data   # Add sample data
    python database_setup.py --reset         # Reset all data (WARNING!)
"""

import argparse
import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import subprocess
from pathlib import Path
import logging
from dotenv import load_dotenv
# Load environment variables from .env file if it exists
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseSetup:
    """Database setup and management utilities"""
    
    def __init__(self):
        self.host = os.getenv('DB_HOST', os.getenv('DB_HOST'))
        self.port = os.getenv('DB_PORT', '5432')
        self.database = os.getenv('DB_NAME', os.getenv('DB_NAME'))
        self.username = os.getenv('DB_USER', os.getenv('DB_USER'))
        self.password = os.getenv('DB_PASSWORD', os.getenv('DB_PASSWORD'))
        
        self.admin_db = 'postgres'  # For creating database
        
    def test_postgresql_installation(self):
        """Test if PostgreSQL is installed and accessible"""
        try:
            # Try to connect to default postgres database
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.admin_db,
                user=self.username,
                password=self.password
            )
            conn.close()
            logger.info("✓ PostgreSQL connection successful")
            return True
        except psycopg2.Error as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            self._show_postgresql_help()
            return False
    
    def _show_postgresql_help(self):
        """Show help for PostgreSQL installation"""
        print("\n" + "="*60)
        print("📋 POSTGRESQL SETUP HELP")
        print("="*60)
        print("""
🔧 INSTALLATION:

Windows:
1. Download from: https://www.postgresql.org/download/windows/
2. Run installer and follow prompts
3. Remember the password you set for 'postgres' user

macOS:
1. Using Homebrew: brew install postgresql
2. Start service: brew services start postgresql
3. Create user: createuser -s postgres

Linux (Ubuntu/Debian):
1. sudo apt update
2. sudo apt install postgresql postgresql-contrib
3. sudo -u postgres createuser --superuser $USER

🔐 ENVIRONMENT VARIABLES:
Set these in your terminal or .env file:

export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=trading_db
export DB_USER=postgres
export DB_PASSWORD=your_postgres_password

💡 TESTING:
After installation, run: python database_setup.py --test
        """)
    
    def create_database(self):
        """Create the trading database if it doesn't exist"""
        try:
            # Connect to postgres database to create new database
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.admin_db,
                user=self.username,
                password=self.password
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
                (self.database,)
            )
            
            if cursor.fetchone():
                logger.info(f"✓ Database '{self.database}' already exists")
            else:
                # Create database
                cursor.execute(f'CREATE DATABASE "{self.database}"')
                logger.info(f"✓ Database '{self.database}' created successfully")
            
            cursor.close()
            conn.close()
            return True
            
        except psycopg2.Error as e:
            logger.error(f"❌ Error creating database: {e}")
            return False
    
    def create_tables(self):
        """Create all tables using the schema"""
        try:
            # Read schema file
            schema_file = Path(__file__).parent / "postgres_trading_schema.sql"
            
            if not schema_file.exists():
                logger.error("❌ Schema file 'postgres_trading_schema.sql' not found")
                logger.info("💡 Please save the PostgreSQL schema artifact as 'postgres_trading_schema.sql'")
                return False
            
            # Connect to our trading database
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password
            )
            cursor = conn.cursor()
            
            # Execute schema
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
            
            cursor.execute(schema_sql)
            conn.commit()
            
            cursor.close()
            conn.close()
            
            logger.info("✓ Database tables created successfully")
            return True
            
        except psycopg2.Error as e:
            logger.error(f"❌ Error creating tables: {e}")
            return False
        except FileNotFoundError:
            logger.error("❌ Schema file not found")
            return False
    
    def test_connection(self):
        """Test connection to trading database"""
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password
            )
            cursor = conn.cursor()
            
            # Test query
            cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
            result = cursor.fetchone()
            table_count = result[0] if result else 0
            
            cursor.close()
            conn.close()
            
            logger.info(f"✓ Connection successful! Found {table_count} tables")
            return True
            
        except psycopg2.Error as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False
    
    def show_table_status(self):
        """Show status of all tables"""
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password
            )
            cursor = conn.cursor()
            
            # Get table information
            cursor.execute("""
                SELECT table_name, 
                       (SELECT COUNT(*) FROM information_schema.columns 
                        WHERE table_name = t.table_name) as column_count
                FROM information_schema.tables t
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            
            tables = cursor.fetchall()
            
            print("\n📊 DATABASE TABLE STATUS:")
            print("-" * 40)
            for table_name, column_count in tables:
                print(f"✓ {table_name}: {column_count} columns")
            
            # Get view information
            cursor.execute("""
                SELECT table_name FROM information_schema.views 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            
            views = cursor.fetchall()
            
            if views:
                print("\n📈 DATABASE VIEWS:")
                print("-" * 40)
                for (view_name,) in views:
                    print(f"✓ {view_name}")
            
            cursor.close()
            conn.close()
            
        except psycopg2.Error as e:
            logger.error(f"❌ Error getting table status: {e}")
    
    def add_sample_data(self):
        """Add sample trading data for testing"""
        try:
            # Import database integration
            from database_integration import TradingDatabase, create_sample_data
            
            db = TradingDatabase()
            create_sample_data(db)
            db.close()
            
            logger.info("✓ Sample data added successfully")
            return True
            
        except ImportError:
            logger.error("❌ Database integration module not found")
            logger.info("💡 Please save the database integration artifact as 'database_integration.py'")
            return False
        except Exception as e:
            logger.error(f"❌ Error adding sample data: {e}")
            return False
    
    def reset_database(self):
        """Reset all data (WARNING: Destructive operation!)"""
        print("\n⚠️  WARNING: This will delete ALL trading data!")
        confirm = input("Type 'RESET' to confirm: ").strip()
        
        if confirm != 'RESET':
            print("❌ Reset cancelled")
            return False
        
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password
            )
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            
            # Truncate all tables
            for table in tables:
                cursor.execute(f'TRUNCATE TABLE "{table}" RESTART IDENTITY CASCADE')
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("✓ Database reset completed")
            return True
            
        except psycopg2.Error as e:
            logger.error(f"❌ Error resetting database: {e}")
            return False
    
    def show_configuration(self):
        """Show current database configuration"""
        print("\n🔧 DATABASE CONFIGURATION:")
        print("-" * 40)
        print(f"Host: {self.host}")
        print(f"Port: {self.port}")
        print(f"Database: {self.database}")
        print(f"Username: {self.username}")
        print(f"Password: {'*' * len(self.password) if self.password else 'Not set'}")
        print()
        
        # Check environment variables
        env_vars = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
        print("Environment Variables:")
        for var in env_vars:
            value = os.getenv(var)
            status = "✓ SET" if value else "❌ NOT SET"
            print(f"  {var}: {status}")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(
        description="Power of 3 Trading Database Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python database_setup.py --create-db     # Full setup
  python database_setup.py --test         # Test connection
  python database_setup.py --config       # Show configuration
        """
    )
    
    parser.add_argument('--create-db', action='store_true', 
                       help='Create database and tables')
    parser.add_argument('--test', action='store_true',
                       help='Test database connection')
    parser.add_argument('--sample-data', action='store_true',
                       help='Add sample trading data')
    parser.add_argument('--reset', action='store_true',
                       help='Reset all data (WARNING: Destructive!)')
    parser.add_argument('--config', action='store_true',
                       help='Show database configuration')
    parser.add_argument('--status', action='store_true',
                       help='Show table status')
    
    args = parser.parse_args()
    
    # Show help if no arguments
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    setup = DatabaseSetup()
    
    if args.config:
        setup.show_configuration()
        return
    
    if args.test:
        print("🧪 TESTING DATABASE CONNECTION")
        print("=" * 40)
        if setup.test_postgresql_installation():
            if setup.test_connection():
                setup.show_table_status()
        return
    
    if args.create_db:
        print("🏗️  CREATING TRADING DATABASE")
        print("=" * 40)
        
        # Step 1: Test PostgreSQL
        if not setup.test_postgresql_installation():
            sys.exit(1)
        
        # Step 2: Create database
        if not setup.create_database():
            sys.exit(1)
        
        # Step 3: Create tables
        if not setup.create_tables():
            sys.exit(1)
        
        # Step 4: Test final connection
        if setup.test_connection():
            print("\n🎉 DATABASE SETUP COMPLETE!")
            print("Your Power of 3 trading database is ready to use.")
            setup.show_table_status()
        else:
            sys.exit(1)
    
    if args.sample_data:
        print("📊 ADDING SAMPLE DATA")
        print("=" * 40)
        setup.add_sample_data()
    
    if args.reset:
        print("🗑️  RESETTING DATABASE")
        print("=" * 40)
        setup.reset_database()
    
    if args.status:
        setup.show_table_status()

if __name__ == "__main__":
    main()