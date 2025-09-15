#!/usr/bin/env python3
"""
Check the current table schema and provide SQL to fix it
"""

from database import engine
from sqlalchemy import text

def check_table_schema():
    """Check the current table schema"""
    try:
        with engine.connect() as connection:
            # Check table structure
            result = connection.execute(text("DESCRIBE embeddings"))
            print("📋 Current embeddings table structure:")
            for row in result:
                print(f"  {row[0]} - {row[1]} - {row[2]} - {row[3]}")
            
            # Check if there are any existing records
            count_result = connection.execute(text("SELECT COUNT(*) FROM embeddings"))
            count = count_result.fetchone()[0]
            print(f"\n📊 Current records in embeddings table: {count}")
            
            # Check vector column specifically
            vector_info = connection.execute(text("""
                SELECT COLUMN_NAME, DATA_TYPE, COLUMN_TYPE 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = 'embeddings' 
                AND COLUMN_NAME = 'vector'
                AND TABLE_SCHEMA = DATABASE()
            """))
            
            vector_row = vector_info.fetchone()
            if vector_row:
                print(f"\n🔍 Vector column info: {vector_row[2]}")
                
                if "1536" in str(vector_row[2]):
                    print("\n⚠️  ISSUE FOUND: Vector column is VECTOR(1536) but we need VECTOR(1024)")
                    print("\n🔧 To fix this, you need to run the following SQL command in your TiDB console:")
                    print("   ALTER TABLE embeddings MODIFY COLUMN vector VECTOR(1024);")
                    print("\n⚠️  WARNING: This will only work if the table is empty or you're okay with data loss!")
                    print("   If you have existing data, you may need to:")
                    print("   1. Backup your data")
                    print("   2. Drop and recreate the table")
                    print("   3. Or create a new table with the correct schema")
                    return False
                elif "1024" in str(vector_row[2]):
                    print("\n✅ Vector column is correctly configured as VECTOR(1024)")
                    return True
                else:
                    print(f"\n❓ Unknown vector column type: {vector_row[2]}")
                    return False
            else:
                print("\n❌ Could not find vector column information")
                return False
                
    except Exception as e:
        print(f"❌ Error checking table schema: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔍 Checking table schema...")
    success = check_table_schema()
    
    if not success:
        print("\n❌ Schema check failed or issues found!")
        print("Please fix the database schema before proceeding.")
    else:
        print("\n✅ Schema check passed!")