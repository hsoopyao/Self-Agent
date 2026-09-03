import sqlite3
import os
from typing import Dict, List, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "../db/memory.db")

def get_connection():
    """获取数据库连接，自动创建表"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

def save_memory(key: str, value: str) -> None:
    """保存或更新一条记忆"""
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO memory (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
        (key, value)
    )
    conn.commit()
    conn.close()

def get_memory(key: str) -> Optional[str]:
    """读取一条记忆，不存在返回 None"""
    conn = get_connection()
    cursor = conn.execute("SELECT value FROM memory WHERE key=?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def get_all_memories() -> Dict[str, str]:
    """获取所有记忆，返回键值对字典"""
    conn = get_connection()
    cursor = conn.execute("SELECT key, value FROM memory ORDER BY key")
    rows = cursor.fetchall()
    conn.close()
    return {key: value for key, value in rows}

def delete_memory(key: str) -> None:
    """删除指定键的记忆"""
    conn = get_connection()
    conn.execute("DELETE FROM memory WHERE key=?", (key,))
    conn.commit()
    conn.close()

def clear_all_memories() -> None:
    """清空所有记忆"""
    conn = get_connection()
    conn.execute("DELETE FROM memory")
    conn.commit()
    conn.close()