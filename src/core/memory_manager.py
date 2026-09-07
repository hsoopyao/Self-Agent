import os
import sqlite3
from typing import List, Dict, Optional

# ---------- 数据库路径 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DB_PATH = os.path.join(PROJECT_ROOT, "./db/memory.db")

def get_connection():
    """获取数据库连接，自动创建 memory 和 categories 表"""
    conn = sqlite3.connect(DB_PATH)
    # 记忆表（键值对）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # 分类表（独立）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            name TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

# ---------- memory 相关函数 ----------
def save_memory(key: str, value: str) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO memory (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
        (key, value)
    )
    conn.commit()
    conn.close()

def get_memory(key: str) -> Optional[str]:
    conn = get_connection()
    cursor = conn.execute("SELECT value FROM memory WHERE key=?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def get_all_memories() -> Dict[str, str]:
    conn = get_connection()
    cursor = conn.execute("SELECT key, value FROM memory ORDER BY key")
    rows = cursor.fetchall()
    conn.close()
    return {key: value for key, value in rows}

def delete_memory(key: str) -> None:
    conn = get_connection()
    conn.execute("DELETE FROM memory WHERE key=?", (key,))
    conn.commit()
    conn.close()

def clear_all_memories() -> None:
    conn = get_connection()
    conn.execute("DELETE FROM memory")
    conn.commit()
    conn.close()

# ---------- 分类相关函数 ----------
def get_categories() -> List[str]:
    """获取所有分类名称（按字母顺序）"""
    conn = get_connection()
    cur = conn.execute("SELECT name FROM categories ORDER BY name")
    rows = cur.fetchall()
    conn.close()
    return [row[0] for row in rows]

def add_category(name: str) -> bool:
    """添加新分类，返回是否成功（名称已存在则失败）"""
    name = name.strip()
    if not name:
        return False
    conn = get_connection()
    try:
        conn.execute("INSERT INTO categories (name) VALUES (?)", (name,))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def delete_category(name: str) -> bool:
    """删除分类（不影响已有文档的元数据）"""
    name = name.strip()
    if not name:
        return False
    conn = get_connection()
    conn.execute("DELETE FROM categories WHERE name = ?", (name,))
    conn.commit()
    conn.close()
    return True