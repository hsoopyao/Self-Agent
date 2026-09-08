import os
import sqlite3
from typing import List, Optional


# ---------- 数据库路径 ----------
# 默认放在项目根目录的 data/ 下，可自行调整
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/ 的父目录
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # 项目根目录
DB_PATH = os.path.join(PROJECT_ROOT, "./db/categories.db")

def get_connection():
    """获取数据库连接，自动创建 categories 表"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            name TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

def get_categories() -> List[str]:
    """获取所有分类名称（按字母顺序）"""
    conn = get_connection()
    cur = conn.execute("SELECT name FROM categories ORDER BY name")
    rows = cur.fetchall()
    conn.close()
    return [row[0] for row in rows]

def add_category(name: str) -> bool:
    """
    添加新分类，若已存在则返回 False。
    name 会自动去除首尾空格。
    """
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
    """删除指定分类（不会影响已有文档）"""
    name = name.strip()
    if not name:
        return False
    conn = get_connection()
    conn.execute("DELETE FROM categories WHERE name = ?", (name,))
    conn.commit()
    conn.close()
    return True

def rename_category(old_name: str, new_name: str) -> bool:
    """
    重命名分类（更新分类名称）。
    注意：仅更新分类表，不会自动更新已存在文档的元数据（但文档中的分类名仍然有效，
    只是下次显示时会作为新的分类出现）。
    """
    old_name = old_name.strip()
    new_name = new_name.strip()
    if not old_name or not new_name:
        return False
    if old_name == new_name:
        return True  # 无需更改
    conn = get_connection()
    try:
        # 先检查新名称是否已存在
        cur = conn.execute("SELECT 1 FROM categories WHERE name = ?", (new_name,))
        if cur.fetchone():
            conn.close()
            return False
        # 更新分类名
        conn.execute("UPDATE categories SET name = ? WHERE name = ?", (new_name, old_name))
        conn.commit()
        conn.close()
        return True
    except Exception:
        conn.close()
        return False

def category_exists(name: str) -> bool:
    """检查分类是否存在"""
    name = name.strip()
    if not name:
        return False
    conn = get_connection()
    cur = conn.execute("SELECT 1 FROM categories WHERE name = ?", (name,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists

def clear_all_categories() -> None:
    """清空所有分类（慎用）"""
    conn = get_connection()
    conn.execute("DELETE FROM categories")
    conn.commit()
    conn.close()