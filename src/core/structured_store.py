import os
import sqlite3
import json
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/ 的父目录
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # 项目根目录
DB_PATH = os.path.join(PROJECT_ROOT, "./db/app.db")


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS requirements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        req_id TEXT NOT NULL,
        title TEXT,
        description TEXT,
        acceptance TEXT,
        source TEXT,
        dependencies TEXT,
        priority TEXT,
        raw_json TEXT,
        updated_at TEXT DEFAULT (datetime('now'))
    )""")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_req_filename ON requirements(filename)"
    )
    conn.commit()
    return conn


def save_requirements(filename: str, items: List[Dict]) -> int:
    """
    覆盖式保存某文档的所有需求条目。
    items 里每条应含 req_id / title / description / acceptance / source 等。
    返回保存的条目数。
    """
    if not items:
        return 0
    with _conn() as conn:
        conn.execute("DELETE FROM requirements WHERE filename = ?", (filename,))
        for it in items:
            conn.execute(
                """INSERT INTO requirements
                   (filename, req_id, title, description, acceptance,
                    source, dependencies, priority, raw_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    filename,
                    it.get("req_id", ""),
                    it.get("title", ""),
                    it.get("description", ""),
                    it.get("acceptance", ""),
                    it.get("source", ""),
                    json.dumps(it.get("dependencies", []), ensure_ascii=False),
                    it.get("priority", ""),
                    json.dumps(it, ensure_ascii=False),
                ),
            )
    return len(items)


def list_requirements(filename: str) -> List[Dict]:
    """返回某文档的所有需求条目（供前端表格展示）"""
    with _conn() as conn:
        cur = conn.execute(
            """SELECT req_id, title, description, acceptance, source, priority
               FROM requirements WHERE filename = ? ORDER BY req_id""",
            (filename,),
        )
        cols = ["req_id", "title", "description", "acceptance", "source", "priority"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def update_requirement(filename: str, req_id: str, data: Dict) -> bool:
    """更新单条需求（供表格编辑后保存）"""
    try:
        with _conn() as conn:
            conn.execute(
                """UPDATE requirements
                   SET title = ?, description = ?, acceptance = ?, source = ?, priority = ?,
                       updated_at = datetime('now')
                   WHERE filename = ? AND req_id = ?""",
                (
                    data.get("title", ""),
                    data.get("description", ""),
                    data.get("acceptance", ""),
                    data.get("source", ""),
                    data.get("priority", ""),
                    filename,
                    req_id,
                ),
            )
        return True
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"更新需求失败: {e}")
        return False


def delete_requirements(filename: str) -> bool:
    """删除某文档的所有需求（删文档时同步调用）"""
    try:
        with _conn() as conn:
            conn.execute("DELETE FROM requirements WHERE filename = ?", (filename,))
        return True
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"删除需求失败: {e}")
        return False