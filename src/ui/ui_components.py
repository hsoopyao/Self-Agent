import streamlit as st
import re

def apply_theme():
    """注入基础卡片样式（不包含主题切换）"""
    st.markdown(
        """
        <style>
        .react-card {
            padding: 0.65rem 0.85rem;
            margin: 0.4rem 0;
            border-left: 4px solid;
            border-radius: 0.4rem;
            overflow-wrap: anywhere;
        }
        .react-thought {
            background-color: #f3f4f6;
            border-left-color: #9ca3af;
            color: #374151;
            font-size: 0.9rem;
            font-style: italic;
        }
        .react-action {
            background-color: #e6f2ff;
            border-left-color: #1976d2;
            color: #123b5d;
            font-family: ui-monospace, monospace;
            font-size: 0.9rem;
        }
        .react-observation {
            background-color: #eaf7ee;
            border-left-color: #388e3c;
            color: #174a2e;
            font-size: 0.95rem;
        }
        .react-card a {
            color: #2563eb !important;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def clean_markdown(text: str) -> str:
    """移除常见的 Markdown 标记（粗体、斜体、标题、列表等）"""
    # 移除粗体/斜体 ** 或 __
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    # 移除标题 #
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # 移除列表符号 - * + 或数字.
    text = re.sub(r'^[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    # 移除代码块标记（如果有）
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # 移除链接 [text](url) 仅保留文本（但此处我们不想破坏出处的链接，所以不处理）
    # 移除多余空行
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()

def render_thought(content: str) -> str:
    return f'<div class="react-card react-thought">{content}</div>'

def render_action(content: str) -> str:
    return f'<div class="react-card react-action">{content}</div>'

def render_observation(content: str) -> str:
    return f'<div class="react-card react-observation">{content}</div>'