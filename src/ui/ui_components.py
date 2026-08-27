import streamlit as st

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

def render_thought(content: str) -> str:
    return f'<div class="react-card react-thought">{content}</div>'

def render_action(content: str) -> str:
    return f'<div class="react-card react-action">{content}</div>'

def render_observation(content: str) -> str:
    return f'<div class="react-card react-observation">{content}</div>'