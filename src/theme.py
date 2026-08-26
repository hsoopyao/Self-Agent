"""页面主题与 ReAct 过程卡片样式。"""

import streamlit as st


THEME_KEY = "ui_theme"
GITHUB_URL = "https://github.com/hsoopyao/self-agent"


THEMES = {
    "light": {
        "app_bg": "#ffffff",
        "sidebar_bg": "#f7f7f8",
        "surface": "#ffffff",
        "surface_alt": "#f8fafc",
        "input_bg": "#ffffff",
        "text": "#111827",
        "muted": "#4b5563",
        "border": "#d1d5db",
        "link": "#2563eb",
        "thought_bg": "#f3f4f6",
        "thought_text": "#374151",
        "thought_border": "#9ca3af",
        "action_bg": "#e6f2ff",
        "action_text": "#123b5d",
        "action_border": "#1976d2",
        "observation_bg": "#eaf7ee",
        "observation_text": "#174a2e",
        "observation_border": "#388e3c",
    },
    "dark": {
        "app_bg": "#0e1117",
        "sidebar_bg": "#161b22",
        "surface": "#1f2937",
        "surface_alt": "#151b23",
        "input_bg": "#111827",
        "text": "#f3f4f6",
        "muted": "#cbd5e1",
        "border": "#374151",
        "link": "#93c5fd",
        "thought_bg": "#272d38",
        "thought_text": "#f3f4f6",
        "thought_border": "#9ca3af",
        "action_bg": "#102a43",
        "action_text": "#dbeafe",
        "action_border": "#60a5fa",
        "observation_bg": "#123429",
        "observation_text": "#dcfce7",
        "observation_border": "#4ade80",
    },
}


def initialize_theme() -> None:
    """为当前会话初始化主题。"""
    if THEME_KEY not in st.session_state:
        st.session_state[THEME_KEY] = "light"


def get_theme() -> str:
    """返回当前主题名称。"""
    return st.session_state.get(THEME_KEY, "light")


def toggle_theme() -> None:
    """在浅色和深色主题之间切换。"""
    st.session_state[THEME_KEY] = "dark" if get_theme() == "light" else "light"


def apply_theme() -> None:
    """将当前主题样式注入 Streamlit 页面。"""
    colors = THEMES[get_theme()]
    st.markdown(
        f"""
        <style>
        :root {{
            --app-bg: {colors['app_bg']};
            --sidebar-bg: {colors['sidebar_bg']};
            --surface: {colors['surface']};
            --surface-alt: {colors['surface_alt']};
            --input-bg: {colors['input_bg']};
            --text-primary: {colors['text']};
            --text-muted: {colors['muted']};
            --ui-border: {colors['border']};
            --ui-link: {colors['link']};
        }}

        .stApp,
        [data-testid="stAppViewContainer"] {{
            background-color: var(--app-bg) !important;
            color: var(--text-primary) !important;
        }}

        [data-testid="stHeader"] {{
            background-color: color-mix(in srgb, var(--app-bg) 92%, transparent) !important;
        }}

        [data-testid="stSidebar"] > div:first-child {{
            background-color: var(--sidebar-bg) !important;
            border-right: 1px solid var(--ui-border);
        }}

        .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        .stApp p, .stApp label,
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stCaptionContainer"] {{
            color: var(--text-primary) !important;
        }}

        .stApp a, [data-testid="stSidebar"] a {{
            color: var(--ui-link);
        }}

        [data-testid="stChatMessage"] {{
            background-color: var(--surface-alt);
            border: 1px solid var(--ui-border);
            border-radius: 0.75rem;
            padding: 0.35rem 0.75rem;
        }}

        [data-testid="stChatInput"],
        [data-testid="stExpander"],
        [data-testid="stFileUploaderDropzone"] {{
            background-color: var(--surface) !important;
            border-color: var(--ui-border) !important;
        }}

        .stApp input, .stApp textarea,
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {{
            background-color: var(--input-bg) !important;
            color: var(--text-primary) !important;
            -webkit-text-fill-color: var(--text-primary) !important;
        }}

        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div {{
            background-color: var(--input-bg) !important;
            color: var(--text-primary) !important;
            border-color: var(--ui-border) !important;
        }}

        [data-testid="stBaseButton-secondary"],
        a[data-testid="stBaseButton-secondary"] {{
            background-color: var(--surface) !important;
            color: var(--text-primary) !important;
            border-color: var(--ui-border) !important;
        }}

        hr {{
            border-color: var(--ui-border) !important;
        }}

        .react-card {{
            padding: 0.65rem 0.85rem;
            margin: 0.4rem 0;
            border-left: 4px solid;
            border-radius: 0.4rem;
            overflow-wrap: anywhere;
        }}

        .react-thought {{
            background-color: {colors['thought_bg']};
            border-left-color: {colors['thought_border']};
            color: {colors['thought_text']} !important;
            font-size: 0.9rem;
            font-style: italic;
        }}

        .react-action {{
            background-color: {colors['action_bg']};
            border-left-color: {colors['action_border']};
            color: {colors['action_text']} !important;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.9rem;
        }}

        .react-observation {{
            background-color: {colors['observation_bg']};
            border-left-color: {colors['observation_border']};
            color: {colors['observation_text']} !important;
            font-size: 0.95rem;
        }}

        .react-card a {{
            color: {colors['link']} !important;
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
