import logging
import os
import sys

import streamlit as st

from src.core.config import init_config
from src.core.logging_config import setup_logging
from src.pages.chats import chat_page
from src.pages.knowledges import main as knowledge_page
from src.pages.analysis_page import requirement_analysis_page   # ← 新增
from src.ui.sidebar import render_sidebar
from src.ui.ui_components import apply_theme

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
st.set_page_config(page_title="智能助手", layout="wide")

setup_logging()
logger = logging.getLogger(__name__)

init_config()
apply_theme()

# 构建侧边栏（保持原样）
update_token_display = render_sidebar()

# ========== 配置导航 ==========
page_chat = st.Page(chat_page, title="聊天", icon="💬")
page_knowledge = st.Page(knowledge_page, title="知识库", icon="📚")
page_analysis = st.Page(requirement_analysis_page, title="需求分析", icon="📋")
page_settings = st.Page("src/pages/settings.py", title="设置", icon="⚙️")

pg = st.navigation(
    [page_chat, page_knowledge, page_analysis, page_settings],
    position="top",
)

pg.run()