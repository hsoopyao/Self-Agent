import os
import sys
import logging
import streamlit as st

from src.core.config import init_config
from src.pages.knowledges import main as knowledge_page
from src.pages.chats import chat_page
from src.retrieval.vectorstore import get_vectorstore
from src.ui.sidebar import render_sidebar
from src.ui.ui_components import apply_theme
from src.core.logging_config import setup_logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
st.set_page_config(page_title="智能助手", layout="centered")

# 初始化日志
setup_logging()
logger = logging.getLogger(__name__)

# 初始化配置与React卡片主题
init_config()
apply_theme()

# 启动时初始化向量库（后台加载）
get_vectorstore()

# 构建侧边栏
update_token_display = render_sidebar()

# ========== 配置导航 ==========
# 创建页面列表
page_chat = st.Page(chat_page, title="聊天", icon="💬")
page_knowledge = st.Page(knowledge_page, title="知识库", icon="📚")
page_settings = st.Page("src/pages/settings.py", title="设置", icon="⚙️")
page_github = st.Page(st.session_state.github_page, title="GitHub", icon="🐙")

# 创建导航（顶部显示）
pg = st.navigation([page_chat, page_knowledge, page_settings, page_github], position="top")

# 运行当前选中的页面
pg.run()

