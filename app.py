import logging
import os
import sys

import streamlit as st

from src.core.config import init_config
from src.core.logging_config import setup_logging
from src.pages.chats import chat_page
from src.pages.knowledges import main as knowledge_page
from src.ui.sidebar import render_sidebar
from src.ui.ui_components import apply_theme

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
st.set_page_config(page_title="智能助手", layout="centered")

# 初始化日志
setup_logging()
logger = logging.getLogger(__name__)

# 初始化配置与React卡片主题
init_config()
apply_theme()

# 构建侧边栏
update_token_display = render_sidebar()

# ========== 配置导航 ==========
# 创建页面列表
page_chat = st.Page(chat_page, title="聊天", icon="💬")
page_knowledge = st.Page(knowledge_page, title="知识库", icon="📚")
page_settings = st.Page("src/pages/settings.py", title="设置", icon="⚙️")

# 创建导航（顶部显示）
pg = st.navigation([page_chat, page_knowledge, page_settings], position="top")

# 运行当前选中的页面
pg.run()
