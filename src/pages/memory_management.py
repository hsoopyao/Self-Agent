import streamlit as st
from src.core.memory_manager import get_all_memories, delete_memory, clear_all_memories

def memory_management_page():
    st.title("🧠 记忆管理")
    st.markdown("这里保存了您告诉我的个人信息，如常用城市、常用影院等。")

    memories = get_all_memories()
    if not memories:
        st.info("当前没有任何记忆。")
        return

    st.write(f"共 {len(memories)} 条记忆：")
    for key, value in memories.items():
        col1, col2, col3 = st.columns([2, 3, 1])
        col1.write(f"**{key}**")
        col2.write(value)
        if col3.button("删除", key=f"del_{key}"):
            delete_memory(key)
            st.rerun()

    if st.button("🗑️ 清空所有记忆", type="secondary"):
        clear_all_memories()
        st.rerun()