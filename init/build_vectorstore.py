#!/usr/bin/env python
"""
一键创建向量数据库脚本（独立执行）
"""
import sys
import os

# 将当前目录加入 sys.path，确保可以导入 src 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval.vectorstore import create_vectorstore

if __name__ == "__main__":
    print("开始构建向量数据库...")
    create_vectorstore()
    print("构建完成！")