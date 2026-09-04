import json
import os
import re
import streamlit as st
import logging, traceback
from typing import Generator, List, Dict, Tuple
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.retrieval.vectorstore import search_with_score
from src.chat.direct_chat import direct_chat_sync
from src.core.llm_client import get_llm
from src.core.config import load_prompt
from src.chat.general_chat import search_results
from src.core.memory_manager import get_memory, save_memory

logger = logging.getLogger(__name__)

# ---------- 辅助函数：提取摘要 ----------
def _summarize_text(text: str, max_sentences: int = 2) -> str:
    """取前 max_sentences 句话作为摘要（按句号、问号、感叹号分隔）"""
    if not text:
        return ""
    # 按中文/英文标点切分句子
    sentences = re.split(r'[。！？.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    summary = "。".join(sentences[:max_sentences])
    if len(sentences) > max_sentences:
        summary += "……"
    return summary

# ---------- 工具函数 ----------
def execute_rag(query: str) -> Tuple[str, str, str]:
    """执行内部知识检索，返回 (摘要, 出处, 完整内容)"""
    has_match, docs, score = search_with_score(query, k=4, score_threshold=0.0)
    if not has_match or not docs:
        return "未找到相关信息", "", ""

    first_doc = docs[0]
    full_text = first_doc.page_content
    # 提取与查询相关的片段作为摘要
    summary = extract_relevant_snippets(full_text, query, max_sentences=3)
    source = first_doc.metadata.get("filename", "未知文档")
    full_content = "\n---\n".join([doc.page_content for doc in docs[:3]])
    return summary, f"📄 {source}", full_content


def execute_web(query: str) -> Tuple[str, str, str]:
    """执行联网搜索，返回 (摘要, 出处文本（含链接）, 完整内容)"""
    try:
        results = search_results(query)
        if not results:
            return "未搜索到相关信息", "", ""

        first = results[0]
        title = first.get("title", "未知标题")
        content = first.get("content", "")
        url = first.get("url", "")

        # 摘要：提取与查询相关的片段
        summary = extract_relevant_snippets(content, query, max_sentences=3)
        # 出处：生成 HTML 可点击链接（如果 URL 存在）
        if url:
            # 注意转义 title 中的特殊字符，避免破坏 HTML
            safe_title = title.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
            source = f'<a href="{url}" target="_blank">{safe_title}</a>'
        else:
            source = title

        full_parts = []
        for i, res in enumerate(results[:5], 1):
            full_parts.append(f"【结果 {i}】{res.get('title', '')}\n{res.get('content', '')}\n来源：{res.get('url', '')}")
        full_content = "\n\n".join(full_parts)
        return summary, source, full_content
    except Exception as e:
        logger.error(f"execute_web 出错: {e}")
        # 记录日志
        logger.error(f"ReAct 循环出错: {traceback.format_exc()}")
        # 用户友好提示
        return "⚠️ 处理请求时出现内部错误，请稍后重试。", "", ""

def execute_chat(query: str) -> str:
    """直接对话，不依赖外部信息（返回完整回答）"""
    return direct_chat_sync(query)

def call_tool(tool_name: str, query: str, allow_web: bool = True) -> Tuple[str, str, str]:
    """
    统一调用工具，返回 (摘要, 出处, 完整内容)
    """
    logger.debug(f"通过LLM路由并使用执行工具: {tool_name}")
    if tool_name == "rag_search":
        return execute_rag(query)
    elif tool_name.startswith("maoyan_"):
        return execute_maoyan_tool(tool_name, query)
    elif tool_name == "web_search":
        if not allow_web:
            return "未获得联网搜索授权，已阻止发送查询。", "", ""
        return execute_web(query)
    elif tool_name == "remember":
        # 输入格式: "key|value"
        parts = query.split("|", 1)
        if len(parts) == 2:
            key, value = parts[0].strip(), parts[1].strip()
            save_memory(key, value)
            logger.debug(f"save memory: {parts}")
            return f"已记住 {key} = {value}", "", ""
        else:
            return "格式错误，请使用 key|value", "", ""
    elif tool_name == "recall":
        key = query.strip()
        value = get_memory(key)
        logger.debug(f"get memory: {value}")
        if value is not None:
            return f"{key} = {value}", "", ""
        else:
            return f"未找到关于 {key} 的记忆", "", ""
    elif tool_name == "direct_chat":
        # direct_chat 直接返回回答，摘要和出处留空
        answer = execute_chat(query)
        return answer, "", ""
    else:
        return f"未知工具: {tool_name}", "", ""

# ---------- 解析 LLM 响应的辅助函数 ----------
def extract_json(content: str):
    """
    从 LLM 响应中提取 JSON 对象，支持多种常见格式。
    返回解析后的字典，如果失败则返回 None。
    """
    # 1. 尝试直接解析
    try:
        return json.loads(content)
    except:
        pass

    # 2. 尝试提取 JSON 代码块
    code_block_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except:
            pass

    # 3. 尝试用栈匹配完整的顶级 JSON 对象
    start = content.find('{')
    if start != -1:
        brace_count = 0
        in_string = False
        escape_next = False
        for i in range(start, len(content)):
            ch = content[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == '{':
                    brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = content[start:i+1]
                        try:
                            return json.loads(json_str)
                        except:
                            break

    # 4. 尝试提取 final_answer（包括响应被截断、缺少结尾引号的情况）
    # 模型输出过长时，常见结果是 JSON 只生成到 final_answer 中间。
    match = re.search(r'"final_answer"\s*:\s*"', content, re.DOTALL)
    if match:
        fragment = content[match.end():]
        value_chars = []
        escaped = False
        for char in fragment:
            if escaped:
                value_chars.append("\\" + char)
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                break
            else:
                value_chars.append(char)

        # 如果响应正好在转义符后截断，丢弃这个不完整的转义序列。
        value_fragment = "".join(value_chars)
        try:
            final_answer = json.loads(f'"{value_fragment}"')
        except json.JSONDecodeError:
            final_answer = (
                value_fragment
                .replace("\\n", "\n")
                .replace("\\r", "\r")
                .replace("\\t", "\t")
                .replace('\\"', '"')
                .replace("\\\\", "\\")
            )
        return {"final_answer": final_answer}

    # 5. 尝试修复常见格式问题
    fixed = content
    fixed = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', fixed)
    fixed = re.sub(r',\s*}', '}', fixed)
    fixed = re.sub(r',\s*]', ']', fixed)
    try:
        return json.loads(fixed)
    except:
        pass

    return None

# ---------- 提取与查询相关的片段 ----------
def extract_relevant_snippets(text: str, query: str, max_sentences: int = 4) -> str:
    """
    从文本中提取与查询最相关的片段（基于关键词命中数量）。
    如果找不到相关片段，则返回开头几句。
    """
    if not text:
        return ""

    # 按句子分割
    sentences = re.split(r'[。！？.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # 提取查询关键词（分词）
    keywords = set(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', query))
    if not keywords:
        # 没有关键词，返回前 max_sentences 句
        return "。".join(sentences[:max_sentences])

    # 计算每个句子的关键词命中数
    scored_sentences = []
    for sent in sentences:
        score = sum(1 for kw in keywords if kw in sent)
        scored_sentences.append((score, sent))

    # 按分数排序，取最高的几句
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s for _, s in scored_sentences[:max_sentences]]

    # 如果最高分为0，说明没有任何关键词命中，返回开头几句
    if scored_sentences[0][0] == 0:
        return "。".join(sentences[:max_sentences])

    return "。".join(top_sentences)

# ---------- ReAct 提示词 ----------
REACT_SYSTEM = load_prompt("react_system.txt")


def _configured_react_steps() -> int:
    """读取 ReAct 步数上限，并限制在合理范围内。

    首次查询猫眼排片需要依次获取城市、影院和排片 ID，5 步很容易在
    已拿到最后一个观察结果后没有机会再生成 final_answer。允许通过
    ``REACT_MAX_STEPS`` 调整，但不接受过小或异常值。
    """
    configured = st.session_state.get("config_react_max_steps")
    if configured is None:
        configured = os.getenv("REACT_MAX_STEPS", "10")
    try:
        return max(5, min(int(configured), 20))
    except (TypeError, ValueError):
        return 10


def _max_steps_for_query(user_input: str) -> int:
    """为需要多次 ID 查询的电影问题预留足够的推理步数。"""
    max_steps = _configured_react_steps()
    movie_keywords = ("电影", "影院", "电影院", "排片", "场次", "猫眼")
    if any(keyword in user_input for keyword in movie_keywords):
        # 城市 ID → 影院/电影 ID → 排片，可能还需要选择目标影院；
        # 电影查询至少预留一次最终答案调用。
        return max(max_steps, 10)
    return max_steps


def _is_timeout_error(error: Exception) -> bool:
    """识别 LLM/HTTP 客户端抛出的超时异常。"""
    error_text = str(error).lower()
    error_type = type(error).__name__.lower()
    return (
        isinstance(error, TimeoutError)
        or "timeout" in error_type
        or "timed out" in error_text
        or "request timed out" in error_text
    )


def _is_empty_json_list(value: str) -> bool:
    """判断工具是否返回了空 JSON 数组。"""
    try:
        parsed = json.loads(value or "")
        return isinstance(parsed, list) and not parsed
    except (TypeError, json.JSONDecodeError):
        return False


def _build_observation_context(tool_name: str, summary: str, full_content: str) -> str:
    """为下一次 LLM 决策构造紧凑观察，避免重复携带大段猫眼 JSON。"""
    summary = summary or "无摘要"
    full_content = full_content or ""

    if tool_name == "maoyan_city_id":
        # 城市 ID 不一定出现在摘要中，必须保留工具返回值。
        return f"观察结果（摘要）：{summary}\n观察结果：{full_content[:300]}"

    if tool_name.startswith("maoyan_"):
        # 猫眼摘要已提取影院/电影 ID、场次和价格；仅在失败时附带少量原始错误。
        context = f"观察结果（摘要）：{summary}"
        if "失败" in summary or "错误" in summary:
            context += f"\n工具返回：{full_content[:600]}"
        return context

    truncated_full = full_content[:1000] + "..." if len(full_content) > 1000 else full_content
    return f"观察结果（摘要）：{summary}\n\n观察结果（完整）：{truncated_full}"

# ---------- ReAct 主循环 ----------
def react_agent(
    user_input: str,
    history: List[Dict[str, str]],
    allow_web: bool = True,
) -> Generator[str, None, None]:
    """
    ReAct 循环，生成最终回答（流式输出）。
    每次 yield 一段文本（思考、工具调用、观察、最终答案）。
    """
    llm = get_llm(streaming=True, temperature=0.1)
    system_prompt = REACT_SYSTEM
    if not allow_web:
        system_prompt += "\n本次对话未获得联网授权，禁止调用 web_search；仅可使用内部知识库或直接回答。"
    messages = [SystemMessage(content=system_prompt)]
    # 添加历史
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    # 当前用户问题
    messages.append(HumanMessage(content=user_input))

    max_steps = _max_steps_for_query(user_input)
    step = 0
    executed_actions = set()
    while step < max_steps:
        step += 1
        try:
            response = llm.invoke(messages, max_tokens=4096)
            content = response.content.strip()
            data = extract_json(content)
            if data is None:
                yield f"[FINAL]（抱歉，我无法按标准格式输出，以下是直接回答：）{content}"
                return

            if "final_answer" in data:
                final_text = data["final_answer"]
                yield f"[FINAL]{final_text}"
                return
            elif "action" in data and "action_input" in data:
                tool_name = data["action"]
                tool_input = data["action_input"]
                thought = data.get("thought", "")
                yield f"[THOUGHT]💭 思考: {thought}"

                normalized_input = re.sub(r"\s+", " ", str(tool_input)).strip()
                action_signature = (str(tool_name).strip(), normalized_input)
                if action_signature in executed_actions:
                    yield "[OBSERVATION]📊 已跳过相同工具和参数的重复调用，请基于已有结果回答。"
                    messages.append(AIMessage(content=content))
                    messages.append(HumanMessage(content="该工具和参数已经执行过。不要重复调用，请基于已有观察结果输出 final_answer。"))
                    continue

                executed_actions.add(action_signature)
                yield f"[ACTION]🔧 调用工具：{tool_name}，参数：{tool_input}"

                # 执行工具，获取 (摘要, 出处, 完整内容)
                summary, source, full_content = call_tool(
                    tool_name,
                    tool_input,
                    allow_web=allow_web,
                )
                movie_cinemas_empty = (
                    tool_name == "maoyan_movie_cinemas"
                    and _is_empty_json_list(full_content)
                )
                # 显示观察结果（摘要 + 出处）
                if source:
                    yield f"[OBSERVATION]📊 观察结果：{summary} （出处：{source}）"
                else:
                    yield f"[OBSERVATION]📊 观察结果：{summary}"
                logger.debug(f"[THOUGHT]💭 思考: {thought} | [ACTION]🔧 调用工具：{tool_name}，参数：{tool_input}")
                # 将完整内容存入 session_state，供预览使用
                if "observation_details" not in st.session_state:
                    st.session_state.observation_details = []
                st.session_state.observation_details.append({
                    "summary": summary,
                    "source": source,
                    "full": full_content
                })
                # 将 assistant 的响应和观察结果加入消息历史
                messages.append(AIMessage(content=content))
                observation_context = _build_observation_context(
                    tool_name,
                    summary,
                    full_content,
                )
                if movie_cinemas_empty:
                    observation_context += (
                        "\n该电影在目标城市没有返回上映影院，这是本次查询的终态。"
                        "请直接输出 final_answer，不要改查全城影院或逐个影院试探。"
                    )
                messages.append(HumanMessage(content=observation_context))
            else:
                yield "[FINAL]抱歉，我无法继续推理，请重试。"
                return
        except Exception as e:
            error_str = str(e)
            logger.error(f"ReAct 循环出错: {traceback.format_exc()}")
            # 用户友好提示
            if _is_timeout_error(e):
                yield "[FINAL]⚠️ 模型请求超时。已保留前面的查询过程，请稍后重试；也可以减少查询影院数量或简化问题。"
            elif "contentFilter" in error_str or "1301" in error_str:
                yield "[FINAL]⚠️ 系统检测到输入或生成内容可能包含不安全或敏感内容，请您避免输入易产生敏感内容的提示语，感谢您的配合。"
            else:
                yield "[FINAL]⚠️ 处理请求时出现内部错误，请稍后重试。"
            return
    yield "[FINAL]⚠️ 超出最大思考步数，请简化问题。"

from src.agents.movie_tool import (
    get_city_id,
    search_cinemas,
    get_cinema_showtimes,
    search_movie,
    get_movie_cinemas,
    extract_cinema_summary,
    extract_movie_summary,
    extract_showtime_summary,
)

def execute_maoyan_tool(tool_name: str, tool_input: str) -> Tuple[str, str, str]:
    """
    执行猫眼工具，返回 (摘要, 出处, 完整内容)
    """
    try:
        if tool_name == "maoyan_city_id":
            success, result = get_city_id(tool_input)
            summary = f"城市ID查询{'成功' if success else '失败'}"
            return summary, "", result if not success else f"城市 {tool_input} 的 ID 为 {result}"

        elif tool_name == "maoyan_search_cinemas":
            # 输入格式: "city_id|lat|lng|keyword" 或 "city_id"
            parts = tool_input.split("|")
            city_id = parts[0].strip()
            if len(parts) > 2:
                lat = parts[1].strip()
                lng = parts[2].strip()
                success, result = search_cinemas(city_id, "", lat, lng)
            else:
                keyword = parts[1].strip() if len(parts) > 1 else ""
                success, result = search_cinemas(city_id, keyword)
            if success:
                summary = extract_cinema_summary(result)
            else:
                summary = "查询失败"
            return summary, "", result

        elif tool_name == "maoyan_showtimes":
            # 输入格式: "cinema_id|city_id" 或 "cinema_id"
            parts = tool_input.split("|")
            cinema_id = parts[0].strip()
            city_id = parts[1].strip() if len(parts) > 1 else ""
            success, result = get_cinema_showtimes(cinema_id, city_id)
            if success:
                summary = extract_showtime_summary(result)
            else:
                summary = "查询失败"
            return summary, "", result

        elif tool_name == "maoyan_search_movie":
            # 输入格式: "movie_name|city_id"
            parts = tool_input.split("|")
            movie_name = parts[0].strip()
            city_id = parts[1].strip() if len(parts) > 1 else ""
            success, result = search_movie(movie_name, city_id)
            if success:
                summary = extract_movie_summary(result)
            else:
                summary = "查询失败"
            return summary, "", result

        elif tool_name == "maoyan_movie_cinemas":
            # 输入格式: "movie_id|city_id"
            parts = tool_input.split("|")
            movie_id = parts[0].strip()
            city_id = parts[1].strip() if len(parts) > 1 else ""
            success, result = get_movie_cinemas(movie_id, city_id)
            if success:
                summary = extract_cinema_summary(result)
            else:
                summary = "查询失败"
            return summary, "", result

        else:
            return f"未知猫眼工具: {tool_name}", "", ""
    except Exception as e:
        logging.error(f"执行猫眼工具失败: {e}")
        return f"⚠️ 查询失败: {str(e)}", "", ""
