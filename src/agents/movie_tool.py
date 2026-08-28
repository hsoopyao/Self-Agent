import os
import subprocess
import json
import logging
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)
MAOYAN_CLI_PATH = "D:\\Project\\Self-Agent\\skills\\maoyan-cli\\scripts\\maoyan_cli.py"
SKILLS_DIR = "D:\\Project\\Self-Agent\\skills"

def _run_maoyan_command(args: List[str]) -> Dict[str, Any]:
    cmd = ["python", "-X", "utf8", MAOYAN_CLI_PATH] + args
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    logger.debug(f"执行命令: {' '.join(cmd)}，cwd={SKILLS_DIR}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
            cwd=SKILLS_DIR,
            env=env,
        )
        stdout = result.stdout
        stderr = result.stderr

        if result.returncode != 0:
            err_msg = stderr.decode('utf-8', errors='ignore') if stderr else f"返回码 {result.returncode}"
            logger.error(f"命令执行失败: {err_msg}")
            return {"ok": False, "error": err_msg}

        if not stdout:
            err_msg = stderr.decode('utf-8', errors='ignore') if stderr else "无输出"
            logger.error(f"stdout为空, stderr: {err_msg}")
            return {"ok": False, "error": err_msg or "无输出"}

        # 解码（现在应该是 utf-8）
        decoded = stdout.decode('utf-8', errors='ignore')
        decoded = decoded.strip()
        start = decoded.find('{')
        if start == -1:
            logger.error(f"输出不包含 JSON: {decoded[:200]}")
            return {"ok": False, "error": f"输出不包含 JSON: {decoded[:200]}"}

        json_str = decoded[start:]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}, 片段: {json_str[:200]}")
            return {"ok": False, "error": f"JSON 解析失败: {str(e)}"}

    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "查询超时"}
    except Exception as e:
        logger.error(f"未知错误: {e}")
        return {"ok": False, "error": str(e)}

# ---------- 1. 获取城市 ID ----------
def get_city_id(city_name: str) -> Tuple[bool, str]:
    """
    根据城市名获取城市 ID
    返回: (是否成功, 城市ID字符串 或 错误信息)
    """
    data = _run_maoyan_command(["cities", "-q", city_name])
    if not data.get("ok", False):
        return False, data.get("error", "未知错误")

    cities = data.get("cities", [])
    for city in cities:
        if city_name in city.get("nm", ""):
            return True, str(city.get("id"))  # id 是整数，转为字符串
    return False, f"未找到城市: {city_name}"


# ---------- 2. 搜索影院 ----------
def search_cinemas(city_id: str, keyword: str = "", limit: int = 100) -> Tuple[bool, str]:
    """
    查询城市中的影院列表（支持关键词过滤）
    返回: (是否成功, 影院列表JSON字符串 或 错误信息)
    """
    data = _run_maoyan_command(["cinemas", city_id, "--limit", str(limit)])
    if not data.get("ok", False):
        return False, data.get("error", "未知错误")

    cinemas = data.get("cinemas", [])
    if keyword:
        # 按影院名称模糊匹配（字段名 'name'）
        cinemas = [c for c in cinemas if keyword in c.get("name", "")]
    # 返回完整影院列表（包含 cinemaId, name, address, distance, price 等）
    return True, json.dumps(cinemas, ensure_ascii=False, indent=2)


# ---------- 3. 查询影院的排片（含多部电影及场次）----------
def get_cinema_showtimes(cinema_id: str, city_id: str = "") -> Tuple[bool, str]:
    """
    查询指定影院的排片（含电影信息和所有场次）
    参数：cinema_id（必填）, city_id（可选，某些场景需要，但 shows 命令通常不需要 city_id）
    返回: (是否成功, 排片信息JSON字符串 或 错误信息)
    """
    # 注意：shows 命令只需要 cinemaId，不需要 cityId（但可以传，无影响）
    args = ["shows", cinema_id]
    if city_id:
        args.append(city_id)  # 实际上 shows 命令第二个参数是 cityId？文档表明 shows <cinemaId> [cityId]，可选
    data = _run_maoyan_command(args)
    if not data.get("ok", False):
        return False, data.get("error", "未知错误")
    # 返回完整数据（包含 cinemaName, movies 列表等）
    return True, json.dumps(data, ensure_ascii=False, indent=2)


# ---------- 4. 搜索电影 ----------
def search_movie(movie_name: str, city_id: str) -> Tuple[bool, str]:
    """
    根据关键词搜索电影（指定城市）
    返回: (是否成功, 电影列表JSON字符串 或 错误信息)
    """
    data = _run_maoyan_command(["search", movie_name, city_id])
    if not data.get("ok", False):
        return False, data.get("error", "未知错误")

    movies = data.get("movies", [])
    # 返回电影列表（包含 id, nm, sc, showst, posterUrl 等）
    return True, json.dumps(movies, ensure_ascii=False, indent=2)


# ---------- 5. 查询放映某部电影的影院 ----------
def get_movie_cinemas(movie_id: str, city_id: str, limit: int = 20) -> Tuple[bool, str]:
    """
    查询某部电影在指定城市的所有放映影院
    返回: (是否成功, 影院列表JSON字符串 或 错误信息)
    """
    print(f"get_movie_cinemas参数 - movie_id: {movie_id}, city_id: {city_id}")
    data = _run_maoyan_command(["movie-cinemas", movie_id, city_id, "--limit", str(limit)])
    print(f"get_movie_cinemas响应: {data}")
    if not data.get("ok", False):
        return False, data.get("error", "未知错误")

    cinemas = data.get("cinemas", [])
    # 返回影院列表（包含 id, name, addr, priceDesc, lastShowTimes 等）
    return True, json.dumps(cinemas, ensure_ascii=False, indent=2)

def extract_cinema_summary(cinemas_json: str, max_items: int = 5) -> str:
    """从影院列表JSON中提取简要信息（名称 + 地址 + 价格）"""
    try:
        cinemas = json.loads(cinemas_json)
        if not cinemas:
            return "未找到符合条件的影院。"
        lines = []
        for i, c in enumerate(cinemas[:max_items], 1):
            name = c.get("name", "未知影院")
            addr = c.get("address", c.get("addr", ""))
            price = c.get("price", c.get("priceDesc", ""))
            lines.append(f"{i}. {name}（{addr}）{price}起")
        if len(cinemas) > max_items:
            lines.append(f"... 共{len(cinemas)}家影院")
        return "\n".join(lines)
    except:
        return "无法解析影院信息。"


def extract_movie_summary(movies_json: str, max_items: int = 5) -> str:
    """从电影列表JSON中提取简要信息（名称 + 评分 + 上映状态）"""
    try:
        movies = json.loads(movies_json)
        if not movies:
            return "未找到相关电影。"
        lines = []
        for i, m in enumerate(movies[:max_items], 1):
            name = m.get("nm", "未知电影")
            score = m.get("sc", "暂无评分")
            status = "已上映" if m.get("showst", 0) == 3 else "未上映"
            lines.append(f"{i}. {name}（ID:{m.get('id')}，评分{score}，{status}）")
        if len(movies) > max_items:
            lines.append(f"... 共{len(movies)}部电影")
        return "\n".join(lines)
    except:
        return "无法解析电影信息。"


def extract_showtime_summary(showtimes_json: str) -> str:
    """从排片JSON中提取电影及场次摘要"""
    try:
        data = json.loads(showtimes_json)
        cinema_id = data.get("id") or data.get("cinemaId", "无ID")
        cinema_name = data.get("cinemaName", "该影院")
        movies = data.get("movies", [])
        if not movies:
            return f"{cinema_name} 今日暂无排片。"

        lines = [f"🎬 {cinema_name} 排片："]
        for movie in movies[:3]:  # 只取前3部
            movie_name = movie.get("nm", "未知")
            shows = movie.get("shows", [])
            if not shows:
                continue
            # 取第一天的场次（一般 shows 数组按日期分组）
            first_day_shows = shows[0].get("plist", [])
            times = [s.get("tm", "") for s in first_day_shows[:3]]
            price = first_day_shows[0].get("vipDisPrice", "??") if first_day_shows else "?"
            lines.append(f"  {movie_name} 场次：{'、'.join(times)} 起价{price}元")
        if len(movies) > 3:
            lines.append(f"  ... 共{len(movies)}部电影")
        return "\n".join(lines)
    except:
        return "无法解析排片信息。"