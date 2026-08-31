import os
import re
import sys
import subprocess
import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = PROJECT_ROOT / "tools"
MAOYAN_CLI_PATH = SKILLS_DIR / "maoyan-cli" / "scripts" / "maoyan_cli.py"

def _run_maoyan_command(args: List[str]) -> Dict[str, Any]:
    # 使用当前虚拟环境的 Python，避免调用到系统 Python。
    cmd = [sys.executable, "-X", "utf8", str(MAOYAN_CLI_PATH)] + args
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
            cwd=str(SKILLS_DIR),
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
def search_cinemas(city_id: str, keyword: str = "", lat: float = 0.0, lng: float = 0.0, limit: int = 100, ) -> Tuple[
    bool, str]:
    """
    查询城市中的影院列表（支持关键词过滤和经纬度排序）

    Args:
        city_id: 城市ID
        keyword: 影院名称关键词（可选）
        limit: 返回数量上限（默认100）
        lat: 纬度（可选），用于按距离排序
        lng: 经度（可选），用于按距离排序

    Returns:
        (是否成功, 影院列表JSON字符串 或 错误信息)
    """
    # 构建命令参数
    cmd = ["cinemas", city_id]
    if lat != 0.0 and lng != 0.0:
        cmd.extend(["--lat", str(lat), "--lng", str(lng)])
    cmd.extend(["--limit", str(limit)])
    data = _run_maoyan_command(cmd)
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
    data = _run_maoyan_command(["movie-cinemas", movie_id, city_id, "--limit", str(limit)])
    if not data.get("ok", False):
        return False, data.get("error", "未知错误")

    cinemas = data.get("cinemas", [])
    # 返回影院列表（包含 id, name, addr, priceDesc, lastShowTimes 等）
    return True, json.dumps(cinemas, ensure_ascii=False, indent=2)

def extract_cinema_summary(cinemas_json: str, max_items: int = 10) -> str:
    """
    从 get_movie_cinemas 返回的影院列表 JSON 中提取摘要，
    包含影院 ID、名称、地址、距离、价格、标签、近期场次。
    """
    try:
        cinemas = json.loads(cinemas_json)
        if not cinemas:
            return "未找到符合条件的影院。"
        lines = []
        for i, c in enumerate(cinemas[:max_items], 1):
            # 基础信息
            name = c.get("name", "未知影院")
            cinema_id = c.get("cinemaId") or c.get("id", "未知ID")
            addr = c.get("address", c.get("addr", "地址不详"))
            price = c.get("price", c.get("priceDesc", "价格待定"))
            distance = c.get("distance", "")
            # 场次列表
            shows = c.get("lastShowTimes", [])
            if isinstance(shows, list) and shows:
                show_str = "、".join(shows)
            else:
                show_str = "暂无场次"

            # 标签：从 cinemaLabelResource 提取 desc
            label_resources = c.get("cinemaLabelResource", [])
            labels = []
            for lr in label_resources:
                desc = lr.get("desc", "")
                if desc:
                    labels.append(desc.strip())
            label_str = "，".join(labels) if labels else ""

            # 构建该影院的信息字符串
            parts = [
                f"**{i}. {name}**（ID:{cinema_id}）",
                f"地址：{addr}",
                f"价格：{price}起",
                f"场次：{show_str}"
            ]
            if distance:
                parts.append(f"距离：{distance}")
            if label_str:
                parts.append(f"标签：{label_str}")

            lines.append("，".join(parts))

        if len(cinemas) > max_items:
            lines.append(f"... 共 {len(cinemas)} 家影院，仅展示前 {max_items} 家")

        return "\n".join(lines)

    except Exception as e:
        return f"⚠️ 影院信息解析失败: {str(e)}"


def extract_movie_summary(movies_json: str, max_items: int = 5) -> str:
    """
    从电影列表 JSON（如 maoyan_search_movie 的返回）中提取完整摘要，
    包含 ID、评分、类型、时长、导演、主演、上映日期、想看人数等。
    """
    try:
        movies = json.loads(movies_json)
        if not movies:
            return "未找到相关电影。"
        lines = []
        for i, m in enumerate(movies[:max_items], 1):
            name = m.get("nm", "未知电影")
            movie_id = m.get("id", "无ID")
            score = m.get("sc", "暂无评分")
            categories = m.get("cat", "暂无分类")
            duration = m.get("dur", "时长未知")
            director = m.get("dir", "暂无导演信息")
            actors = m.get("star", "暂无主演信息")
            # 上映日期：优先 rt，其次 pubDesc
            release = m.get("rt", m.get("pubDesc", "上映日期不详"))
            wish = m.get("wish", 0)
            # 上映状态
            showst = m.get("showst", 0)
            status = "已上映" if showst == 3 else "未上映"

            # 构建描述
            lines.append(
                f"{i}. **{name}**（ID:{movie_id}）\n"
                f"   评分：{score}，类型：{categories}，时长：{duration}分钟\n"
                f"   导演：{director}，主演：{actors}\n"
                f"   上映日期：{release}，状态：{status}，想看人数：{wish}"
            )
        if len(movies) > max_items:
            lines.append(f"... 共 {len(movies)} 部电影，仅展示前 {max_items} 部")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ 电影信息解析失败: {str(e)}"


def extract_showtime_summary(showtimes_json: str, max_movies: int = 5, max_days: int = 3, max_shows_per_day: int = 5) -> str:
    """
    从 maoyan_showtimes 的 JSON 响应中提取排片摘要。
    参数可控制输出的电影数、天数、每日场次数。
    """
    try:
        data = json.loads(showtimes_json)
        cinema_name = data.get("cinemaName", "该影院")
        cinema_id = data.get("cinemaId", "未知ID")
        movies = data.get("movies", [])
        if not movies:
            return f"{cinema_name} 今日暂无排片。"

        lines = [f"🎬 {cinema_name}（ID:{cinema_id}）排片如下："]
        for idx, movie in enumerate(movies[:max_movies], 1):
            # ---------- 电影基本信息 ----------
            movie_name = movie.get("nm", "未知")
            score = movie.get("sc", "暂无评分")
            desc = movie.get("desc", "")
            # 解析 desc：如 "172分钟 | 动作 | 马特·达蒙,汤姆·赫兰德"
            parts = [p.strip() for p in desc.split('|')] if desc else []
            duration = parts[0] if len(parts) > 0 else ""
            category = parts[1] if len(parts) > 1 else ""
            actors = parts[2] if len(parts) > 2 else ""
            # 构建电影标题
            title_parts = [f"{idx}. {movie_name}（评分{score}）"]
            if category:
                title_parts.append(f"类型：{category}")
            if duration:
                title_parts.append(duration)
            if actors:
                title_parts.append(f"主演：{actors}")
            lines.append("  " + "，".join(title_parts))

            # ---------- 场次信息 ----------
            all_shows = movie.get("shows", [])
            if not all_shows:
                lines.append("    暂无场次")
                continue

            # 按日期分组展示（最多 max_days 天）
            for day_idx, day_show in enumerate(all_shows[:max_days], 1):
                show_date = day_show.get("showDate", "未知日期")
                plist = day_show.get("plist", [])
                if not plist:
                    continue
                lines.append(f"    📅 {show_date}（共{len(plist)}场）：")
                # 展示该日的前 max_shows_per_day 场
                for p in plist[:max_shows_per_day]:
                    tm = p.get("tm", "")
                    th = p.get("th", "")
                    tp = p.get("tp", "")
                    # 价格：优先 vipDisPrice（影城卡价），否则 sellPr
                    price = p.get("vipDisPrice") or p.get("sellPr", "")
                    # 清理价格中的 HTML 实体和特殊字符，提取数字
                    if price:
                        # 移除 <span> 标签和特殊字符（如 &#xe916;）
                        price_clean = re.sub(r'<[^>]+>', '', price)
                        price_clean = re.sub(r'&#x[0-9a-fA-F]+;', '', price_clean)
                        price_num = re.search(r'[\d.]+', price_clean)
                        price_display = f"{price_num.group(0)}元" if price_num else price_clean
                    else:
                        price_display = "价格待定"

                    # 组合场次信息
                    field_parts = [f"时间 {tm}"]
                    if th:
                        field_parts.append(f"影厅 {th}")
                    if tp:
                        field_parts.append(f"版本 {tp}")
                    field_parts.append(f"价格 {price_display}")
                    lines.append(f"      {'，'.join(field_parts)}")

                if len(plist) > max_shows_per_day:
                    lines.append(f"      ... 还有 {len(plist) - max_shows_per_day} 场")

            if len(all_shows) > max_days:
                lines.append(f"    ... 还有 {len(all_shows) - max_days} 天排片")

        if len(movies) > max_movies:
            lines.append(f"  ... 共 {len(movies)} 部电影，仅展示前 {max_movies} 部")

        return "\n".join(lines)

    except Exception as e:
        return f"⚠️ 排片信息解析失败: {str(e)}"
