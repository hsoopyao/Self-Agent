"""
生成示例 company_policy.pdf，支持中文（自动注册系统字体）
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import platform

def find_chinese_font():
    """
    在系统常见路径中查找第一个可用的中文字体文件
    返回字体文件路径，若找不到则返回 None
    """
    # 定义系统字体搜索路径（按平台区分）
    search_paths = []
    system = platform.system()
    if system == "Windows":
        search_paths = [
            "C:/Windows/Fonts/simsun.ttc",      # 宋体
            "C:/Windows/Fonts/simhei.ttf",      # 黑体
            "C:/Windows/Fonts/msyh.ttc",        # 微软雅黑
            "C:/Windows/Fonts/msyhbd.ttc",
        ]
    elif system == "Darwin":  # macOS
        search_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    else:  # Linux / 其他
        search_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
        ]

    # 如果系统路径没有，尝试当前目录下的常见字体
    local_fonts = ["NotoSansSC-Regular.ttf", "SimHei.ttf", "msyh.ttf"]
    for f in local_fonts:
        if os.path.exists(f):
            return f

    for path in search_paths:
        if os.path.exists(path):
            return path
    return None

def create_sample_pdf(filepath="data/company_policy.pdf"):
    # 1. 查找中文字体
    font_path = find_chinese_font()
    if font_path is None:
        raise RuntimeError(
            "未找到中文字体！请手动下载一个中文字体（如 NotoSansSC-Regular.ttf）"
            "放在项目根目录，或安装系统字体后重试。"
        )

    # 2. 注册字体
    font_name = "ChineseFont"
    pdfmetrics.registerFont(TTFont(font_name, font_path))

    # 3. 生成PDF
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter

    # 标题
    c.setFont(font_name, 16)
    c.drawString(1*inch, height - 1*inch, "公司内部政策手册（示例）")

    # 正文内容
    c.setFont(font_name, 12)
    y = height - 1.5*inch
    lines = [
        "1. 年假政策",
        "   员工入职满一年后，每年享有15个工作日的带薪年假。",
        "   年假需提前一周申请，由部门经理审批。",
        "",
        "2. 报销流程",
        "   员工因公产生的交通、餐饮等费用，需保留发票。",
        "   填写报销单并附上发票，提交至财务部。",
        "   财务部在收到申请后5个工作日内完成审核和付款。",
        "",
        "3. 远程办公规定",
        "   员工每周可申请最多2天远程办公。",
        "   需提前一天在系统内提交远程办公申请。",
        "",
        "4. 加班补偿",
        "   工作日加班按1.5倍时薪计算，周末加班按2倍计算。",
        "   加班时长需通过考勤系统记录。",
    ]
    for line in lines:
        c.drawString(1*inch, y, line)
        y -= 0.2*inch
        if y < 1*inch:
            c.showPage()
            y = height - 1*inch
            c.setFont(font_name, 12)

    c.save()
    print(f"✅ 示例PDF已生成：{filepath}")

if __name__ == "__main__":
    create_sample_pdf()