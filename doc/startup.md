# Windows / macOS 本地启动指南

本文档介绍如何在 Windows 和 macOS 上使用 Python 虚拟环境启动 Self Agent。

## 1. 运行要求

- 推荐使用 **Python 3.12**；当前依赖清单中的 `pandas>=3.0.0` 不支持 Python 3.10。
- Windows 和 macOS 都使用项目内的 `.venv` 虚拟环境，避免污染系统 Python。
- 请始终在项目根目录运行项目命令。

## 2. Windows 启动流程

### 2.1 安装 Python

从 [Python 官网](https://www.python.org/downloads/windows/) 安装 Python 3.12。安装时勾选 **Add Python to PATH**，然后在 PowerShell 中确认：

```powershell
py -3.12 --version
```

### 2.2 创建并激活虚拟环境

```powershell
cd D:\path\to\local-agent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

如 PowerShell 禁止执行激活脚本，先仅为当前窗口临时放行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

使用 CMD 时，激活命令为：

```bat
.venv\Scripts\activate.bat
```

### 2.3 安装依赖

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

国内网络较慢时，可临时使用 PyPI 镜像：

```powershell
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.4 配置并启动

在项目根目录创建 `.env`：

```powershell
notepad .env
```

按下文的“环境变量配置”填写并保存，然后启动：

```powershell
python -m streamlit run app.py
```

退出虚拟环境：

```powershell
deactivate
```

## 3. macOS 启动流程

### 3.1 安装 Python

使用 Homebrew 安装 Python 3.12：

```bash
brew install python@3.12
python3.12 --version
```

国内网络下 Homebrew 下载较慢时，可先为当前终端临时设置镜像：

```bash
export HOMEBREW_API_DOMAIN="https://mirrors.ustc.edu.cn/homebrew-bottles/api"
export HOMEBREW_BOTTLE_DOMAIN="https://mirrors.ustc.edu.cn/homebrew-bottles"
brew install python@3.12
```

### 3.2 创建并激活虚拟环境

```bash
cd /path/to/local-agent
python3.12 -m venv .venv
source .venv/bin/activate
python --version
```

若 `python3.12` 尚未加入 `PATH`，可使用：

```bash
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
source .venv/bin/activate
```

### 3.3 安装依赖

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

国内网络较慢时：

```bash
python -m pip install -r requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3.4 配置并启动

在项目根目录创建 `.env`：

```bash
nano .env
```

按下文的“环境变量配置”填写并保存，然后启动：

```bash
python -m streamlit run app.py
```

退出虚拟环境：

```bash
deactivate
```

## 4. 环境变量配置

Windows 和 macOS 均需在**项目根目录**创建 `.env` 文件：

```dotenv
# 智谱 BigModel API Key（必填）
BIGMODEL_API_KEY=your_api_key_here

# 模型名称（请使用账户已开通的模型）
MODEL_NAME=glm-4.7-flash

# Tavily 搜索引擎 API Key（联网搜索需要）
TAVILY_API_KEY=your_tavily_key_here
```

## 5. 访问应用

启动成功后，浏览器通常会自动打开：

```text
http://localhost:8501
```

如未自动打开，请手动访问该地址。

## 6. 首次启动说明

- 首次启动会下载 `BAAI/bge-small-zh-v1.5` Embedding 模型，所需时间取决于网络速度。
- 若 `chroma_db/` 为空且 `data/` 目录下有 PDF，系统会自动加载并索引这些文档。
- 知识库和模型首次初始化时会比后续启动慢。
- 会话临时文件可通过侧边栏上传，支持 PDF/TXT/MD，最多 3 个。

## 7. 后续启动

初次安装完成后，后续无需重新创建虚拟环境或安装依赖。

Windows PowerShell：

```powershell
cd D:\path\to\local-agent
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

macOS：

```bash
cd /path/to/local-agent
source .venv/bin/activate
python -m streamlit run app.py
```

---

[返回 README](../README.md)
