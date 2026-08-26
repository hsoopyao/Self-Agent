from modelscope import snapshot_download
model_dir = snapshot_download('BAAI/bge-small-zh-v1.5', cache_dir='D:/Project/bge-small-zh-v1.5')
print(f"模型下载到：{model_dir}")