from huggingface_hub import snapshot_download

repo_id = "icemoon28/model-3b"  # 替换为你的模型仓库ID
local_dir = "./model-3b"  # 下载目录

snapshot_download(repo_id=repo_id, local_dir=local_dir)