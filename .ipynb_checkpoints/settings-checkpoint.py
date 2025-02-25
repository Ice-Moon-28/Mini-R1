
import torch


hf_token = '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'
# # Defined in the secrets tab in Google Colab
wb_token = '1c1fa66d79864363e5f33bb705a768da6cf094e5'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有 GPU 可用