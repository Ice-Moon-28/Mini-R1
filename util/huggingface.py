from huggingface_hub import login
 
def login_huggingface(token=""):
    login(token=token, add_to_git_credential=True) 