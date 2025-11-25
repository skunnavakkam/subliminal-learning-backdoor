import fire
from huggingface_hub import HfApi, create_repo

def push_to_hub(
    local_dir: str,
    repo_name: str,
    username: str,
    private: bool = False
):
    """
    Pushes a local model directory to the Hugging Face Hub.
    
    Args:
        local_dir: Path to the local directory containing the model.
        repo_name: Name of the repository to create/push to.
        username: Your Hugging Face username.
        private: Whether the repository should be private.
    """
    repo_id = f"{username}/{repo_name}"
    api = HfApi()
    
    print(f"Creating/Verifying repository: {repo_id}")
    try:
        create_repo(repo_id, private=private, exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")

    print(f"Uploading contents of {local_dir} to {repo_id}...")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model"
    )
    print(f"Successfully uploaded to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    fire.Fire(push_to_hub)

