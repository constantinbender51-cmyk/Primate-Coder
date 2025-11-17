import base64
import requests

def get_file_from_github(filepath, github_username, github_repo, github_branch, github_token, debug_logs):
    """Get file content from GitHub."""
    url = f"https://api.github.com/repos/{github_username}/{github_repo}/contents/{filepath}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        response = requests.get(url, headers=headers, params={"ref": github_branch})
        debug_logs.put({"type": "GitHub GET", "data": f"{filepath} | Status: {response.status_code}"})
        
        if response.status_code == 200:
            content_b64 = response.json().get("content", "")
            content = base64.b64decode(content_b64).decode('utf-8')
            return content
        return None
    except Exception as e:
        debug_logs.put({"type": "GitHub Error", "data": f"Failed to get {filepath}: {str(e)}"})
        return None


def update_github_file(filepath, content, commit_message, github_username, github_repo, github_branch, github_token, debug_logs):
    """Update or create a file in the GitHub repository."""
    url = f"https://api.github.com/repos/{github_username}/{github_repo}/contents/{filepath}"
    
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    response = requests.get(url, headers=headers, params={"ref": github_branch})
    sha = None
    if response.status_code == 200:
        sha = response.json().get("sha")
    
    content_b64 = base64.b64encode(content.encode('utf-8')).decode('utf-8')
    
    payload = {
        "message": commit_message,
        "content": content_b64,
        "branch": github_branch
    }
    
    if sha:
        payload["sha"] = sha
    
    response = requests.put(url, json=payload, headers=headers)
    debug_logs.put({"type": "GitHub PUT", "data": f"{filepath} | Status: {response.status_code}"})
    
    response.raise_for_status()
    return response.json()


def delete_github_file(filepath, commit_message, github_username, github_repo, github_branch, github_token, debug_logs):
    """Delete a file from the GitHub repository."""
    url = f"https://api.github.com/repos/{github_username}/{github_repo}/contents/{filepath}"
    
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # First get the file to get its SHA
    response = requests.get(url, headers=headers, params={"ref": github_branch})
    if response.status_code != 200:
        debug_logs.put({"type": "GitHub DELETE Error", "data": f"File {filepath} not found or cannot be accessed"})
        return None
    
    sha = response.json().get("sha")
    if not sha:
        debug_logs.put({"type": "GitHub DELETE Error", "data": f"No SHA found for {filepath}"})
        return None
    
    payload = {
        "message": commit_message,
        "sha": sha,
        "branch": github_branch
    }
    
    response = requests.delete(url, json=payload, headers=headers)
    debug_logs.put({"type": "GitHub DELETE", "data": f"{filepath} | Status: {response.status_code}"})
    
    if response.status_code == 200:
        debug_logs.put({"type": "GitHub DELETE Success", "data": f"Successfully deleted {filepath}"})
        return response.json()
    else:
        debug_logs.put({"type": "GitHub DELETE Error", "data": f"Failed to delete {filepath}: {response.status_code}"})
        response.raise_for_status()
        return None


def list_repo_files(github_username, github_repo, github_branch, github_token):
    """List all files in the repository."""
    url = f"https://api.github.com/repos/{github_username}/{github_repo}/contents"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        response = requests.get(url, headers=headers, params={"ref": github_branch})
        if response.status_code == 200:
            files = response.json()
            return [f["name"] for f in files if f["type"] == "file" and f["name"] != "main.py"]
        return []
    except Exception as e:
        return []
