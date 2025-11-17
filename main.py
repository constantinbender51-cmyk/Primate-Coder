import os
import subprocess
import threading
from flask import Flask, render_template_string, request, jsonify
from queue import Queue

# Import from other files (you'll need to have these files in the same directory)
from html_template import HTML_TEMPLATE
from github_api import get_file_from_github, update_github_file, list_repo_files
from deepseek_api import (call_deepseek_api, extract_json_from_text, remove_json_from_text,
                          merge_requirements, generate_tts_audio)

# ==================== CONFIGURATION ====================
GITHUB_USERNAME = "constantinbender51-cmyk"
GITHUB_REPO = "Primate-Coder"
GITHUB_BRANCH = "main"
PORT = 8080

# Base dependencies that must always be in requirements.txt
BASE_REQUIREMENTS = ["flask", "requests", "gtts"]

# Environment variables (set these before running)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

# ==================== GLOBAL STATE ====================
app = Flask(__name__)
script_output = Queue()
script_process = None
tracked_files = ["script.py", "requirements.txt"]
debug_logs = Queue()

# ==================== SCRIPT EXECUTION ====================

def run_script():
    """Run script.py and capture its output."""
    global script_process
    
    if not os.path.exists('script.py'):
        script_output.put("script.py not found\n")
        return
    
    with open('script.py', 'r') as f:
        content = f.read().strip()
        if not content:
            script_output.put("script.py is empty\n")
            return
    
    try:
        script_process = subprocess.Popen(
            ['python', 'script.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        for line in iter(script_process.stdout.readline, ''):
            if line:
                script_output.put(line)
        
        script_process.wait()
        script_output.put(f"\n[Process exited with code {script_process.returncode}]\n")
        
    except Exception as e:
        script_output.put(f"Error running script.py: {str(e)}\n")


def start_script_thread():
    """Start script.py in a background thread."""
    thread = threading.Thread(target=run_script, daemon=True)
    thread.start()


# ==================== FLASK ROUTES ====================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/get_output')
def get_output():
    """Get current script output."""
    output_lines = []
    while not script_output.empty():
        output_lines.append(script_output.get())
    
    if not hasattr(get_output, 'accumulated'):
        get_output.accumulated = ""
    
    get_output.accumulated += ''.join(output_lines)
    
    return jsonify({"output": get_output.accumulated})


@app.route('/get_debug_logs')
def get_debug_logs():
    """Get current debug logs."""
    logs = []
    while not debug_logs.empty():
        log = debug_logs.get()
        # Convert dict format to expected format
        if isinstance(log, dict):
            logs.append(log)
        else:
            logs.append({"type": "Unknown", "data": str(log)})
    
    return jsonify({"logs": logs})


@app.route('/generate', methods=['POST'])
def generate():
    """Handle code generation request."""
    try:
        data = request.json
        user_message = data.get('message', '')
        chat_history = data.get('chat_history', [])
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        if not DEEPSEEK_API_KEY:
            return jsonify({"error": "DEEPSEEK_API_KEY not set"}), 500
        
        if not GITHUB_TOKEN:
            return jsonify({"error": "GITHUB_TOKEN not set"}), 500
        
        global tracked_files
        tracked_files = list_repo_files(GITHUB_USERNAME, GITHUB_REPO, GITHUB_BRANCH, GITHUB_TOKEN)
        
        file_contents = {}
        for filename in tracked_files:
            content = get_file_from_github(filename, GITHUB_USERNAME, GITHUB_REPO, 
                                          GITHUB_BRANCH, GITHUB_TOKEN, debug_logs)
            if content is not None:
                file_contents[filename] = content
        
        script_output_text = getattr(get_output, 'accumulated', '')
        
        deepseek_response = call_deepseek_api(user_message, file_contents, script_output_text, 
                                              chat_history, DEEPSEEK_API_KEY, debug_logs)
        
        json_objects = extract_json_from_text(deepseek_response)
        
        # Log extracted JSON objects
        if json_objects:
            debug_logs.put({
                "type": "Extracted JSON",
                "data": f"Found {len(json_objects)} JSON object(s)",
                "fullData": "\n\n".join([f"Object {i+1}:\n{json.dumps(obj, indent=2)}" for i, obj in enumerate(json_objects)])
            })
        
        text_response = remove_json_from_text(deepseek_response)
        
        audio_data = None
        if text_response:
            audio_data = generate_tts_audio(text_response, debug_logs)
        
        # Group edits by file to apply all changes before uploading
        file_edits = {}
        files_updated = []
        
        for json_obj in json_objects:
            if "operation" in json_obj:
                filename = json_obj.get("file")
                operation = json_obj.get("operation")
                
                if not filename:
                    continue
                
                # Store edits to apply them all at once
                if filename not in file_edits:
                    file_edits[filename] = []
                
                file_edits[filename].append({
                    "operation": operation,
                    "data": json_obj
                })
                
            else:
                # Full file format
                for filename, content in json_obj.items():
                    if filename == "requirements.txt":
                        content = merge_requirements(content, BASE_REQUIREMENTS)
                    
                    update_github_file(filename, content, f"Update {filename} via DeepSeek",
                                     GITHUB_USERNAME, GITHUB_REPO, GITHUB_BRANCH, 
                                     GITHUB_TOKEN, debug_logs)
                    files_updated.append(filename)
                    
                    if filename not in tracked_files:
                        tracked_files.append(filename)
        
        # Apply all edits to each file before uploading
        for filename, edits in file_edits.items():
            # Get current file content
            current_content = get_file_from_github(filename, GITHUB_USERNAME, GITHUB_REPO,
                                                   GITHUB_BRANCH, GITHUB_TOKEN, debug_logs)
            if current_content is None:
                debug_logs.put({
                    "type": "Edit Error",
                    "data": f"Could not retrieve {filename} from GitHub"
                })
                continue
            
            # Log the original file for debugging
            debug_logs.put({
                "type": "File Before Edits",
                "data": f"{filename}: {len(current_content)} characters, {len(current_content.split(chr(10)))} lines",
                "fullData": f"Original content of {filename}:\n{current_content}"
            })
            
            # CRITICAL FIX: Sort edits by line number in DESCENDING order
            # This ensures we edit from bottom to top, so line numbers stay valid
            def get_sort_key(edit):
                operation = edit["operation"]
                data = edit["data"]
                if operation == "replace_lines":
                    return data.get("start_line", 0)
                elif operation == "insert_at_line":
                    return data.get("line", 0)
                return 0
            
            # Sort in descending order (highest line numbers first)
            edits_sorted = sorted(edits, key=get_sort_key, reverse=True)
            
            debug_logs.put({
                "type": "Edit Sorting", 
                "data": f"{filename}: Applying {len(edits_sorted)} edit(s) from bottom to top",
                "fullData": "\n".join([f"Edit {i+1}: {e['operation']} at line {get_sort_key(e)}" for i, e in enumerate(edits_sorted)])
            })
            
            # Apply each edit sequentially from bottom to top
            for edit_idx, edit in enumerate(edits_sorted):
                operation = edit["operation"]
                data = edit["data"]
                
                if operation == "replace_lines":
                    start_line = data.get("start_line")
                    end_line = data.get("end_line")
                    content = data.get("content", "")
                    
                    if start_line is None or end_line is None:
                        continue
                    
                    lines = current_content.split('\n')
                    total_lines = len(lines)
                    
                    debug_logs.put({
                        "type": "Replace Lines",
                        "data": f"{filename}: Replacing lines {start_line}-{end_line} (file has {total_lines} lines)",
                        "fullData": f"Original lines {start_line}-{end_line}:\n" + 
                                   "\n".join([f"{i}: {lines[i-1]}" for i in range(start_line, min(end_line+1, total_lines+1)) if i-1 < len(lines)]) +
                                   f"\n\nNew content:\n{content}"
                    })
                    
                    new_lines = content.split('\n') if content else []
                    # Python slice: [start_line-1:end_line] replaces lines start_line through end_line (inclusive)
                    lines[start_line-1:end_line] = new_lines
                    current_content = '\n'.join(lines)
                    
                elif operation == "insert_at_line":
                    line_number = data.get("line")
                    content = data.get("content", "")
                    
                    if line_number is None:
                        continue
                    
                    lines = current_content.split('\n')
                    total_lines = len(lines)
                    
                    debug_logs.put({
                        "type": "Insert at Line",
                        "data": f"{filename}: Inserting at line {line_number} (file has {total_lines} lines)",
                        "fullData": f"Content to insert at line {line_number}:\n{content}"
                    })
                    
                    new_lines = content.split('\n') if content else []
                    lines[line_number-1:line_number-1] = new_lines
                    current_content = '\n'.join(lines)
            
            # Log the modified file for debugging
            debug_logs.put({
                "type": "File After Edits",
                "data": f"{filename}: {len(current_content)} characters, {len(current_content.split(chr(10)))} lines",
                "fullData": f"Modified content of {filename}:\n{current_content}"
            })
            
            # Handle requirements.txt merging
            if filename == "requirements.txt":
                current_content = merge_requirements(current_content, BASE_REQUIREMENTS)
            
            # Upload once with all edits applied
            update_github_file(filename, current_content, f"Update {filename} via DeepSeek",
                             GITHUB_USERNAME, GITHUB_REPO, GITHUB_BRANCH, 
                             GITHUB_TOKEN, debug_logs)
            files_updated.append(filename)
            
            if filename not in tracked_files:
                tracked_files.append(filename)
        
        return jsonify({
            "success": True,
            "deepseek_response": text_response if text_response else "",
            "files_updated": files_updated,
            "audio": audio_data
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        debug_logs.put({
            "type": "Server Error", 
            "data": str(e),
            "fullData": error_details
        })
        return jsonify({"error": str(e)}), 500


@app.route('/new_session', methods=['POST'])
def new_session():
    """Start a new session by clearing script.py."""
    try:
        if not GITHUB_TOKEN:
            return jsonify({"error": "GITHUB_TOKEN not set"}), 500
        
        update_github_file("script.py", "", "Clear script.py for new session",
                         GITHUB_USERNAME, GITHUB_REPO, GITHUB_BRANCH, 
                         GITHUB_TOKEN, debug_logs)
        
        if hasattr(get_output, 'accumulated'):
            get_output.accumulated = ""
        
        debug_logs.put({"type": "Session Reset", "data": "New session started"})
        
        return jsonify({"success": True})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    print("=" * 60)
    print("🐵 PRIMATE CODER - Starting...")
    print("=" * 60)
    print(f"Repository: {GITHUB_USERNAME}/{GITHUB_REPO}")
    print(f"Branch: {GITHUB_BRANCH}")
    print(f"Port: {PORT}")
    print()
    print("Environment variables:")
    print(f"  DEEPSEEK_API_KEY: {'✓ Set' if DEEPSEEK_API_KEY else '✗ Not set'}")
    print(f"  GITHUB_TOKEN: {'✓ Set' if GITHUB_TOKEN else '✗ Not set'}")
    print()
    print("Starting script.py execution...")
    print("=" * 60)
    
    start_script_thread()
    
    app.run(host='0.0.0.0', port=PORT, debug=False)
