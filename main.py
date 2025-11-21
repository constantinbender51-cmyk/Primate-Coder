import os
import subprocess
import threading
import time
from flask import Flask, render_template_string, request, jsonify
from queue import Queue

# Import from other files
from github_api import get_file_from_github, update_github_file, list_repo_files, delete_github_file
from deepseek_api import (call_deepseek_api, extract_json_from_text, remove_json_from_text,
                          merge_requirements, generate_tts_audio, analyze_script_output)

# Import Railway integration
try:
    from railway_integration import get_deployment_status, start_railway_monitor
    RAILWAY_AVAILABLE = True
except ImportError:
    RAILWAY_AVAILABLE = False
    print("Warning: Railway integration module not found")

# ==================== CONFIGURATION ====================
GITHUB_USERNAME = "constantinbender51-cmyk"
GITHUB_REPO = "Primate-Coder"
GITHUB_BRANCH = "main"
PORT = 8080
MAX_AUTO_RETRY_ATTEMPTS = 5

BASE_REQUIREMENTS = ["flask", "requests", "gtts"]

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

# ==================== GLOBAL STATE ====================
app = Flask(__name__)
script_output = Queue()
script_process = None
script_exit_code = None
script_is_running = False
tracked_files = ["script.py", "requirements.txt"]
debug_logs = Queue()
auto_retry_in_progress = False
auto_retry_messages = Queue()

# ==================== HTML TEMPLATE ====================
# Load the updated HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Primate Coder</title>
    <script src="https://cdn.jsdelivr.net/npm/marked@4.3.0/marked.min.js"></script>
    <style>
        /* Include the updated CSS from the HTML template artifact */
    </style>
</head>
<body>
    <!-- Include the updated HTML from the HTML template artifact -->
</body>
</html>
"""

# ==================== SCRIPT EXECUTION ====================

def run_script():
    """Run script.py and capture its output."""
    global script_process, script_exit_code, script_is_running
    
    if not os.path.exists('script.py'):
        script_output.put("script.py not found\n")
        script_is_running = False
        script_exit_code = -1
        return
    
    with open('script.py', 'r') as f:
        content = f.read().strip()
        if not content:
            script_output.put("script.py is empty\n")
            script_is_running = False
            script_exit_code = -1
            return
    
    try:
        script_is_running = True
        script_exit_code = None
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
        script_exit_code = script_process.returncode
        script_output.put(f"\n[Process exited with code {script_exit_code}]\n")
        script_is_running = False
        debug_logs.put({
            "type": "Script Completed",
            "data": f"script.py finished with exit code {script_exit_code}"
        })
        
    except Exception as e:
        script_output.put(f"Error running script.py: {str(e)}\n")
        script_is_running = False
        script_exit_code = -1
        debug_logs.put({
            "type": "Script Error",
            "data": f"script.py error: {str(e)}"
        })


def start_script_thread():
    """Start script.py in a background thread."""
    thread = threading.Thread(target=run_script, daemon=True)
    thread.start()


def wait_for_script_completion(timeout=300):
    """Wait for script to complete execution with timeout."""
    start_time = time.time()
    debug_logs.put({
        "type": "Wait for Script",
        "data": f"Waiting for script to complete (timeout: {timeout}s)..."
    })
    
    while (time.time() - start_time) < timeout:
        if not script_is_running and script_exit_code is not None:
            debug_logs.put({
                "type": "Script Completed",
                "data": f"Script finished with exit code {script_exit_code}"
            })
            return True
        time.sleep(1)
    
    debug_logs.put({
        "type": "Wait Timeout",
        "data": f"Script did not complete within {timeout} seconds"
    })
    return False


# ==================== FLASK ROUTES ====================

@app.route('/')
def index():
    # Use the updated HTML template from the artifact
    try:
        with open('html_template.py', 'r') as f:
            exec(f.read(), globals())
            return render_template_string(HTML_TEMPLATE)
    except:
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
        logs.append(debug_logs.get())
    
    return jsonify({"logs": logs})


@app.route('/get_auto_retry_messages')
def get_auto_retry_messages():
    """Get auto-retry status messages for the frontend."""
    messages = []
    while not auto_retry_messages.empty():
        messages.append(auto_retry_messages.get())
    
    return jsonify({"messages": messages})


@app.route('/generate', methods=['POST'])
def generate():
    """Handle code generation request."""
    global auto_retry_in_progress
    
    try:
        data = request.json
        user_message = data.get('message', '')
        chat_history = data.get('chat_history', [])
        is_auto_retry = data.get('is_auto_retry', False)
        retry_attempt = data.get('retry_attempt', 0)
        
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
                                              chat_history, DEEPSEEK_API_KEY, debug_logs, is_auto_retry)
        
        json_objects = extract_json_from_text(deepseek_response)
        text_response = remove_json_from_text(deepseek_response)
        
        audio_data = None
        if text_response and not is_auto_retry:
            audio_data = generate_tts_audio(text_response, debug_logs)
        
        file_edits = {}
        files_updated = []
        files_deleted = []
        
        for json_obj in json_objects:
            if "operation" in json_obj:
                filename = json_obj.get("file")
                operation = json_obj.get("operation")
                
                if not filename:
                    continue
                
                if operation == "delete":
                    delete_github_file(filename, f"Delete {filename} via DeepSeek",
                                     GITHUB_USERNAME, GITHUB_REPO, GITHUB_BRANCH,
                                     GITHUB_TOKEN, debug_logs)
                    files_deleted.append(filename)
                    if filename in tracked_files:
                        tracked_files.remove(filename)
                    continue
                
                if filename not in file_edits:
                    file_edits[filename] = []
                
                file_edits[filename].append({
                    "operation": operation,
                    "data": json_obj
                })
                
            else:
                for filename, content in json_obj.items():
                    if filename == "requirements.txt":
                        content = merge_requirements(content, BASE_REQUIREMENTS)
                    
                    update_github_file(filename, content, f"Update {filename} via DeepSeek",
                                     GITHUB_USERNAME, GITHUB_REPO, GITHUB_BRANCH, 
                                     GITHUB_TOKEN, debug_logs)
                    files_updated.append(filename)
                    
                    if filename not in tracked_files:
                        tracked_files.append(filename)
        
        for filename, edits in file_edits.items():
            current_content = get_file_from_github(filename, GITHUB_USERNAME, GITHUB_REPO,
                                                   GITHUB_BRANCH, GITHUB_TOKEN, debug_logs)
            if current_content is None:
                continue
            
            def get_sort_key(edit):
                operation = edit["operation"]
                data = edit["data"]
                if operation == "replace_lines":
                    return data.get("start_line", 0)
                elif operation == "insert_at_line":
                    return data.get("line", 0)
                return 0
            
            edits_sorted = sorted(edits, key=get_sort_key, reverse=True)
            
            debug_logs.put({
                "type": "Edit Sorting", 
                "data": f"{filename}: Applying {len(edits_sorted)} edits from bottom to top"
            })
            
            for edit in edits_sorted:
                operation = edit["operation"]
                data = edit["data"]
                
                if operation == "replace_lines":
                    start_line = data.get("start_line")
                    end_line = data.get("end_line")
                    content = data.get("content", "")
                    
                    if start_line is None or end_line is None:
                        continue
                    
                    debug_logs.put({
                        "type": "Replace Lines",
                        "data": f"{filename}: Lines {start_line}-{end_line}"
                    })
                    
                    lines = current_content.split('\n')
                    new_lines = content.split('\n') if content else []
                    lines[start_line-1:end_line] = new_lines
                    current_content = '\n'.join(lines)
                    
                elif operation == "insert_at_line":
                    line_number = data.get("line")
                    content = data.get("content", "")
                    
                    if line_number is None:
                        continue
                    
                    debug_logs.put({
                        "type": "Insert at Line",
                        "data": f"{filename}: Line {line_number}"
                    })
                    
                    lines = current_content.split('\n')
                    new_lines = content.split('\n') if content else []
                    lines[line_number-1:line_number-1] = new_lines
                    current_content = '\n'.join(lines)
            
            if filename == "requirements.txt":
                current_content = merge_requirements(current_content, BASE_REQUIREMENTS)
            
            update_github_file(filename, current_content, f"Update {filename} via DeepSeek",
                             GITHUB_USERNAME, GITHUB_REPO, GITHUB_BRANCH, 
                             GITHUB_TOKEN, debug_logs)
            files_updated.append(filename)
            
            if filename not in tracked_files:
                tracked_files.append(filename)
        
        script_confirmed_working = "SCRIPT_WORKING_CORRECTLY" in deepseek_response.upper()
        
        response_data = {
            "success": True,
            "deepseek_response": text_response if text_response else "",
            "files_updated": files_updated,
            "files_deleted": files_deleted,
            "audio": audio_data,
            "script_confirmed_working": script_confirmed_working,
            "is_auto_retry": is_auto_retry,
            "retry_attempt": retry_attempt
        }
        
        if (files_updated or files_deleted) and not is_auto_retry and not script_confirmed_working:
            threading.Thread(
                target=auto_retry_loop,
                args=(chat_history, user_message, text_response),
                daemon=True
            ).start()
        
        return jsonify(response_data)
        
    except Exception as e:
        debug_logs.put({"type": "Server Error", "data": str(e)})
        return jsonify({"error": str(e)}), 500


def auto_retry_loop(chat_history, original_user_message, assistant_response):
    """Auto-retry loop that analyzes script output and fixes issues."""
    global auto_retry_in_progress, script_exit_code, auto_retry_messages, script_is_running
    
    if auto_retry_in_progress:
        debug_logs.put({
            "type": "Auto-Retry",
            "data": "Already in progress, skipping duplicate"
        })
        return
    
    auto_retry_in_progress = True
    
    try:
        debug_logs.put({
            "type": "Auto-Retry",
            "data": "Waiting for Railway redeployment and script to start..."
        })
        auto_retry_messages.put({
            "type": "status",
            "content": "⏳ Waiting for Railway to redeploy and script to start..."
        })
        
        time.sleep(20)
        
        wait_start_time = time.time()
        while not script_is_running and (time.time() - wait_start_time) < 600:
            time.sleep(2)
        
        if not script_is_running:
            debug_logs.put({
                "type": "Auto-Retry",
                "data": "Script did not start running within 600 seconds"
            })
            auto_retry_messages.put({
                "type": "error",
                "content": "⚠️ Script did not start running after deployment"
            })
            return
        
        debug_logs.put({
            "type": "Auto-Retry",
            "data": "Script is now running, waiting for completion..."
        })
        
        for attempt in range(1, MAX_AUTO_RETRY_ATTEMPTS + 1):
            debug_logs.put({
                "type": "Auto-Retry",
                "data": f"Attempt {attempt}/{MAX_AUTO_RETRY_ATTEMPTS} - Waiting for script to complete..."
            })
            auto_retry_messages.put({
                "type": "auto-retry",
                "content": f"🔄 Auto-Retry Attempt {attempt}/{MAX_AUTO_RETRY_ATTEMPTS}: Waiting for script to complete..."
            })
            
            completed = wait_for_script_completion(timeout=3000)
            
            if not completed:
                debug_logs.put({
                    "type": "Auto-Retry Timeout",
                    "data": f"Attempt {attempt}: Script did not complete within timeout"
                })
                auto_retry_messages.put({
                    "type": "error",
                    "content": f"⏱️ Timeout: Script did not complete within 5 minutes"
                })
                break
            
            debug_logs.put({
                "type": "Auto-Retry",
                "data": f"Attempt {attempt}: Script completed with exit code {script_exit_code}"
            })
            
            script_output_text = getattr(get_output, 'accumulated', '')
            
            if not script_output_text or script_output_text.strip() == "Waiting for script.py output...":
                debug_logs.put({
                    "type": "Auto-Retry",
                    "data": f"Attempt {attempt}: No output captured, skipping analysis"
                })
                auto_retry_messages.put({
                    "type": "error",
                    "content": f"❌ No output captured from script"
                })
                break
            
            debug_logs.put({
                "type": "Auto-Retry",
                "data": f"Attempt {attempt}: Analyzing output with DeepSeek..."
            })
            auto_retry_messages.put({
                "type": "status",
                "content": f"🤖 Analyzing script output with DeepSeek..."
            })
            
            global tracked_files
            tracked_files = list_repo_files(GITHUB_USERNAME, GITHUB_REPO, GITHUB_BRANCH, GITHUB_TOKEN)
            
            file_contents = {}
            for filename in tracked_files:
                content = get_file_from_github(filename, GITHUB_USERNAME, GITHUB_REPO,
                                              GITHUB_BRANCH, GITHUB_TOKEN, debug_logs)
                if content is not None:
                    file_contents[filename] = content
            
            retry_chat_history = list(chat_history)
            retry_chat_history.append({"role": "user", "content": original_user_message})
            retry_chat_history.append({"role": "assistant", "content": assistant_response})
            
            analysis_response = analyze_script_output(
                script_output_text,
                file_contents,
                retry_chat_history,
                DEEPSEEK_API_KEY,
                debug_logs,
                attempt
            )
            
            auto_retry_messages.put({
                "type": "assistant",
                "content": analysis_response
            })
            
            if "SCRIPT_WORKING_CORRECTLY" in analysis_response.upper():
                debug_logs.put({
                    "type": "Auto-Retry Success",
                    "data": f"Script confirmed working after {attempt} attempt(s)"
                })
                auto_retry_messages.put({
                    "type": "system",
                    "content": f"✅ Auto-retry complete: Script is working correctly after {attempt} attempt(s)!"
                })
                break
            
            json_objects = extract_json_from_text(analysis_response)
            
            if not json_objects:
                debug_logs.put({
                    "type": "Auto-Retry",
                    "data": f"Attempt {attempt}: No fixes provided by DeepSeek"
                })
                auto_retry_messages.put({
                    "type": "system",
                    "content": f"ℹ️ No code fixes needed"
                })
                break
            
            auto_retry_messages.put({
                "type": "status",
                "content": f"🔧 Applying fixes from DeepSeek..."
            })
            
            files_updated = apply_deepseek_fixes(json_objects, file_contents)
            
            if files_updated:
                debug_logs.put({
                    "type": "Auto-Retry",
                    "data": f"Attempt {attempt}: Applied fixes to {', '.join(files_updated)}"
                })
                auto_retry_messages.put({
                    "type": "system",
                    "content": f"📝 Updated files: {', '.join(files_updated)}"
                })
                auto_retry_messages.put({
                    "type": "system",
                    "content": f"🚀 Redeploying to Railway..."
                })
                
                time.sleep(300)
                
                wait_start_time = time.time()
                script_is_running = False
                while not script_is_running and (time.time() - wait_start_time) < 600:
                    time.sleep(2)
                
                if not script_is_running:
                    auto_retry_messages.put({
                        "type": "error",
                        "content": "⚠️ Script did not restart after fixes"
                    })
                    break
            else:
                break
        
        if attempt >= MAX_AUTO_RETRY_ATTEMPTS:
            auto_retry_messages.put({
                "type": "error",
                "content": f"⚠️ Reached maximum retry attempts ({MAX_AUTO_RETRY_ATTEMPTS}). Please check the output manually."
            })
        
    except Exception as e:
        debug_logs.put({
            "type": "Auto-Retry Error",
            "data": str(e)
        })
        auto_retry_messages.put({
            "type": "error",
            "content": f"❌ Auto-retry error: {str(e)}"
        })
    finally:
        auto_retry_in_progress = False


def apply_deepseek_fixes(json_objects, file_contents):
    """Apply fixes from DeepSeek's analysis."""
    files_updated = []
    file_edits = {}
    
    for json_obj in json_objects:
        if "operation" in json_obj:
            filename = json_obj.get("file")
            operation = json_obj.get("operation")
            
            if not filename:
                continue
            
            if operation == "delete":
                delete_github_file(filename, f"Auto-fix: Delete {filename}",
                                 GITHUB_USERNAME, GITHUB_REPO, GITHUB_BRANCH,
                                 GITHUB_TOKEN, debug_logs)
                files_updated.append(filename)
                continue
            
            if filename not in file_edits:
                file_edits[filename] = []
            
            file_edits[filename].append({
                "operation": operation,
                "data": json_obj
            })
        else:
            for filename, content in json_obj.items():
                if filename == "requirements.txt":
                    content = merge_requirements(content, BASE_REQUIREMENTS)
                
                update_github_file(filename, content, f"Auto-fix: Update {filename}",
                                 GITHUB_USERNAME, GITHUB_REPO, GITHUB_BRANCH,
                                 GITHUB_TOKEN, debug_logs)
                files_updated.append(filename)
    
    for filename, edits in file_edits.items():
        current_content = get_file_from_github(filename, GITHUB_USERNAME, GITHUB_REPO,
                                               GITHUB_BRANCH, GITHUB_TOKEN, debug_logs)
        if current_content is None:
            continue
        
        def get_sort_key(edit):
            operation = edit["operation"]
            data = edit["data"]
            if operation == "replace_lines":
                return data.get("start_line", 0)
            elif operation == "insert_at_line":
                return data.get("line", 0)
            return 0
        
        edits_sorted = sorted(edits, key=get_sort_key, reverse=True)
        
        for edit in edits_sorted:
            operation = edit["operation"]
            data = edit["data"]
            
            if operation == "replace_lines":
                start_line = data.get("start_line")
                end_line = data.get("end_line")
                content = data.get("content", "")
                
                if start_line is None or end_line is None:
                    continue
                
                lines = current_content.split('\n')
                new_lines = content.split('\n') if content else []
                lines[start_line-1:end_line] = new_lines
                current_content = '\n'.join(lines)
                
            elif operation == "insert_at_line":
                line_number = data.get("line")
                content = data.get("content", "")
                
                if line_number is None:
                    continue
                
                lines = current_content.split('\n')
                new_lines = content.split('\n') if content else []
                lines[line_number-1:line_number-1] = new_lines
                current_content = '\n'.join(lines)
        
        if filename == "requirements.txt":
            current_content = merge_requirements(current_content, BASE_REQUIREMENTS)
        
        update_github_file(filename, current_content, f"Auto-fix: Update {filename}",
                         GITHUB_USERNAME, GITHUB_REPO, GITHUB_BRANCH,
                         GITHUB_TOKEN, debug_logs)
        files_updated.append(filename)
    
    return files_updated


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
    
    # Start Railway monitoring if available
    if RAILWAY_AVAILABLE:
        print(f"  RAILWAY_API_KEY: {'✓ Set' if os.environ.get('RAILWAY_API_KEY') else '✗ Not set'}")
        print(f"  RAILWAY_PROJECT_ID: {'✓ Set' if os.environ.get('RAILWAY_PROJECT_ID') else '✗ Not set'}")
        if os.environ.get('RAILWAY_API_KEY') and os.environ.get('RAILWAY_PROJECT_ID'):
            start_railway_monitor(debug_logs)
            print("  Railway monitoring: ✓ Started")
    
    print()
    print("Starting script.py execution...")
    print("=" * 60)
    
    start_script_thread()
    
    app.run(host='0.0.0.0', port=PORT, debug=False)
