import os
import json
import re
import base64
import subprocess
import threading
import time
from flask import Flask, render_template_string, request, jsonify
import requests
from queue import Queue
from io import BytesIO
from gtts import gTTS

# ==================== CONFIGURATION ====================
GITHUB_USERNAME = "constantinbender51-cmyk"
GITHUB_REPO = "Primate-Coder"
GITHUB_BRANCH = "main"
RAILWAY_PROJECT_ID = "your-project-id"  # Optional, for future use
PORT = 8080

# Base dependencies that must always be in requirements.txt
BASE_REQUIREMENTS = ["flask", "requests", "gtts"]

# Environment variables (set these before running)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

# ==================== GLOBAL STATE ====================
app = Flask(__name__)
script_output = Queue()  # Thread-safe queue for script output
script_process = None
tracked_files = ["script.py", "requirements.txt"]  # Files to track and send to DeepSeek

# ==================== HTML TEMPLATE ====================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Primate Coder</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            background: #fafafa;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border: 1px solid #e8e8e8;
            display: flex;
            flex-direction: column;
            height: 90vh;
        }
        .header {
            background: #ffffff;
            color: #1a1a1a;
            padding: 20px 30px;
            text-align: center;
            border-bottom: 1px solid #e8e8e8;
        }
        .header h1 {
            font-size: 1.5em;
            margin-bottom: 5px;
            font-weight: 500;
            color: #1a1a1a;
        }
        .header h1 .highlight {
            color: #4a9eff;
        }
        .header p {
            color: #666666;
            font-size: 0.85em;
            font-weight: 300;
        }
        .main-content {
            display: flex;
            flex: 1;
            min-height: 0;
            overflow-x: auto;
            overflow-y: hidden;
        }
        .output-panel {
            flex: 1;
            min-width: 400px;
            display: flex;
            flex-direction: column;
            border-right: 1px solid #e8e8e8;
            background: #ffffff;
            overflow: hidden;
        }
        .output-header {
            background: #ffffff;
            color: #1a1a1a;
            padding: 15px;
            font-weight: 400;
            border-bottom: 1px solid #e8e8e8;
            font-size: 0.9em;
        }
        .output-content {
            flex: 1;
            overflow-y: auto;
            overflow-x: auto;
            padding: 15px;
            font-family: 'SF Mono', 'Monaco', 'Consolas', 'Courier New', monospace;
            font-size: 0.85em;
            color: #4a9eff;
            white-space: pre-wrap;
            word-wrap: break-word;
            min-height: 0;
            background: #f8f9fa;
            border: 1px solid #e8e8e8;
            margin: 10px;
        }
        .chat-panel {
            flex: 1;
            min-width: 400px;
            display: flex;
            flex-direction: column;
            background: #ffffff;
            overflow: hidden;
        }
        .chat-header {
            background: #ffffff;
            padding: 15px;
            font-weight: 400;
            border-bottom: 1px solid #e8e8e8;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.9em;
            color: #1a1a1a;
        }
        .header-left {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        #ttsToggle {
            background: #ffffff;
            color: #666666;
            border: 1px solid #d0d0d0;
            font-size: 0.85em;
            padding: 6px 12px;
            cursor: pointer;
            transition: all 0.2s;
        }
        #ttsToggle:hover {
            border-color: #888888;
            color: #888888;
        }
        #ttsToggle.active {
            background: #888888;
            color: white;
            border-color: #888888;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            min-height: 0;
            background: #ffffff;
        }
        .message {
            margin-bottom: 15px;
            padding: 12px 15px;
            border: 1px solid #e8e8e8;
            animation: fadeIn 0.3s;
            font-size: 0.9em;
            line-height: 1.6;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .user-message {
            background: #f8f9fa;
            color: #1a1a1a;
            margin-left: 20%;
            border-color: #d0d0d0;
        }
        .assistant-message {
            background: #f5f5f5;
            color: #4a4a4a;
            margin-right: 20%;
            border-color: #d0d0d0;
        }
        .status-message {
            background: #ffffff;
            color: #888888;
            text-align: center;
            font-size: 0.85em;
            border-color: #e8e8e8;
            font-style: italic;
        }
        .error-message {
            background: #fff5f5;
            color: #666666;
            border-color: #d0d0d0;
            border-left-width: 3px;
        }
        .success-message {
            background: #f8f9fa;
            color: #666666;
            border-color: #d0d0d0;
            border-left-width: 3px;
        }
        .chat-input-area {
            padding: 20px;
            background: white;
            border-top: 1px solid #e8e8e8;
        }
        .input-wrapper {
            display: flex;
            gap: 10px;
        }
        #userInput {
            flex: 1;
            padding: 12px;
            border: 1px solid #d0d0d0;
            font-size: 0.9em;
            resize: vertical;
            min-height: 50px;
            font-family: inherit;
            background: #ffffff;
            color: #1a1a1a;
            transition: border-color 0.2s;
        }
        #userInput:focus {
            outline: none;
            border-color: #888888;
        }
        .btn {
            padding: 12px 24px;
            border: 1px solid #d0d0d0;
            font-size: 0.9em;
            font-weight: 400;
            cursor: pointer;
            transition: all 0.2s;
            background: #ffffff;
            color: #1a1a1a;
        }
        .btn:hover {
            border-color: #888888;
            color: #888888;
        }
        .btn:disabled {
            background: #f5f5f5;
            color: #cccccc;
            border-color: #e8e8e8;
            cursor: not-allowed;
        }
        #sendBtn {
            background: #888888;
            color: white;
            border-color: #888888;
        }
        #sendBtn:hover {
            background: #666666;
            border-color: #666666;
            color: white;
        }
        #sendBtn:disabled {
            background: #e8e8e8;
            border-color: #e8e8e8;
        }
        #newSessionBtn {
            background: #ffffff;
            color: #666666;
            border-color: #d0d0d0;
            font-size: 0.85em;
            padding: 8px 16px;
        }
        #newSessionBtn:hover {
            border-color: #888888;
            color: #1a1a1a;
        }
        .loading {
            display: inline-block;
            width: 12px;
            height: 12px;
            border: 1px solid #e8e8e8;
            border-top: 1px solid #888888;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 8px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐵 Prima<span class="highlight">t</span>e Coder</h1>
            <p>AI-powered code generation with DeepSeek</p>
        </div>
        <div class="main-content">
            <div class="output-panel">
                <div class="output-header">📟 Script Output (script.py)</div>
                <div class="output-content" id="outputContent">Waiting for script.py output...</div>
            </div>
            <div class="chat-panel">
                <div class="chat-header">
                    <div class="header-left">
                        <span>💬 Chat with DeepSeek</span>
                        <button id="ttsToggle" class="btn active" onclick="toggleTTS()">🔊 TTS On</button>
                    </div>
                    <button id="newSessionBtn" class="btn" onclick="startNewSession()">🔄 Start New Session</button>
                </div>
                <div class="chat-messages" id="chatMessages"></div>
                <div class="chat-input-area">
                    <div class="input-wrapper">
                        <textarea id="userInput" placeholder="Describe what you want to build..."></textarea>
                        <button id="sendBtn" class="btn" onclick="sendMessage()">Send</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let shouldAutoScroll = true;
        let chatHistory = [];  // Client-side chat history storage
        let ttsEnabled = true; // TTS enabled by default
        let currentAudio = null; // Track currently playing audio
        
        // Load TTS preference from localStorage
        const savedTTSPref = localStorage.getItem('primateTTSEnabled');
        if (savedTTSPref !== null) {
            ttsEnabled = savedTTSPref === 'true';
            updateTTSButton();
        }
        
        // Load chat history from localStorage on page load
        const savedHistory = localStorage.getItem('primateChatHistory');
        if (savedHistory) {
            try {
                chatHistory = JSON.parse(savedHistory);
                // Restore chat messages to UI
                chatHistory.forEach(msg => {
                    if (msg.role === 'user') {
                        addMessage(msg.content, 'user', false);
                    } else if (msg.role === 'assistant') {
                        addMessage('🤖 DeepSeek: ' + msg.content, 'assistant', false);
                    }
                });
            } catch (e) {
                console.error('Error loading chat history:', e);
                chatHistory = [];
            }
        }
        
        function toggleTTS() {
            ttsEnabled = !ttsEnabled;
            localStorage.setItem('primateTTSEnabled', ttsEnabled);
            updateTTSButton();
            
            // Stop current audio if disabling
            if (!ttsEnabled && currentAudio) {
                currentAudio.pause();
                currentAudio = null;
            }
        }
        
        function updateTTSButton() {
            const btn = document.getElementById('ttsToggle');
            if (ttsEnabled) {
                btn.textContent = '🔊 TTS On';
                btn.classList.add('active');
            } else {
                btn.textContent = '🔇 TTS Off';
                btn.classList.remove('active');
            }
        }
        
        function playAudio(audioData) {
            if (!ttsEnabled) return;
            
            // Stop any currently playing audio
            if (currentAudio) {
                currentAudio.pause();
            }
            
            currentAudio = new Audio(audioData);
            currentAudio.play().catch(err => {
                console.error('Error playing audio:', err);
            });
            
            currentAudio.onended = () => {
                currentAudio = null;
            };
        }
        
        // Add scroll detection to output panel
        const outputDiv = document.getElementById('outputContent');
        outputDiv.addEventListener('scroll', function() {
            const scrollThreshold = 50; // pixels from bottom
            const distanceFromBottom = outputDiv.scrollHeight - outputDiv.scrollTop - outputDiv.clientHeight;
            shouldAutoScroll = distanceFromBottom < scrollThreshold;
        });
        
        // Poll for script output
        setInterval(async () => {
            try {
                const response = await fetch('/get_output');
                const data = await response.json();
                if (data.output) {
                    outputDiv.textContent = data.output;
                    // Only auto-scroll if user is near the bottom
                    if (shouldAutoScroll) {
                        outputDiv.scrollTop = outputDiv.scrollHeight;
                    }
                }
            } catch (error) {
                console.error('Error fetching output:', error);
            }
        }, 1000);

        function addMessage(content, type, saveToHistory = true) {
            const chatMessages = document.getElementById('chatMessages');
            const msg = document.createElement('div');
            msg.className = `message ${type}-message`;
            msg.innerHTML = content;
            chatMessages.appendChild(msg);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function saveChatHistory() {
            // Limit to last 20 messages to avoid localStorage limits
            if (chatHistory.length > 20) {
                chatHistory = chatHistory.slice(-20);
            }
            localStorage.setItem('primateChatHistory', JSON.stringify(chatHistory));
        }

        async function sendMessage() {
            const input = document.getElementById('userInput');
            const btn = document.getElementById('sendBtn');
            const message = input.value.trim();
            
            if (!message) return;
            
            addMessage(message, 'user');
            
            // Add to chat history
            chatHistory.push({
                role: 'user',
                content: message
            });
            saveChatHistory();
            
            input.value = '';
            btn.disabled = true;
            
            addMessage('<span class="loading"></span>Processing your request...', 'status');
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        message: message,
                        chat_history: chatHistory
                    })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    addMessage('❌ Error: ' + data.error, 'error');
                } else {
                    if (data.deepseek_response) {
                        addMessage('🤖 DeepSeek: ' + data.deepseek_response, 'assistant');
                        
                        // Add assistant response to chat history
                        chatHistory.push({
                            role: 'assistant',
                            content: data.deepseek_response
                        });
                        saveChatHistory();
                        
                        // Play TTS audio if available
                        if (data.audio && ttsEnabled) {
                            playAudio(data.audio);
                        }
                    }
                    if (data.files_updated && data.files_updated.length > 0) {
                        addMessage('✅ Updated files: ' + data.files_updated.join(', '), 'success');
                        addMessage('🚀 Files pushed to GitHub. Railway will redeploy automatically...', 'success');
                    }
                }
            } catch (error) {
                addMessage('❌ Error: ' + error.message, 'error');
            }
            
            btn.disabled = false;
        }

        async function startNewSession() {
            if (!confirm('This will clear script.py and chat history. Continue?')) {
                return;
            }
            
            try {
                const response = await fetch('/new_session', { method: 'POST' });
                const data = await response.json();
                
                if (data.success) {
                    // Clear chat history
                    chatHistory = [];
                    localStorage.removeItem('primateChatHistory');
                    
                    // Clear chat UI
                    document.getElementById('chatMessages').innerHTML = '';
                    
                    addMessage('🔄 New session started. script.py and chat history cleared.', 'success');
                } else {
                    addMessage('❌ Error: ' + data.error, 'error');
                }
            } catch (error) {
                addMessage('❌ Error: ' + error.message, 'error');
            }
        }

        document.getElementById('userInput').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>
"""

# ==================== SCRIPT EXECUTION ====================

def run_script():
    """Run script.py and capture its output."""
    global script_process
    
    # Check if script.py exists and is not empty
    if not os.path.exists('script.py'):
        script_output.put("script.py not found\n")
        return
    
    with open('script.py', 'r') as f:
        content = f.read().strip()
        if not content:
            script_output.put("script.py is empty\n")
            return
    
    try:
        # Run script.py as subprocess
        script_process = subprocess.Popen(
            ['python', 'script.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Read output line by line
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


# ==================== GITHUB API ====================

def get_file_from_github(filepath):
    """Get file content from GitHub."""
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents/{filepath}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        response = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
        if response.status_code == 200:
            content_b64 = response.json().get("content", "")
            content = base64.b64decode(content_b64).decode('utf-8')
            return content
        return None
    except Exception as e:
        print(f"Error getting {filepath} from GitHub: {e}")
        return None


def update_github_file(filepath, content, commit_message):
    """Update or create a file in the GitHub repository."""
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents/{filepath}"
    
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Get current file SHA if it exists
    response = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
    sha = None
    if response.status_code == 200:
        sha = response.json().get("sha")
    
    # Encode content to base64
    content_b64 = base64.b64encode(content.encode('utf-8')).decode('utf-8')
    
    payload = {
        "message": commit_message,
        "content": content_b64,
        "branch": GITHUB_BRANCH
    }
    
    if sha:
        payload["sha"] = sha
    
    response = requests.put(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


def list_repo_files():
    """List all files in the repository."""
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        response = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
        if response.status_code == 200:
            files = response.json()
            return [f["name"] for f in files if f["type"] == "file" and f["name"] != "main.py"]
        return []
    except Exception as e:
        print(f"Error listing repo files: {e}")
        return []


# ==================== DEEPSEEK API ====================

def call_deepseek_api(user_message, file_contents, script_output_text, chat_history):
    """Call DeepSeek API with coding agent prompt."""
    
    system_prompt = """You are a coding agent with the ability to create and edit files.

IMPORTANT WORKFLOW:
1. Before making ANY changes, explain what you plan to do
2. Ask the user for confirmation (e.g., "Should I proceed with these changes?")
3. Wait for the user to confirm (they might say "yes", "confirm", "go ahead", or provide modifications)
4. Only after receiving confirmation, provide the file changes in JSON format

IMPORTANT: The main executable file is 'script.py' which will be run automatically. When you create or modify code, put it in script.py.

CRITICAL - DEPENDENCY MANAGEMENT:
- Whenever you write code that uses external packages (imports), you MUST update requirements.txt
- Always include ALL packages needed by your code in requirements.txt
- Do NOT include flask, requests, or gtts in requirements.txt as they are already available
- If you use packages like numpy, pandas, beautifulsoup4, selenium, pillow, opencv-python, matplotlib, scikit-learn, etc., you MUST add them to requirements.txt
- Format: one package per line, optionally with version (e.g., "numpy==1.24.0" or just "numpy")

To create or edit files, include JSON objects in your response with this format:
{
  "filename.py": "file content here",
  "requirements.txt": "package1\\npackage2",
  "style.css": "css content here"
}

You can create any files needed (script.py, requirements.txt, index.html, style.css, etc.).

When updating requirements.txt, only include additional packages needed by script.py. Do not include flask, requests, or gtts as they are already available.

Current files in the repository:
""" + "\n".join([f"- {name}: {len(content)} characters" for name, content in file_contents.items()])

    if script_output_text:
        system_prompt += f"\n\nCurrent script.py output:\n{script_output_text}"

    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Include file contents in the user message
    context = "\n\n".join([f"=== {name} ===\n{content}" for name, content in file_contents.items()])
    full_message = f"{context}\n\n=== User Request ===\n{user_message}"
    
    # Build messages array with chat history
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history from client
    messages.extend(chat_history)
    
    # Add current user message
    messages.append({"role": "user", "content": full_message})
    
    payload = {
        "model": "deepseek-coder",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 8000
    }
    
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    
    data = response.json()
    if "choices" in data and len(data["choices"]) > 0:
        return data["choices"][0]["message"]["content"]
    
    raise Exception("No valid response from DeepSeek")


def extract_json_from_text(text):
    """Extract all JSON objects from text."""
    json_objects = []
    brace_count = 0
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                json_str = text[start_idx:i+1]
                try:
                    obj = json.loads(json_str)
                    json_objects.append(obj)
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    
    return json_objects


def remove_json_from_text(text):
    """Remove JSON objects from text to get plain text response."""
    result = text
    brace_count = 0
    start_idx = -1
    ranges_to_remove = []
    
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                json_str = text[start_idx:i+1]
                try:
                    json.loads(json_str)
                    ranges_to_remove.append((start_idx, i+1))
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    
    # Remove ranges in reverse order
    for start, end in reversed(ranges_to_remove):
        result = result[:start] + result[end:]
    
    return result.strip()


def merge_requirements(deepseek_requirements):
    """Merge DeepSeek requirements with base requirements."""
    lines = [line.strip() for line in deepseek_requirements.split('\n') if line.strip()]
    
    # Add base requirements if not present
    for base_req in BASE_REQUIREMENTS:
        if not any(line.lower().startswith(base_req.lower()) for line in lines):
            lines.insert(0, base_req)
    
    return '\n'.join(lines)


def generate_tts_audio(text):
    """Generate TTS audio from text using gTTS."""
    try:
        # Remove emojis and special characters that might cause issues
        clean_text = text.strip()
        
        if not clean_text:
            return None
        
        # Generate speech
        tts = gTTS(text=clean_text, lang='en', slow=False)
        
        # Save to BytesIO buffer
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Convert to base64
        audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')
        
        return f"data:audio/mp3;base64,{audio_base64}"
    
    except Exception as e:
        print(f"TTS Error: {e}")
        return None


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
    
    # Store accumulated output
    if not hasattr(get_output, 'accumulated'):
        get_output.accumulated = ""
    
    get_output.accumulated += ''.join(output_lines)
    
    return jsonify({"output": get_output.accumulated})


@app.route('/generate', methods=['POST'])
def generate():
    """Handle code generation request."""
    try:
        data = request.json
        user_message = data.get('message', '')
        chat_history = data.get('chat_history', [])  # Receive chat history from client
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        if not DEEPSEEK_API_KEY:
            return jsonify({"error": "DEEPSEEK_API_KEY not set"}), 500
        
        if not GITHUB_TOKEN:
            return jsonify({"error": "GITHUB_TOKEN not set"}), 500
        
        # Get all tracked files from repo
        global tracked_files
        tracked_files = list_repo_files()
        
        file_contents = {}
        for filename in tracked_files:
            content = get_file_from_github(filename)
            if content is not None:
                file_contents[filename] = content
        
        # Get current script output
        script_output_text = getattr(get_output, 'accumulated', '')
        
        # Call DeepSeek API with chat history
        deepseek_response = call_deepseek_api(user_message, file_contents, script_output_text, chat_history)
        
        # Extract JSON objects from response
        json_objects = extract_json_from_text(deepseek_response)
        
        # Get plain text response (without JSON)
        text_response = remove_json_from_text(deepseek_response)
        
        # Generate TTS audio for DeepSeek response
        audio_data = None
        if text_response:
            audio_data = generate_tts_audio(text_response)
        
        # Update files on GitHub
        files_updated = []
        for json_obj in json_objects:
            for filename, content in json_obj.items():
                # Special handling for requirements.txt
                if filename == "requirements.txt":
                    content = merge_requirements(content)
                
                update_github_file(filename, content, f"Update {filename} via DeepSeek")
                files_updated.append(filename)
                
                # Add to tracked files if new
                if filename not in tracked_files:
                    tracked_files.append(filename)
        
        return jsonify({
            "success": True,
            "deepseek_response": text_response if text_response else "Files updated successfully",
            "files_updated": files_updated,
            "audio": audio_data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/new_session', methods=['POST'])
def new_session():
    """Start a new session by clearing script.py."""
    try:
        if not GITHUB_TOKEN:
            return jsonify({"error": "GITHUB_TOKEN not set"}), 500
        
        # Clear script.py
        update_github_file("script.py", "", "Clear script.py for new session")
        
        # Reset output
        if hasattr(get_output, 'accumulated'):
            get_output.accumulated = ""
        
        # Note: Chat history is now managed client-side
        
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
    
    # Start script.py in background
    start_script_thread()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=PORT, debug=False)
