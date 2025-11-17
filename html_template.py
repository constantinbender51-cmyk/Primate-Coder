HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Primate Coder</title>
    <script src="https://cdn.jsdelivr.net/npm/marked@4.3.0/marked.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            background: #2a2a2a;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: #1a1a1a;
            border: 1px solid #3a3a3a;
            display: flex;
            flex-direction: column;
            height: 90vh;
        }
        .header {
            background: #1a1a1a;
            color: #ffffff;
            padding: 20px 30px;
            text-align: center;
            border-bottom: 1px solid #3a3a3a;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header-center {
            flex: 1;
        }
        .header h1 {
            font-size: 1.5em;
            margin-bottom: 5px;
            font-weight: 500;
            color: #ffffff;
        }
        .header h1 .highlight {
            color: #FF1669C5;
        }
        .header p {
            color: #888888;
            font-size: 0.85em;
            font-weight: 300;
        }
        .header-right {
            position: relative;
        }
        .dropdown {
            position: relative;
            display: inline-block;
        }
        .dropdown-button {
            background: #1a1a1a;
            color: #888888;
            border: 1px solid #3a3a3a;
            padding: 8px 16px;
            cursor: pointer;
            font-size: 0.85em;
            transition: all 0.2s;
        }
        .dropdown-button:hover {
            border-color: #FF1669C5;
            color: #FF1669C5;
        }
        .dropdown-content {
            display: none;
            position: absolute;
            right: 0;
            background: #1a1a1a;
            min-width: 180px;
            border: 1px solid #3a3a3a;
            z-index: 1000;
            margin-top: 5px;
        }
        .dropdown-content.show {
            display: block;
        }
        .dropdown-item {
            padding: 10px 15px;
            cursor: pointer;
            border-bottom: 1px solid #3a3a3a;
            color: #888888;
            transition: all 0.2s;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .dropdown-item:last-child {
            border-bottom: none;
        }
        .dropdown-item:hover {
            background: #2a2a2a;
            color: #FF1669C5;
        }
        .dropdown-item.active {
            color: #FF1669C5;
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
            border-right: 1px solid #3a3a3a;
            background: #1a1a1a;
            overflow: hidden;
        }
        .output-header {
            background: #1a1a1a;
            color: #ffffff;
            padding: 15px;
            font-weight: 400;
            border-bottom: 1px solid #3a3a3a;
            font-size: 0.9em;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .debug-toggle-btn {
            background: transparent;
            border: 1px solid #3a3a3a;
            color: #888888;
            padding: 4px 10px;
            cursor: pointer;
            font-size: 0.75em;
            transition: all 0.2s;
        }
        .debug-toggle-btn:hover {
            border-color: #FF1669C5;
            color: #FF1669C5;
        }
        .debug-toggle-btn.active {
            background: #FF1669C5;
            border-color: #FF1669C5;
            color: #ffffff;
        }
        .output-content {
            flex: 1;
            overflow-y: auto;
            overflow-x: auto;
            padding: 15px;
            font-family: 'SF Mono', 'Monaco', 'Consolas', 'Courier New', monospace;
            font-size: 0.85em;
            color: #FF1669C5;
            white-space: pre-wrap;
            word-wrap: break-word;
            min-height: 0;
            background: #0a0a0a;
            border: 1px solid #3a3a3a;
            margin: 10px;
        }
        .debug-console {
            flex: 1;
            overflow-y: auto;
            overflow-x: auto;
            padding: 15px;
            font-family: 'SF Mono', 'Monaco', 'Consolas', 'Courier New', monospace;
            font-size: 0.75em;
            color: #888888;
            white-space: pre-wrap;
            word-wrap: break-word;
            min-height: 0;
            background: #0a0a0a;
            border: 1px solid #3a3a3a;
            margin: 10px;
            display: none;
        }
        .debug-console.active {
            display: block;
        }
        .debug-entry {
            margin-bottom: 15px;
            padding: 10px;
            border-left: 3px solid #FF1669C5;
            background: #1a1a1a;
        }
        .debug-timestamp {
            color: #FF1669C5;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .debug-type {
            color: #00ff88;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .debug-data {
            color: #ffffff;
            margin-top: 5px;
        }
        .debug-full-data {
            margin-top: 10px;
            padding: 10px;
            background: #0a0a0a;
            border: 1px solid #3a3a3a;
            color: #888888;
            max-height: 300px;
            overflow: auto;
            display: none;
        }
        .debug-full-data.visible {
            display: block;
        }
        .debug-full-data pre {
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .debug-expand-btn {
            color: #FF1669C5;
            cursor: pointer;
            font-size: 0.9em;
            margin-top: 5px;
            display: inline-block;
        }
        .debug-expand-btn:hover {
            text-decoration: underline;
        }
        .chat-panel {
            flex: 1;
            min-width: 400px;
            display: flex;
            flex-direction: column;
            background: #2a2a2a;
            overflow: hidden;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            min-height: 0;
            background: #2a2a2a;
        }
        .message {
            margin-bottom: 15px;
            padding: 12px 15px;
            animation: fadeIn 0.3s;
            font-size: 0.9em;
            line-height: 1.6;
            color: #ffffff;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .user-message {
            background: transparent;
            color: #ffffff;
            margin-left: 20%;
        }
        .assistant-message {
            background: transparent;
            color: #ffffff;
            margin-right: 20%;
            border: 1px solid #FF1669C5;
            padding: 12px 15px;
        }
        .assistant-message code {
            background: #1a1a1a;
            padding: 2px 6px;
            border-radius: 3px;
            color: #FF1669C5;
            font-size: 0.9em;
        }
        .assistant-message pre {
            background: #1a1a1a;
            padding: 12px;
            border-radius: 4px;
            overflow-x: auto;
            margin: 10px 0;
        }
        .assistant-message pre code {
            background: transparent;
            padding: 0;
            color: #FF1669C5;
        }
        .assistant-message h1,
        .assistant-message h2,
        .assistant-message h3,
        .assistant-message h4 {
            margin: 10px 0 5px 0;
            font-weight: 600;
        }
        .assistant-message h1 { font-size: 1.3em; }
        .assistant-message h2 { font-size: 1.2em; }
        .assistant-message h3 { font-size: 1.1em; }
        .assistant-message h4 { font-size: 1em; }
        .assistant-message p {
            margin: 8px 0;
        }
        .assistant-message ul,
        .assistant-message ol {
            margin: 8px 0;
            padding-left: 25px;
        }
        .assistant-message li {
            margin: 4px 0;
        }
        .assistant-message strong {
            font-weight: 600;
        }
        .assistant-message em {
            font-style: italic;
        }
        .assistant-message a {
            color: #FF1669C5;
            text-decoration: underline;
        }
        .assistant-message a:hover {
            color: #ff3388;
        }
        .assistant-message blockquote {
            border-left: 3px solid #FF1669C5;
            padding-left: 15px;
            margin: 10px 0;
            color: #888888;
        }
        .status-message {
            background: transparent;
            color: #888888;
            text-align: center;
            font-size: 0.85em;
            font-style: italic;
        }
        .error-message {
            background: transparent;
            color: #ff4444;
            border-left: 3px solid #ff4444;
            padding-left: 12px;
        }
        .success-message {
            background: transparent;
            color: #00ff88;
            border-left: 3px solid #00ff88;
            padding-left: 12px;
        }
        .chat-input-area {
            padding: 20px;
            background: #2a2a2a;
            border-top: 1px solid #3a3a3a;
        }
        .input-wrapper {
            display: flex;
            gap: 10px;
        }
        #userInput {
            flex: 1;
            padding: 12px;
            border: 1px solid #3a3a3a;
            font-size: 0.9em;
            resize: vertical;
            min-height: 50px;
            font-family: inherit;
            background: #1a1a1a;
            color: #ffffff;
            transition: border-color 0.2s;
        }
        #userInput:focus {
            outline: none;
            border-color: #FF1669C5;
        }
        .btn {
            padding: 12px 24px;
            border: 1px solid #3a3a3a;
            font-size: 0.9em;
            font-weight: 400;
            cursor: pointer;
            transition: all 0.2s;
            background: #1a1a1a;
            color: #ffffff;
        }
        .btn:hover {
            border-color: #FF1669C5;
            color: #FF1669C5;
        }
        .btn:disabled {
            background: #0a0a0a;
            color: #3a3a3a;
            border-color: #3a3a3a;
            cursor: not-allowed;
        }
        #sendBtn {
            background: #ffffff;
            color: #1a1a1a;
            border-color: #1a1a1a;
        }
        #sendBtn:hover {
            background: #f0f0f0;
            border-color: #1a1a1a;
            color: #1a1a1a;
        }
        #sendBtn:disabled {
            background: #3a3a3a;
            border-color: #3a3a3a;
        }
        .loading {
            display: inline-block;
            width: 12px;
            height: 12px;
            border: 1px solid #3a3a3a;
            border-top: 1px solid #FF1669C5;
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
            <div></div>
            <div class="header-center">
                <h1>🐵 Prima<span class="highlight">t</span>e Coder</h1>
                <p>AI-powered code generation with DeepSeek</p>
            </div>
            <div class="header-right">
                <div class="dropdown">
                    <button class="dropdown-button" onclick="toggleDropdown()">⚙️ Options ▼</button>
                    <div class="dropdown-content" id="dropdownMenu">
                        <div class="dropdown-item" onclick="toggleTTS()">
                            <span id="ttsLabel">🔊 TTS</span>
                            <span id="ttsStatus">On</span>
                        </div>
                        <div class="dropdown-item" onclick="toggleDebug()">
                            <span>🐛 Debug</span>
                            <span id="debugStatus">Off</span>
                        </div>
                        <div class="dropdown-item" onclick="clearMemory()">
                            <span>🧹 Clear Memory</span>
                        </div>
                        <div class="dropdown-item" onclick="startNewSession()">
                            <span>🔄 New Session</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="main-content">
            <div class="output-panel">
                <div class="output-header">📟 Script Output (script.py)</div>
                <div class="output-content" id="outputContent">Waiting for script.py output...</div>
                <div class="output-header">
                    <span>🐛 Debug Console</span>
                    <button class="debug-toggle-btn" id="verboseToggle" onclick="toggleVerbose()">Show Full Payloads</button>
                </div>
                <div class="debug-console" id="debugConsole">No debug logs yet...</div>
            </div>
            <div class="chat-panel">
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
        let chatHistory = [];
        let ttsEnabled = true;
        let debugMode = false;
        let verboseMode = false;
        let currentAudio = null;
        let debugLogs = [];
        
        // Configure marked.js
        marked.setOptions({ 
            breaks: true, 
            gfm: true 
        });
        
        // Load TTS preference
        const savedTTSPref = localStorage.getItem('primateTTSEnabled');
        if (savedTTSPref !== null) {
            ttsEnabled = savedTTSPref === 'true';
            updateTTSStatus();
        }
        
        // Load Debug preference
        const savedDebugPref = localStorage.getItem('primateDebugEnabled');
        if (savedDebugPref !== null) {
            debugMode = savedDebugPref === 'true';
            updateDebugStatus();
            if (debugMode) {
                document.getElementById('debugConsole').classList.add('active');
            }
        }
        
        // Load Verbose preference
        const savedVerbosePref = localStorage.getItem('primateVerboseEnabled');
        if (savedVerbosePref !== null) {
            verboseMode = savedVerbosePref === 'true';
            updateVerboseStatus();
        }
        
        // Load chat history
        const savedHistory = localStorage.getItem('primateChatHistory');
        if (savedHistory) {
            try {
                chatHistory = JSON.parse(savedHistory);
                chatHistory.forEach(msg => {
                    if (msg.role === 'user') {
                        addMessage(msg.content, 'user', false);
                    } else if (msg.role === 'assistant') {
                        const htmlContent = marked.parse(msg.content);
                        addMessage('🤖 DeepSeek: ' + htmlContent, 'assistant', false);
                    } else if (msg.role === 'system') {
                        addMessage(msg.content, 'success', false);
                    }
                });
            } catch (e) {
                console.error('Error loading chat history:', e);
                chatHistory = [];
            }
        }
        
        function toggleDropdown() {
            document.getElementById('dropdownMenu').classList.toggle('show');
        }
        
        // Close dropdown when clicking outside
        window.onclick = function(event) {
            if (!event.target.matches('.dropdown-button')) {
                const dropdowns = document.getElementsByClassName('dropdown-content');
                for (let i = 0; i < dropdowns.length; i++) {
                    dropdowns[i].classList.remove('show');
                }
            }
        }
        
        function toggleTTS() {
            ttsEnabled = !ttsEnabled;
            localStorage.setItem('primateTTSEnabled', ttsEnabled);
            updateTTSStatus();
            if (!ttsEnabled && currentAudio) {
                currentAudio.pause();
                currentAudio = null;
            }
        }
        
        function updateTTSStatus() {
            document.getElementById('ttsStatus').textContent = ttsEnabled ? 'On' : 'Off';
        }
        
        function toggleDebug() {
            debugMode = !debugMode;
            localStorage.setItem('primateDebugEnabled', debugMode);
            updateDebugStatus();
            const console = document.getElementById('debugConsole');
            if (debugMode) {
                console.classList.add('active');
            } else {
                console.classList.remove('active');
            }
        }
        
        function updateDebugStatus() {
            document.getElementById('debugStatus').textContent = debugMode ? 'On' : 'Off';
        }
        
        function toggleVerbose() {
            verboseMode = !verboseMode;
            localStorage.setItem('primateVerboseEnabled', verboseMode);
            updateVerboseStatus();
            updateDebugConsole();
        }
        
        function updateVerboseStatus() {
            const btn = document.getElementById('verboseToggle');
            if (verboseMode) {
                btn.classList.add('active');
                btn.textContent = 'Hide Full Payloads';
            } else {
                btn.classList.remove('active');
                btn.textContent = 'Show Full Payloads';
            }
        }
        
        function clearMemory() {
            if (!confirm('Clear chat history? Files will not be affected.')) return;
            chatHistory = [];
            localStorage.removeItem('primateChatHistory');
            document.getElementById('chatMessages').innerHTML = '';
            addMessage('🧹 Chat memory cleared.', 'success');
        }
        
        function addDebugLog(type, data, fullData = null) {
            const timestamp = new Date().toLocaleTimeString();
            debugLogs.push({ timestamp, type, data, fullData });
            if (debugLogs.length > 100) debugLogs.shift();
            updateDebugConsole();
        }
        
        function toggleFullData(index) {
            const elem = document.getElementById('debug-full-' + index);
            if (elem) {
                elem.classList.toggle('visible');
            }
        }
        
        function updateDebugConsole() {
            const console = document.getElementById('debugConsole');
            if (debugLogs.length === 0) {
                console.innerHTML = 'No debug logs yet...';
                return;
            }
            let html = '';
            debugLogs.forEach((log, index) => {
                html += '<div class="debug-entry">' +
                        '<div class="debug-timestamp">⏱ ' + log.timestamp + '</div>' +
                        '<div class="debug-type">📡 ' + log.type + '</div>' +
                        '<div class="debug-data">' + log.data + '</div>';
                
                if (log.fullData && verboseMode) {
                    html += '<div class="debug-expand-btn" onclick="toggleFullData(' + index + ')">▼ Show Full Data</div>' +
                            '<div class="debug-full-data" id="debug-full-' + index + '">' +
                            '<pre>' + escapeHtml(log.fullData) + '</pre>' +
                            '</div>';
                }
                
                html += '</div>';
            });
            console.innerHTML = html;
            if (debugMode) {
                console.scrollTop = console.scrollHeight;
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function playAudio(audioData) {
            if (!ttsEnabled) return;
            if (currentAudio) currentAudio.pause();
            currentAudio = new Audio(audioData);
            currentAudio.play().catch(err => {
                console.error('Error playing audio:', err);
                addDebugLog('TTS Error', 'Failed to play audio');
            });
            currentAudio.onended = () => currentAudio = null;
            addDebugLog('TTS Generated', 'Audio playing');
        }
        
        // Output panel scroll detection
        const outputDiv = document.getElementById('outputContent');
        outputDiv.addEventListener('scroll', function() {
            const distanceFromBottom = outputDiv.scrollHeight - outputDiv.scrollTop - outputDiv.clientHeight;
            shouldAutoScroll = distanceFromBottom < 50;
        });
        
        // Poll for script output
        setInterval(async () => {
            try {
                const response = await fetch('/get_output');
                const data = await response.json();
                if (data.output) {
                    outputDiv.textContent = data.output;
                    if (shouldAutoScroll) {
                        outputDiv.scrollTop = outputDiv.scrollHeight;
                    }
                }
            } catch (error) {
                console.error('Error fetching output:', error);
            }
        }, 1000);
        
        // Poll for debug logs
        setInterval(async () => {
            try {
                const response = await fetch('/get_debug_logs');
                const data = await response.json();
                if (data.logs && data.logs.length > 0) {
                    data.logs.forEach(log => addDebugLog(log.type, log.data));
                }
            } catch (error) {
                console.error('Error fetching debug logs:', error);
            }
        }, 1000);
        
        function addMessage(content, type, saveToHistory = true) {
            const chatMessages = document.getElementById('chatMessages');
            const msg = document.createElement('div');
            msg.className = 'message ' + type + '-message';
            msg.innerHTML = content;
            chatMessages.appendChild(msg);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            // Save system messages to chat history
            if (saveToHistory && (type === 'success' || type === 'error')) {
                const textContent = content.replace(/<[^>]*>/g, '').replace(/[✅❌🚀🧹🔄]/g, '').trim();
                chatHistory.push({ role: 'system', content: textContent });
                saveChatHistory();
            }
            
            return msg;
        }
        
        function saveChatHistory() {
            if (chatHistory.length > 30) {
                chatHistory = chatHistory.slice(-30);
            }
            localStorage.setItem('primateChatHistory', JSON.stringify(chatHistory));
        }
        
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const btn = document.getElementById('sendBtn');
            const message = input.value.trim();
            
            if (!message) return;
            
            addMessage(message, 'user');
            chatHistory.push({ role: 'user', content: message });
            saveChatHistory();
            
            input.value = '';
            btn.disabled = true;
            
            const statusMsg = addMessage('<span class="loading"></span>Processing your request...', 'status');
            
            const requestPayload = { message, chat_history: chatHistory };
            addDebugLog(
                'Client → Server', 
                'POST /generate | Message length: ' + message.length + ' | History items: ' + chatHistory.length,
                JSON.stringify(requestPayload, null, 2)
            );
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestPayload)
                });
                
                // Log response status
                addDebugLog('Server → Client', 'Status: ' + response.status + ' ' + response.statusText);
                
                // Get raw response text first
                const responseText = await response.text();
                addDebugLog(
                    'Server Response (Raw)',
                    'Length: ' + responseText.length + ' bytes',
                    responseText
                );
                
                statusMsg.remove();
                
                // Try to parse as JSON
                let data;
                try {
                    data = JSON.parse(responseText);
                    addDebugLog(
                        'Server Response (Parsed)',
                        'Successfully parsed JSON',
                        JSON.stringify(data, null, 2)
                    );
                } catch (parseError) {
                    addDebugLog(
                        'JSON Parse Error',
                        'Failed to parse response: ' + parseError.message,
                        'Parse error at position: ' + parseError.message
                    );
                    addMessage('❌ Error: Invalid JSON response from server', 'error');
                    btn.disabled = false;
                    return;
                }
                
                if (data.error) {
                    addDebugLog('Error Response', data.error);
                    addMessage('❌ Error: ' + data.error, 'error');
                } else {
                    if (data.files_updated && data.files_updated.length > 0) {
                        addDebugLog('GitHub', 'Updated: ' + data.files_updated.join(', '));
                        addMessage('✅ Updated files: ' + data.files_updated.join(', '), 'success');
                        addMessage('🚀 Files pushed to GitHub. Railway redeploying...', 'success');
                        
                        if (data.deepseek_response) {
                            const htmlContent = marked.parse(data.deepseek_response);
                            addMessage('🤖 DeepSeek: ' + htmlContent, 'assistant');
                            chatHistory.push({ role: 'assistant', content: data.deepseek_response });
                            saveChatHistory();
                            if (data.audio && ttsEnabled) playAudio(data.audio);
                        }
                    } else if (data.deepseek_response) {
                        const htmlContent = marked.parse(data.deepseek_response);
                        addMessage('🤖 DeepSeek: ' + htmlContent, 'assistant');
                        chatHistory.push({ role: 'assistant', content: data.deepseek_response });
                        saveChatHistory();
                        if (data.audio && ttsEnabled) playAudio(data.audio);
                    }
                }
            } catch (error) {
                statusMsg.remove();
                addDebugLog(
                    'Client Fetch Error', 
                    error.message,
                    'Error Name: ' + error.name + '\nError Message: ' + error.message + '\nStack: ' + error.stack
                );
                addMessage('❌ Network Error: ' + error.message, 'error');
            }
            
            btn.disabled = false;
        }
        
        async function startNewSession() {
            if (!confirm('Clear script.py and chat history?')) return;
            
            try {
                const response = await fetch('/new_session', { method: 'POST' });
                const data = await response.json();
                
                if (data.success) {
                    chatHistory = [];
                    localStorage.removeItem('primateChatHistory');
                    document.getElementById('chatMessages').innerHTML = '';
                    addDebugLog('Session Reset', 'New session started');
                    addMessage('🔄 New session started.', 'success');
                } else {
                    addMessage('❌ Error: ' + data.error, 'error');
                }
            } catch (error) {
                addMessage('❌ Error: ' + error.message, 'error');
            }
        }
        
        // Enter key to send message
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
