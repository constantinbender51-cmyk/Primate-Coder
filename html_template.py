
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Primate Coder</title>
    <script src="https://cdn.jsdelivr.net/npm/marked@4.3.0/marked.min.js"></script>
    <style>
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f5f5f5;
            --bg-tertiary: #e8e8e8;
            --text-primary: #1a1a1a;
            --text-secondary: #666666;
            --text-tertiary: #999999;
            --accent: #1669C5;
            --accent-hover: #1457a8;
            --border: #d0d0d0;
            --success: #00aa66;
            --error: #dd4444;
            --shadow: rgba(0, 0, 0, 0.1);
        }

        [data-theme="dark"] {
            --bg-primary: #1a1a1a;
            --bg-secondary: #2a2a2a;
            --bg-tertiary: #0a0a0a;
            --text-primary: #ffffff;
            --text-secondary: #cccccc;
            --text-tertiary: #888888;
            --accent: #1669C5;
            --accent-hover: #1e7de6;
            --border: #3a3a3a;
            --success: #00ff88;
            --error: #ff4444;
            --shadow: rgba(0, 0, 0, 0.3);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            -webkit-tap-highlight-color: transparent;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg-secondary);
            color: var(--text-primary);
            overflow: hidden;
        }

        .app {
            display: flex;
            flex-direction: column;
            height: 100vh;
            max-width: 100%;
            margin: 0 auto;
            background: var(--bg-primary);
        }

        .header {
            background: var(--bg-primary);
            border-bottom: 1px solid var(--border);
            padding: 12px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header h1 {
            font-size: 1.1rem;
            font-weight: 500;
            color: var(--text-primary);
        }

        .highlight {
            color: var(--accent);
        }

        .menu-btn {
            background: transparent;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            padding: 4px 8px;
            color: var(--text-secondary);
            min-width: 44px;
            min-height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .menu-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            z-index: 200;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.2s, visibility 0.2s;
        }

        .menu-overlay.show {
            opacity: 1;
            visibility: visible;
        }

        .menu {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: var(--bg-primary);
            border-top: 1px solid var(--border);
            border-radius: 16px 16px 0 0;
            z-index: 201;
            transform: translateY(100%);
            transition: transform 0.3s;
            box-shadow: 0 -4px 12px var(--shadow);
        }

        .menu.show {
            transform: translateY(0);
        }

        .menu-header {
            padding: 16px;
            border-bottom: 1px solid var(--border);
            font-weight: 500;
            color: var(--text-primary);
        }

        .menu-item {
            padding: 16px;
            border-bottom: 1px solid var(--border);
            cursor: pointer;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 12px;
            min-height: 56px;
            transition: background 0.2s;
        }

        .menu-item:active {
            background: var(--bg-secondary);
        }

        .menu-item:last-child {
            border-bottom: none;
        }

        /* Updated tabs positioning */
        .tabs {
            display: flex;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border);
            justify-content: flex-end;
            padding: 0 16px;
        }

        .tab {
            padding: 8px 16px;
            background: transparent;
            border: none;
            border-bottom: 2px solid transparent;
            color: var(--text-secondary);
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s;
            min-height: 40px;
        }

        .tab.active {
            color: var(--accent);
            border-bottom-color: var(--accent);
            font-weight: 500;
        }

        .content {
            flex: 1;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .view {
            flex: 1;
            overflow-y: auto;
            display: none;
            flex-direction: column;
        }

        .view.active {
            display: flex;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .message {
            padding: 12px 16px;
            border-radius: 8px;
            max-width: 85%;
            line-height: 1.5;
            font-size: 0.95rem;
            word-wrap: break-word;
        }

        .message.user {
            background: var(--accent);
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 4px;
        }

        .message.assistant {
            background: var(--bg-secondary);
            color: var(--text-primary);
            margin-right: auto;
            border-bottom-left-radius: 4px;
        }

        .message.assistant code {
            background: var(--bg-tertiary);
            padding: 2px 6px;
            border-radius: 3px;
            color: var(--accent);
            font-size: 0.9em;
        }

        .message.assistant pre {
            background: var(--bg-tertiary);
            padding: 12px;
            border-radius: 4px;
            overflow-x: auto;
            margin: 10px 0;
        }

        .message.assistant pre code {
            background: transparent;
            padding: 0;
        }

        .message.system {
            background: transparent;
            color: var(--success);
            font-size: 0.85rem;
            text-align: center;
            margin: 0 auto;
        }

        .message.status {
            background: transparent;
            color: var(--text-tertiary);
            font-size: 0.85rem;
            font-style: italic;
            text-align: center;
            margin: 0 auto;
        }

        .message.error {
            background: transparent;
            color: var(--error);
            font-size: 0.85rem;
            margin: 0 auto;
            text-align: center;
        }

        .output-container {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            background: var(--bg-tertiary);
            font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
            font-size: 0.85rem;
            color: var(--accent);
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .debug-container {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            background: var(--bg-tertiary);
            font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        .debug-entry {
            margin-bottom: 16px;
            padding: 12px;
            background: var(--bg-primary);
            border-left: 3px solid var(--accent);
            border-radius: 4px;
        }

        .debug-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .debug-timestamp {
            color: var(--accent);
            font-weight: 600;
        }

        .debug-type {
            color: var(--success);
            font-weight: 600;
            margin-bottom: 6px;
        }

        .debug-data {
            color: var(--text-primary);
        }

        .debug-expand-btn {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.75rem;
            transition: all 0.2s;
        }

        .debug-expand-btn:hover {
            border-color: var(--accent);
            color: var(--accent);
        }

        .debug-expand-btn.expanded {
            background: var(--accent);
            color: white;
            border-color: var(--accent);
        }

        .debug-full-data {
            margin-top: 10px;
            padding: 10px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 4px;
            max-height: 400px;
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
            font-size: 0.75rem;
        }

        .input-area {
            padding: 16px;
            background: var(--bg-primary);
            border-top: 1px solid var(--border);
        }

        .input-wrapper {
            display: flex;
            gap: 8px;
            align-items: flex-end;
        }

        .input {
            flex: 1;
            padding: 12px;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--bg-secondary);
            color: var(--text-primary);
            font-size: 1rem;
            font-family: inherit;
            resize: none;
            min-height: 48px;
            max-height: 120px;
        }

        .input:focus {
            outline: none;
            border-color: var(--accent);
        }

        .send-btn {
            padding: 12px 20px;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            min-height: 48px;
            min-width: 72px;
            transition: background 0.2s;
        }

        .send-btn:active {
            background: var(--accent-hover);
        }

        .send-btn:disabled {
            background: var(--border);
            color: var(--text-tertiary);
            cursor: not-allowed;
        }

        .loading {
            display: inline-block;
            width: 12px;
            height: 12px;
            border: 2px solid var(--border);
            border-top: 2px solid var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 8px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @media (min-width: 768px) {
            .app {
                max-width: 1200px;
                margin: 0 auto;
                height: 95vh;
                margin-top: 2.5vh;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 24px var(--shadow);
            }

            .header {
                border-radius: 8px 8px 0 0;
                padding: 16px 24px;
            }

            .header h1 {
                font-size: 1.3rem;
            }

            .tabs {
                display: flex;
                padding: 0 24px;
            }

            .tab {
                padding: 10px 20px;
                font-size: 0.95rem;
            }

            .content {
                flex-direction: row;
            }

            .view {
                display: flex;
                flex: 1;
                border-right: 1px solid var(--border);
            }

            .view:last-child {
                border-right: none;
            }

            .view.active {
                display: flex;
            }

            .chat-messages {
                padding: 24px;
            }

            .message {
                max-width: 70%;
            }

            .output-container, .debug-container {
                padding: 24px;
            }

            .input-area {
                padding: 20px 24px;
            }
        }
    </style>
</head>
<body>
    <div class="app">
        <div class="header">
            <h1>Prima<span class="highlight">t</span>e Coder</h1>
            <button class="menu-btn" onclick="toggleMenu()">⚙️</button>
        </div>

        <div class="menu-overlay" id="menuOverlay" onclick="toggleMenu()"></div>
        <div class="menu" id="menu">
            <div class="menu-header">Options</div>
            <div class="menu-item" onclick="toggleTheme()">
                <span id="themeIcon">☀️</span>
                <span id="themeLabel">Switch to Light Mode</span>
            </div>
            <div class="menu-item" onclick="clearMemory()">
                <span>🧹</span>
                <span>Clear Memory</span>
            </div>
            <div class="menu-item" onclick="startNewSession()">
                <span>🔄</span>
                <span>New Session</span>
            </div>
        </div>

        <div class="content">
            <div class="view active" id="chatView">
                <div class="chat-messages" id="chatMessages"></div>
            </div>

            <div class="view" id="outputView">
                <div class="output-container" id="outputContent">Waiting for script.py output...</div>
            </div>

            <div class="view" id="debugView">
                <div class="debug-container" id="debugConsole">No debug logs yet...</div>
            </div>
        </div>

        <!-- Tabs moved to bottom right above input area -->
        <div class="tabs">
            <button class="tab active" onclick="switchView('chat')">Chat</button>
            <button class="tab" onclick="switchView('output')">Output</button>
            <button class="tab" onclick="switchView('debug')">Debug</button>
        </div>

        <div class="input-area">
            <div class="input-wrapper">
                <textarea 
                    class="input" 
                    id="userInput" 
                    placeholder="Describe what you want to build..."
                    rows="1"
                ></textarea>
                <button class="send-btn" id="sendBtn" onclick="sendMessage()">Send</button>
            </div>
        </div>
    </div>

    <script>
        let chatHistory = [];
        let theme = localStorage.getItem('primateTheme') || 'dark';
        let activeView = 'chat';
        let debugLogs = [];
        let shouldAutoScroll = true;

        // Configure marked.js
        marked.setOptions({ breaks: true, gfm: true });

        // Initialize theme
        document.documentElement.setAttribute('data-theme', theme);
        updateThemeUI();

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
                        addMessage(htmlContent, 'assistant', false);
                    } else if (msg.role === 'system') {
                        addMessage(msg.content, 'system', false);
                    }
                });
            } catch (e) {
                console.error('Error loading chat history:', e);
                chatHistory = [];
            }
        }

        function toggleMenu() {
            const menu = document.getElementById('menu');
            const overlay = document.getElementById('menuOverlay');
            menu.classList.toggle('show');
            overlay.classList.toggle('show');
        }

        function toggleTheme() {
            theme = theme === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem('primateTheme', theme);
            updateThemeUI();
            toggleMenu();
        }

        function updateThemeUI() {
            const icon = document.getElementById('themeIcon');
            const label = document.getElementById('themeLabel');
            if (theme === 'dark') {
                icon.textContent = '☀️';
                label.textContent = 'Switch to Light Mode';
            } else {
                icon.textContent = '🌙';
                label.textContent = 'Switch to Dark Mode';
            }
        }

        function switchView(view) {
            activeView = view;
            document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            
            if (view === 'chat') {
                document.getElementById('chatView').classList.add('active');
                document.querySelectorAll('.tab')[0].classList.add('active');
            } else if (view === 'output') {
                document.getElementById('outputView').classList.add('active');
                document.querySelectorAll('.tab')[1].classList.add('active');
            } else if (view === 'debug') {
                document.getElementById('debugView').classList.add('active');
                document.querySelectorAll('.tab')[2].classList.add('active');
            }
        }

        function clearMemory() {
            if (!confirm('Clear chat history?')) return;
            chatHistory = [];
            localStorage.removeItem('primateChatHistory');
            document.getElementById('chatMessages').innerHTML = '';
            addMessage('Chat memory cleared', 'system');
            toggleMenu();
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
                    addMessage('New session started', 'system');
                } else {
                    addMessage('Error: ' + data.error, 'error');
                }
            } catch (error) {
                addMessage('Error: ' + error.message, 'error');
            }
            toggleMenu();
        }

        function addMessage(content, type, saveToHistory = true) {
            const chatMessages = document.getElementById('chatMessages');
            const msg = document.createElement('div');
            msg.className = 'message ' + type;
            msg.innerHTML = content;
            chatMessages.appendChild(msg);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            if (saveToHistory && (type === 'system' || type === 'error')) {
                const textContent = content.replace(/<[^>]*>/g, '').trim();
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

        function addDebugLog(type, data, fullData = null) {
            const timestamp = new Date().toLocaleTimeString();
            debugLogs.push({ timestamp, type, data, fullData });
            if (debugLogs.length > 50) debugLogs.shift();
            updateDebugConsole();
        }

        function updateDebugConsole() {
            const console = document.getElementById('debugConsole');
            if (debugLogs.length === 0) {
                console.innerHTML = 'No debug logs yet...';
                return;
            }
            
            let html = '';
            debugLogs.forEach((log, index) => {
                html += '<div class="debug-entry">';
                html += '<div class="debug-header">';
                html += '<div><div class="debug-timestamp">' + log.timestamp + '</div>';
                html += '<div class="debug-type">' + log.type + '</div></div>';
                if (log.fullData) {
                    html += '<button class="debug-expand-btn" onclick="toggleDebugData(' + index + ')">Show Details</button>';
                }
                html += '</div>';
                html += '<div class="debug-data">' + escapeHtml(log.data) + '</div>';
                if (log.fullData) {
                    html += '<div class="debug-full-data" id="debug-full-' + index + '">';
                    html += '<pre>' + escapeHtml(log.fullData) + '</pre>';
                    html += '</div>';
                }
                html += '</div>';
            });
            console.innerHTML = html;
            console.scrollTop = console.scrollHeight;
        }

        function toggleDebugData(index) {
            const elem = document.getElementById('debug-full-' + index);
            const btn = document.querySelectorAll('.debug-expand-btn')[debugLogs.length - 1 - index];
            if (elem) {
                elem.classList.toggle('visible');
                btn.classList.toggle('expanded');
                btn.textContent = elem.classList.contains('visible') ? 'Hide Details' : 'Show Details';
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
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
                'POST /generate | Message length: ' + message.length + ' chars',
                JSON.stringify(requestPayload, null, 2)
            );
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestPayload)
                });
                
                addDebugLog('Server → Client', 'Status: ' + response.status + ' ' + response.statusText);
                
                const responseText = await response.text();
                addDebugLog(
                    'Server Response (Raw)',
                    'Length: ' + responseText.length + ' bytes',
                    responseText
                );
                
                statusMsg.remove();
                
                let data;
                try {
                    data = JSON.parse(responseText);
                    addDebugLog(
                        'Server Response (Parsed)',
                        'Successfully parsed JSON response',
                        JSON.stringify(data, null, 2)
                    );
                } catch (parseError) {
                    addDebugLog(
                        'JSON Parse Error',
                        'Failed to parse response: ' + parseError.message,
                        responseText
                    );
                    addMessage('Error: Invalid JSON response from server', 'error');
                    btn.disabled = false;
                    return;
                }
                
                if (data.error) {
                    addDebugLog('Error Response', data.error);
                    addMessage('Error: ' + data.error, 'error');
                } else {
                    if (data.files_updated && data.files_updated.length > 0) {
                        addDebugLog('Files Updated', 'Updated: ' + data.files_updated.join(', '));
                        addMessage('Updated files: ' + data.files_updated.join(', '), 'system');
                        addMessage('Files pushed to GitHub. Railway redeploying...', 'system');
                    }
                    
                    if (data.deepseek_response) {
                        const htmlContent = marked.parse(data.deepseek_response);
                        addMessage(htmlContent, 'assistant');
                        chatHistory.push({ role: 'assistant', content: data.deepseek_response });
                        saveChatHistory();
                    }
                }
            } catch (error) {
                statusMsg.remove();
                addDebugLog(
                    'Client Fetch Error', 
                    error.message,
                    'Error: ' + error.name + '\\nMessage: ' + error.message + '\\nStack: ' + error.stack
                );
                addMessage('Network Error: ' + error.message, 'error');
            }
            
            btn.disabled = false;
        }

        // Enter key to send
        document.getElementById('userInput').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Poll for script output
        const outputDiv = document.getElementById('outputContent');
        outputDiv.addEventListener('scroll', function() {
            const distanceFromBottom = outputDiv.scrollHeight - outputDiv.scrollTop - outputDiv.clientHeight;
            shouldAutoScroll = distanceFromBottom < 50;
        });

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
    </script>
</body>
</html>
"""
