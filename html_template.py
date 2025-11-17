HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>👣 Primate Coder🐾</title>
    <script src="https://cdn.jsdelivr.net/npm/marked@4.3.0/marked.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
        
        :root {
            --bg-primary: #faf8f3;
            --bg-secondary: #f0ebe0;
            --bg-tertiary: #e8e0d0;
            --bg-gradient: linear-gradient(135deg, #faf8f3 0%, #f5f0e8 100%);
            --text-primary: #2d3436;
            --text-secondary: #636e72;
            --text-tertiary: #b2bec3;
            --accent: #00b894;
            --accent-hover: #00a383;
            --accent-light: #55efc4;
            --primate-brown: #8b6f47;
            --primate-dark: #5a4a32;
            --jungle-green: #27ae60;
            --border: #dfe6e9;
            --success: #00b894;
            --error: #ff7675;
            --shadow: rgba(45, 52, 54, 0.1);
            --shadow-strong: rgba(45, 52, 54, 0.2);
            --card-bg: rgba(255, 255, 255, 0.8);
        }

        [data-theme="dark"] {
            --bg-primary: #1a2f1a;
            --bg-secondary: #243324;
            --bg-tertiary: #2d3e2d;
            --bg-gradient: linear-gradient(135deg, #1a2f1a 0%, #243324 100%);
            --text-primary: #ecf0f1;
            --text-secondary: #b2bec3;
            --text-tertiary: #636e72;
            --accent: #55efc4;
            --accent-hover: #81ecdc;
            --accent-light: #a8f5e5;
            --primate-brown: #a98f6d;
            --primate-dark: #8b7355;
            --jungle-green: #2ecc71;
            --border: #3d4f3d;
            --success: #55efc4;
            --error: #ff7675;
            --shadow: rgba(0, 0, 0, 0.3);
            --shadow-strong: rgba(0, 0, 0, 0.5);
            --card-bg: rgba(36, 51, 36, 0.8);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            -webkit-tap-highlight-color: transparent;
        }

        body {
            font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg-gradient);
            color: var(--text-primary);
            overflow: hidden;
            position: fixed;
            width: 100%;
            height: 100%;
        }

        .app {
            display: flex;
            flex-direction: column;
            height: 100vh;
            height: 100dvh;
            max-width: 100%;
            margin: 0 auto;
            background: var(--bg-primary);
            backdrop-filter: blur(10px);
        }

        .header {
            background: linear-gradient(135deg, var(--accent) 0%, var(--jungle-green) 100%);
            padding: 12px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
            min-height: 60px;
            box-shadow: 0 4px 20px var(--shadow);
            position: relative;
            overflow: hidden;
        }

        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
            animation: shimmer 3s infinite;
        }

        @keyframes shimmer {
            0%, 100% { transform: translateX(-100%); }
            50% { transform: translateX(100%); }
        }

        .header-content {
            display: flex;
            align-items: center;
            gap: 12px;
            position: relative;
            z-index: 1;
        }

        .logo-container {
            display: flex;
            align-items: center;
            gap: 8px;
            animation: bounceIn 0.6s ease-out;
        }

        @keyframes bounceIn {
            0% { transform: scale(0.3); opacity: 0; }
            50% { transform: scale(1.05); }
            70% { transform: scale(0.9); }
            100% { transform: scale(1); opacity: 1; }
        }

        .paw-icon {
            font-size: 1.8rem;
            animation: rotate 3s ease-in-out infinite;
            display: inline-block;
        }

        @keyframes rotate {
            0%, 100% { transform: rotate(-5deg); }
            50% { transform: rotate(5deg); }
        }

        .header h1 {
            font-size: 1.2rem;
            font-weight: 700;
            color: white;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            letter-spacing: 0.5px;
        }

        .menu-btn {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            border: 2px solid rgba(255, 255, 255, 0.3);
            font-size: 1.5rem;
            cursor: pointer;
            padding: 8px 12px;
            color: white;
            min-width: 48px;
            min-height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            transition: all 0.3s ease;
            position: relative;
            z-index: 1;
        }

        .menu-btn:hover {
            background: rgba(255, 255, 255, 0.3);
            transform: scale(1.05);
        }

        .menu-btn:active {
            transform: scale(0.95);
        }

        .menu-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(5px);
            z-index: 200;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s, visibility 0.3s;
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
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border-top: 1px solid var(--border);
            border-radius: 24px 24px 0 0;
            z-index: 201;
            transform: translateY(100%);
            transition: transform 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
            box-shadow: 0 -8px 32px var(--shadow-strong);
            max-height: 80vh;
            overflow-y: auto;
        }

        .menu.show {
            transform: translateY(0);
        }

        .menu-header {
            padding: 20px;
            border-bottom: 2px solid var(--border);
            font-weight: 600;
            font-size: 1.1rem;
            color: var(--text-primary);
            background: linear-gradient(135deg, var(--accent) 0%, var(--jungle-green) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .menu-item {
            padding: 18px 20px;
            border-bottom: 1px solid var(--border);
            cursor: pointer;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 14px;
            min-height: 60px;
            transition: all 0.3s ease;
            font-weight: 500;
        }

        .menu-item:hover {
            background: var(--bg-secondary);
            transform: translateX(5px);
        }

        .menu-item:active {
            background: var(--bg-tertiary);
        }

        .menu-item:last-child {
            border-bottom: none;
        }

        .menu-item span:first-child {
            font-size: 1.5rem;
        }

        .tabs {
            display: flex;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border);
            justify-content: flex-end;
            padding: 0 16px;
            flex-shrink: 0;
            gap: 8px;
        }

        .tab {
            padding: 10px 18px;
            background: transparent;
            border: none;
            border-bottom: 3px solid transparent;
            color: var(--text-secondary);
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            min-height: 40px;
            border-radius: 8px 8px 0 0;
        }

        .tab:hover {
            color: var(--accent);
            background: var(--bg-tertiary);
        }

        .tab.active {
            color: var(--accent);
            border-bottom-color: var(--accent);
            background: var(--bg-primary);
        }

        .content {
            flex: 1;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            min-height: 0;
            background: var(--bg-gradient);
        }

        .view {
            flex: 1;
            overflow-y: auto;
            display: none;
            flex-direction: column;
            min-height: 0;
        }

        .view.active {
            display: flex;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px 16px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            -webkit-overflow-scrolling: touch;
        }

        .message-wrapper {
            display: flex;
            gap: 12px;
            align-items: flex-start;
            animation: messageSlideIn 0.4s ease-out;
        }

        @keyframes messageSlideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .message-wrapper.user {
            flex-direction: row-reverse;
        }

        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            flex-shrink: 0;
            box-shadow: 0 2px 8px var(--shadow);
        }

        .avatar.user-avatar {
            background: linear-gradient(135deg, var(--primate-brown) 0%, var(--primate-dark) 100%);
        }

        .avatar.ai-avatar {
            background: linear-gradient(135deg, var(--accent) 0%, var(--jungle-green) 100%);
            animation: pulse 2s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        .message {
            padding: 14px 18px;
            border-radius: 18px;
            max-width: 75%;
            line-height: 1.5;
            font-size: 0.9rem;
            word-wrap: break-word;
            white-space: pre-wrap;
            box-shadow: 0 2px 12px var(--shadow);
            position: relative;
        }

        .message-wrapper.user .message {
            background: linear-gradient(135deg, var(--primate-brown) 0%, var(--primate-dark) 100%);
            color: white;
            border-bottom-right-radius: 4px;
        }

        .message-wrapper.assistant .message {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            color: var(--text-primary);
            border-bottom-left-radius: 4px;
            border: 1px solid var(--border);
        }

        .message-wrapper.assistant .message p {
            margin: 10px 0;
        }

        .message-wrapper.assistant .message p:first-child {
            margin-top: 0;
        }

        .message-wrapper.assistant .message p:last-child {
            margin-bottom: 0;
        }

        .message-wrapper.assistant .message code {
            background: var(--bg-tertiary);
            padding: 3px 8px;
            border-radius: 6px;
            color: var(--accent);
            font-size: 0.85em;
            font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
            border: 1px solid var(--border);
        }

        .message-wrapper.assistant .message pre {
            background: var(--bg-tertiary);
            padding: 16px;
            border-radius: 12px;
            overflow-x: auto;
            margin: 12px 0;
            border: 1px solid var(--border);
            box-shadow: inset 0 2px 8px var(--shadow);
        }

        .message-wrapper.assistant .message pre code {
            background: transparent;
            padding: 0;
            border: none;
        }

        .message-wrapper.assistant .message ul,
        .message-wrapper.assistant .message ol {
            margin: 10px 0;
            padding-left: 24px;
        }

        .message-wrapper.assistant .message li {
            margin: 6px 0;
        }

        .message.system {
            background: linear-gradient(135deg, var(--success) 0%, var(--jungle-green) 100%);
            color: white;
            font-size: 0.85rem;
            text-align: center;
            margin: 0 auto;
            max-width: 90%;
            font-weight: 500;
            box-shadow: 0 4px 12px rgba(0, 184, 148, 0.3);
        }

        .message.status {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            color: var(--text-secondary);
            font-size: 0.85rem;
            font-style: italic;
            text-align: center;
            margin: 0 auto;
            border: 1px solid var(--border);
        }

        .message.error {
            background: linear-gradient(135deg, var(--error) 0%, #e17055 100%);
            color: white;
            font-size: 0.85rem;
            margin: 0 auto;
            text-align: center;
            max-width: 90%;
            font-weight: 500;
            box-shadow: 0 4px 12px rgba(255, 118, 117, 0.3);
        }

        .message.auto-retry {
            background: linear-gradient(135deg, var(--accent-light) 0%, var(--accent) 100%);
            color: var(--text-primary);
            font-size: 0.85rem;
            text-align: center;
            margin: 0 auto;
            max-width: 90%;
            padding: 12px 18px;
            border-left: 4px solid var(--accent);
            font-weight: 500;
            box-shadow: 0 4px 12px rgba(0, 184, 148, 0.2);
        }

        .typing-indicator {
            display: flex;
            gap: 6px;
            padding: 16px;
        }

        .typing-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent);
            animation: typing 1.4s infinite;
        }

        .typing-dot:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing-dot:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.7; }
            30% { transform: translateY(-10px); opacity: 1; }
        }

        .output-container {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            background: var(--bg-tertiary);
            font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
            font-size: 0.8rem;
            color: var(--accent);
            white-space: pre-wrap;
            word-wrap: break-word;
            -webkit-overflow-scrolling: touch;
            border-radius: 12px;
            margin: 12px;
            box-shadow: inset 0 2px 8px var(--shadow);
        }

        .debug-container {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            background: var(--bg-tertiary);
            font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
            font-size: 0.75rem;
            color: var(--text-secondary);
            -webkit-overflow-scrolling: touch;
        }

        .debug-entry {
            margin-bottom: 14px;
            padding: 12px;
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border-left: 4px solid var(--accent);
            border-radius: 8px;
            box-shadow: 0 2px 8px var(--shadow);
            transition: all 0.3s ease;
        }

        .debug-entry:hover {
            transform: translateX(4px);
            box-shadow: 0 4px 12px var(--shadow);
        }

        .debug-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            flex-wrap: wrap;
            gap: 8px;
        }

        .debug-timestamp {
            color: var(--accent);
            font-weight: 700;
            font-size: 0.7rem;
        }

        .debug-type {
            color: var(--success);
            font-weight: 700;
            margin-bottom: 6px;
            font-size: 0.75rem;
        }

        .debug-data {
            color: var(--text-primary);
            word-wrap: break-word;
            font-size: 0.75rem;
        }

        .debug-expand-btn {
            background: var(--accent);
            border: none;
            color: white;
            padding: 6px 12px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.7rem;
            font-weight: 600;
            transition: all 0.3s ease;
            white-space: nowrap;
            box-shadow: 0 2px 6px rgba(0, 184, 148, 0.3);
        }

        .debug-expand-btn:hover {
            background: var(--accent-hover);
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(0, 184, 148, 0.4);
        }

        .debug-expand-btn:active {
            transform: translateY(0);
        }

        .debug-expand-btn.expanded {
            background: var(--primate-brown);
            box-shadow: 0 2px 6px rgba(139, 111, 71, 0.3);
        }

        .debug-full-data {
            margin-top: 10px;
            padding: 10px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 8px;
            max-height: 300px;
            overflow: auto;
            display: none;
            box-shadow: inset 0 2px 6px var(--shadow);
        }

        .debug-full-data.visible {
            display: block;
        }

        .debug-full-data pre {
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 0.7rem;
        }

        .input-area {
            padding: 16px 20px;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border-top: 1px solid var(--border);
            flex-shrink: 0;
            position: relative;
            padding-bottom: max(16px, env(safe-area-inset-bottom));
            box-shadow: 0 -4px 20px var(--shadow);
        }

        .input-wrapper {
            display: flex;
            gap: 10px;
            align-items: flex-end;
        }

        .input {
            flex: 1;
            padding: 14px 16px;
            border: 2px solid var(--border);
            border-radius: 16px;
            background: var(--bg-primary);
            color: var(--text-primary);
            font-size: 0.95rem;
            font-family: 'Poppins', sans-serif;
            resize: none;
            min-height: 52px;
            max-height: 30vh;
            overflow-y: auto;
            line-height: 1.5;
            transition: all 0.3s ease;
            box-shadow: inset 0 2px 4px var(--shadow);
        }

        .input:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(0, 184, 148, 0.1), inset 0 2px 4px var(--shadow);
        }

        .send-btn {
            padding: 14px 24px;
            background: linear-gradient(135deg, var(--accent) 0%, var(--jungle-green) 100%);
            color: white;
            border: none;
            border-radius: 16px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            min-height: 52px;
            min-width: 80px;
            transition: all 0.3s ease;
            flex-shrink: 0;
            box-shadow: 0 4px 12px rgba(0, 184, 148, 0.3);
        }

        .send-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0, 184, 148, 0.4);
        }

        .send-btn:active {
            transform: translateY(0);
        }

        .send-btn:disabled {
            background: var(--border);
            color: var(--text-tertiary);
            cursor: not-allowed;
            box-shadow: none;
            transform: none;
        }

        .loading {
            display: inline-block;
            width: 14px;
            height: 14px;
            border: 3px solid var(--border);
            border-top: 3px solid var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 8px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-secondary);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--accent);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent-hover);
        }

        /* Desktop styles */
        @media (min-width: 768px) {
            body {
                position: static;
            }

            .app {
                max-width: 1200px;
                margin: 0 auto;
                height: 95vh;
                margin-top: 2.5vh;
                border-radius: 20px;
                overflow: hidden;
                box-shadow: 0 20px 60px var(--shadow-strong);
            }

            .header {
                border-radius: 20px 20px 0 0;
                padding: 16px 28px;
                min-height: 70px;
            }

            .header h1 {
                font-size: 1.5rem;
            }

            .paw-icon {
                font-size: 2rem;
            }

            .tabs {
                padding: 0 24px;
            }

            .tab {
                padding: 12px 24px;
                font-size: 0.9rem;
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
                padding: 28px 24px;
                gap: 20px;
            }

            .message {
                max-width: 65%;
                font-size: 0.95rem;
            }

            .avatar {
                width: 40px;
                height: 40px;
                font-size: 1.3rem;
            }

            .output-container, .debug-container {
                padding: 24px;
                font-size: 0.85rem;
            }

            .input-area {
                padding: 24px 28px;
            }

            .input {
                font-size: 1rem;
                padding: 16px 20px;
            }

            .send-btn {
                font-size: 1rem;
                padding: 16px 28px;
            }
        }

        /* iOS safe area support */
        @supports (padding: max(0px)) {
            .app {
                padding-left: env(safe-area-inset-left);
                padding-right: env(safe-area-inset-right);
            }
            
            .header {
                padding-top: max(12px, env(safe-area-inset-top));
            }
        }
    </style>
</head>
<body>
    <div class="app">
        <div class="header">
            <div class="header-content">
                <div class="logo-container">
                    <span class="paw-icon">👣</span>
                    <h1>Primate Coder</h1>
                    <span class="paw-icon">🐾</span>
                </div>
            </div>
            <button class="menu-btn" onclick="toggleMenu()">⚙️</button>
        </div>

        <div class="menu-overlay" id="menuOverlay" onclick="toggleMenu()"></div>
        <div class="menu" id="menu">
            <div class="menu-header">⚡ Options</div>
            <div class="menu-item" onclick="toggleTheme()">
                <span id="themeIcon">☀️</span>
                <span id="themeLabel">Switch to Light Mode</span>
            </div>
            <div class="menu-item" onclick="toggleTTS()">
                <span id="ttsMenuIcon">🔊</span>
                <span id="ttsMenuLabel">TTS: On</span>
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

        <div class="tabs">
            <button class="tab active" onclick="switchView('chat')">💬 Chat</button>
            <button class="tab" onclick="switchView('output')">📊 Output</button>
            <button class="tab" onclick="switchView('debug')">🔍 Debug</button>
        </div>

        <div class="input-area">
            <div class="input-wrapper">
                <textarea 
                    class="input" 
                    id="userInput" 
                    placeholder="🌿 Describe what you want to build..."
                    rows="1"
                ></textarea>
                <button class="send-btn" id="sendBtn" onclick="sendMessage()">🚀 Send</button>
            </div>
        </div>
    </div>

    <script>
        let chatHistory = [];
        let theme = localStorage.getItem('primateTheme') || 'dark';
        let activeView = 'chat';
        let debugLogs = [];
        let shouldAutoScroll = true;
        let ttsEnabled = true;
        let currentAudio = null;
        let isProcessing = false;

        // Configure marked.js
        marked.setOptions({ breaks: true, gfm: true });

        // Initialize theme
        document.documentElement.setAttribute('data-theme', theme);
        updateThemeUI();

        // Load TTS preference
        const savedTTSPref = localStorage.getItem('primateTTSEnabled');
        if (savedTTSPref !== null) {
            ttsEnabled = savedTTSPref === 'true';
        }
        updateTTSUI();

        // Load chat history
        const savedHistory = localStorage.getItem('primateChatHistory');
        if (savedHistory) {
            try {
                chatHistory = JSON.parse(savedHistory);
                chatHistory.forEach(msg => {
                    if (msg.role === 'user') {
                        addMessage(msg.content, 'user', false);
                    } else if (msg.role === 'assistant') {
                        addMessage(msg.content, 'assistant', false);
                    } else if (msg.role === 'system') {
                        addMessageSimple(msg.content, 'system', false);
                    }
                });
            } catch (e) {
                console.error('Error loading chat history:', e);
                chatHistory = [];
            }
        } else {
            // Show welcome message
            addMessageSimple('👋 Welcome to Primate Coder! Describe what you want to build and I\'ll help you create it.', 'system', false);
        }

        // Auto-resize textarea
        const textarea = document.getElementById('userInput');
        textarea.addEventListener('input', function() {
            this.style.height = '52px';
            this.style.height = Math.min(this.scrollHeight, window.innerHeight * 0.3) + 'px';
        });

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

        function toggleTTS() {
            ttsEnabled = !ttsEnabled;
            localStorage.setItem('primateTTSEnabled', ttsEnabled);
            updateTTSUI();
            
            if (!ttsEnabled && currentAudio) {
                currentAudio.pause();
                currentAudio = null;
            }
            
            addDebugLog('TTS Toggled', 'TTS is now ' + (ttsEnabled ? 'enabled' : 'disabled'));
            
            const menu = document.getElementById('menu');
            if (menu.classList.contains('show')) {
                toggleMenu();
            }
        }

        function updateTTSUI() {
            const menuIcon = document.getElementById('ttsMenuIcon');
            const menuLabel = document.getElementById('ttsMenuLabel');
            
            if (ttsEnabled) {
                menuIcon.textContent = '🔊';
                menuLabel.textContent = 'TTS: On';
            } else {
                menuIcon.textContent = '🔇';
                menuLabel.textContent = 'TTS: Off';
            }
        }

        function playAudio(audioData) {
            if (!ttsEnabled || !audioData) return;
            
            if (currentAudio) {
                currentAudio.pause();
                currentAudio = null;
            }
            
            try {
                currentAudio = new Audio(audioData);
                currentAudio.play().catch(err => {
                    console.error('Error playing audio:', err);
                    addDebugLog('TTS Error', 'Failed to play audio: ' + err.message);
                });
                
                currentAudio.onended = () => {
                    currentAudio = null;
                    addDebugLog('TTS Completed', 'Audio playback finished');
                };
                
                addDebugLog('TTS Started', 'Playing audio response');
            } catch (error) {
                console.error('Error creating audio:', error);
                addDebugLog('TTS Error', 'Failed to create audio: ' + error.message);
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
            addMessageSimple('🧹 Chat memory cleared', 'system');
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
                    addMessageSimple('🔄 New session started', 'system');
                } else {
                    addMessageSimple('❌ Error: ' + data.error, 'error');
                }
            } catch (error) {
                addMessageSimple('❌ Error: ' + error.message, 'error');
            }
            toggleMenu();
        }

        function addMessage(content, type, saveToHistory = true) {
            const chatMessages = document.getElementById('chatMessages');
            const wrapper = document.createElement('div');
            wrapper.className = 'message-wrapper ' + type;
            
            // Add avatar
            const avatar = document.createElement('div');
            avatar.className = 'avatar ' + (type === 'user' ? 'user-avatar' : 'ai-avatar');
            avatar.textContent = type === 'user' ? '👤' : '🦍';
            wrapper.appendChild(avatar);
            
            // Add message
            const msg = document.createElement('div');
            msg.className = 'message';
            
            if (type === 'assistant') {
                const htmlContent = marked.parse(content);
                msg.innerHTML = htmlContent;
            } else {
                msg.textContent = content;
            }
            
            wrapper.appendChild(msg);
            chatMessages.appendChild(wrapper);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            if (saveToHistory && type !== 'system' && type !== 'error' && type !== 'status') {
                if (type === 'user') {
                    chatHistory.push({ role: 'user', content: content });
                } else if (type === 'assistant') {
                    chatHistory.push({ role: 'assistant', content: content });
                }
                saveChatHistory();
            }
            
            return msg;
        }

        function addMessageSimple(content, type, saveToHistory = true) {
            const chatMessages = document.getElementById('chatMessages');
            const msg = document.createElement('div');
            msg.className = 'message ' + type;
            msg.innerHTML = content;
            chatMessages.appendChild(msg);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            if (saveToHistory) {
                const textContent = content.replace(/<[^>]*>/g, '').trim();
                chatHistory.push({ role: 'system', content: textContent });
                saveChatHistory();
            }
            
            return msg;
        }

        function showTypingIndicator() {
            const chatMessages = document.getElementById('chatMessages');
            const wrapper = document.createElement('div');
            wrapper.className = 'message-wrapper assistant';
            wrapper.id = 'typing-indicator';
            
            const avatar = document.createElement('div');
            avatar.className = 'avatar ai-avatar';
            avatar.textContent = '🦍';
            wrapper.appendChild(avatar);
            
            const indicator = document.createElement('div');
            indicator.className = 'message';
            indicator.innerHTML = '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';
            wrapper.appendChild(indicator);
            
            chatMessages.appendChild(wrapper);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            return wrapper;
        }

        function removeTypingIndicator() {
            const indicator = document.getElementById('typing-indicator');
            if (indicator) {
                indicator.remove();
            }
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
                html += '<div class="debug-type">' + escapeHtml(log.type) + '</div></div>';
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
            const buttons = document.querySelectorAll('.debug-expand-btn');
            const btn = buttons[index];
            if (elem && btn) {
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
            if (isProcessing) return;
            
            const input = document.getElementById('userInput');
            const btn = document.getElementById('sendBtn');
            const message = input.value.trim();
            
            if (!message) return;
            
            isProcessing = true;
            
            addMessage(message, 'user');
            
            input.value = '';
            input.style.height = '52px';
            btn.disabled = true;
            
            const typingIndicator = showTypingIndicator();
            
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
                
                removeTypingIndicator();
                
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
                    addMessageSimple('❌ Error: Invalid JSON response from server', 'error');
                    btn.disabled = false;
                    isProcessing = false;
                    return;
                }
                
                if (data.error) {
                    addDebugLog('Error Response', data.error);
                    addMessageSimple('❌ Error: ' + data.error, 'error');
                } else {
                    if (data.files_updated && data.files_updated.length > 0) {
                        addDebugLog('Files Updated', 'Updated: ' + data.files_updated.join(', '));
                        addMessageSimple('📝 Updated files: ' + data.files_updated.join(', '), 'system');
                        addMessageSimple('🚀 Files pushed to GitHub. Railway redeploying...', 'system');
                        
                        if (!data.script_confirmed_working) {
                            addMessageSimple('🔄 Auto-retry enabled: Will monitor script output and fix any errors automatically', 'auto-retry');
                        }
                    }
                    
                    if (data.files_deleted && data.files_deleted.length > 0) {
                        addDebugLog('Files Deleted', 'Deleted: ' + data.files_deleted.join(', '));
                        addMessageSimple('🗑️ Deleted files: ' + data.files_deleted.join(', '), 'system');
                    }
                    
                    if (data.deepseek_response) {
                        addMessage(data.deepseek_response, 'assistant');
                        
                        if (data.audio) {
                            playAudio(data.audio);
                        }
                    }
                    
                    if (data.is_auto_retry) {
                        if (data.script_confirmed_working) {
                            addMessageSimple('✅ Auto-retry complete: Script is now working correctly!', 'system');
                        } else {
                            addMessageSimple('🔄 Auto-retry attempt ' + data.retry_attempt + ': Applying fixes...', 'auto-retry');
                        }
                    }
                }
            } catch (error) {
                removeTypingIndicator();
                addDebugLog(
                    'Client Fetch Error', 
                    error.message,
                    'Error: ' + error.name + '\\nMessage: ' + error.message + '\\nStack: ' + error.stack
                );
                addMessageSimple('❌ Network Error: ' + error.message, 'error');
            }
            
            btn.disabled = false;
            isProcessing = false;
        }

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
                    data.logs.forEach(log => {
                        if (log.fullData !== undefined) {
                            addDebugLog(log.type, log.data, log.fullData);
                        } else {
                            addDebugLog(log.type, log.data);
                        }
                    });
                }
            } catch (error) {
                console.error('Error fetching debug logs:', error);
            }
        }, 1000);

        // Poll for auto-retry messages
        setInterval(async () => {
            try {
                const response = await fetch('/get_auto_retry_messages');
                const data = await response.json();
                if (data.messages && data.messages.length > 0) {
                    data.messages.forEach(msg => {
                        if (msg.type === 'assistant') {
                            addMessage(msg.content, 'assistant', false);
                        } else if (msg.type === 'system') {
                            addMessageSimple(msg.content, 'system', false);
                        } else if (msg.type === 'error') {
                            addMessageSimple(msg.content, 'error', false);
                        } else if (msg.type === 'status') {
                            addMessageSimple(msg.content, 'status', false);
                        } else if (msg.type === 'auto-retry') {
                            addMessageSimple(msg.content, 'auto-retry', false);
                        }
                    });
                }
            } catch (error) {
                console.error('Error fetching auto-retry messages:', error);
            }
        }, 1000);
    </script>
</body>
</html>
"""
