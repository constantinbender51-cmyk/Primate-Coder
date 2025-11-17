import React, { useState, useEffect, useRef } from 'react';

const PrimateCoder = () => {
  const [theme, setTheme] = useState(() => localStorage.getItem('primateTheme') || 'dark');
  const [activeView, setActiveView] = useState('chat');
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [output, setOutput] = useState('Waiting for script.py output...');
  const [showMenu, setShowMenu] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  useEffect(() => {
    const saved = localStorage.getItem('primateChatHistory');
    if (saved) {
      try {
        const history = JSON.parse(saved);
        setChatHistory(history);
        const msgs = history.map(msg => ({
          content: msg.content,
          type: msg.role === 'user' ? 'user' : msg.role === 'assistant' ? 'assistant' : 'system'
        }));
        setMessages(msgs);
      } catch (e) {
        console.error('Error loading history:', e);
      }
    }
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const toggleTheme = () => {
    const newTheme = theme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
    localStorage.setItem('primateTheme', newTheme);
    setShowMenu(false);
  };

  const clearMemory = () => {
    if (!confirm('Clear chat history?')) return;
    setChatHistory([]);
    setMessages([]);
    localStorage.removeItem('primateChatHistory');
    setShowMenu(false);
  };

  const newSession = async () => {
    if (!confirm('Clear script.py and chat history?')) return;
    setChatHistory([]);
    setMessages([]);
    localStorage.removeItem('primateChatHistory');
    setMessages(prev => [...prev, { content: 'New session started', type: 'system' }]);
    setShowMenu(false);
  };

  const addMessage = (content, type) => {
    setMessages(prev => [...prev, { content, type }]);
  };

  const sendMessage = async () => {
    if (!message.trim() || isLoading) return;

    const userMsg = message.trim();
    addMessage(userMsg, 'user');
    
    const newHistory = [...chatHistory, { role: 'user', content: userMsg }];
    setChatHistory(newHistory);
    localStorage.setItem('primateChatHistory', JSON.stringify(newHistory));

    setMessage('');
    setIsLoading(true);
    addMessage('Processing...', 'status');

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const response = {
        deepseek_response: 'This is a simulated response. In production, this would call your actual API.',
        files_updated: []
      };

      setMessages(prev => prev.filter(m => m.type !== 'status'));
      
      addMessage(response.deepseek_response, 'assistant');
      const updatedHistory = [...newHistory, { role: 'assistant', content: response.deepseek_response }];
      setChatHistory(updatedHistory);
      localStorage.setItem('primateChatHistory', JSON.stringify(updatedHistory));
      
    } catch (error) {
      setMessages(prev => prev.filter(m => m.type !== 'status'));
      addMessage(`Error: ${error.message}`, 'error');
    }

    setIsLoading(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="app">
      <style>{`
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

        .tabs {
          display: flex;
          background: var(--bg-secondary);
          border-bottom: 1px solid var(--border);
        }

        .tab {
          flex: 1;
          padding: 12px;
          background: transparent;
          border: none;
          border-bottom: 2px solid transparent;
          color: var(--text-secondary);
          font-size: 0.95rem;
          cursor: pointer;
          transition: all 0.2s;
          min-height: 48px;
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
        }

        .view.active {
          display: flex;
          flex-direction: column;
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
            display: none;
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

          .output-container {
            padding: 24px;
          }

          .input-area {
            padding: 20px 24px;
          }
        }
      `}</style>

      <div className="header">
        <h1>Prima<span className="highlight">t</span>e Coder</h1>
        <button className="menu-btn" onClick={() => setShowMenu(!showMenu)}>
          ⚙️
        </button>
      </div>

      <div className={`menu-overlay ${showMenu ? 'show' : ''}`} onClick={() => setShowMenu(false)} />
      <div className={`menu ${showMenu ? 'show' : ''}`}>
        <div className="menu-header">Options</div>
        <div className="menu-item" onClick={toggleTheme}>
          {theme === 'dark' ? '☀️' : '🌙'} Switch to {theme === 'dark' ? 'Light' : 'Dark'} Mode
        </div>
        <div className="menu-item" onClick={clearMemory}>
          🧹 Clear Memory
        </div>
        <div className="menu-item" onClick={newSession}>
          🔄 New Session
        </div>
      </div>

      <div className="tabs">
        <button 
          className={`tab ${activeView === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveView('chat')}
        >
          Chat
        </button>
        <button 
          className={`tab ${activeView === 'output' ? 'active' : ''}`}
          onClick={() => setActiveView('output')}
        >
          Output
        </button>
      </div>

      <div className="content">
        <div className={`view ${activeView === 'chat' ? 'active' : ''}`}>
          <div className="chat-messages">
            {messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.type}`}>
                {msg.content}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
          <div className="input-area">
            <div className="input-wrapper">
              <textarea
                className="input"
                placeholder="Describe what you want to build..."
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={1}
              />
              <button 
                className="send-btn" 
                onClick={sendMessage}
                disabled={isLoading || !message.trim()}
              >
                Send
              </button>
            </div>
          </div>
        </div>

        <div className={`view ${activeView === 'output' ? 'active' : ''}`}>
          <div className="output-container">
            {output}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PrimateCoder;
