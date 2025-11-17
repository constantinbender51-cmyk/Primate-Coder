import json
import re
import base64
import requests
from io import BytesIO
from gtts import gTTS

def call_deepseek_api(user_message, file_contents, script_output_text, chat_history, deepseek_api_key, debug_logs):
    """Call DeepSeek API with coding agent prompt."""
    
    system_prompt = """You are a coding agent with the ability to create and edit files.

IMPORTANT WORKFLOW:
1. WAIT for user confirmation before providing JSON with file changes

IMPORTANT: The main executable file is 'script.py' which will be run automatically.

CRITICAL - DEPENDENCY MANAGEMENT:
- Whenever you write code that uses external packages (imports), you MUST update requirements.txt
- Always include ALL packages needed by your code in requirements.txt
- Do NOT include flask, requests, or gtts in requirements.txt as they are already available
- Format: one package per line, optionally with version (e.g., "numpy==1.24.0" or just "numpy")

FILE EDITING - THREE FORMATS:

CRITICAL - JSON STRUCTURE:
Each JSON object must be a TOP-LEVEL object, NOT nested inside filename keys.

❌ WRONG - Do NOT nest operations:
{
  "script.py": {
    "file": "script.py",
    "operation": "replace_lines",
    "start_line": 10,
    "end_line": 20,
    "content": "code"
  }
}

✅ CORRECT - Operations are top-level:
{
  "file": "script.py",
  "operation": "replace_lines",
  "start_line": 10,
  "end_line": 20,
  "content": "new code"
}

CRITICAL - LINE NUMBER CLARITY:
When using line editing operations, line numbers are INCLUSIVE on BOTH ends:
- start_line: 10 means line 10 IS INCLUDED in the edit
- end_line: 20 means line 20 IS INCLUDED in the edit
- Lines 10, 11, 12, ..., 19, 20 will ALL be replaced with the new content
- Line numbers are 1-indexed (first line is line 1, not line 0)

Example: To replace lines 5 through 8 (inclusive):
{
  "file": "script.py",
  "operation": "replace_lines",
  "start_line": 5,
  "end_line": 8,
  "content": "new code here"
}
This will replace lines 5, 6, 7, AND 8 with the new content.

For SMALL files (<100 lines) or NEW files, use FULL CONTENT format:
{
  "filename.py": "complete file content here"
}

For LARGE files (>100 lines) with edits, use LINE RANGE format:
{
  "file": "script.py",
  "operation": "replace_lines",
  "start_line": 45,
  "end_line": 67,
  "content": "new code here"
}

For INSERTING code at a specific line, use INSERT format:
{
  "file": "script.py",
  "operation": "insert_at_line",
  "line": 45,
  "content": "new code to insert"
}

For MULTIPLE file operations, use SEPARATE JSON objects:
{
  "file": "script.py",
  "operation": "replace_lines",
  "start_line": 10,
  "end_line": 15,
  "content": "code"
}
{
  "requirements.txt": "numpy\\npandas"
}

Current files in the repository:
"""
    
    # Filter out component files from file_contents display
    hidden_files = ['main.py', 'html_template.py', 'github_api.py', 'deepseek_api.py']
    visible_files = {name: content for name, content in file_contents.items() if name not in hidden_files}
    
    system_prompt += "\n".join([f"- {name}: {len(content)} characters" for name, content in visible_files.items()])

    if script_output_text:
        system_prompt += f"\n\nCurrent script.py output:\n{script_output_text}"

    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {deepseek_api_key}",
        "Content-Type": "application/json"
    }
    
    # Only show visible files in context
    context_parts = []
    for name, content in visible_files.items():
        lines = content.split('\n')
        if len(lines) > 50:
            numbered_content = '\n'.join([f"{i+1}: {line}" for i, line in enumerate(lines)])
            context_parts.append(f"=== {name} (with line numbers) ===\n{numbered_content}")
        else:
            context_parts.append(f"=== {name} ===\n{content}")
    
    context = "\n\n".join(context_parts)
    full_message = f"{context}\n\n=== User Request ===\n{user_message}"
    
    # Include system prompt in messages for chat history
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": full_message})
    
    payload = {
        "model": "deepseek-coder",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 8000
    }
    
    debug_logs.put({"type": "→ DeepSeek API", "data": f"Sending request | Messages: {len(messages)}"})
    
    response = requests.post(url, json=payload, headers=headers)
    debug_logs.put({"type": "← DeepSeek API", "data": f"Status: {response.status_code}"})
    
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
    """Remove JSON objects and code fences from text."""
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
    
    for start, end in reversed(ranges_to_remove):
        result = result[:start] + result[end:]
    
    result = re.sub(r'```[\w]*\n?', '', result)
    result = re.sub(r'```', '', result)
    
    return result.strip()


def merge_requirements(deepseek_requirements, base_requirements):
    """Merge DeepSeek requirements with base requirements."""
    lines = [line.strip() for line in deepseek_requirements.split('\n') if line.strip()]
    
    for base_req in base_requirements:
        if not any(line.lower().startswith(base_req.lower()) for line in lines):
            lines.insert(0, base_req)
    
    return '\n'.join(lines)


def generate_tts_audio(text, debug_logs):
    """Generate TTS audio from text using gTTS."""
    try:
        clean_text = text.strip()
        if not clean_text:
            return None
        
        debug_logs.put({"type": "TTS Generation", "data": "Generating audio..."})
        
        tts = gTTS(text=clean_text, lang='en', slow=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')
        debug_logs.put({"type": "TTS Success", "data": "Audio generated"})
        
        return f"data:audio/mp3;base64,{audio_base64}"
    
    except Exception as e:
        debug_logs.put({"type": "TTS Error", "data": str(e)})
        return None
