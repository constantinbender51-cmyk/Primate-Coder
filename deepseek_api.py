import json
import re
import base64
import requests
from io import BytesIO
from gtts import gTTS

def call_deepseek_api(user_message, file_contents, script_output_text, chat_history, deepseek_api_key, debug_logs):
    """Call DeepSeek API with coding agent prompt."""
    
    system_prompt = """You are a coding agent with the ability to create, edit, and delete files.

IMPORTANT WORKFLOW:
1. ASK before EVERY code change for user confirmation before providing JSON with file changes and only proceed once the user has confirmed 

IMPORTANT: The main executable file is 'script.py' which will be run automatically.

CRITICAL - DEPENDENCY MANAGEMENT:
- Whenever you write code that uses external packages (imports), you MUST update requirements.txt
- Always include ALL packages needed by your code in requirements.txt
- Do NOT include flask, requests, or gtts in requirements.txt as they are already available
- Format: one package per line, optionally with version (e.g., "numpy==1.24.0" or just "numpy")

FILE EDITING - FOUR FORMATS:

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

CRITICAL - LINE NUMBER EXPLANATION:
Line numbers are 1-indexed (first line is line 1, NOT line 0).
When you see numbered lines in the file context, use those EXACT numbers.

For replace_lines operation:
- start_line and end_line are BOTH INCLUSIVE
- To replace lines 10 through 15, use: start_line: 10, end_line: 15
- This will replace lines 10, 11, 12, 13, 14, AND 15 (all 6 lines)
- The content you provide will replace ALL these lines

Example from the file:
10: def my_function():
11:     x = 5
12:     y = 10
13:     return x + y

To replace lines 11-12 ONLY (the two variable assignments):
{
  "file": "script.py",
  "operation": "replace_lines",
  "start_line": 11,
  "end_line": 12,
  "content": "    x = 100\\n    y = 200"
}

Result:
10: def my_function():
11:     x = 100
12:     y = 200
13:     return x + y

IMPORTANT: Count the line numbers carefully! If you see:
5: import os
6: import sys
7: 
8: def main():

And you want to add an import after line 6, you have two options:

Option 1 - Insert at line 7:
{
  "file": "script.py",
  "operation": "insert_at_line",
  "line": 7,
  "content": "import json"
}

Option 2 - Replace the empty line 7:
{
  "file": "script.py",
  "operation": "replace_lines",
  "start_line": 7,
  "end_line": 7,
  "content": "import json\\n"
}

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

For INSERTING code at a specific line (pushes existing lines down), use INSERT format:
{
  "file": "script.py",
  "operation": "insert_at_line",
  "line": 45,
  "content": "new code to insert"
}

For DELETING a file, use DELETE format:
{
  "file": "filename.py",
  "operation": "delete"
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
  "file": "unused_file.py",
  "operation": "delete"
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
        # Always show line numbers for better accuracy
        numbered_content = '\n'.join([f"{i+1}: {line}" for i, line in enumerate(lines)])
        context_parts.append(f"=== {name} (lines 1-{len(lines)}) ===\n{numbered_content}")
    
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
    debug_logs.put({
        "type": "DeepSeek Request Payload", 
        "data": f"Payload size: {len(str(payload))} chars",
        "fullData": json.dumps(payload, indent=2)
    })
    
    response = requests.post(url, json=payload, headers=headers)
    debug_logs.put({"type": "← DeepSeek API", "data": f"Status: {response.status_code}"})
    
    response.raise_for_status()
    
    data = response.json()
    
    # Log the FULL DeepSeek API response
    debug_logs.put({
        "type": "DeepSeek Full API Response", 
        "data": f"Response size: {len(str(data))} chars",
        "fullData": json.dumps(data, indent=2)
    })
    
    if "choices" in data and len(data["choices"]) > 0:
        response_content = data["choices"][0]["message"]["content"]
        # Log the DeepSeek response content separately
        debug_logs.put({
            "type": "DeepSeek Response Content", 
            "data": f"Length: {len(response_content)} characters",
            "fullData": response_content
        })
        return response_content
    
    raise Exception("No valid response from DeepSeek")

def extract_json_from_text(text):
    """Extract all JSON objects from text, respecting strings."""
    json_objects = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '{':
            start = i
            brace_count = 1
            in_string = False
            escaped = False
            j = i + 1
            while j < n and brace_count > 0:
                char = text[j]
                if escaped:
                    escaped = False
                elif in_string:
                    if char == '\\':
                        escaped = True
                    elif char == '"':
                        in_string = False
                else:
                    if char == '"':
                        in_string = True
                    elif char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                j += 1
            if brace_count == 0:
                json_str = text[start:j]
                try:
                    obj = json.loads(json_str)
                    json_objects.append(obj)
                    i = j  # Skip to after valid JSON
                except json.JSONDecodeError:
                    i = start + 1  # Invalid, continue
            else:
                i = start + 1  # Unbalanced, continue
        else:
            i += 1
    return json_objects

def remove_json_from_text(text):
    """Remove JSON objects and code fences from text, respecting strings."""
    ranges_to_remove = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '{':
            start = i
            brace_count = 1
            in_string = False
            escaped = False
            j = i + 1
            while j < n and brace_count > 0:
                char = text[j]
                if escaped:
                    escaped = False
                elif in_string:
                    if char == '\\':
                        escaped = True
                    elif char == '"':
                        in_string = False
                else:
                    if char == '"':
                        in_string = True
                    elif char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                j += 1
            if brace_count == 0:
                json_str = text[start:j]
                try:
                    json.loads(json_str)
                    ranges_to_remove.append((start, j))
                    i = j  # Skip to after valid JSON
                except json.JSONDecodeError:
                    i = start + 1  # Invalid, continue
            else:
                i = start + 1  # Unbalanced, continue
        else:
            i += 1
    
    # Remove the ranges in reverse order
    result = text
    for start, end in reversed(ranges_to_remove):
        result = result[:start] + result[end:]
    
    # Remove code fences
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
