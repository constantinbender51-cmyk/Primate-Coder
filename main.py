from flask import Flask, render_template_string
import requests
import os

app = Flask(__name__)

# IMPORTANT: Replace with your actual Google Drive file ID
# You can find this in the shareable link: https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing
GOOGLE_DRIVE_FILE_ID = 'YOUR_GOOGLE_DRIVE_FILE_ID'

# Construct the direct download URL for Google Drive
# This URL format is for files that are not explicitly shared with specific users.
# If your file has restricted sharing, you might need to use Google Drive API.
DOWNLOAD_URL = f'https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}'

# Define the HTML template
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>Google Drive File Snippet</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        h1 { color: #333; }
        pre { background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>First 5,000 Characters of Google Drive File</h1>
    <pre>{{ file_snippet }}</pre>
    {% if error %}
        <p style="color: red;">Error: {{ error }}</p>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def display_file_snippet():
    file_snippet = ""
    error_message = None

    try:
        # Use requests to download the file content
        response = requests.get(DOWNLOAD_URL, stream=True)

        # Check for non-success status codes (e.g., 404, 403)
        if response.status_code != 200:
            error_message = f"Failed to download file. Status code: {response.status_code}"
            if response.status_code == 404:
                error_message += " - File not found. Please check the file ID and sharing settings."
            elif response.status_code == 403:
                error_message += " - Permission denied. Ensure the file is shared publicly or you have access."
            else:
                error_message += " - An unexpected error occurred."
        else:
            # Read the content chunk by chunk to avoid loading large files entirely into memory
            # We only need the first 5000 characters, so we can stop early.
            content = b''
            for chunk in response.iter_content(chunk_size=1024):
                content += chunk
                if len(content) >= 5000:
                    break

            # Decode the content assuming UTF-8 encoding
            try:
                file_snippet = content[:5000].decode('utf-8', errors='replace')
            except UnicodeDecodeError:
                file_snippet = "Could not decode the file content as UTF-8. Displaying raw bytes snippet." # Fallback
                file_snippet = content[:5000].hex() # Display hex if decoding fails

    except requests.exceptions.RequestException as e:
        error_message = f"Network or request error: {e}"
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"

    return render_template_string(HTML_TEMPLATE, file_snippet=file_snippet, error=error_message)

if __name__ == '__main__':
    # IMPORTANT: Replace 'YOUR_GOOGLE_DRIVE_FILE_ID' with the actual ID of your file.
    if GOOGLE_DRIVE_FILE_ID == 'YOUR_GOOGLE_DRIVE_FILE_ID':
        print("\n!!! IMPORTANT !!!")
        print("Please replace 'YOUR_GOOGLE_DRIVE_FILE_ID' in the script with your actual Google Drive file ID.")
        print("You can find this ID in the shareable link of your Google Drive file.")
        print("Example: https://drive.google.com/file/d/THIS_IS_THE_FILE_ID/view?usp=sharing")
        print("The application will not work until this is updated.")
        print("\nStarting the server anyway, but it will likely show an error.")

    print("Starting Flask web server on http://127.0.0.1:8080")
    app.run(host='0.0.0.0', port=8080)
