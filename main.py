from flask import Flask
import requests

app = Flask(__name__)

@app.route('/')
def index():
    file_id = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
    url = f'https://drive.google.com/uc?export=download&id={file_id}'

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes

        content = ''
        # Read the file content in chunks
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                content += chunk.decode('utf-8', errors='ignore')
                if len(content) >= 5000:
                    break

        return f'<pre>{content[:5000]}</pre>'

    except requests.exceptions.RequestException as e:
        return f'Error downloading file: {e}'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
