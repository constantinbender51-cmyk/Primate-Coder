from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    languages = [
        "Hello, World! (English)",
        "¡Hola, Mundo! (Spanish)",
        "Bonjour le monde ! (French)",
        "Hallo Welt! (German)",
        "Ciao mondo! (Italian)",
        "Olá Mundo! (Portuguese)",
        "Привет, мир! (Russian)",
        "你好，世界！ (Chinese)",
        "こんにちは世界 (Japanese)",
        "안녕하세요 세계 (Korean)",
        "नमस्ते दुनिया (Hindi)",
        "Salam dunia (Malay)",
        "Sawasdee Krub/Ka Lok (Thai)",
        "Merhaba Dünya (Turkish)",
        "Halo Dunia (Indonesian)",
        "Dia duit, a Dhomhain! (Irish)",
        "Hoi wêreld! (Frisian)",
        "Shkruaj botë! (Albanian)",
        "Halo Mlimwengu! (Swahili)",
        "Héy dats world! (Limburgish)"
    ]
    output = "<h1>Hello World in 20 Languages</h1>"
    for greeting in languages:
        output += f"<p>{greeting}</p>"
    return output

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)