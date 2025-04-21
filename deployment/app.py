from flask import Flask, request, render_template, jsonify
import os, requests

app = Flask(__name__)

LLM1_URL = "http://localhost:8000/v1/chat/completions"
LLM2_URL = "http://localhost:8080/v1/chat/completions"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/medical', methods=['POST'])
def medical():
    data = request.get_json(force=True)
    prompt = data.get('input', '')

    r1 = requests.post(LLM1_URL, json={'prompt': prompt}, headers=HEADERS, timeout=15)
    r2 = requests.post(LLM2_URL, json={'prompt': prompt}, headers=HEADERS, timeout=15)
    agent1 = r1.json().get('text', '')
    agent2 = r2.json().get('text', '')


    verified_output = output_verifier(agent1, agent2)

    return jsonify({
        'agent1':  agent1,
        'agent2':  agent2,
        'verified': "—",        
        'final':   verified_output
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
