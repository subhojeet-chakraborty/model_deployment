from flask import Flask, request, jsonify
import json
import requests

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        url = "https://ocr.asprise.com/api/v1/receipt"
        res = requests.post(url, data={
            'api_key': 'TEST',
            'recognizer': 'auto',
            'ref_no': 'oct_python_123',
        },
        files={
            'file': file.read()
        })

        response_data = json.loads(res.text)
        return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
