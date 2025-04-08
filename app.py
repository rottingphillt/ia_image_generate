from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import base64

app = Flask(__name__)
CORS(app)

HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0" # Substitua pelo modelo Fooocus se disponível na Inference API

headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

@app.route('/generate_image', methods=['POST'])
def generate_image():
    data = request.get_json()
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    payload = {"inputs": prompt}
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        image_bytes = response.content
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        return jsonify({"image": base64_image})
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error communicating with Hugging Face API: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)