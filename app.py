import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from VIRT_ZUPAN_RF_api import VirtualniZupan

app = Flask(__name__)
CORS(app)

# Initialize Virtual Mayor
zupan = VirtualniZupan()

@app.route('/')
def home():
    return jsonify({
        "message": "Virtualni župan Rače-Fram API",
        "status": "running",
        "endpoints": {
            "/ask": "POST - Ask a question",
            "/": "GET - This page"
        }
    })

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        # Get answer from Virtual Mayor
        answer = zupan.odgovori(question, 'web_session')
        
        return jsonify({
            "question": question,
            "answer": answer,
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
