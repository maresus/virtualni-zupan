import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from VIRT_ZUPAN_RF_api import VirtualniZupan

app = Flask(__name__)
CORS(app)

# Initialize Virtual Mayor
zupan = VirtualniZupan()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        # Get answer from Virtual Mayor
        answer = zupan.odgovori(question, 'web_session')
        
        # Filter contact responses
        if 'kontakt' in question.lower() and 'občin' in question.lower():
            answer = """**Kontakt Občine Rače-Fram:**

📞 **Telefon:** +386 2 609 60 10
📍 **Naslov:** Grajski trg 14, 2327 Rače"""
        
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