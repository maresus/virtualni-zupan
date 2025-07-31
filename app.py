import os
import uuid
from flask import Flask, request, jsonify, render_template, session, send_from_directory
from flask_cors import CORS # Nov import
from dotenv import load_dotenv
from VIRT_ZUPAN_RF_api import VirtualniZupan

load_dotenv()
app = Flask(__name__)
# Dovolimo dostop z vseh domen za naš widget
CORS(app, resources={r"/ask": {}, r"/widget.js": {}}) 
app.secret_key = os.getenv("FLASK_SECRET_KEY", "prosim-spremeni-to-skrivno-vrednost")

ZUPAN_INSTANCE = None
def get_zupan():
    global ZUPAN_INSTANCE
    if ZUPAN_INSTANCE is None:
        print("Prva zahteva uporabnika: Inicializacija instance virtualnega župana...")
        ZUPAN_INSTANCE = VirtualniZupan()
    return ZUPAN_INSTANCE

@app.route('/')
def home():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    # Uporabimo novo, poenostavljeno predlogo za iframe
    return render_template('chat_widget.html') 

@app.route('/ask', methods=['POST'])
def ask():
    zupan = get_zupan()
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    uporabnikov_id = session['session_id']
    data = request.get_json()
    uporabnikovo_vprasanje = data.get('question', '')

    if not uporabnikovo_vprasanje:
        return jsonify({'answer': 'Prosim, vnesite vprašanje.'})
    
    odgovor_zupana = zupan.odgovori(uporabnikovo_vprasanje, session_id=uporabnikov_id)
    
    return jsonify({'answer': odgovor_zupana})

# --- DODANA NOVA POT ZA WIDGET ---
@app.route('/widget.js')
def serve_widget():
    # Pošljemo našo glavno JavaScript datoteko za widget
    return send_from_directory('static', 'widget.js')
# ------------------------------------