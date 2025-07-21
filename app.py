import os
import uuid # Uvozimo knjižnico za generiranje unikatnih ID-jev
from flask import Flask, request, jsonify, render_template, session # Dodamo 'session'
from dotenv import load_dotenv
from VIRT_ZUPAN_RF_api import VirtualniZupan

load_dotenv()
app = Flask(__name__)
# DODANO: Skrivni ključ je nujen za delovanje sej (sessions) za sledenje uporabnikom
# V .env datoteko lahko dodate vrstico: FLASK_SECRET_KEY='nekaj-zelo-skrivnega'
app.secret_key = os.getenv("FLASK_SECRET_KEY", "privzeta-skrivna-vrednost-za-vsak-slucaj")

# Vaša obstoječa "vrhunska" logika za leno nalaganje ostaja nedotaknjena
ZUPAN_INSTANCE = None
def get_zupan():
    global ZUPAN_INSTANCE
    if ZUPAN_INSTANCE is None:
        print("Prva zahteva uporabnika: Inicializacija instance virtualnega župana...")
        ZUPAN_INSTANCE = VirtualniZupan()
    return ZUPAN_INSTANCE

@app.route('/')
def home():
    # DODANO: Ob prvem obisku ustvarimo unikatno kodo za uporabnika in jo shranimo v piškotek
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    zupan = get_zupan()
    
    # DODANO: Poskrbimo, da ima uporabnik vedno kodo seje
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    uporabnikov_id = session['session_id']
    
    data = request.get_json()
    uporabnikovo_vprasanje = data.get('question', '')
    if not uporabnikovo_vprasanje:
        return jsonify({'answer': 'Prosim, vnesite vprašanje.'})
    
    # SPREMENJENO: Županu posredujemo tako vprašanje kot kodo uporabnika
    odgovor_zupana = zupan.odgovori(uporabnikovo_vprasanje, session_id=uporabnikov_id)
    
    return jsonify({'answer': odgovor_zupana})