import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from VIRT_ZUPAN_RF_api import VirtualniZupan

load_dotenv()
app = Flask(__name__)

# --- POPOLNOMA NOV, ROBUSTEN PRISTOP ---
# Ustvarimo samo "škatlo" za župana, ne kličemo ga takoj.
ZUPAN_INSTANCE = None

def get_zupan():
    """
    Funkcija, ki zagotovi, da se župan ustvari samo enkrat - ob prvi zahtevi.
    To prepreči časovno zamudo ob zagonu strežnika.
    """
    global ZUPAN_INSTANCE
    if ZUPAN_INSTANCE is None:
        print("Prva zahteva uporabnika: Inicializacija instance virtualnega župana...")
        ZUPAN_INSTANCE = VirtualniZupan()
    return ZUPAN_INSTANCE
# --- KONEC NOVEGA PRISTOPA ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    zupan = get_zupan() # Kličemo župana šele tukaj!
    data = request.get_json()
    uporabnikovo_vprasanje = data.get('question', '')
    if not uporabnikovo_vprasanje:
        return jsonify({'answer': 'Prosim, vnesite vprašanje.'})
    odgovor_zupana = zupan.odgovori(uporabnikovo_vprasanje)
    return jsonify({'answer': odgovor_zupana})