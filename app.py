import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# --- DINAMIČNA KONFIGURACIJA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

from VIRT_ZUPAN_RF_api import VirtualniZupan

# --- INICIALIZACIJA APLIKACIJE ---
app = Flask(__name__)

print("Inicializacija instance virtualnega župana za spletno aplikacijo...")
zupan = VirtualniZupan()
print("Virtualni župan je pripravljen za sprejemanje spletnih zahtev.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    uporabnikovo_vprasanje = data.get('question', '')
    if not uporabnikovo_vprasanje:
        return jsonify({'answer': 'Prosim, vnesite vprašanje.'})
    odgovor_zupana = zupan.odgovori(uporabnikovo_vprasanje)
    return jsonify({'answer': odgovor_zupana})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)