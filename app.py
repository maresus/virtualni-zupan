import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# --- DINAMIČNA KONFIGURACIJA ---
# Skripta sama ugotovi, v kateri mapi se nahaja, da lahko pravilno naloži .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

# Uvozimo razred VirtualniZupan iz vaše glavne skripte
# Ime datoteke mora biti VIRT_ZUPAN_RF_api.py
from VIRT_ZUPAN_RF_api import VirtualniZupan

# --- INICIALIZACIJA APLIKACIJE ---
app = Flask(__name__)

# Naložimo župana samo enkrat ob zagonu strežnika
print("Inicializacija instance virtualnega župana za spletno aplikacijo...")
zupan = VirtualniZupan()
print("Virtualni župan je pripravljen za sprejemanje spletnih zahtev.")


# --- SPLETNE POTI (ROUTES) ---

# Glavna pot ('/'), ki uporabniku prikaže spletno stran s klepetom
@app.route('/')
def home():
    # Funkcija render_template poišče datoteko index.html v mapi 'templates'
    return render_template('index.html')

# "Skrita" pot ('/ask'), ki jo bo klicala spletna stran za postavljanje vprašanj
@app.route('/ask', methods=['POST'])
def ask():
    # Preberemo vprašanje, ki ga je poslal uporabnik v JSON formatu
    data = request.get_json()
    uporabnikovo_vprasanje = data.get('question', '')
    
    if not uporabnikovo_vprasanje:
        return jsonify({'answer': 'Prosim, vnesite vprašanje.'})

    # Pokličemo metodo za odgovor iz razreda VirtualniZupan
    odgovor_zupana = zupan.odgovori(uporabnikovo_vprasanje)
    
    # Odgovor pošljemo nazaj na spletno stran v JSON formatu
    return jsonify({'answer': odgovor_zupana})


# Ta del omogoča, da aplikacijo zaženemo direktno z ukazom "python app.py" za lokalno testiranje
if __name__ == '__main__':
    # Uporabimo port 5001, da se izognemo morebitnim konfliktom
    app.run(host='0.0.0.0', port=5001, debug=True)