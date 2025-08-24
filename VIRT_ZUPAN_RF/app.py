import os
import uuid
from flask import Flask, request, jsonify, render_template, session, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

# Uvozi vse potrebne komponente
from .VIRT_ZUPAN_RF_api import (
    VirtualniZupan, 
    KnowledgeBaseManager, 
    TrafficService, 
    WasteService, 
    RuleBasedService, 
    ModelRouter, 
    SystemLogger, 
    ThreadSafeCache,
    PROMET_FILTER_LOKACIJE,
    LOCAL_LOG_DIR
)

load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/ask": {}, r"/widget.js": {}}) 
app.secret_key = os.getenv("FLASK_SECRET_KEY", "prosim-spremeni-to-skrivno-vrednost")

ZUPAN_INSTANCE = None

def create_full_zupan_instance():
    """Kreira polno instanco VirtualniZupan z vsemi servisi"""
    try:
        print("🚀 Sestavljam virtualni župan z vsemi servisi...")
        
        # 1. Osnovni servisi
        cache = ThreadSafeCache()
        logger = SystemLogger(LOCAL_LOG_DIR)
        
        # 2. Knowledge Base
        kb = KnowledgeBaseManager()
        if not kb.load():
            print("❌ KRITIČNA NAPAKA: Knowledge base se ni naložil!")
            raise RuntimeError("Knowledge base failed to load")
        
        # 3. OpenAI Client & Router
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        router = ModelRouter(openai_client)
        
        # 4. Specializirani servisi
        traffic = TrafficService(cache=cache, location_keywords=PROMET_FILTER_LOKACIJE)
        waste = WasteService(kb)
        rules = RuleBasedService(kb)
        
        # 5. Glavni Virtualni Župan z dependency injection
        zupan = VirtualniZupan(
            kb_manager=kb,
            traffic_service=traffic,
            waste_service=waste,
            rule_service=rules,
            model_router=router,
            logger=logger
        )
        
        print("✅ Virtualni župan je pripravljen z vsemi servisi!")
        return zupan
        
    except Exception as e:
        print(f"❌ NAPAKA PRI SESTAVLJANJU ŽUPANA: {e}")
        import traceback
        traceback.print_exc()
        raise

def get_zupan():
    global ZUPAN_INSTANCE
    if ZUPAN_INSTANCE is None:
        print("Prva zahteva uporabnika: Inicializacija instance virtualnega župana...")
        # NAMESTO: ZUPAN_INSTANCE = VirtualniZupan()
        # UPORABI: Polno inicializacijo z vsemi servisi
        ZUPAN_INSTANCE = create_full_zupan_instance()
    return ZUPAN_INSTANCE

@app.route('/')
def home():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('chat_widget.html') 

@app.route('/ask', methods=['POST'])
def ask():
    try:
        zupan = get_zupan()
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        uporabnikov_id = session['session_id']
        data = request.get_json()
        uporabnikovo_vprasanje = data.get('question', '')

        if not uporabnikovo_vprasanje:
            return jsonify({'answer': 'Prosim, vnesite vprašanje.'})
        
        print(f"📝 Obdelavam: {uporabnikovo_vprasanje[:50]}...")
        odgovor_zupana = zupan.odgovori(uporabnikovo_vprasanje, session_id=uporabnikov_id)
        
        return jsonify({'answer': odgovor_zupana})
    
    except Exception as e:
        print(f"❌ NAPAKA v /ask: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'answer': 'Prišlo je do napake pri obdelavi vprašanja. Poskusite ponovno.'
        }), 500

@app.route('/widget.js')
def serve_widget():
    return send_from_directory('static', 'widget.js')

@app.route('/health')
def health():
    """Health check za Render monitoring"""
    try:
        # Preveri osnovne komponente
        zupan = get_zupan()
        if not zupan or not zupan.kb_manager:
            return jsonify({'status': 'unhealthy', 'error': 'Services not ready'}), 500
        
        return jsonify({
            'status': 'healthy',
            'version': '40.5',
            'services': {
                'kb_loaded': bool(zupan.kb_manager.collection),
                'model_router': bool(zupan.model_router),
                'traffic_service': bool(zupan.traffic_service),
                'waste_service': bool(zupan.waste_service),
                'rule_service': bool(zupan.rule_service),
                'logger': bool(zupan.logger)
            }
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/stats')
def stats():
    """Sistem statistika"""
    try:
        zupan = get_zupan()
        return jsonify(zupan.get_system_stats())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)