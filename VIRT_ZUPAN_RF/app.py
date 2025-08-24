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
    LOCAL_LOG_DIR,
    normalize_text,
    _has_waste_intent,
    _has_traffic_intent,
    get_canonical_waste_type,
    WASTE_TYPE_VARIANTS
)

load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/ask": {}, r"/widget.js": {}, r"/debug": {}, r"/db-info": {}, r"/add-staff": {}, r"/check-staff": {}}) 
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

# ============================================================================
# DEBUG ENDPOINTS - ZA ODKRIVANJE RAZLIK MED LOKALNO IN RENDER
# ============================================================================

@app.route('/debug-env')
def debug_env():
    """Pokaže environment informacije"""
    try:
        from . import VIRT_ZUPAN_RF_api as api_module
        
        return jsonify({
            'environment': {
                'ENV_TYPE': os.getenv('ENV_TYPE', 'NOT_SET'),
                'DATA_DIR': getattr(api_module, 'DATA_DIR', 'NOT_SET'),
                'CHROMA_DB_PATH': getattr(api_module, 'CHROMA_DB_PATH', 'NOT_SET'),
                'PRIMARY_GEN_MODEL': getattr(api_module, 'PRIMARY_GEN_MODEL', 'NOT_SET'),
                'ALT_GEN_MODELS': getattr(api_module, 'ALT_GEN_MODELS', []),
                'ECONOMY_MODE': getattr(api_module, 'ECONOMY_MODE', False),
                'FORCE_MODEL': getattr(api_module, 'FORCE_MODEL', ''),
                'LLM_DEBUG': getattr(api_module, 'LLM_DEBUG', False)
            },
            'files_exist': {
                'chroma_db_exists': os.path.exists(getattr(api_module, 'CHROMA_DB_PATH', '')),
            },
            'python_info': {
                'version': str(os.sys.version),
                'platform': str(os.sys.platform)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/db-info')
def db_info():
    """Pokaže informacije o bazi podatkov"""
    try:
        zupan = get_zupan()
        
        # Osnove info
        total_docs = zupan.kb_manager.collection.count() if zupan.kb_manager.collection else 0
        
        # Waste documents
        waste_docs = zupan.kb_manager.get_waste_documents()
        waste_count = len(waste_docs.get('documents', [])) if waste_docs else 0
        
        # Sample metadatov
        sample_meta = []
        if waste_docs and waste_docs.get('metadatas'):
            sample_meta = waste_docs['metadatas'][:5]  # Prvi 5 metadatov
        
        return jsonify({
            'database': {
                'total_documents': total_docs,
                'waste_documents': waste_count,
                'sample_metadata': sample_meta
            },
            'collection_name': zupan.kb_manager.collection_name if hasattr(zupan.kb_manager, 'collection_name') else 'N/A'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/debug-query', methods=['POST'])
def debug_query():
    """Debug specifične poizvedbe"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Manjka query parameter'})
        
        zupan = get_zupan()
        
        # Osnovne analize
        normalized = normalize_text(query)
        is_waste = _has_waste_intent(normalized)
        is_traffic = _has_traffic_intent(normalized)
        waste_type = get_canonical_waste_type(normalized) if is_waste else None
        
        debug_info = {
            'original_query': query,
            'normalized': normalized,
            'intent_detection': {
                'is_waste': is_waste,
                'is_traffic': is_traffic,
                'detected_waste_type': waste_type
            }
        }
        
        # Če je waste query, dodaj podrobnosti
        if is_waste:
            # Location extraction logic (kopiran iz waste service)
            waste_type_stopwords = {normalize_text(k) for k in WASTE_TYPE_VARIANTS.keys()}
            for variants in WASTE_TYPE_VARIANTS.values():
                for v in variants:
                    waste_type_stopwords.add(normalize_text(v))
            
            extra_stop = {
                "kdaj", "je", "naslednji", "odvoz", "odpadkov", "odpadke", "smeti", "na", "v",
                "za", "kako", "kateri", "katera", "kaj", "kje", "rumene", "rumena", "kanta",
                "kante", "ulici", "cesti"
            }
            odstrani = waste_type_stopwords.union(extra_stop)
            
            raw_tokens = [t for t in normalized.split() if t and t not in odstrani]
            
            location_phrases = []
            for size in (3, 2, 1):
                for i in range(len(raw_tokens) - size + 1):
                    phrase = " ".join(raw_tokens[i:i + size])
                    if phrase and phrase not in location_phrases:
                        location_phrases.append(phrase)
            
            debug_info['waste_processing'] = {
                'raw_tokens': raw_tokens,
                'location_phrases': location_phrases,
                'removed_stopwords': list(odstrani)
            }
        
        # Dodaj actual response
        actual_response = zupan.odgovori(query, "debug_session")
        debug_info['actual_response'] = actual_response
        
        return jsonify(debug_info)
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        })

@app.route('/test-queries')
def test_queries():
    """Testira ključne poizvedbe"""
    test_cases = [
        "kdaj je odvoz stekla pod terasami",
        "kdaj je odvoz stekla na požegu", 
        "kdo je župan občine",
        "kaj pa pod terasami"
    ]
    
    try:
        zupan = get_zupan()
        results = {}
        
        for query in test_cases:
            try:
                response = zupan.odgovori(query, f"test_{hash(query)}")
                results[query] = {
                    'response': response,
                    'length': len(response),
                    'status': 'success'
                }
            except Exception as e:
                results[query] = {
                    'error': str(e),
                    'status': 'error'
                }
        
        return jsonify({
            'test_results': results,
            'total_tests': len(test_cases),
            'passed': len([r for r in results.values() if r.get('status') == 'success'])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

# ============================================================================
# STAFF MANAGEMENT ENDPOINTS - DODAJANJE ZAPOSLENIH V CHROMADB
# ============================================================================

@app.route('/add-staff', methods=['POST'])
def add_staff():
    """Endpoint za dodajanje podatkov o zaposlenih v ChromaDB"""
    
    # Podatki o zaposlenih
    staff_data = [
        {"text": "Direktorica občinske uprave Občine Rače-Fram je mag. Karmen Kotnik. Njen kontakt je karmen.kotnik@race-fram.si ali 02 609 60 19.", "metadata": {"kategorija": "Zaposleni", "vir": "Imenik zaposlenih", "tip": "direktor", "oseba": "Karmen Kotnik"}},
        {"text": "Višja svetovalka za prostorsko planiranje in premoženjske zadeve je Suzana Pungartnik. Njen kontakt je suzana.pungartnik@race-fram.si ali 02 609 60 15.", "metadata": {"kategorija": "Zaposleni", "vir": "Imenik zaposlenih", "tip": "prostorsko_planiranje", "oseba": "Suzana Pungartnik"}},
        {"text": "Računovodja VII/II je Rosvita Robar. Njen kontakt je rosvita.robar@race-fram.si ali 02 609 60 14.", "metadata": {"kategorija": "Zaposleni", "vir": "Imenik zaposlenih", "tip": "racunovodja", "oseba": "Rosvita Robar"}},
        {"text": "Višja svetovalka za investicije in investicijsko vzdrževanje je Mateja Frešer. Njen kontakt je mateja.freser@race-fram.si ali 02 609 60 23.", "metadata": {"kategorija": "Zaposleni", "vir": "Imenik zaposlenih", "tip": "investicije", "oseba": "Mateja Frešer"}},
        {"text": "Višja svetovalka za okolje, kmetijstvo, turizem in civilno zaščito je Tanja Kosi. Njen kontakt je tanja.kosi@race-fram.si ali 02 609 60 24.", "metadata": {"kategorija": "Zaposleni", "vir": "Imenik zaposlenih", "tip": "turizem", "oseba": "Tanja Kosi"}},
        {"text": "Višja svetovalka za delovne in splošne pravne zadeve ter javna naročila je Anja Čelan. Njen kontakt je anja.celan@race-fram.si ali 02 609 60 27.", "metadata": {"kategorija": "Zaposleni", "vir": "Imenik zaposlenih", "tip": "pravne_zadeve", "oseba": "Anja Čelan"}},
        {"text": "Višja svetovalka za družbene dejavnosti je Monika Skledar. Njen kontakt je monika.skledar@race-fram.si ali 02 609 60 28.", "metadata": {"kategorija": "Zaposleni", "vir": "Imenik zaposlenih", "tip": "druzbene_dejavnosti", "oseba": "Monika Skledar"}},
        {"text": "Referentka za računovodstvo in šport je Klaudia Sovdat. Njen kontakt je klaudia.sovdat@race-fram.si ali 02 609 60 12.", "metadata": {"kategorija": "Zaposleni", "vir": "Imenik zaposlenih", "tip": "sport", "oseba": "Klaudia Sovdat"}},
        {"text": "Referentka – tajnica je Marjetka Kristl. Njen kontakt je marjetka.kristl@race-fram.si ali 02 609 60 10.", "metadata": {"kategorija": "Zaposleni", "vir": "Imenik zaposlenih", "tip": "tajnica", "oseba": "Marjetka Kristl"}},
        {"text": "Administrator V je Jožica Medved. Njen kontakt je jozica.medved@race-fram.si ali 02 609 60 13.", "metadata": {"kategorija": "Zaposleni", "vir": "Imenik zaposlenih", "tip": "administrator", "oseba": "Jožica Medved"}},
        {"text": "Vodja režijskega obrata je Gregor Ovnik. Njegov kontakt je gregor.ovnik@race-fram.si ali 02 609 60 25. Njegova mobilna številka je 051 815 947.", "metadata": {"kategorija": "Zaposleni", "vir": "Imenik zaposlenih", "tip": "rezijski_obrat", "oseba": "Gregor Ovnik"}},
        {"text": "Delovodja v režijskem obratu je Vojko Kmetec. Njegova telefonska številka je 051 367 478.", "metadata": {"kategorija": "Zaposleni", "vir": "Imenik zaposlenih", "tip": "rezijski_obrat", "oseba": "Vojko Kmetec"}},
        {"text": "Delovodja v režijskem obratu je Ludvik Stupan. Njegova telefonska številka je 051 661 850.", "metadata": {"kategorija": "Zaposleni", "vir": "Imenik zaposlenih", "tip": "rezijski_obrat", "oseba": "Ludvik Stupan"}},
        {"text": "Inštalater v režijskem obratu je Aleš Šmelc. Njegova telefonska številka je 041 534 402.", "metadata": {"kategorija": "Zaposleni", "vir": "Imenik zaposlenih", "tip": "rezijski_obrat", "oseba": "Aleš Šmelc"}},
        {"text": "Inštalater v režijskem obratu je Miran Grašič. Njegova telefonska številka je 031 418 031.", "metadata": {"kategorija": "Zaposleni", "vir": "Imenik zaposlenih", "tip": "rezijski_obrat", "oseba": "Miran Grašič"}}
    ]
    
    try:
        zupan = get_zupan()
        collection = zupan.kb_manager.collection
        
        if not collection:
            return jsonify({'error': 'ChromaDB collection ni na voljo'}), 500
        
        # Preveri če podatki že obstajajo
        try:
            existing = collection.get(where={"kategorija": "Zaposleni"}, limit=1)
            if existing and existing.get('documents') and len(existing['documents']) > 0:
                return jsonify({
                    'message': 'Podatki o zaposlenih že obstajajo',
                    'existing_count': len(existing['documents'])
                })
        except Exception as e:
            print(f"Warning: Could not check existing staff: {e}")
        
        # Dodaj nove podatke
        texts = [item["text"] for item in staff_data]
        metadatas = [item["metadata"] for item in staff_data] 
        ids = [f"staff_{i}_{item['metadata']['oseba'].replace(' ', '_').lower()}" for i, item in enumerate(staff_data)]
        
        collection.add(
            documents=texts,
            metadatas=metadatas, 
            ids=ids
        )
        
        # Test iskanja
        try:
            test_results = collection.query(
                query_texts=["kdo je zadolžen za turizem"],
                n_results=3,
                where={"kategorija": "Zaposleni"}
            )
            
            test_query_count = len(test_results.get('documents', [[]])[0]) if test_results.get('documents') else 0
            sample_result = test_results.get('documents', [[]])[0][:1] if test_results.get('documents') else []
        except Exception as e:
            print(f"Warning: Test query failed: {e}")
            test_query_count = 0
            sample_result = []
        
        return jsonify({
            'message': f'Successfully added {len(staff_data)} staff records',
            'total_documents': collection.count(),
            'test_query_results': test_query_count,
            'sample_result': sample_result
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/check-staff')
def check_staff():
    """Preveri ali so podatki o zaposlenih v bazi"""
    try:
        zupan = get_zupan()
        collection = zupan.kb_manager.collection
        
        if not collection:
            return jsonify({'error': 'ChromaDB collection ni na voljo'})
        
        # Poizkusi najti zaposlene
        try:
            staff_results = collection.get(
                where={"kategorija": "Zaposleni"}, 
                limit=20,
                include=["documents", "metadatas"]
            )
        except Exception as e:
            return jsonify({
                'error': f'Could not query staff: {str(e)}',
                'staff_count': 0,
                'sample_staff': [],
                'sample_metadata': []
            })
        
        return jsonify({
            'staff_count': len(staff_results.get('documents', [])),
            'sample_staff': staff_results.get('documents', [])[:3],
            'sample_metadata': staff_results.get('metadatas', [])[:3]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/reset-staff', methods=['POST']) 
def reset_staff():
    """Počisti in znova doda podatke o zaposlenih"""
    try:
        zupan = get_zupan()
        collection = zupan.kb_manager.collection
        
        if not collection:
            return jsonify({'error': 'ChromaDB collection ni na voljo'}), 500
        
        # Poizkusi pobrisati obstoječe zaposlene
        try:
            existing = collection.get(where={"kategorija": "Zaposleni"})
            if existing and existing.get('ids'):
                collection.delete(ids=existing['ids'])
                deleted_count = len(existing['ids'])
            else:
                deleted_count = 0
        except Exception as e:
            print(f"Warning: Could not delete existing staff: {e}")
            deleted_count = 0
        
        # Klic add_staff funkcije
        add_result = add_staff()
        
        if add_result[1] == 200:  # Če je add_staff uspešen
            result_data = add_result[0].get_json()
            result_data['deleted_count'] = deleted_count
            result_data['message'] = f"Reset completed: deleted {deleted_count}, added {len(staff_data)}"
            return jsonify(result_data)
        else:
            return add_result
            
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)