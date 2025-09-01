import os
import time
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from VIRT_ZUPAN_RF_api import VirtualniZupan

app = Flask(__name__)
CORS(app)

# Initialize Virtual Mayor
zupan = VirtualniZupan()

# Session memory za kontekst pogovorov
session_memory = {}
MAX_SESSION_HISTORY = 3

# Kategorije vprašanj za boljšo logiko
QUESTION_CATEGORIES = {
    'jedilnik': ['malica', 'kosilo', 'jedilnik', 'hrana', 'oš', 'šola'],
    'odvoz': ['odvoz', 'smeti', 'odpadki', 'steklo', 'papir', 'embalaža', 'bio'],
    'promet': ['cesta', 'zapora', 'promet', 'dela', 'zastoj'],
    'kontakt': ['kontakt', 'telefon', 'naslov', 'uradne ure'],
    'občina_info': ['prebivalci', 'župan', 'občina', 'ustanovitev'],
    'prevoz': ['avtobus', 'prevoz', 'vozni red', 'šolski'],
    'splošno': ['kje', 'kdo', 'kaj', 'kdaj', 'kako', 'zakaj']
}

def get_session_context(session_id):
    """Pridobi kontekst zadnjih vprašanj v seji"""
    if session_id in session_memory:
        return session_memory[session_id][-MAX_SESSION_HISTORY:]
    return []

def update_session_memory(session_id, question, answer):
    """Posodobi spomin seje"""
    if session_id not in session_memory:
        session_memory[session_id] = []
    
    session_memory[session_id].append({
        'question': question,
        'answer': answer[:200],  # Skrajšano za spomin
        'timestamp': time.time()
    })
    
    # Obdrži le zadnjih MAX_SESSION_HISTORY vprašanj
    if len(session_memory[session_id]) > MAX_SESSION_HISTORY:
        session_memory[session_id] = session_memory[session_id][-MAX_SESSION_HISTORY:]

def categorize_question(question):
    """Kategoriziraj vprašanje"""
    question_lower = question.lower()
    
    for category, keywords in QUESTION_CATEGORIES.items():
        if any(keyword in question_lower for keyword in keywords):
            return category
    return 'splošno'

def detect_context_continuation(question, context):
    """Zazna ali je vprašanje nadaljevanje prejšnjega konteksta"""
    question_lower = question.lower()
    
    # Kratka vprašanja so verjetno nadaljevanja
    if len(question.split()) <= 3 and context:
        if any(word in question_lower for word in ['kaj', 'pa', 'kdaj', 'kje', 'kako']):
            return True
    
    return False

def enhance_question_with_context(question, context):
    """Izboljšaj vprašanje s kontekstom"""
    if not context:
        return question
    
    last_context = context[-1]
    
    # Če je kratko vprašanje, dodaj kontekst
    if len(question.split()) <= 3:
        if 'odvoz' in last_context['answer'].lower():
            return f"Na podlagi prejšnjega vprašanja o odvozu: {question}"
        elif any(school in last_context['answer'].lower() for school in ['oš', 'šola', 'malica']):
            return f"Na podlagi prejšnjega vprašanja o šoli: {question}"
        elif 'ulica' in last_context['answer'].lower():
            return f"Na podlagi prejšnje ulice: {question}"
    
    return question

def create_intelligent_response(question, raw_answer, category, context):
    """Ustvari bolj inteligenten in človeški odgovor"""
    
    # Prepoznaj nelogične kombinacije
    if category == 'občina_info' and 'promet' in raw_answer.lower():
        return ("Oprostite, vaše vprašanje se nanaša na občinske informacije, "
               "vendar sem našel podatke o prometu. Lahko pojasnite kaj vas zanima? "
               "Na primer: informacije o občini, kontaktni podatki, ali prometno stanje?")
    
    if 'žal nimam specifičnega kontakta' in raw_answer:
        # Zamenjaj generičen odgovor z bolj koristnim
        if category == 'jedilnik':
            return ("Za informacije o jedilnikih lahko kontaktirate:\n"
                   "• OŠ Rače: 02 787 81 20\n"
                   "• OŠ Fram: 02 787 82 30\n"
                   "Ali pa obiščete spletni strani šol.")
        elif category == 'prevoz':
            return ("Za informacije o prevozih se obrnite na:\n"
                   "• Šole (OŠ Rače: 02 787 81 20, OŠ Fram: 02 787 82 30)\n"
                   "• Občino Rače-Fram: 02 609 60 10\n"
                   "Lahko preverite tudi spletne strani šol za vozne rede.")
        else:
            return ("Za to informacijo se lahko obrnete na:\n"
                   "📞 Občina Rače-Fram: 02 609 60 10\n"
                   "📧 info@race-fram.si\n"
                   "Radi vam bodo pomagali!")
    
    # Dodaj osebni pridih k standardnim odgovorom
    if 'prebivalcev' in raw_answer:
        return raw_answer.replace('Občina Rače-Fram ima', 'Naša občina šteje')
    
    if category == 'kontakt' and 'občin' in question.lower():
        return """**Kontakt Občine Rače-Fram:**

📞 **Telefon:** 02 609 60 10  
📍 **Naslov:** Grajski trg 14, 2327 Rače  

Uradne ure:
• Ponedeljek: 8:00–12:00 in 13:00–15:00  
• Sreda: 8:00–12:00 in 13:00–17:00  
• Petek: 8:00–13:00"""
    
    return raw_answer

def validate_question_logic(question, category):
    """Preveri logičnost vprašanja"""
    question_lower = question.lower()
    
    # Preveri nelogične kombinacije
    if 'zaprta občina' in question_lower and category == 'promet':
        return ("Mislite verjetno na zapore cest, ne na zaprtje občinske uprave? "
               "Trenutno stanje prometa lahko preverim, za uradne ure občine pa se obrnite na 02 609 60 10.")
    
    if category == 'jedilnik' and not any(school in question_lower for school in ['oš', 'šola', 'rače', 'fram']):
        if len(question.split()) <= 4:  # Kratko vprašanje
            return f"Za jedilnik katere šole vas zanima - OŠ Rače ali OŠ Fram?"
    
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"error": "Prosim, zastavite vprašanje."}), 400
        
        # Pridobi session ID (iz headera ali ustvari novega)
        session_id = request.headers.get('X-Session-ID', f"web_{int(time.time())}")
        
        # Pridobi kontekst seje
        context = get_session_context(session_id)
        
        # Kategoriziraj vprašanje
        category = categorize_question(question)
        
        # Preveri logičnost vprašanja
        logic_check = validate_question_logic(question, category)
        if logic_check:
            return jsonify({
                "question": question,
                "answer": logic_check,
                "status": "clarification"
            })
        
        # Izboljšaj vprašanje s kontekstom če je potrebno
        if detect_context_continuation(question, context):
            enhanced_question = enhance_question_with_context(question, context)
        else:
            enhanced_question = question
        
        # Pridobi odgovor od Virtualnega župana
        print(f"🤖 Processing: '{enhanced_question}' (category: {category})")
        raw_answer = zupan.odgovori(enhanced_question, session_id)
        
        # Ustvari inteligentnejši odgovor
        final_answer = create_intelligent_response(question, raw_answer, category, context)
        
        # Posodobi spomin seje
        update_session_memory(session_id, question, final_answer)
        
        return jsonify({
            "question": question,
            "answer": final_answer,
            "status": "success",
            "session_id": session_id,
            "category": category
        })
    
    except Exception as e:
        print(f"❌ Error in ask_question: {e}")
        return jsonify({
            "error": "Oprostite, prišlo je do tehnične težave. Poskusite kasneje ali pokličite občino na 02 609 60 10.",
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint za monitoring"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(session_memory)
    })

# Cleanup job za stare session-e (opcijsko)
def cleanup_old_sessions():
    """Počisti stare session-e"""
    current_time = time.time()
    old_sessions = []
    
    for session_id, history in session_memory.items():
        if history and current_time - history[-1]['timestamp'] > 3600:  # 1 ura
            old_sessions.append(session_id)
    
    for session_id in old_sessions:
        del session_memory[session_id]
    
    print(f"🧹 Cleaned up {len(old_sessions)} old sessions")

# Zaženi cleanup vsakih 30 minut (opcijsko)
import threading
import time

def periodic_cleanup():
    while True:
        time.sleep(1800)  # 30 minut
        cleanup_old_sessions()

# Zaženi background cleanup
cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Enhanced Virtual Mayor on port {port}")
    print(f"📊 Session memory initialized")
    app.run(host='0.0.0.0', port=port, debug=False)