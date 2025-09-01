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

# Simplified session memory - just store conversation history
session_conversations = {}
MAX_HISTORY = 5

def add_to_conversation(session_id, question, answer):
    """Add Q&A pair to conversation history"""
    if session_id not in session_conversations:
        session_conversations[session_id] = []
    
    session_conversations[session_id].append({
        'question': question,
        'answer': answer[:300],  # Limit answer length for context
        'timestamp': time.time()
    })
    
    # Keep only last MAX_HISTORY exchanges
    if len(session_conversations[session_id]) > MAX_HISTORY:
        session_conversations[session_id] = session_conversations[session_id][-MAX_HISTORY:]

def build_context_prompt(session_id, new_question):
    """Build enhanced prompt with conversation history"""
    
    # Get conversation history
    history = session_conversations.get(session_id, [])
    
    if not history:
        # No context - just the question
        return new_question
    
    # Build conversation context
    context_lines = []
    for exchange in history[-3:]:  # Last 3 exchanges
        context_lines.append(f"Uporabnik: {exchange['question']}")
        context_lines.append(f"Župan: {exchange['answer'][:150]}...")
    
    context_text = "\n".join(context_lines)
    
    # Enhanced prompt with instructions for the LLM
    enhanced_prompt = f"""KONTEKST POGOVORA (zadnja vprašanja):
{context_text}

NOVO VPRAŠANJE: {new_question}

NAVODILA ZA ODGOVOR:
- Če je novo vprašanje kratko ali nejasno ("kaj pa...", "kdaj...", "kje...", "kako..."), se verjetno navezuje na prejšnji pogovor
- Uporabi kontekst iz prejšnjih vprašanj za boljše razumevanje
- Če vprašanje ni dovolj jasno, povprašaj za pojasnilo
- Odgovori kot pravi župan občine - osebno, prijazno, s poznavanjem lokalnih razmer
- Uporabi "naša občina", "naši občani" namesto "občina Rače-Fram"

Odgovori direktno na vprašanje:"""
    
    return enhanced_prompt

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
        
        # Get or generate session ID
        session_id = request.headers.get('X-Session-ID', f"web_{int(time.time())}")
        
        print(f"DEBUG: Session {session_id[:15]}... | Question: '{question}'")
        print(f"DEBUG: Active conversations: {len(session_conversations)}")
        
        # Build enhanced prompt with context
        enhanced_prompt = build_context_prompt(session_id, question)
        
        print(f"DEBUG: Using enhanced prompt: {len(enhanced_prompt)} chars")
        
        # Get answer from Virtual Mayor with enhanced context
        answer = zupan.odgovori(enhanced_prompt, session_id)
        
        # Special handling for contact info to keep it simple
        if 'kontakt' in question.lower() and 'občin' in question.lower():
            answer = """**Kontakt Občine Rače-Fram:**

📞 **Telefon:** 02 609 60 10
📍 **Naslov:** Grajski trg 14, 2327 Rače

Uradne ure:
• Ponedeljek: 8:00–12:00 in 13:00–15:00
• Sreda: 8:00–12:00 in 13:00–17:00  
• Petek: 8:00–13:00"""
        
        # Add this exchange to conversation history
        add_to_conversation(session_id, question, answer)
        
        print(f"DEBUG: Answer generated, length: {len(answer)} chars")
        
        return jsonify({
            "question": question,
            "answer": answer,
            "status": "success",
            "session_id": session_id
        })
    
    except Exception as e:
        print(f"ERROR in ask_question: {e}")
        return jsonify({
            "error": "Oprostite, prišlo je do tehnične težave. Poskusite kasneje ali pokličite občino na 02 609 60 10.",
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_conversations": len(session_conversations)
    })

# Cleanup old conversations periodically
def cleanup_old_conversations():
    """Remove conversations older than 1 hour"""
    current_time = time.time()
    old_sessions = []
    
    for session_id, history in session_conversations.items():
        if history and current_time - history[-1]['timestamp'] > 3600:  # 1 hour
            old_sessions.append(session_id)
    
    for session_id in old_sessions:
        del session_conversations[session_id]
    
    if old_sessions:
        print(f"Cleaned up {len(old_sessions)} old conversation sessions")

# Run cleanup every 30 minutes
import threading
def periodic_cleanup():
    while True:
        time.sleep(1800)  # 30 minutes
        cleanup_old_conversations()

cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Contextual Virtual Mayor on port {port}")
    print(f"📝 Conversation tracking initialized")
    app.run(host='0.0.0.0', port=port, debug=False)