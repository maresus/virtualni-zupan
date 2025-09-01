import os
import time
import json
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from VIRT_ZUPAN_RF_api import VirtualniZupan

app = Flask(__name__)
CORS(app)

# Initialize Virtual Mayor
print("🚀 Initializing Virtual Mayor system...")
zupan = VirtualniZupan()

# Session memory storage
session_conversations = {}
MAX_HISTORY = 4  # Keep last 4 exchanges

# File-based session backup (optional persistence)
SESSION_BACKUP_FILE = "session_backup.json"

def load_session_backup():
    """Load sessions from file if exists"""
    try:
        if os.path.exists(SESSION_BACKUP_FILE):
            with open(SESSION_BACKUP_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Only load recent sessions (last 2 hours)
                cutoff = time.time() - 7200
                for session_id, history in data.items():
                    if history and history[-1].get('timestamp', 0) > cutoff:
                        session_conversations[session_id] = history
            print(f"📂 Loaded {len(session_conversations)} active sessions from backup")
    except Exception as e:
        print(f"⚠️ Could not load session backup: {e}")

def save_session_backup():
    """Save current sessions to file"""
    try:
        with open(SESSION_BACKUP_FILE, 'w', encoding='utf-8') as f:
            json.dump(session_conversations, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save session backup: {e}")

def add_to_conversation(session_id, question, answer):
    """Add Q&A pair to conversation history"""
    if session_id not in session_conversations:
        session_conversations[session_id] = []
    
    session_conversations[session_id].append({
        'question': question,
        'answer': answer[:200],  # Truncate for context efficiency
        'timestamp': time.time()
    })
    
    # Keep only recent exchanges
    if len(session_conversations[session_id]) > MAX_HISTORY:
        session_conversations[session_id] = session_conversations[session_id][-MAX_HISTORY:]

def build_contextual_question(session_id, new_question):
    """Build question with conversation context using existing system logic"""
    
    # Get conversation history
    history = session_conversations.get(session_id, [])
    
    if not history:
        return new_question
    
    # Convert to format expected by existing preoblikuj_vprasanje_s_kontekstom
    zgodovina_pogovora = []
    for exchange in history[-3:]:  # Last 3 exchanges
        zgodovina_pogovora.append((exchange['question'], exchange['answer']))
    
    # Use the existing context transformation function
    try:
        contextual_question = zupan.preoblikuj_vprasanje_s_kontekstom(
            zgodovina_pogovora, 
            new_question
        )
        return contextual_question
    except Exception as e:
        print(f"Context transformation error: {e}")
        return new_question

def is_short_followup_question(question):
    """Detect if this is likely a follow-up question"""
    question_lower = question.lower().strip()
    
    # Short questions that are likely follow-ups
    followup_patterns = [
        'kaj pa', 'kako pa', 'kdaj pa', 'kje pa', 'kdo pa',
        'kaj v', 'kako v', 'kdaj v', 'kje v', 
        'kaj za', 'kako za', 'kdaj za',
        'kaj o', 'kako o', 'kdaj o',
        'oš rače', 'oš fram', 'osnovni šoli',
        'bistriška', 'bistriski', 'bistriska',  # DODANO
        'mlinska', 'framska', 'grajski',       # DODANO
        'kaj tam', 'kaj pri', 'kaj na'         # DODANO
    ]
    
    # Lokacijske besede brez konteksta
    location_only = [
        'bistriška cesta', 'mlinska cesta', 'framska cesta', 
        'grajski trg', 'mariborska cesta'
    ]
    
    # Check if question is short and contains follow-up patterns
    word_count = len(question.split())
    contains_pattern = any(pattern in question_lower for pattern in followup_patterns)
    contains_location = any(loc in question_lower for loc in location_only)
    
    return word_count <= 4 or contains_pattern or contains_location

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
        
        print(f"🔍 Session: {session_id[:12]}... | Q: '{question}'")
        print(f"📊 Active sessions: {len(session_conversations)}")
        
        # Check if this is likely a follow-up question
        is_followup = is_short_followup_question(question)
        
        if is_followup and session_id in session_conversations:
            print("🔗 Detected follow-up question, adding context...")
            contextual_question = build_contextual_question(session_id, question)
            print(f"🎯 Contextual Q: '{contextual_question}'")
        else:
            contextual_question = question
        
        # Get answer from Virtual Mayor
        answer = zupan.odgovori(contextual_question, session_id)
        
        # Special handling for contact info
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
        
        print(f"✅ Answer generated ({len(answer)} chars)")
        
        return jsonify({
            "question": question,
            "answer": answer,
            "status": "success",
            "session_id": session_id,
            "context_used": is_followup
        })
    
    except Exception as e:
        print(f"❌ ERROR in ask_question: {e}")
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
        "active_conversations": len(session_conversations),
        "total_exchanges": sum(len(history) for history in session_conversations.values())
    })

@app.route('/debug-session/<session_id>', methods=['GET'])
def debug_session(session_id):
    """Debug endpoint to view session history"""
    history = session_conversations.get(session_id, [])
    return jsonify({
        "session_id": session_id,
        "history": history,
        "count": len(history)
    })

def cleanup_old_conversations():
    """Remove conversations older than 2 hours"""
    current_time = time.time()
    old_sessions = []
    
    for sid, history in session_conversations.items():
        if history and current_time - history[-1]['timestamp'] > 7200:  # 2 hours
            old_sessions.append(sid)
    
    for sid in old_sessions:
        del session_conversations[sid]
    
    if old_sessions:
        print(f"🧹 Cleaned up {len(old_sessions)} old sessions")
        save_session_backup()  # Save after cleanup

# Periodic cleanup
import threading
def periodic_cleanup():
    while True:
        time.sleep(3600)  # Every hour
        cleanup_old_conversations()

# Load previous sessions on startup
load_session_backup()

# Start cleanup thread
cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()

# Save sessions on app shutdown
import atexit
atexit.register(save_session_backup)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Contextual Virtual Mayor on port {port}")
    print(f"📝 Session tracking active with {len(session_conversations)} loaded sessions")
    app.run(host='0.0.0.0', port=port, debug=False)