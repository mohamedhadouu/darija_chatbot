# Fix pour le conflit OpenMP - DOIT ÊTRE EN PREMIER
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, Response
from flask_cors import CORS
import json
import logging
from datetime import datetime
import uuid
import ssl
import threading
import time
import base64
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration
UPLOAD_FOLDER = 'Source Files'
AUDIO_FOLDER = 'audio_files'
RECORDINGS_FOLDER = 'recordings'
CERT_FILE = 'cert.pem'
KEY_FILE = 'key.pem'

# Créer les dossiers nécessaires
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(RECORDINGS_FOLDER, exist_ok=True)

from openai import OpenAI
from Main import Main

# Initialisation des composants
main_system: Main 
app = Flask(__name__, static_url_path='', static_folder='.')
CORS(app)
def init_system():
    """Initialise le système principal avec OpenAI embeddings"""
    global main_system
    try:
        logger.info("Initialisation du système avec OpenAI embeddings...")

        # Vérifier la clé API OpenAI
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Clé API OpenAI manquante dans le fichier .env")

        # Créer un client OpenAI
        client = OpenAI(api_key=openai_api_key)
        
        # Initialiser le système principal avec le nouveau modèle
        main_system = Main(
            client=client,
            model_name="text-embedding-3-small",  # Nouveau modèle
            embedding_dimensions=1536,            # Nouvelles dimensions
            force_reindex=False         # Forcer la réindexation pour la migration
        )

        logger.info("Système initialisé avec succès!")
        logger.info(f"Modèle d'embeddings: text-embedding-3-small (1536D)")
        logger.info("Configuration: Transcription en français/darija latine pour meilleure compatibilité LLM")
        return True
    except Exception as e:
        logger.exception(f"Erreur d'initialisation: {e}")
        return False

def normalize_darija_for_tts(text):
    """
    Normalise le texte darija pour améliorer la synthèse vocale française
    """
    if not text:
        return text
    
    # Mapping des chiffres arabes et lettres darija
    darija_mappings = {
        '3': 'aa',      # عين -> e 
        '7': 'h',      # حاء -> h
        '9': 'k',      # قاف -> k
        'kh': 'k',     # خ -> k pour TTS
        'gh': 'g',     # غ -> g pour TTS  
        'w': 'ou',     # و -> ou
        'y': 'i' ,
        
        # Mots darija courants -> phonétique française
        'bghit': 'beghit',
        'bezaf': 'bézaf',
        'chwiya': 'chouiya',
        'mzyan': 'mezian',
        'kayn': 'kayin',
        'makainch': 'makaynch',
        'allah': 'ala',
        'inchallah': 'inchala',
        'machallah': 'machala',
    }
    
    normalized_text = text
    
    # Appliquer les mappings
    for darija_char, french_sound in darija_mappings.items():
        if darija_char.isdigit():
            # Remplacer seulement si entouré de lettres
            import re
            pattern = r'(?<=[a-zA-Z])' + re.escape(darija_char) + r'(?=[a-zA-Z])'
            normalized_text = re.sub(pattern, french_sound, normalized_text)
        else:
            normalized_text = normalized_text.replace(darija_char, french_sound)
    
    # Nettoyer
    normalized_text = ' '.join(normalized_text.split())
    
    return normalized_text

def enhance_darija_transcription(text):
    """
    Améliore la transcription darija avec corrections courantes
    """
    if not text:
        return text
    
    # Corrections darija courantes
    corrections = {
        'wach': 'wach', 'chkoun': 'chkoun', 'fin': 'fin', 'kifach': 'kifach',
        'bghit': 'bghit', 'kayn': 'kayn', 'makainch': 'makainch',
        'bezaf': 'bezaf', 'chwiya': 'chwiya', 'mzyan': 'mzyan',
        'daba': 'daba', 'ghda': 'ghda', 'salam': 'salam alikom',
        'hada': 'hada', 'hadik': 'hadik', 'ach': 'ach'
    }
    
    processed_text = text.lower()
    
    for wrong, correct in corrections.items():
        import re
        pattern = r'\b' + re.escape(wrong.lower()) + r'\b'
        processed_text = re.sub(pattern, correct, processed_text)
    
    return ' '.join(processed_text.split())

def transcribe_audio_darija_latin(audio_path):
    with open(audio_path, "rb") as f:
        transcript = main_system.client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="ar"
        )
    return transcript.text

def post_process_darija_transcription(text):
    """
    Post-traitement pour améliorer la transcription darija en alphabet latin
    """
    if not text:
        return text
    
    # Corrections courantes pour la darija transcrite en français
    corrections = {
        # Salutations courantes
        'salam': 'salam alikom',
        'ahlan': 'ahlan wa sahlan',
        'marhaba': 'marhaba',
        
        # Mots darija courants mal transcrits
        'wach': 'wach',  # "quoi" en darija
        'fin': 'fin',    # "où" en darija
        'kifach': 'kifach',  # "comment" en darija
        'chkoun': 'chkoun',  # "qui" en darija
        'imta': 'imta',  # "quand" en darija
        'ach': 'ach',    # "qu'est-ce que" en darija
        'hada': 'hada',  # "ceci" en darija
        'hadik': 'hadik', # "cela" en darija
        
        # Mots français/darija mixtes
        'bghit': 'bghit',  # "je veux" en darija
        'kayn': 'kayn',    # "il y a" en darija
        'makainch': 'makainch',  # "il n'y a pas" en darija
        'bezaf': 'bezaf',  # "beaucoup" en darija
        'chwiya': 'chwiya', # "un peu" en darija
        
        # Corrections de transcription commune
        'inchallah': 'inchallah',
        'machallah': 'machallah',
        'hamdulillah': 'hamdulillah',
    }
    
    # Appliquer les corrections (insensible à la casse)
    processed_text = text
    for wrong, correct in corrections.items():
        # Remplacer en préservant la casse
        import re
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        processed_text = pattern.sub(correct, processed_text)
    
    # Nettoyer les espaces multiples
    processed_text = ' '.join(processed_text.split())
    
    return processed_text

def generate_ssl_cert():
    """Génère un certificat SSL auto-signé pour le développement"""
    if not os.path.exists(CERT_FILE) or not os.path.exists(KEY_FILE):
        logger.info("Génération du certificat SSL...")
        os.system(f'openssl req -x509 -newkey rsa:4096 -keyout {KEY_FILE} -out {CERT_FILE} -days 365 -nodes -subj "/C=MA/ST=Fes/L=Fes/O=Agriculture/CN=localhost"')

@app.route('/audio_files/<filename>')
def serve_audio(filename):
    path = os.path.join("audio_files", filename)
    if not os.path.exists(path):
        return jsonify({'error': 'Fichier audio introuvable'}), 404
    return send_from_directory('audio_files', filename)

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Vérifie l'état du système et de l'accès à l'API OpenAI"""
    if main_system:
        try:
            health = main_system.check_system_health() if hasattr(main_system, "check_system_health") else {}

            # Vérification de la disponibilité API OpenAI
            try:
                main_system.client.models.list()  # requête test
                health["openai_api_status"] = "ok"
            except Exception as e:
                health["openai_api_status"] = f"error: {str(e)}"

            # Informations supplémentaires pour la migration
            health["embedding_model"] = "text-embedding-3-small"
            health["embedding_dimensions"] = 1536
            health["transcription_mode"] = "openai_whisper_api"
            health["migration_status"] = "completed"
            health["timestamp"] = datetime.now().isoformat()
            return jsonify(health)
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f"Erreur lors du contrôle de santé: {e}"
            }), 500
    else:
        return jsonify({'status': 'error', 'message': 'Système non initialisé'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint pour les requêtes texte"""
    try:
        data = request.get_json()
        query = data.get('message', '').strip()
        
        if not query:
            return jsonify({'error': 'Message vide'}), 400
        
        if not main_system:
            return jsonify({'error': 'Système non initialisé'}), 500
        
        logger.info(f"Requête reçue: {query}")
        
        # Traitement de la requête avec le nouveau système d'embeddings
        result = main_system.process_query(query)
        
        # Génération de la réponse via LLM
        response_generator = main_system.llm_responder.generate_response(
            query, 
            result.get('response', [])
        )
        
        # Récupération de la réponse complète
        full_response = ""
        for chunk in response_generator:
            if chunk.startswith("data: "):
                chunk_data = json.loads(chunk[6:])
                if 'content' in chunk_data:
                    full_response = chunk_data['content']
                    break
        
        return jsonify({
            'response': full_response,
            'sources': result.get('sources', []),
            'debug_info': result.get('debug_info', {}),
            'embedding_model': 'text-embedding-3-small',  # Ajout de l'info du modèle
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erreur dans /chat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload_document', methods=['POST'])
def upload_document():
    """Endpoint pour l'upload et traitement automatique de documents"""
    try:
        if 'document' not in request.files:
            return jsonify({'error': 'Aucun fichier document'}), 400
        
        doc_file = request.files['document']
        if doc_file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        # Extensions autorisées
        allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md', '.xlsx', '.xls', '.csv'}
        filename = str(doc_file.filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Format non supporté: {file_ext}'}), 400
        
        # Sauvegarde du fichier dans Source Files
        safe_filename = f"{uuid.uuid4().hex}_{doc_file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, safe_filename)
        doc_file.save(file_path)
        
        logger.info(f"Document sauvegardé: {file_path}")
        
        # Attendre un peu que le fichier soit complètement écrit
        import time
        time.sleep(0.5)
        
        # Le système d'embedding traite automatiquement les nouveaux fichiers
        # On attend que le traitement soit terminé en surveillant le nombre d'embeddings
        initial_count = main_system.embedding_system.collection.count_documents({})
        
        # Attendre que le traitement soit terminé (max 30 secondes)
        max_wait = 30
        wait_time = 0
        processed = False
        
        while wait_time < max_wait and not processed:
            time.sleep(1)
            wait_time += 1
            current_count = main_system.embedding_system.collection.count_documents({})
            
            # Si le nombre d'embeddings a augmenté, le fichier a été traité
            if current_count > initial_count:
                processed = True
                break
        
        if processed:
            new_count = main_system.embedding_system.collection.count_documents({})
            return jsonify({
                'status': 'success',
                'message': 'Document traité avec succès',
                'original_filename': doc_file.filename,
                'saved_filename': safe_filename,
                'embedding_count': new_count,
                'embedding_model': 'text-embedding-3-small',
                'processing_time': f"{wait_time} secondes",
                'timestamp': datetime.now().isoformat()
            })
        else:
            # Le fichier a été sauvé mais pas encore traité
            return jsonify({
                'status': 'saved',
                'message': 'Document sauvegardé, traitement en cours...',
                'original_filename': doc_file.filename,
                'saved_filename': safe_filename,
                'note': 'Le document sera disponible pour les recherches dans quelques instants',
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"Erreur dans /upload_document: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/check_processing_status')
def check_processing_status():
    """Vérifie si des documents sont en cours de traitement"""
    try:
        # Obtenir le nombre actuel d'embeddings
        current_count = main_system.embedding_system.collection.count_documents({})
        
        # Obtenir les informations sur le dernier document traité
        last_doc = main_system.embedding_system.collection.find_one(
            {},
            sort=[("processing_date", -1)]
        )
        
        return jsonify({
            'embedding_count': current_count,
            'last_processed': last_doc.get("processing_date") if last_doc else None,
            'last_filename': last_doc.get("file_path") if last_doc else None,
            'embedding_model': last_doc.get("model_name", "text-embedding-3-small") if last_doc else "text-embedding-3-small",
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erreur dans /check_processing_status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/list_documents')
def list_documents():
    """Liste tous les documents dans Source Files"""
    try:
        documents = []
        
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    documents.append({
                        'filename': filename,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        
        # Trier par date de modification (plus récent en premier)
        documents.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({
            'count': len(documents),
            'documents': documents,
            'embedding_model': 'text-embedding-3-small',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erreur dans /list_documents: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/reindex', methods=['POST'])
def reindex():
    """Force un réindexage complet avec le nouveau modèle"""
    try:
        logger.info("Démarrage du réindexage avec text-embedding-3-small...")
        main_system.reindex_all()
        
        new_count = main_system.embedding_system.collection.count_documents({})
        
        return jsonify({
            'status': 'success',
            'message': 'Réindexage terminé avec text-embedding-3-small',
            'embedding_count': new_count,
            'embedding_model': 'text-embedding-3-small',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erreur dans /reindex: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """Endpoint pour l'upload et transcription d'audio - version API OpenAI"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'Aucun fichier audio'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        # Sauvegarde temporaire du fichier audio
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_audio_{uuid.uuid4().hex}.wav")
        audio_file.save(temp_path)
        logger.info(f"Transcription de l'audio: {temp_path}")
        
        # ✅ Transcription avec OpenAI Whisper API
        try:
            transcript_text = main_system.client.audio.transcriptions.create(
                model="whisper-1",
                file=open(temp_path, "rb"),
                language="ar"
            ).text
        except Exception as transcribe_err:
            logger.error(f"Erreur transcription: {transcribe_err}")
            return jsonify({'error': f'Erreur de transcription: {transcribe_err}'}), 500
        
        # Nettoyage
        os.remove(temp_path)

        if not transcript_text.strip():
            return jsonify({'error': 'Transcription vide'}), 400
        
        logger.info(f"Transcription (Whisper API): {transcript_text}")
        
        # Traitement de la transcription avec le nouveau système d'embeddings
        query_result = main_system.process_query(transcript_text)
        response_generator = main_system.llm_responder.generate_response(
            transcript_text,
            query_result.get('response', [])
        )

        full_response = ""
        for chunk in response_generator:
            if chunk.startswith("data: "):
                chunk_data = json.loads(chunk[6:])
                if 'content' in chunk_data:
                    full_response = chunk_data['content']
                    break

        return jsonify({
            'response': full_response,
            'sources': query_result.get('sources', []),
            'embedding_model': 'text-embedding-3-small',
            'timestamp': datetime.now().isoformat(),
            'detected_language': "ar",
            'transcription_method': "openai_whisper"
        })

    except Exception as e:
        logger.error(f"Erreur dans /upload_audio: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    """Endpoint pour la synthèse vocale avec OpenAI TTS - Darija compatible"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        language = data.get('language', 'darija_latin')

        if not text:
            return jsonify({'error': 'Texte vide'}), 400

        # ✅ Normalisation darija (si nécessaire)
        normalized_text = normalize_darija_for_tts(text)

        logger.info(f"TTS darija: '{text}' -> '{normalized_text}'")

        # 🔊 Synthèse vocale avec OpenAI
        response = main_system.client.audio.speech.create(
            model="tts-1",
            voice="nova",  # onyx, echo, fable, etc.
            input=normalized_text
        )

        # ✅ Sauvegarde de l'audio
        output_path = os.path.join("audio_files", f"tts_{uuid.uuid4().hex}.mp3")
        with open(output_path, "wb") as f:
            f.write(response.content)

        # ✅ Réponse JSON avec chemin du fichier
        return jsonify({"audio_url": f"/{output_path}"})

    except Exception as e:
        logger.error(f"Erreur dans /text_to_speech: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/record_audio', methods=['POST'])
def record_audio():
    """Endpoint pour l'enregistrement audio via base64 - version OpenAI STT"""
    try:
        data = request.get_json()
        audio_data = data.get('audio_data')

        if not audio_data:
            return jsonify({'error': 'Aucune donnée audio'}), 400

        # Décoder le base64
        audio_bytes = base64.b64decode(audio_data.split(',')[1])

        # Sauvegarde de l'enregistrement
        recording_filename = f"recording_{uuid.uuid4().hex}.wav"
        recording_path = os.path.join(RECORDINGS_FOLDER, recording_filename)
        with open(recording_path, 'wb') as f:
            f.write(audio_bytes)

        # Transcription via OpenAI Whisper
        transcription = transcribe_audio_darija_latin(recording_path)

        if not transcription.strip():
            return jsonify({'error': 'Transcription vide'}), 400

        # Génération de la réponse avec le nouveau système d'embeddings
        query_result = main_system.process_query(transcription)
        response_generator = main_system.llm_responder.generate_response(
            transcription, 
            query_result.get('response', [])
        )

        response_text = ""
        for chunk in response_generator:
            if chunk.startswith("data: "):
                chunk_data = json.loads(chunk[6:])
                if 'content' in chunk_data:
                    response_text = chunk_data['content']
                    break

        return jsonify({
            'response': response_text,
            'sources': query_result.get('sources', []),
            'recording_url': f'/audio/{recording_filename}',
            'recording_filename': recording_filename,
            'transcription_method': 'openai_whisper',
            'embedding_model': 'text-embedding-3-small',
            'detected_language': 'ar',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Erreur dans /record_audio: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stream_chat', methods=['POST'])
def stream_chat():
    """Endpoint pour le streaming de réponses"""
    try:
        data = request.get_json()
        query = data.get('message', '').strip()
        
        if not query:
            return jsonify({'error': 'Message vide'}), 400
        
        def generate():
            try:
                # Recherche de contexte avec le nouveau système d'embeddings
                result = main_system.process_query(query)
                
                # Génération streaming
                for chunk in main_system.llm_responder.generate_response(query, result.get('response', [])):
                    yield chunk
                    
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(generate(), mimetype='text/plain')
        
    except Exception as e:
        logger.error(f"Erreur dans /stream_chat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup')
def cleanup():
    """Nettoie les fichiers temporaires"""
    try:
        # Nettoyage des fichiers audio anciens (> 1 heure)
        current_time = time.time()
        cleaned_files = 0
        
        for folder in [AUDIO_FOLDER, RECORDINGS_FOLDER]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):
                        file_age = current_time - os.path.getctime(file_path)
                        if file_age > 3600:  # 1 heure
                            os.remove(file_path)
                            cleaned_files += 1
                            logger.info(f"Fichier nettoyé: {filename}")
        
        return jsonify({
            'message': f'Nettoyage effectué - {cleaned_files} fichiers supprimés',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Erreur dans /cleanup: {e}")
        return jsonify({'error': str(e)}), 500


def periodic_cleanup():
    """Nettoyage périodique des fichiers temporaires"""
    while True:
        time.sleep(3600)  # Toutes les heures
        try:
            cleanup()
        except Exception as e:
            logger.error(f"Erreur nettoyage périodique: {e}")

# Template HTML amélioré avec contrôles audio - SANS AFFICHAGE TRANSCRIPTION
def create_template():
    """Crée le template HTML amélioré avec contrôles audio - SANS AFFICHAGE TRANSCRIPTION"""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    html_content = '''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pôle Digital Agriculture - Assistant IA</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; color: #2c5f2d; }
        .darija-info { background: #e8f5e8; border: 1px solid #4caf50; border-radius: 5px; padding: 10px; margin-bottom: 20px; font-size: 12px; color: #2e7d32; }
        .darija-help { background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 10px; margin-bottom: 15px; font-size: 11px; }
        .darija-examples { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
        .darija-example { background: #f8f9fa; padding: 5px; border-radius: 3px; font-family: monospace; }
        .chat-container { height: 500px; border: 1px solid #ddd; border-radius: 5px; overflow-y: auto; padding: 10px; margin-bottom: 20px; background: #f9f9f9; }
        .input-group { display: flex; gap: 10px; margin-bottom: 10px; }
        input[type="text"] { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        button { padding: 10px 15px; background: #2c5f2d; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 12px; }
        button:hover { background: #1a4a1c; }
        button:disabled { background: #6c757d; cursor: not-allowed; }
        button.small { padding: 5px 8px; font-size: 10px; margin-left: 5px; min-width: auto; }
        .audio-controls { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
        .audio-controls button { min-width: 110px; }
        .message { margin-bottom: 15px; padding: 10px; border-radius: 5px; position: relative; }
        .user-message { background: #e3f2fd; text-align: right; }
        .bot-message { background: #f1f8e9; white-space: pre-line; }
        .audio-player { margin-top: 10px; }
        .status { padding: 10px; margin-bottom: 10px; border-radius: 5px; }
        .status.success { background: #d4edda; color: #155724; }
        .status.error { background: #f8d7da; color: #721c24; }
        .status.info { background: #d1ecf1; color: #0c5460; }
        .document-info { font-size: 11px; color: #666; margin-top: 5px; }
        .message-controls { position: absolute; top: 5px; right: 5px; display: flex; gap: 3px; }
        .recording-audio { margin-top: 10px; }
        .recording-audio audio { width: 100%; max-width: 300px; }
        .progress { display: none; background: #ffc107; color: #212529; padding: 8px; border-radius: 3px; margin: 5px 0; }
        
        /* Styles pour les contrôles audio */
        .tts-controls { display: flex; gap: 3px; align-items: center; }
        .play-btn { background: #28a745; }
        .play-btn:hover { background: #218838; }
        .play-btn.playing { background: #dc3545; }
        .play-btn.playing:hover { background: #c82333; }
        .pause-btn { background: #ffc107; color: #212529; }
        .pause-btn:hover { background: #e0a800; }
        .stop-btn { background: #dc3545; }
        .stop-btn:hover { background: #c82333; }
        
        /* Animation pour indiquer la lecture */
        .playing-indicator { 
            animation: pulse 1.5s infinite; 
            background: #dc3545 !important;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
            <div class="container">
        <div class="header">
            <h1>🌾 Pôle Digital Agriculture</h1>
            <p>Assistant IA spécialisé en agriculture marocaine</p>
        </div>
        
        <div id="chat" class="chat-container">
            <div class="message bot-message">
                <div class="message-controls">
                    <div class="tts-controls">
                        <button class="small play-btn" onclick="playTTS(this, \`Salam alikom! Ana msa3dek f l9adaya dial zra3a. Kifach ymkn li nsa3dek?\`)">▶️</button> # type: ignore
                    </div>
                </div>
                <strong>Assistant:</strong> Salam alikom! Ana msa3dek f l9adaya dial zra3a. Kifach ymkn li nsa3dek?
                <div class="document-info">Assistant vocal prêt</div>
            </div>
        </div>
        
        <div class="audio-controls">
            <input type="file" id="audioFile" accept="audio/*" style="display: none;">
            <input type="file" id="documentFile" accept=".pdf,.docx,.doc,.txt,.md,.xlsx,.xls,.csv" style="display: none;">
            <button onclick="document.getElementById('documentFile').click()">📄 Upload Document</button>
            <button id="recordBtn" onclick="isPreviewMode ? startNewRecording() : toggleRecording()">🎤 Enregistrer</button>
            <button onclick="listDocuments()">📋 List Docs</button>
        </div>
        
        <div class="input-group">
            <input type="text" id="messageInput" placeholder="Tapez votre message..." onkeypress="handleKeyPress(event)">
            <button onclick="sendMessage()">Envoyer</button>
            <button onclick="toggleTTS()">🔊 TTS</button>
        </div>
        
        <div id="progress" class="progress">
            Traitement en cours...
        </div>
    </div>

    <script>
        let isRecording = false;
        let mediaRecorder;
        let audioChunks = [];
        let ttsEnabled = true;
        let messageCounter = 0;
        
        // Gestion des audios
        let currentAudio = null;
        let currentPlayButton = null;
        let allAudioElements = new Set();
        
        // Gestion de la prévisualisation
        let recordedBlob = null;
        let previewAudio = null;
        let isPreviewMode = false;

        // Vérification de l'état au chargement
        window.onload = function() {
            checkHealth();
        };

        async function checkHealth() {
            try {
                const response = await fetch('/health');
                const data = await response.json();
                
                if (response.ok) {
                    console.log('Système prêt');
                } else {
                    console.error('Erreur système:', data.message);
                }
            } catch (error) {
                console.error('Erreur de connexion:', error.message);
            }
        }

        function showProgress(message) {
            const progress = document.getElementById('progress');
            progress.style.display = 'block';
            progress.textContent = message || 'Traitement en cours...';
        }

        function hideProgress() {
            document.getElementById('progress').style.display = 'none';
        }

        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            addMessage('user', message);
            input.value = '';
            
            showProgress('⏳ Traitement...');
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                hideProgress();
                
                if (response.ok) {
                    addMessage('bot', data.response);
                    
                    // TTS automatique si activé
                    if (ttsEnabled) {
                        setTimeout(() => {
                            const lastMessage = document.querySelector('.bot-message:last-child');
                            if (lastMessage) {
                                const playBtn = lastMessage.querySelector('.play-btn');
                                if (playBtn) {
                                    playTTS(playBtn, data.response);
                                }
                            }
                        }, 500);
                    }
                } else {
                    addMessage('bot', `❌ Erreur: ${data.error}`);
                }
            } catch (error) {
                hideProgress();
                addMessage('bot', `❌ Erreur de connexion: ${error.message}`);
            }
        }

        function addMessage(sender, content, extraData = null) {
            const chat = document.getElementById('chat');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            messageDiv.id = `msg-${++messageCounter}`;
            
            let messageHTML = `<strong>${sender === 'user' ? 'Vous' : 'Assistant'}:</strong> ${content}`;
            
            // Ajouter les contrôles pour les messages bot
            if (sender === 'bot') {
                messageHTML = `
                    <div class="message-controls">
                        <div class="tts-controls">
                            <button class="small play-btn" onclick="playTTS(this, \`${content}\`)">▶️</button>
                            <button class="small pause-btn" onclick="pauseCurrentAudio()">⏸️</button>
                            <button class="small stop-btn" onclick="stopCurrentAudio()">⏹️</button>
                        </div>
                    </div>
                    ${messageHTML}
                `;
            }
            
            // Ajouter l'audio d'enregistrement si disponible
            if (extraData && extraData.recording_url) {
                messageHTML += `
                    <div class="recording-audio">
                        <p><em>Enregistrement original:</em></p>
                        <audio controls>
                            <source src="${extraData.recording_url}" type="audio/wav">
                            Votre navigateur ne supporte pas l'audio.
                        </audio>
                    </div>
                `;
            }

            // Ajouter un indicateur discret de traitement vocal si c'est de l'audio
            if (extraData && extraData.transcription_method) {
                messageHTML += `<div class="document-info">🎤 Message vocal traité</div>`;
            }
            
            messageDiv.innerHTML = messageHTML;
            chat.appendChild(messageDiv);
            chat.scrollTop = chat.scrollHeight;
        }

        // Fonction pour nettoyer le texte avant TTS
        function cleanTextForTTS(text) {
            if (!text || typeof text !== 'string') return '';
            
            var cleaned = text;
            
            // 1. Nettoyage HTML de base (comme votre version qui marche)
            cleaned = cleaned.replace(/<[^>]*>/g, '');
            
            // 2. Remplacer les entités HTML courantes (comme votre version qui marche)
            cleaned = cleaned.replace(/&lt;/g, '<');
            cleaned = cleaned.replace(/&gt;/g, '>');
            cleaned = cleaned.replace(/&amp;/g, '&');
            cleaned = cleaned.replace(/&quot;/g, '"');
            cleaned = cleaned.replace(/&#39;/g, "'");
            cleaned = cleaned.replace(/&nbsp;/g, ' ');
            
            // 3. NOUVEAU: Amélioration de la ponctuation pour TTS
            // Ajouter des pauses naturelles après la ponctuation forte
            cleaned = cleaned.replace(/([.!?])\s*/g, '$1 '); // Point, exclamation, interrogation
            cleaned = cleaned.replace(/([,:;])\s*/g, '$1 '); // Virgule, deux-points, point-virgule
            
            // NOUVEAU: Gérer les points de suspension
            cleaned = cleaned.replace(/\.{3,}/g, '... '); // Normaliser les points de suspension
            
            // NOUVEAU: Améliorer les parenthèses pour une meilleure lecture
            cleaned = cleaned.replace(/\(\s*/g, '( '); // Espace après (
            cleaned = cleaned.replace(/\s*\)/g, ' )'); // Espace avant )
            
            // NOUVEAU: Gérer les guillemets
            cleaned = cleaned.replace(/"\s*/g, '" '); // Espace après "
            cleaned = cleaned.replace(/\s*"/g, ' "'); // Espace avant "
            
            // NOUVEAU: Normaliser les tirets
            cleaned = cleaned.replace(/\s*[-–—]\s*/g, ' - '); // Normaliser tous types de tirets
            
            // 4. Supprimer les caractères problématiques (comme votre version qui marche)
            cleaned = cleaned.replace(/[<>]/g, '');
            cleaned = cleaned.replace(/[\[\]{}]/g, '');
            cleaned = cleaned.replace(/[★☆]/g, '');
            
            // 5. Supprimer les emojis communs (comme votre version qui marche)
            cleaned = cleaned.replace(/[🌾📄🎤📋🔊🔇⏳💬❌✅⚠️📊⏱️⏰🎙️📁🔤📝]/g, '');
            
            // 6. NOUVEAU: Gestion des abréviations courantes pour TTS
            const abbreviations = {
                'M.': 'Monsieur',
                'Mme.': 'Madame', 
                'Dr.': 'Docteur',
                'etc.': 'etcetera',
                'vs.': 'contre'
            };
            
            for (let abbr in abbreviations) {
                // Utilisation de split/join plus sûre que regex complexes
                cleaned = cleaned.split(abbr).join(abbreviations[abbr]);
            }
            
            // 7. NOUVEAU: Gestion des nombres et unités
            cleaned = cleaned.replace(/(\d+)\s*%/g, '$1 pour cent');
            cleaned = cleaned.replace(/(\d+)\s*€/g, '$1 euros');
            cleaned = cleaned.replace(/(\d+)\s*\$/g, '$1 dollars');
            cleaned = cleaned.replace(/(\d+)\s*h\s*(\d+)/g, '$1 heure $2');
            cleaned = cleaned.replace(/(\d+)\s*:\s*(\d+)/g, '$1 heure $2');
            
            // 8. Nettoyer les espaces multiples (comme votre version qui marche)
            cleaned = cleaned.replace(/\s+/g, ' ');
            
            // 9. Mapping darija pour TTS (comme votre version qui marche)
            const darijaMap = {
                '3': 'aa', '7': 'h', '9': 'k', 'kh': 'k', 'gh': 'g', 'w': 'ou', 'y': 'i',
                'bghit': 'beghit', 'bezaf': 'bézaf', 'kayn': 'kayin'
            };
            
            for (let darija in darijaMap) {
                cleaned = cleaned.replace(new RegExp(darija, 'gi'), darijaMap[darija]);
            }
            
            // 10. NOUVEAU: S'assurer d'une ponctuation finale pour une meilleure lecture TTS
            cleaned = cleaned.trim();
            if (cleaned && !/[.!?]$/.test(cleaned)) {
                cleaned += '.';
            }
            
            return cleaned;
        }

        async function playTTS(button, text) {
            if (!button || button.disabled) return;
            
            const cleanText = cleanTextForTTS(text);
            
            if (!cleanText || cleanText.length < 2) {
                console.warn('Texte trop court ou vide après nettoyage:', cleanText);
                return;
            }
            
            // Si c'est le même bouton et qu'un audio existe, toggle play/pause
            if (currentAudio && currentPlayButton === button) {
                if (currentAudio.paused) {
                    currentAudio.play();
                    return;
                } else {
                    currentAudio.pause();
                    return;
                }
            }
            
            // Arrêter l'audio actuel s'il y en a un autre
            stopCurrentAudio();
            
            button.disabled = true;
            button.textContent = '⏳';
            button.classList.add('playing-indicator');
            
            try {
                const response = await fetch('/text_to_speech', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: cleanText, language: 'darija_latin' })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    currentAudio = new Audio(data.audio_url);
                    currentPlayButton = button;
                    allAudioElements.add(currentAudio);
                    
                    enableAudioControls();
                    
                    currentAudio.onloadstart = () => {
                        button.disabled = false;
                        button.textContent = '▶️';
                        button.classList.remove('playing-indicator');
                    };
                    
                    currentAudio.onplay = () => {
                        button.textContent = '⏸️';
                        button.classList.add('playing');
                        enableAudioControls();
                    };
                    
                    currentAudio.onpause = () => {
                        button.textContent = '▶️';
                        button.classList.remove('playing');
                    };
                    
                    currentAudio.onended = () => {
                        resetAudioControls();
                    };
                    
                    currentAudio.onerror = () => {
                        resetAudioControls();
                        button.textContent = '❌';
                    };
                    
                    await currentAudio.play();
                    
                } else {
                    button.disabled = false;
                    button.textContent = '❌';
                    button.classList.remove('playing-indicator');
                }
            } catch (error) {
                console.error('Erreur TTS:', error);
                button.disabled = false;
                button.textContent = '❌';
                button.classList.remove('playing-indicator');
            }
        }

        function pauseCurrentAudio() {
            if (currentAudio && !currentAudio.paused) {
                currentAudio.pause();
            } else if (currentAudio && currentAudio.paused) {
                currentAudio.play();
            }
        }

        function stopCurrentAudio() {
            if (currentAudio) {
                currentAudio.pause();
                currentAudio.currentTime = 0;
                resetAudioControls();
            }
        }

        function stopAllAudio() {
            allAudioElements.forEach(audio => {
                if (audio && !audio.paused) {
                    audio.pause();
                    audio.currentTime = 0;
                }
            });
            
            stopCurrentAudio();
            
            document.querySelectorAll('.play-btn').forEach(btn => {
                btn.disabled = false;
                btn.textContent = '▶️';
                btn.classList.remove('playing', 'playing-indicator');
            });
            
            disableAudioControls();
            
            currentAudio = null;
            currentPlayButton = null;
        }

        function enableAudioControls() {
            document.querySelectorAll('.pause-btn, .stop-btn').forEach(btn => {
                btn.disabled = false;
            });
        }

        function disableAudioControls() {
            document.querySelectorAll('.pause-btn, .stop-btn').forEach(btn => {
                btn.disabled = true;
            });
        }

        function resetAudioControls() {
            if (currentPlayButton) {
                currentPlayButton.disabled = false;
                currentPlayButton.textContent = '▶️';
                currentPlayButton.classList.remove('playing', 'playing-indicator');
            }
            
            disableAudioControls();
            currentAudio = null;
            currentPlayButton = null;
        }

        async function toggleRecording() {
            const btn = document.getElementById('recordBtn');
            
            if (!isRecording && !isPreviewMode) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    
                    mediaRecorder.ondataavailable = (event) => {
                        audioChunks.push(event.data);
                    };
                    
                    mediaRecorder.onstop = () => {
                        recordedBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        mediaRecorder.stream.getTracks().forEach(track => track.stop());
                        showPreviewMode();
                    };
                    
                    mediaRecorder.start();
                    isRecording = true;
                    btn.textContent = '⏹️ Arrêter';
                    btn.style.background = '#dc3545';
                    showProgress('🎤 Enregistrement en cours... Cliquez pour arrêter');
                    
                } catch (error) {
                    alert('Erreur microphone: ' + error.message);
                }
            } else if (isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                hideProgress();
            }
        }

        function showPreviewMode() {
            isPreviewMode = true;
            const btn = document.getElementById('recordBtn');
            
            btn.textContent = '🎤 Nouvel Enregistrement';
            btn.style.background = '#6c757d';
            
            createPreviewInterface();
        }

        function createPreviewInterface() {
            const existingPreview = document.getElementById('previewInterface');
            if (existingPreview) {
                existingPreview.remove();
            }
            
            const previewDiv = document.createElement('div');
            previewDiv.id = 'previewInterface';
            previewDiv.style.cssText = `
                background: #e8f5e8;
                border: 2px solid #28a745;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                display: flex;
                flex-direction: column;
                gap: 10px;
                animation: slideIn 0.3s ease;
            `;
            
            previewDiv.innerHTML = `
                <div style="display: flex; align-items: center; gap: 10px; font-weight: bold; color: #28a745;">
                    🎧 Prévisualisation
                </div>
                
                <div style="display: flex; gap: 10px; align-items: center;">
                    <button id="previewPlayBtn" onclick="playPreview()" style="background: #28a745; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;">
                        ▶️ Écouter
                    </button>
                    <button id="previewStopBtn" onclick="stopPreview()" style="background: #ffc107; color: #212529; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;" disabled>
                        ⏹️ Arrêter
                    </button>
                    <span id="previewStatus" style="color: #6c757d; font-size: 14px;">Prêt pour traitement</span>
                </div>
                
                <div style="display: flex; gap: 10px;">
                    <button onclick="sendRecording()" style="background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; flex: 1;">
                        📤 Envoyer
                    </button>
                    <button onclick="cancelRecording()" style="background: #dc3545; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; flex: 1;">
                        🗑️ Annuler
                    </button>
                </div>
            `;
            
            const audioControls = document.querySelector('.audio-controls');
            audioControls.parentNode.insertBefore(previewDiv, audioControls.nextSibling);
            
            if (!document.getElementById('previewStyles')) {
                const style = document.createElement('style');
                style.id = 'previewStyles';
                style.textContent = `
                    @keyframes slideIn {
                        from { opacity: 0; transform: translateY(-10px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                `;
                document.head.appendChild(style);
            }
        }

        function playPreview() {
            if (!recordedBlob) return;
            
            const playBtn = document.getElementById('previewPlayBtn');
            const stopBtn = document.getElementById('previewStopBtn');
            const status = document.getElementById('previewStatus');
            
            if (previewAudio && !previewAudio.paused) {
                previewAudio.pause();
                playBtn.textContent = '▶️ Reprendre';
                stopBtn.disabled = true;
                status.textContent = 'En pause';
                return;
            }
            
            if (previewAudio && previewAudio.paused && previewAudio.currentTime > 0) {
                previewAudio.play();
                playBtn.textContent = '⏸️ Pause';
                stopBtn.disabled = false;
                status.textContent = 'Lecture en cours...';
                return;
            }
            
            const audioUrl = URL.createObjectURL(recordedBlob);
            previewAudio = new Audio(audioUrl);
            
            previewAudio.onplay = () => {
                playBtn.textContent = '⏸️ Pause';
                stopBtn.disabled = false;
                status.textContent = 'Lecture en cours...';
            };
            
            previewAudio.onpause = () => {
                playBtn.textContent = '▶️ Reprendre';
                stopBtn.disabled = true;
                status.textContent = 'En pause';
            };
            
            previewAudio.onended = () => {
                playBtn.textContent = '▶️ Réécouter';
                stopBtn.disabled = true;
                status.textContent = 'Lecture terminée';
                URL.revokeObjectURL(audioUrl);
            };
            
            previewAudio.onerror = () => {
                playBtn.textContent = '❌ Erreur';
                stopBtn.disabled = true;
                status.textContent = 'Erreur de lecture';
            };
            
            previewAudio.play();
        }

        function stopPreview() {
            if (previewAudio) {
                previewAudio.pause();
                previewAudio.currentTime = 0;
                
                const playBtn = document.getElementById('previewPlayBtn');
                const stopBtn = document.getElementById('previewStopBtn');
                const status = document.getElementById('previewStatus');
                
                playBtn.textContent = '▶️ Écouter';
                stopBtn.disabled = true;
                status.textContent = 'Arrêté';
            }
        }

        async function sendRecording() {
            if (!recordedBlob) return;
            
            showProgress('⏳ Traitement vocal en cours...');
            
            const reader = new FileReader();
            reader.onloadend = async () => {
                try {
                    const response = await fetch('/record_audio', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ audio_data: reader.result })
                    });
                    
                    const data = await response.json();
                    hideProgress();
                    
                    if (response.ok) {
                        // Ajouter le message utilisateur SANS la transcription
                        addMessage('user', '🎤 Message vocal', {
                            recording_url: data.recording_url,
                            transcription_method: data.transcription_method
                        });
                        
                        // Ajouter la réponse
                        if (data.response) {
                            addMessage('bot', data.response);
                            
                            if (ttsEnabled) {
                                setTimeout(() => {
                                    const lastMessage = document.querySelector('.bot-message:last-child');
                                    if (lastMessage) {
                                        const playBtn = lastMessage.querySelector('.play-btn');
                                        if (playBtn) {
                                            playTTS(playBtn, data.response);
                                        }
                                    }
                                }, 500);
                            }
                        }
                        
                        cancelRecording();
                        
                    } else {
                        addMessage('bot', `❌ Erreur: ${data.error}`);
                    }
                } catch (error) {
                    hideProgress();
                    addMessage('bot', `❌ Erreur: ${error.message}`);
                }
            };
            reader.readAsDataURL(recordedBlob);
        }

        function cancelRecording() {
            stopPreview();
            
            recordedBlob = null;
            previewAudio = null;
            isPreviewMode = false;
            
            const previewInterface = document.getElementById('previewInterface');
            if (previewInterface) {
                previewInterface.style.animation = 'slideOut 0.3s ease';
                setTimeout(() => {
                    previewInterface.remove();
                }, 300);
            }
            
            const btn = document.getElementById('recordBtn');
            btn.textContent = '🎤 Enregistrer';
            btn.style.background = '#2c5f2d';
            
            if (!document.getElementById('slideOutAnimation')) {
                const style = document.createElement('style');
                style.id = 'slideOutAnimation';
                style.textContent = `
                    @keyframes slideOut {
                        from { opacity: 1; transform: translateY(0); }
                        to { opacity: 0; transform: translateY(-10px); }
                    }
                `;
                document.head.appendChild(style);
            }
        }

        function startNewRecording() {
            cancelRecording();
            setTimeout(() => {
                toggleRecording();
            }, 100);
        }

        function toggleTTS() {
            ttsEnabled = !ttsEnabled;
            const btn = event.target;
            btn.textContent = ttsEnabled ? '🔊 TTS' : '🔇 TTS';
            btn.style.background = ttsEnabled ? '#2c5f2d' : '#6c757d';
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        async function cleanup() {
            try {
                const response = await fetch('/cleanup');
                const data = await response.json();
                
                if (response.ok) {
                    console.log('Nettoyage:', data.message);
                }
            } catch (error) {
                console.error('Erreur nettoyage:', error);
            }
        }

        // Upload de fichier audio
        document.getElementById('audioFile').addEventListener('change', async function(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('audio', file);
            
            showProgress('⏳ Traitement du fichier audio...');
            
            try {
                const response = await fetch('/upload_audio', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                hideProgress();
                
                if (response.ok) {
                    // Ajouter le message utilisateur SANS la transcription
                    addMessage('user', `📁 Fichier audio: ${file.name}`, {
                        transcription_method: data.transcription_method
                    });
                    
                    if (data.response) {
                        addMessage('bot', data.response);
                        
                        if (ttsEnabled) {
                            setTimeout(() => {
                                const lastMessage = document.querySelector('.bot-message:last-child');
                                if (lastMessage) {
                                    const playBtn = lastMessage.querySelector('.play-btn');
                                    if (playBtn) {
                                        playTTS(playBtn, data.response);
                                    }
                                }
                            }, 500);
                        }
                    }
                } else {
                    addMessage('bot', `❌ Erreur: ${data.error}`);
                }
            } catch (error) {
                hideProgress();
                addMessage('bot', `❌ Erreur: ${error.message}`);
            }
            
            event.target.value = '';
        });

        // Upload de document
        document.getElementById('documentFile').addEventListener('change', async function(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('document', file);
            
            showProgress(`⏳ Upload de "${file.name}" en cours...`);
            addMessage('bot', `📄 Upload du document "${file.name}" en cours...`);
            
            try {
                const response = await fetch('/upload_document', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    if (data.status === 'success') {
                        hideProgress();
                        updateLastMessage(`✅ Document "${data.original_filename}" traité et indexé avec succès! 
📊 Total: ${data.embedding_count} documents dans la base
⏱️ Temps de traitement: ${data.processing_time}
⏰ Traité le: ${new Date(data.timestamp).toLocaleString()}

Le document est maintenant disponible pour les recherches.`);
                        
                        checkHealth();
                    } else if (data.status === 'saved') {
                        updateLastMessage(`📄 Document "${data.original_filename}" sauvegardé avec succès!
⏳ Le traitement automatique est en cours...

${data.note || 'Le document sera bientôt disponible pour les recherches.'}`);
                        
                        monitorProcessingStatus(data.original_filename);
                    }
                } else {
                    hideProgress();
                    updateLastMessage(`❌ Erreur lors de l'upload: ${data.error}`);
                }
            } catch (error) {
                hideProgress();
                updateLastMessage(`❌ Erreur de connexion: ${error.message}`);
            }
            
            event.target.value = '';
        });

        async function monitorProcessingStatus(filename, maxAttempts = 30) {
            let attempts = 0;
            const checkInterval = setInterval(async () => {
                attempts++;
                
                try {
                    const response = await fetch('/check_processing_status');
                    const data = await response.json();
                    
                    if (response.ok) {
                        if (data.last_filename && data.last_filename.includes(filename.split('.')[0])) {
                            hideProgress();
                            clearInterval(checkInterval);
                            
                            updateLastMessage(`✅ Document "${filename}" traité et indexé avec succès! 
📊 Total: ${data.embedding_count} documents dans la base
⏰ Traité le: ${new Date(data.last_processed).toLocaleString()}

Le document est maintenant disponible pour les recherches.`);
                            
                            checkHealth();
                            return;
                        }
                    }
                } catch (error) {
                    console.error('Erreur surveillance:', error);
                }
                
                if (attempts >= maxAttempts) {
                    hideProgress();
                    clearInterval(checkInterval);
                    updateLastMessage(`⚠️ Le document "${filename}" a été sauvegardé mais le statut de traitement est incertain.
Utilisez le bouton "📋 List Docs" pour vérifier l'état du système.`);
                }
            }, 2000);
        }

        function updateLastMessage(content) {
            const messages = document.querySelectorAll('.bot-message');
            const lastMessage = messages[messages.length - 1];
            if (lastMessage) {
                lastMessage.innerHTML = `
                    <div class="message-controls">
                        <div class="tts-controls">
                            <button class="small play-btn" onclick="playTTS(this, \`${content}\`)">▶️</button>
                            <button class="small pause-btn" onclick="pauseCurrentAudio()">⏸️</button>
                            <button class="small stop-btn" onclick="stopCurrentAudio()">⏹️</button>
                        </div>
                    </div>
                    <strong>Assistant:</strong> ${content}
                `;
            }
        }

        async function listDocuments() {
            showProgress('📋 Chargement de la liste des documents...');
            
            try {
                const response = await fetch('/list_documents');
                const data = await response.json();
                hideProgress();
                
                if (response.ok) {
                    let docList = `📋 Documents indexés (${data.count}):\n\n`;
                    
                    if (data.documents.length === 0) {
                        docList += 'Aucun document trouvé dans le système.';
                    } else {
                        data.documents.forEach((doc, index) => {
                            const sizeKB = Math.round(doc.size / 1024);
                            const date = new Date(doc.modified).toLocaleString();
                            docList += `${index + 1}. ${doc.filename}\n`;
                            docList += `   📊 Taille: ${sizeKB} KB | 📅 Modifié: ${date}\n\n`;
                        });
                    }
                    
                    addMessage('bot', docList);
                } else {
                    addMessage('bot', `❌ Erreur: ${data.error}`);
                }
            } catch (error) {
                hideProgress();
                addMessage('bot', `❌ Erreur: ${error.message}`);
            }
        }

        // Fonction utilitaire pour détecter si l'utilisateur quitte la page
        window.addEventListener('beforeunload', function() {
            if (isRecording) {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
            
            stopAllAudio();
            
            if (previewAudio) {
                previewAudio.pause();
                previewAudio = null;
            }
        });

        // Auto-cleanup périodique côté client
        setInterval(async () => {
            try {
                await fetch('/cleanup');
            } catch (error) {
                console.log('Auto-cleanup échoué:', error);
            }
        }, 1800000); // 30 minutes
        
        // Raccourcis clavier
        document.addEventListener('keydown', function(event) {
            // Ctrl/Cmd + Space pour arrêter tous les audios
            if ((event.ctrlKey || event.metaKey) && event.code === 'Space') {
                event.preventDefault();
                stopAllAudio();
            }
            
            // Échap pour arrêter l'audio actuel
            if (event.key === 'Escape') {
                stopCurrentAudio();
            }
        });
    </script>
</body>
</html>
    '''
    
    with open(os.path.join(template_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == '__main__':
    # Créer le template HTML au démarrage
    create_template()
    
    # Génération du certificat SSL si nécessaire
    generate_ssl_cert()
    
    # Initialisation du système
    if not init_system():
        logger.error("Échec de l'initialisation du système")
        exit(1)
    
    # Démarrage du nettoyage périodique
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()
    
    # Configuration SSL
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(CERT_FILE, KEY_FILE)
    
    logger.info("🚀 Démarrage du serveur HTTPS sur https://localhost:5000")
    logger.info("📱 Interface disponible sur https://localhost:5000")
    logger.info("🔒 Certificat SSL auto-signé généré")
    logger.info("🎙️ NOUVEAU: Transcription optimisée pour Darija Latine/Français")
    logger.info("🔤 Configuration: Alphabet latin pour meilleure compatibilité LLM")
    logger.info("📄 Traitement automatique des documents uploadés")
    logger.info("🎙️ Sauvegarde des enregistrements pour réécoute")
    logger.info("🔊 Contrôles audio complets (Play/Pause/Stop)")
    logger.info("⌨️ Raccourcis: Ctrl+Space (Stop All), Échap (Stop Current)")
    logger.info("🌍 Langues supportées: Darija (latin), Français, mélange des deux")
    
    # Démarrage du serveur
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        ssl_context=context,
        threaded=True
    )