from Embedding import Embedding
from Retriever_s import Retriever
from LLM import LLMResponder
import os
import logging
from typing import List, Dict, Optional
from pymongo import MongoClient
import torch
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder


# Charger les variables d'environnement
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Main:
    def __init__(
        self,
        client,
        source_dir: str = "Source Files",
        chunk_size: int = 150,
        mongo_uri: str = "mongodb://localhost:27017/",
        db_name: str = "embeddings_db",
        collection_name: str = "files",
        faiss_index_path: str = "faiss_index.index",
        model_name: str = "text-embedding-3-large",  # Modèle OpenAI par défaut
        similarity_threshold: float = 0.1,
        use_gpu: bool = True,
        embedding_dimensions: int = 1536,  # Dimensions pour text-embedding-3-small
        force_reindex: bool = False
        ):
        self.client = client 
        self.context = []
        self.index = None
        self.source_dir = os.path.abspath(source_dir)
        self.faiss_index_path = os.path.abspath(faiss_index_path)
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.use_gpu = use_gpu
        self.embedding_dimensions = embedding_dimensions
        
        # Vérifier la clé API OpenAI
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("Clé API OpenAI manquante. Vérifiez votre fichier .env")

        # Initialisation des composants
        self.embedding_system = self._init_embedding_system()
        self.retriever = self._init_retriever()
        self.llm_responder = LLMResponder()
        self.force_reindex = force_reindex
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")
        logger.info(f"Système initialisé avec succès - Modèle: {self.model_name}")

    def _init_embedding_system(self) -> Embedding:
        return Embedding(
            directory=self.source_dir,
            model_name=self.model_name,
            chunk_size=150,
            faiss_index_path=self.faiss_index_path,
            force_reindex=True,  # Forcer la réindexation lors du changement de modèle
            use_gpu=self.use_gpu,
            embedding_dimensions=self.embedding_dimensions )
    
    def _init_retriever(self) -> Retriever:
        return Retriever(
            faiss_index_path=self.faiss_index_path,
            model_name=self.model_name,
            similarity_threshold=self.similarity_threshold,
            use_gpu=self.use_gpu,
            embedding_dimensions=self.embedding_dimensions
        )

    def check_system_health(self) -> Dict:
        """Vérifie l'état du système"""
        health = {
            "model_name": self.model_name,
            "embedding_dimensions": self.embedding_dimensions,
            "embedding_count": self.embedding_system.collection.count_documents({}),
            "faiss_index_size": self.embedding_system.index.ntotal,
            "source_dir": self.source_dir,
            "last_processed": self._get_last_processed_date(),
            "gpu_available": torch.cuda.is_available(),
            "gpu_used": self.use_gpu and torch.cuda.is_available(),
            "openai_api_configured": bool(os.getenv('OPENAI_API_KEY'))
        }
        
        if self.use_gpu and torch.cuda.is_available():
            health.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory": f"{torch.cuda.memory_allocated()/1024**2:.2f}MB / {torch.cuda.memory_reserved()/1024**2:.2f}MB"
            })
        
        return health

    def _get_last_processed_date(self) -> Optional[str]:
        last_doc = self.embedding_system.collection.find_one(
            {},
            sort=[("processing_date", -1)]
        )
        return last_doc.get("processing_date") if last_doc else None
    
    
    def process_query(self, query: str, top_k: int = 15) -> Dict:
        """Pipeline intelligent : détection langue, salutation, traduction, embeddings"""
        logger.info(f"[PROCESS] Nouvelle requête: '{query}'")
        
        # Détection de la langue
        detected_lang = self.llm_responder.detect_language_advanced(query)
        detected_lang = "darija_latin"
        logger.info(f"[LANGUE] Détectée : {detected_lang}")

        # Salutation ?
        if self.llm_responder.is_greeting(query):
            logger.info("[SALUTATION] Salutation détectée.")
            return {
                "status": "SALUTATION",
                "query": query,
                "context": None,
                "detected_lang": detected_lang,
                "is_greet": True,
                "sources": [],
                "debug_info": {}
            }

        # Traduction en français pour le système d'embeddings
        translated_query = self.llm_responder.translate_query_to_french(query)
        logger.info(f"[TRADUCTION] -> {translated_query}")

        # Recherche vectorielle
        try:
            search_results = self.retriever.search(translated_query, top_k)

            # Reranking si activé
            # Reranking si activé
            if hasattr(self, 'reranker') and self.reranker:
                # Préparation des paires (query, document)
                pairs = [(translated_query, doc["text"]) for doc in search_results]
                
                # Prédiction des scores
                scores = self.reranker.predict(pairs)

                # Attribution des scores aux résultats
                for i, score in enumerate(scores):
                    search_results[i]["score"] = float(score)

                # Tri par score décroissant
                search_results = sorted(search_results, key=lambda x: x["score"], reverse=True)
                
                logger.info("[RERANKING] Résultats rerankés.")


            if not search_results:
                return {
                    "status": "AUCUN_RESULTAT",
                    "message": "Aucun résultat trouvé",
                    "response": "Désolé, aucun document pertinent n'a été trouvé.",
                    "context": [],
                    "detected_lang": detected_lang,
                    "is_greet": False,
                    "sources": [],
                    "debug_info": {}
                }

            context = [res["text"] for res in search_results]

            return {
                "status": "SUCCES",
                "query": query,
                "context": context,
                "detected_lang": detected_lang,
                "is_greet": False,
                "sources": self._format_sources(search_results),
                "debug_info": {
                    "model_used": self.model_name,
                    "embedding_dimensions": self.embedding_dimensions,
                    "top_score": search_results[0]['score'],
                    "gpu_used": self.use_gpu and torch.cuda.is_available()
                }
            }

        except Exception as e:
            logger.error(f"[ERREUR] Recherche échouée: {str(e)}")
            return {
                "status": "ERREUR",
                "message": str(e),
                "response": "Une erreur est survenue pendant la recherche.",
                "context": [],
                "detected_lang": detected_lang,
                "is_greet": False,
                "sources": [],
                "debug_info": {}
            }


    def _format_sources(self, results: List[Dict]) -> List[Dict]:
        return [{
            "source": res["source"],
            "excerpt": res["text"][:200] + "...",
            "confidence": round(res["score"], 3)
        } for res in results]

    def reindex_all(self):
        logger.warning("Lancement d'un réindexage complet...")
        self.embedding_system.close()
        self.embedding_system = Embedding(
            directory=self.source_dir,
            model_name=self.model_name,
            chunk_size=150,
            faiss_index_path=self.faiss_index_path,
            force_reindex=True,
            use_gpu=self.use_gpu,
            embedding_dimensions=self.embedding_dimensions
        )
        logger.info("Réindexage terminé")

    def close(self):
        self.embedding_system.close()
        self.retriever.close()
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Système arrêté")
