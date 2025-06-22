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
        model_name: str = "text-embedding-3-small",  # Modèle OpenAI par défaut
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
        logger.info(f"Système initialisé avec succès - Modèle: {self.model_name}")

    def _init_embedding_system(self) -> Embedding:
        return Embedding(
            directory=self.source_dir,
            model_name=self.model_name,
            chunk_size=150,
            faiss_index_path=self.faiss_index_path,
            force_reindex=True,  # Forcer la réindexation lors du changement de modèle
            use_gpu=self.use_gpu,
            embedding_dimensions=self.embedding_dimensions
        )
    
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
    
    def get_response_for_query(self, query: str) -> str:
        result = self.process_query(query)
        return result.get("response", "Désolé, je n'ai pas trouvé de réponse pertinente.")

    def process_query(self, query: str, top_k: int = 15) -> Dict:
        logger.info(f"Traitement de la requête: '{query}' avec modèle {self.model_name}")
        
        try:
            search_results = self.retriever.search(query, top_k)
            
            print("\n=== Résultats de la recherche ===")
            print(f"Requête: '{query}'")
            print(f"Modèle: {self.model_name}")
            print(f"Nombre de résultats trouvés: {len(search_results)}\n")
            
            for i, result in enumerate(search_results, 1):
                print(f"Résultat #{i}:")
                print(f"Score: {result['score']:.4f}")
                print(f"Source: {result['source']}")
                print(f"Page: {result['page_number']}/{result['total_pages']}")
                print(f"Extrait: {result['text'][:200]}...\n")
            
            if not search_results:
                logger.warning("Aucun résultat trouvé")
                return {
                    "status": "AUCUN_RESULTAT",
                    "message": "Aucune correspondance trouvée",
                    "response": "Désolé, aucun résultat trouvé pour votre requête."
                }

            context = [res["text"] for res in search_results]
            llm_response = self.llm_responder.generate_response(query, context)
            
            return {
                "status": "SUCCES",
                "query": query,
                "response": llm_response,
                "sources": self._format_sources(search_results),
                "debug_info": {
                    "model_used": self.model_name,
                    "embedding_dimensions": self.embedding_dimensions,
                    "candidates_processed": len(search_results),
                    "top_score": search_results[0]["score"] if search_results else 0,
                    "gpu_used": self.use_gpu and torch.cuda.is_available()
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur traitement requête: {str(e)}")
            return {
                "status": "ERREUR",
                "message": str(e),
                "response": "Une erreur est survenue lors du traitement."
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