import os
import faiss
import numpy as np
import torch
from pymongo import MongoClient
from typing import List, Dict, Optional
import logging
import re
from bson import ObjectId
from openai import OpenAI
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27017/",
        db_name: str = "embeddings_db",
        collection_name: str = "files",
        faiss_index_path: str = "faiss_index.index",
        model_name: str = "text-embedding-3-small",  # Modèle OpenAI
        similarity_threshold: float = 0.01,
        top_k: int = 15,
        use_gpu: bool = True,
        embedding_dimensions: int = 1536  # Dimensions par défaut pour text-embedding-3-small
    ):
        # Configuration du device (GPU uniquement pour FAISS)
        self.use_gpu = use_gpu
        self.device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"
        
        # Configuration du modèle OpenAI
        self.model_name = model_name
        self.embedding_dimensions = embedding_dimensions
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

        # Initialisation du client OpenAI
        self.openai_client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Vérifier que la clé API est présente
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("Clé API OpenAI manquante. Vérifiez votre fichier .env")

        # Initialisation FAISS
        self.faiss_index_path = os.path.abspath(faiss_index_path)
        self.index = self._initialize_faiss_index()

        # Connexion MongoDB
        self.client = MongoClient(mongo_uri)
        self.collection = self.client[db_name][collection_name]
        
        logger.info(f"Retriever initialisé - Modèle: {self.model_name}, Device FAISS: {self.device}")

    def _initialize_faiss_index(self) -> faiss.Index:
        """Charge ou crée l'index FAISS avec les bonnes dimensions"""
        if os.path.exists(self.faiss_index_path):
            logger.info("Chargement de l'index FAISS existant")
            index = faiss.read_index(self.faiss_index_path)
            
            # Vérification du type d'index
            if not isinstance(index, faiss.IndexFlatIP):
                raise ValueError("L'index FAISS doit être de type IndexFlatIP pour la similarité cosinus")
            
            # Vérification des dimensions
            if index.d != self.embedding_dimensions:
                logger.warning(f"Dimensions de l'index ({index.d}) différentes du modèle ({self.embedding_dimensions})")
                logger.info("Création d'un nouvel index avec les bonnes dimensions")
                index = faiss.IndexFlatIP(self.embedding_dimensions)
                faiss.write_index(index, self.faiss_index_path)
        else:
            logger.info(f"Création d'un nouvel index FAISS avec {self.embedding_dimensions} dimensions")
            index = faiss.IndexFlatIP(self.embedding_dimensions)
            faiss.write_index(index, self.faiss_index_path)
        
        # Configuration GPU pour FAISS si disponible
        if self.use_gpu and torch.cuda.is_available():
            try:
                res = faiss.StandardGpuResources() # type: ignore[attr-defined]
                index = faiss.index_cpu_to_gpu(res, 0, index) # type: ignore[attr-defined]
                logger.info("FAISS configuré pour utiliser le GPU")
            except Exception as e:
                logger.warning(f"Impossible d'utiliser le GPU pour FAISS: {str(e)}")
        
        return index

    def _preprocess_query(self, query: str) -> str:
        """Nettoyage de la requête utilisateur"""
        query = query.lower().strip()
        query = re.sub(r'[^\w\séèàçâêîôûäëïöüÿ]', '', query)
        return query

    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Génère l'embedding pour la requête avec l'API OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                input=query,
                model=self.model_name,
                dimensions=self.embedding_dimensions
            )
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            # Normalisation L2 pour la similarité cosinus
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de l'embedding: {str(e)}")
            raise e

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Recherche contextuelle avec gestion par page/slide complète"""
        query = self._preprocess_query(query)
        
        search_top_k = top_k if top_k is not None else self.top_k

        # Génération de l'embedding avec OpenAI
        query_embedding = self._generate_query_embedding(query)

        # Recherche FAISS
        similarities, indices = self.index.search(np.expand_dims(query_embedding, axis=0), search_top_k) # type: ignore

        results = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if similarity < self.similarity_threshold:
                continue

            doc = self._get_document_by_index(int(idx))
            if doc:
                result = self._format_result(doc, idx, similarity)
                # Ajouter des métadonnées supplémentaires
                result.update({
                    "file_path": doc["file_path"],
                    "processing_date": doc["processing_date"].isoformat() if doc.get("processing_date") else None,
                    "model_used": doc.get("model_name", "unknown")
                })
                results.append(result)

        return sorted(results, key=lambda x: x["score"], reverse=True)[:search_top_k]

    def _get_document_by_index(self, index: int) -> Optional[Dict]:
        """Récupère le document MongoDB avec vérification d'intégrité"""
        try:
            doc = self.collection.find_one({"embedding_ids": {"$in": [index]}})
            if not doc:
                logger.warning(f"Aucun document trouvé pour l'index {index}")
                return None

            # Vérification de la cohérence des données
            chunk_index = doc["embedding_ids"].index(index)
            if chunk_index >= len(doc["chunks"]):
                logger.error(f"Index de chunk invalide dans le document {doc['_id']}")
                return None

            return doc
        except Exception as e:
            logger.error(f"Erreur de récupération du document: {str(e)}")
            return None

    def _format_result(self, doc: Dict, index: int, similarity: float) -> Dict:
        """Formate les résultats de recherche pour une page/slide complète"""
        chunk_index = doc["embedding_ids"].index(index)
        return {
            "score": float(similarity),
            "text": doc["chunks"][chunk_index],
            "source": doc["file_name"],
            "page_number": chunk_index + 1,
            "document_id": str(doc["_id"]),
            "total_pages": doc["chunk_count"]
        }
    
    def explain_result(self, document_id: str, page_number: int) -> Dict:
        """Retourne le contenu complet d'une page avec son contexte"""
        doc = self.collection.find_one({"_id": ObjectId(document_id)})
        if not doc:
            return {}
        
        # Convertir le numéro de page en index
        page_index = page_number - 1
        
        # Vérifier que l'index est valide
        if page_index < 0 or page_index >= len(doc["chunks"]):
            return {}
            
        return {
            "page_content": doc["chunks"][page_index],
            "metadata": {
                "file_path": doc["file_path"],
                "processing_date": doc["processing_date"],
                "current_page": page_number,
                "total_pages": doc["chunk_count"],
                "file_name": doc["file_name"],
                "model_used": doc.get("model_name", "unknown"),
                "embedding_dimensions": doc.get("embedding_dimensions", "unknown")
            }
        }

    def get_model_stats(self) -> Dict:
        """Retourne les statistiques sur les modèles utilisés"""
        pipeline = [
            {
                "$group": {
                    "_id": "$model_name",
                    "count": {"$sum": 1},
                    "total_chunks": {"$sum": "$chunk_count"},
                    "avg_dimensions": {"$avg": "$embedding_dimensions"}
                }
            }
        ]
        
        stats = list(self.collection.aggregate(pipeline))
        return {
            "models_used": stats,
            "current_model": self.model_name,
            "current_dimensions": self.embedding_dimensions
        }

    def close(self):
        """Nettoyage des ressources"""
        self.client.close()
        logger.info("Connexion MongoDB fermée")

    def __repr__(self):
        return (
            f"<Retriever: "
            f"Modèle={self.model_name} "
            f"Dimensions={self.embedding_dimensions} "
            f"Index_size={self.index.ntotal} vecteurs "
            f"Device={self.device}>"
        )