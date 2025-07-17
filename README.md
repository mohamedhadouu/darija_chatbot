
# 🧠 Assistant Vocal Intelligent en Darija Marocaine pour l'Agriculture🌾

## 🎯 Objectif du projet

Ce projet de fin d'études a été réalisé dans le cadre du Master "Machine Learning Avancé & Applications" en partenariat avec le **Pôle Digital de l’Agriculture**. Il vise à démocratiser l’accès à l’information agricole pour les agriculteurs marocains via un **assistant vocal intelligent** capable de comprendre et répondre en **darija marocaine**, aussi bien écrite que parlée.

L’assistant repose sur une architecture **RAG** (Retrieval-Augmented Generation) combinée à un pipeline de traitement vocal complet (reconnaissance vocale + synthèse vocale), afin d’offrir une interaction **fluide**, **contextualisée** et **multilingue**.

---

## 🚀 Fonctionnalités Clés

- 🔍 **Recherche documentaire intelligente** dans des documents techniques agricoles.
- 🗣️ **Interaction vocale** bidirectionnelle (parole vers texte, texte vers parole).
- 🌐 **Multilingue** : support du darija, français, arabe.
- 📄 Téléversement de documents personnalisés (PDF, DOCX, images).
- 🤖 Génération de réponses contextuelles via GPT-4.
- 🧠 Adaptation dynamique aux dialectes marocains.
- 🔊 Synthèse vocale en darija pour les utilisateurs peu alphabétisés.

---

## 🧱 Architecture Technique

```
Utilisateur (voix ou texte)
        ↓
[Interface Web HTML/CSS/JS]
        ↓
[Backend Flask (Python)]
        ↓
[Whisper STT (si audio)]
        ↓
[Détection de la langue + Filtrage de salutation]
        ├──▶ [Réponse de salutation directe]
        ↓
[Traduction + Reformulation de la requête → français (via GPT)]
        ↓
[Vectorisation → text-embedding-3-large]
        ↓
[Recherche contextuelle (FAISS + MongoDB)]
        ↓
[Re-ranking des résultats (via GPT ou BGE-reranker)]
        ↓
[Génération de réponse avec GPT-4 (Prompt multilingue)]
        ↓
[OpenAI TTS (Synthèse vocale en darija/français)]
        ↓
[Réponse vocale jouée + Texte affiché sur l’interface]

```

---

## ⚙️ Technologies Utilisées

- **Langage** : Python, HTML, CSS, JavaScript
- **Backend** : Flask
- **NLP & IA** : OpenAI GPT-4, Whisper, FAISS, OpenAI TTS
- **Base de données** : MongoDB (NoSQL) + FAISS (index vectoriel)
- **OCR** : Tesseract
- **Interface web** : responsive et intuitive
- **Plateformes** : Kaggle, Visual Studio Code

---

## 🧪 Étapes d'Exécution

### 1. 📦 Installation des dépendances

```bash
pip install -r requirements.txt
```

### 2. 🔧 Configuration

- Créez un fichier `.env` avec vos clés API :
```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
```

- Vérifiez les chemins vers vos modèles et documents.

### 3. ▶️ Lancement de l’application

```bash
python app.py
```

- L'interface sera disponible sur [http://localhost:5000](http://localhost:5000)

---

## 📁 Organisation du Projet

```
.
├── app.py                  # Backend principal
├── static/                 # Fichiers JS, CSS, images
├── templates/              # Pages HTML
├── data/                   # Documents à indexer
├── modules/                # Transcription, vectorisation, génération
├── index/                  # Index FAISS & MongoDB
└── README.md               # Ce fichier
```

---

## 📣 Impact

Ce projet est une **initiative inclusive** qui valorise :
- le **patrimoine linguistique marocain**,
- l’**agriculture numérique**,
- et l’**accès à l’information pour les communautés rurales**.

Il constitue une **preuve de concept** pour des assistants spécialisés en darija dans d’autres domaines (santé, éducation...).

---

## 👨‍💻 Auteur

**HADOU Mohamed**  
Étudiant Master ML Avancé & Applications  
Université Sidi Mohamed Ben Abdellah, Fès  
📍 Projet réalisé au Pôle Digital de l’Agriculture  
📆 Avril – Juillet 2025  

---


## 📌 Remarques

Ce projet est une preuve de concept. L’accès à l’API OpenAI nécessite une clé personnelle ou organisationnelle. Une version open-source complète est en cours d'adaptation avec des modèles alternatifs (Gemma, Mistral...).
