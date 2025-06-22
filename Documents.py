import os
import fitz  # Pour les PDF
import hashlib
from pptx import Presentation  # Pour les PPTX
from docx import Document as DocxDocument  # Pour les DOCX
from PIL import Image  # Pour les images
import pytesseract

class Documents:
    def __init__(self, directory="Source Files", hash_file="directory_hash.txt"):
        self.directory = directory
        self.hash_file = hash_file
        os.makedirs(self.directory, exist_ok=True)

    def _compute_directory_hash(self):
        """Compute a hash based on file names, sizes, and modification times in the directory."""
        hash_md5 = hashlib.md5()
        files_list = sorted([x for x in os.listdir(self.directory) if x.endswith(('.pdf', '.pptx', '.docx', '.png', '.jpg', '.jpeg'))])

        for file in files_list:
            file_path = os.path.join(self.directory, file)
            file_stat = os.stat(file_path)
            # Update hash with file name, size, and modification time
            hash_md5.update(file.encode('utf-8'))
            hash_md5.update(str(file_stat.st_size).encode('utf-8'))
            hash_md5.update(str(file_stat.st_mtime).encode('utf-8'))
            
        return hash_md5.hexdigest()

    def _load_previous_hash(self):
        """Load the previously saved hash from file, if it exists."""
        try:
            if os.path.exists(self.hash_file):
                with open(self.hash_file, "r") as f:
                    return f.read().strip()
        except IOError as e:
            print(f"Error reading hash file: {e}")
        return None

    def _save_current_hash(self, current_hash):
        """Save the current hash to the hash file."""
        with open(self.hash_file, "w") as f:
            f.write(current_hash)

    def detect_change(self):
        """Determine if there has been a change in the directory."""
        current_hash = self._compute_directory_hash()
        previous_hash = self._load_previous_hash()

        if current_hash != previous_hash:
            self._save_current_hash(current_hash)
            return True
        return False
    
    def load_magic(self):
        print("Chargement des fichiers...")
        data = []  # Liste pour stocker les données des fichiers

       
        files_list = [x for x in os.listdir(self.directory) 
                    if x.endswith(('.pdf', '.pptx', '.docx', '.png', '.jpg', '.jpeg', *audio_extensions, *video_extensions))]

        if not files_list:
            print("❌ Aucun fichier trouvé dans le répertoire.")
            return data

        # Charger chaque fichier
        for file in files_list:
            file_path = os.path.join(self.directory, file)
            try:
                if file.endswith('.pdf'):
                    doc = fitz.open(file_path)
                    data.append({
                        "file_name": file,
                        "file_path": file_path,
                        "type": "pdf",
                        "content": doc
                    })
                elif file.endswith('.pptx'):
                    ppt = Presentation(file_path)
                    data.append({
                        "file_name": file,
                        "file_path": file_path,
                        "type": "pptx",
                        "content": ppt
                    })
                elif file.endswith('.docx'):
                    docx = DocxDocument(file_path)
                    data.append({
                        "file_name": file,
                        "file_path": file_path,
                        "type": "docx",
                        "content": docx
                    })
                elif file.lower().endswith(('.png', '.jpg', '.jpeg','.bmp','.gif')):
                    img = Image.open(file_path)
                    text = pytesseract.image_to_string(img)
                    data.append({
                        "file_name": file,
                        "file_path": file_path,
                        "type": "image",
                        "content": text
                    })
                # Cas audio - Chargement brut sans traitement
                elif file.lower().endswith(audio_extensions):
                    data.append({
                        "file_name": file,
                        "file_path": file_path,
                        "type": "audio",
                        "content": None  # Ou le contenu brut si nécessaire
                    })
                # Cas vidéo - Chargement brut sans traitement
                elif file.lower().endswith(video_extensions):
                    data.append({
                        "file_name": file,
                        "file_path": file_path,
                        "type": "video",
                        "content": None  # Ou le contenu brut si nécessaire
                    })
                print(f"✅ Fichier chargé : {file}")
            except Exception as e:
                print(f"❌ Erreur lors du chargement du fichier {file}: {e}")

        return data