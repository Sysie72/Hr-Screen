# Backend HR-Screen

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload
```

Lancer ces commandes depuis ce dossier. Les données sont dans `data/`, les modèles sauvegardés dans `models/` et le code métier dans `src/`.

`POST /predict-batch` accepte jusqu'à 200 CV (`PDF`, `DOCX`, `TXT`, `CSV`, `ZIP`). Le code d'extraction est dans `src/resume_extraction.py`; il utilise exclusivement des règles explicites pour les compétences et l'expérience déclarée, sans inférer d'attributs sensibles.
