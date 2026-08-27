# HR-Screen

Prototype universitaire de présélection de CV. Il ne doit pas être utilisé pour automatiser une décision de recrutement : **l’IA propose, l’humain dispose**.

Le projet est séparé en deux applications faciles à démarrer :

- `backend/` : API FastAPI, pipeline ML, données et modèles.
- `frontend/` : interface React.

## Démarrage du backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload
```

L'API est accessible sur `http://localhost:8000/docs`. Au premier démarrage, elle crée les modèles depuis le jeu synthétique, en réservant 20 % des données pour le test. Placez ensuite un CSV réel dans `backend/data/` : colonnes requises `resume_text`, `years_experience`, `is_selected`; `audit_group` est optionnelle mais nécessaire à l'audit.

Avant toute analyse, le RH crée ou sélectionne une offre : description, compétences obligatoires, compétences souhaitées et expérience minimale. Les CV sont ensuite comparés à **cette offre**, et non à un profil générique. Le score affiché est un score d'adéquation sur 100, composé de la couverture des compétences obligatoires (50 %), souhaitées (20 %), de l'expérience (15 %) et de la proximité textuelle avec la description (15 %) ; ce n'est pas une probabilité d'embauche.

L'interface permet aussi l'analyse par lot de 200 CV maximum (`PDF`, `DOCX`, `TXT`, `CSV` ou archive `ZIP`). Le backend extrait le texte, les années d'expérience déclarées et des compétences à l'aide de règles Regex auditables. Les routes associées sont `POST /jobs`, `GET /jobs`, `POST /predict` et `POST /predict-batch`.

## Démarrage du frontend

Dans un second terminal :

```powershell
cd frontend
npm install
npm run dev
```

Le client React s'ouvre habituellement sur `http://localhost:5173`. Les routes API principales sont `/predict`, `/evaluation`, `/fairness-audit` et `/baseline-terms`.
