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

## Démarrage du frontend

Dans un second terminal :

```powershell
cd frontend
npm install
npm run dev
```

Le client React s'ouvre habituellement sur `http://localhost:5173`. Les routes API principales sont `/predict`, `/evaluation`, `/fairness-audit` et `/baseline-terms`.
