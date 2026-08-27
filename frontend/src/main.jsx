import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const splitSkills = value => value.split(",").map(item => item.trim()).filter(Boolean);

function App() {
  const [jobs, setJobs] = useState([]);
  const [jobId, setJobId] = useState("");
  const [job, setJob] = useState({title: "Data Scientist", description: "Analyser des données avec Python et SQL.", required_skills: "Python, SQL", preferred_skills: "Machine learning", min_years_experience: 2});
  const [resumeText, setResumeText] = useState("Développeuse Python avec 4 ans d'expérience en analyse de données, SQL, scikit-learn et déploiement d'API.");
  const [years, setYears] = useState(4);
  const [result, setResult] = useState(null);
  const [batchFiles, setBatchFiles] = useState([]);
  const [batchResult, setBatchResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function loadJobs() {
    const response = await fetch(`${API_URL}/jobs`);
    if (response.ok) setJobs(await response.json());
  }
  useEffect(() => { loadJobs().catch(() => setError("Impossible de joindre l'API.")); }, []);
  async function createJob(event) {
    event.preventDefault(); setLoading(true); setError("");
    try {
      const response = await fetch(`${API_URL}/jobs`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({...job, required_skills: splitSkills(job.required_skills), preferred_skills: splitSkills(job.preferred_skills), min_years_experience: Number(job.min_years_experience)})});
      if (!response.ok) throw new Error("L'offre n'a pas pu être créée.");
      const created = await response.json(); setJobs(old => [...old, created]); setJobId(created.id);
    } catch (err) { setError(err.message); } finally { setLoading(false); }
  }

  async function submit(event) {
    event.preventDefault(); if (!jobId) return setError("Sélectionnez ou créez une offre avant l'analyse."); setLoading(true); setError("");
    try {
      const response = await fetch(`${API_URL}/predict`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({resume_text: resumeText, years_experience: Number(years), job_id: jobId})});
      if (!response.ok) throw new Error("La prédiction n'a pas pu être obtenue.");
      setResult(await response.json());
    } catch (err) { setError(err.message); } finally { setLoading(false); }
  }
  async function submitBatch(event) {
    event.preventDefault();
    if (!jobId) return setError("Sélectionnez ou créez une offre avant l'analyse.");
    if (!batchFiles.length) return;
    setLoading(true); setError("");
    try {
      const form = new FormData();
      form.append("job_id", jobId);
      batchFiles.forEach(file => form.append("files", file));
      const response = await fetch(`${API_URL}/predict-batch`, {method: "POST", body: form});
      if (!response.ok) throw new Error((await response.json()).detail || "Le traitement par lot a échoué.");
      setBatchResult(await response.json());
    } catch (err) { setError(err.message); } finally { setLoading(false); }
  }
  return <main>
    <header><h1>HR-Screen</h1><p>Prototype universitaire de présélection explicable</p></header>
    <aside>L’IA propose, l’humain dispose. Toute recommandation doit être contrôlée par un recruteur qualifié.</aside>
    <section className="job-panel"><h2>1. Offre à analyser</h2><label>Offre active<select value={jobId} onChange={e => setJobId(e.target.value)}><option value="">— Créez ou sélectionnez une offre —</option>{jobs.map(item => <option key={item.id} value={item.id}>{item.title}</option>)}</select></label><details><summary>Créer une offre</summary><form className="job-form" onSubmit={createJob}><label>Intitulé<input value={job.title} onChange={e => setJob({...job, title: e.target.value})} required /></label><label>Description<textarea value={job.description} onChange={e => setJob({...job, description: e.target.value})} required /></label><label>Compétences obligatoires (virgules)<input value={job.required_skills} onChange={e => setJob({...job, required_skills: e.target.value})} /></label><label>Compétences souhaitées (virgules)<input value={job.preferred_skills} onChange={e => setJob({...job, preferred_skills: e.target.value})} /></label><label>Expérience minimale<input type="number" min="0" value={job.min_years_experience} onChange={e => setJob({...job, min_years_experience: e.target.value})} /></label><button disabled={loading}>Enregistrer l’offre</button></form></details></section>
    <form onSubmit={submit}><h2>2. Analyse individuelle</h2><label>Texte du CV<textarea value={resumeText} onChange={e => setResumeText(e.target.value)} required minLength="10" /></label><label>Années d’expérience<input type="number" min="0" max="70" value={years} onChange={e => setYears(e.target.value)} required /></label><button disabled={loading || !jobId}>{loading ? "Analyse…" : "Évaluer pour cette offre"}</button></form>
    <form className="batch" onSubmit={submitBatch}><h2>Traitement en masse</h2><p>Téléversez jusqu’à 200 CV au format PDF, DOCX, TXT, CSV ou une archive ZIP. Les compétences et l’expérience sont extraites automatiquement par règles Regex.</p><label>Fichiers CV<input type="file" accept=".pdf,.docx,.txt,.csv,.zip" multiple onChange={e => setBatchFiles(Array.from(e.target.files))} /></label>{batchFiles.length > 0 && <p>{batchFiles.length} fichier(s) sélectionné(s)</p>}<button disabled={loading || !batchFiles.length}>{loading ? "Extraction et analyse…" : "Analyser le lot"}</button></form>
    {error && <p className="error">{error}</p>}
    {result && result.adequation && <section><h2>Résultat pour : {result.offre.titre}</h2><article><strong>Score d’adéquation : {result.adequation.score_adequation} / 100</strong><p>{result.adequation.nature_score}</p><p>Compétences obligatoires trouvées : {result.adequation.competences_obligatoires_trouvees.join(", ") || "Aucune"}</p><p>Compétences obligatoires manquantes : {result.adequation.competences_obligatoires_manquantes.join(", ") || "Aucune"}</p><p>Expérience : {result.adequation.experience_cv} ans ; minimum de l’offre : {result.adequation.experience_minimale_offre} ans.</p><p>Proximité textuelle avec la description : {result.adequation.proximite_description} %.</p></article></section>}
    {batchResult && <section><h2>Résultats du lot : {batchResult.offre.titre}</h2><p>{batchResult.total_analyse} CV analysé(s), classés par score de correspondance à cette offre.</p><div className="table-wrap"><table><thead><tr><th>Fichier</th><th>Score</th><th>Obligatoires trouvées</th><th>Obligatoires manquantes</th><th>Expérience</th></tr></thead><tbody>{batchResult.resultats.map(item => <tr key={item.fichier}><td>{item.fichier}</td><td>{item.score_adequation} / 100</td><td>{item.competences_obligatoires_trouvees.join(", ") || "—"}</td><td>{item.competences_obligatoires_manquantes.join(", ") || "—"}</td><td>{item.experience_cv} / {item.experience_minimale_offre} ans</td></tr>)}</tbody></table></div>{batchResult.erreurs_extraction.length > 0 && <p className="error">{batchResult.erreurs_extraction.length} fichier(s) n’ont pas pu être lus.</p>}</section>}
  </main>;
}
createRoot(document.getElementById("root")).render(<App />);
