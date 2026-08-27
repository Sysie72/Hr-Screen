import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const [resumeText, setResumeText] = useState("Développeuse Python avec 4 ans d'expérience en analyse de données, SQL, scikit-learn et déploiement d'API.");
  const [years, setYears] = useState(4);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event) {
    event.preventDefault(); setLoading(true); setError("");
    try {
      const response = await fetch(`${API_URL}/predict`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({resume_text: resumeText, years_experience: Number(years)})});
      if (!response.ok) throw new Error("La prédiction n'a pas pu être obtenue.");
      setResult(await response.json());
    } catch (err) { setError(err.message); } finally { setLoading(false); }
  }
  return <main>
    <header><h1>HR-Screen</h1><p>Prototype universitaire de présélection explicable</p></header>
    <aside>L’IA propose, l’humain dispose. Toute recommandation doit être contrôlée par un recruteur qualifié.</aside>
    <form onSubmit={submit}><label>Texte du CV<textarea value={resumeText} onChange={e => setResumeText(e.target.value)} required minLength="10" /></label><label>Années d’expérience<input type="number" min="0" max="70" value={years} onChange={e => setYears(e.target.value)} required /></label><button disabled={loading}>{loading ? "Analyse…" : "Analyser le CV"}</button></form>
    {error && <p className="error">{error}</p>}
    {result && <section><h2>Résultats</h2><div className="cards">{["baseline", "advanced"].map(name => <article key={name}><h3>Modèle {name === "baseline" ? "interprétable" : "avancé"}</h3><strong>{result[name].decision}</strong><p>Probabilité de sélection : {(result[name].probabilite_selection * 100).toFixed(1)} %</p></article>)}</div><h2>Explication locale</h2><p>{result.explication.methode} — score : {(result.explication.probabilite_selection * 100).toFixed(1)} %</p><ul>{result.explication.facteurs.map((factor, index) => <li key={index}>{factor.facteur} : {factor.sens} ({factor.impact})</li>)}</ul></section>}
  </main>;
}
createRoot(document.getElementById("root")).render(<App />);
