import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { BatchAnalysis } from "./components/BatchAnalysis";
import { BatchResults } from "./components/BatchResults";
import { JobSidebar } from "./components/JobSidebar";
import { ProcessingState } from "./components/ProcessingState";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const splitSkills = (value) =>
  value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
const initialJob = {
  title: "Data Scientist",
  description: "Analyser des données avec Python et SQL.",
  required_skills: "Python, SQL",
  preferred_skills: "Machine learning",
  min_years_experience: 2,
};

function App() {
  const [jobs, setJobs] = useState([]);
  const [jobId, setJobId] = useState("");
  const [job, setJob] = useState(initialJob);
  const [batchFiles, setBatchFiles] = useState([]);
  const [batchResult, setBatchResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [page, setPage] = useState("analysis");
  useEffect(() => {
    fetch(`${API_URL}/jobs`)
      .then((response) => (response.ok ? response.json() : Promise.reject()))
      .then(setJobs)
      .catch(() => setError("Impossible de joindre l'API."));
  }, []);
  async function createJob(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_URL}/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...job,
          required_skills: splitSkills(job.required_skills),
          preferred_skills: splitSkills(job.preferred_skills),
          min_years_experience: Number(job.min_years_experience),
        }),
      });
      if (!response.ok) throw new Error("L'offre n'a pas pu être créée.");
      const created = await response.json();
      setJobs((currentJobs) => [...currentJobs, created]);
      setJobId(created.id);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }
  async function submitBatch(event) {
    event.preventDefault();
    if (!jobId)
      return setError("Sélectionnez ou créez une offre avant l'analyse.");
    if (!batchFiles.length) return;
    setLoading(true);
    setIsAnalyzing(true);
    setError("");
    try {
      const form = new FormData();
      form.append("job_id", jobId);
      batchFiles.forEach((file) => form.append("files", file));
      const response = await fetch(`${API_URL}/predict-batch`, {
        method: "POST",
        body: form,
      });
      if (!response.ok)
        throw new Error(
          (await response.json()).detail || "Le traitement par lot a échoué.",
        );
      setBatchResult(await response.json());
      setPage("results");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setIsAnalyzing(false);
    }
  }
  return (
    <div className="app-shell">
      {page === "analysis" ? (
        <div className="workspace">
          <JobSidebar
            jobs={jobs}
            job={job}
            jobId={jobId}
            loading={loading}
            onJobChange={setJob}
            onJobSelect={setJobId}
            onSubmit={createJob}
          />
          <main className="main-content">
          {error && (
            <p className="alert alert-error" role="alert">
              {error}
            </p>
          )}
            {isAnalyzing ? (
              <ProcessingState fileCount={batchFiles.length} />
            ) : (
              <BatchAnalysis
                files={batchFiles}
                hasJob={Boolean(jobId)}
                onFilesChange={setBatchFiles}
                onSubmit={submitBatch}
              />
            )}
          </main>
        </div>
      ) : (
        <main className="results-page">
          <button className="back-button" onClick={() => setPage("analysis")}>← Nouvelle analyse</button>
          <BatchResults result={batchResult} />
        </main>
      )}
    </div>
  );
}
createRoot(document.getElementById("root")).render(<App />);
