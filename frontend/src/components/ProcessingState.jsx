export function ProcessingState({ fileCount }) {
  return <section className="processing-state" aria-live="polite"><div className="processing-orbit" aria-hidden="true"><i /><i /><i /></div><h1>Analyse en cours</h1><p>{fileCount} CV en cours de traitement</p></section>;
}
