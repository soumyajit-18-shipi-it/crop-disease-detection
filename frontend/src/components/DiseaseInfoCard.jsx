function DiseaseInfoCard({ symptoms, treatment }) {
  return (
    <div className="info-grid">
      <article>
        <h3>Symptoms</h3>
        <p>{symptoms}</p>
      </article>
      <article>
        <h3>Recommended Treatment</h3>
        <p>{treatment}</p>
      </article>
    </div>
  );
}

export default DiseaseInfoCard;
