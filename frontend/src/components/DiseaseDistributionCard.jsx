import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

const COLORS = ["#166534", "#0f766e", "#1d4ed8", "#7c3aed", "#b45309", "#be123c"];
const formatClass = (value) => value.replaceAll("___", " · ").replaceAll("__", " ").replaceAll("_", " ");

function DiseaseDistributionCard({ distribution }) {
  if (distribution.length === 0) {
    return <article className="card distribution-card empty-card"><h2 className="card-title">Disease distribution</h2><p>No disease-classified scans are available.</p></article>;
  }
  const chartData = distribution.map((item) => ({ ...item, name: formatClass(item.class_name), value: item.count }));
  return (
    <article className="card distribution-card">
      <h2 className="card-title">Disease distribution</h2>
      <div className="pie-container" aria-label="Disease scan counts">
        <ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={chartData} dataKey="value" nameKey="name" innerRadius={45} outerRadius={76} stroke="none">{chartData.map((item, index) => <Cell key={item.class_name} fill={COLORS[index % COLORS.length]} />)}</Pie><Tooltip formatter={(value, _name, item) => [`${value} scans (${item.payload.percentage.toFixed(1)}%)`, item.payload.name]} /></PieChart></ResponsiveContainer>
      </div>
      <ul className="distribution-legend">{chartData.map((item, index) => <li className="legend-item" key={item.class_name}><span className="legend-dot" style={{ background: COLORS[index % COLORS.length] }}></span><span>{item.name}: {item.count} ({item.percentage.toFixed(1)}%)</span></li>)}</ul>
    </article>
  );
}

export default DiseaseDistributionCard;
