import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { getHistory } from "../services/api.js";

function Dashboard() {
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    getHistory(100)
      .then(setHistory)
      .catch((err) => setError(err.message))
      .finally(() => setIsLoading(false));
  }, []);

  const chartData = useMemo(() => {
    const counts = history.reduce((acc, item) => {
      acc[item.predicted_class] = (acc[item.predicted_class] || 0) + 1;
      return acc;
    }, {});
    return Object.entries(counts).map(([name, count]) => ({ name: name.replaceAll("_", " "), count }));
  }, [history]);

  if (isLoading) return <section className="text-panel skeleton" />;
  if (error) return <section className="error-banner">{error}</section>;

  return (
    <section className="dashboard-layout">
      <div className="text-panel">
        <h1>Scan History</h1>
        <p>Recent predictions recorded by the FastAPI service.</p>
      </div>
      <div className="chart-panel">
        <h2>Disease Frequency</h2>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} interval={0} angle={-25} textAnchor="end" height={90} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" fill="#2f7d4f" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="history-table">
        {history.map((item) => (
          <div className="history-row" key={item.id}>
            <div className="thumb-placeholder" />
            <span>{new Date(item.timestamp).toLocaleString()}</span>
            <strong>{item.predicted_class.replaceAll("_", " ")}</strong>
            <span>{Math.round(item.confidence * 100)}%</span>
          </div>
        ))}
        {history.length === 0 && <p>No scans recorded yet.</p>}
      </div>
    </section>
  );
}

export default Dashboard;
