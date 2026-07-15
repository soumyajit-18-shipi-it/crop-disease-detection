import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { getHistory } from "../services/api.js";

function formatName(name = "") {
  return name.replaceAll("_", " ");
}

function cropFromClass(name = "") {
  return name.split("_")[0] || "Unknown";
}

function barColor(name = "") {
  const lower = name.toLowerCase();
  if (lower.includes("healthy")) return "var(--leaf)";
  if (lower.includes("late") || lower.includes("severe")) return "var(--rust)";
  return "var(--gold)";
}

function Dashboard() {
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    getHistory(100)
      .then(setHistory)
      .catch(() => setError("Scan history is unavailable. Check that the backend is running."))
      .finally(() => setIsLoading(false));
  }, []);

  const chartData = useMemo(() => {
    const counts = history.reduce((acc, item) => {
      acc[item.predicted_class] = (acc[item.predicted_class] || 0) + 1;
      return acc;
    }, {});
    return Object.entries(counts).map(([name, count]) => ({ name, label: formatName(name), count }));
  }, [history]);

  const stats = useMemo(() => {
    const total = history.length;
    const avg = total ? history.reduce((sum, item) => sum + item.confidence, 0) / total : 0;
    const mostCommon = chartData.slice().sort((a, b) => b.count - a.count)[0]?.label || "None";
    const healthy = history.filter((item) => item.predicted_class.toLowerCase().includes("healthy")).length;
    const ratio = total ? `${healthy}/${total - healthy}` : "0/0";
    return { total, avg, mostCommon, ratio };
  }, [history, chartData]);

  if (isLoading) return <section className="text-panel skeleton dashboard-loading" />;
  if (error) return <section className="error-banner">{error}</section>;

  return (
    <section className="dashboard-layout">
      <div className="section-heading">
        <p className="eyebrow"><span></span>Scan history</p>
        <h1>Review what the backend has recorded.</h1>
        <p>Every card below is loaded from the API history table.</p>
      </div>

      <div className="stats-row">
        <article><strong>{stats.total}</strong><span>Total scans</span></article>
        <article><strong>{stats.mostCommon}</strong><span>Most common</span></article>
        <article><strong>{Math.round(stats.avg * 100)}%</strong><span>Avg confidence</span></article>
        <article><strong>{stats.ratio}</strong><span>Healthy / diseased</span></article>
      </div>

      {history.length === 0 ? (
        <div className="text-panel designed-empty">
          <p className="eyebrow"><span></span>Empty field log</p>
          <h2>No scans yet.</h2>
          <p>Upload a leaf image from the scanner page and the backend will record it here.</p>
          <span className="empty-leaf">LL</span>
        </div>
      ) : (
        <>
          <div className="chart-panel">
            <h2>Disease frequency</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(20,32,22,0.12)" />
                <XAxis dataKey="label" tick={{ fontSize: 11, fill: "#7C7256" }} interval={0} angle={-24} textAnchor="end" height={90} />
                <YAxis allowDecimals={false} tick={{ fill: "#7C7256" }} />
                <Tooltip cursor={{ fill: "rgba(79,139,92,0.08)" }} />
                <Bar dataKey="count" radius={[8, 8, 0, 0]}>
                  {chartData.map((entry) => <Cell key={entry.name} fill={barColor(entry.name)} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="history-cards">
            {history.map((item) => (
              <article className="disease-card" key={item.id}>
                <div className="history-thumb" aria-hidden="true">{cropFromClass(item.predicted_class).slice(0, 2).toUpperCase()}</div>
                <div>
                  <span className="chip">{cropFromClass(item.predicted_class)}</span>
                  <h3>{formatName(item.predicted_class)}</h3>
                  <time>{new Date(item.timestamp).toLocaleString()}</time>
                </div>
                <strong className="history-confidence">{Math.round(item.confidence * 100)}%</strong>
              </article>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

export default Dashboard;
