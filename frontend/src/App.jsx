import { useState } from "react";

import About from "./pages/About.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Home from "./pages/Home.jsx";

const pages = {
  home: Home,
  dashboard: Dashboard,
  about: About
};

function App() {
  const [activePage, setActivePage] = useState("home");
  const Page = pages[activePage];

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand-button" onClick={() => setActivePage("home")}>
          Crop Disease Detection
        </button>
        <nav aria-label="Primary navigation">
          <button className={activePage === "home" ? "active" : ""} onClick={() => setActivePage("home")}>
            Scan
          </button>
          <button className={activePage === "dashboard" ? "active" : ""} onClick={() => setActivePage("dashboard")}>
            Dashboard
          </button>
          <button className={activePage === "about" ? "active" : ""} onClick={() => setActivePage("about")}>
            About
          </button>
        </nav>
      </header>
      <main>
        <Page />
      </main>
    </div>
  );
}

export default App;
