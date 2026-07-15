const navItems = [
  { id: "dashboard", label: "Dashboard" },
  { id: "scan", label: "New scan" },
  { id: "history", label: "Scan history" },
  { id: "profile", label: "Profile" },
];

function Sidebar({ activePage, onNavigate }) {
  return (
    <nav className="sidebar" aria-label="Main navigation">
      <span className="sidebar-label">Workspace</span>
      {navItems.map(({ id, label }) => (
        <button key={id} type="button" className={`nav-item${activePage === id ? " active" : ""}`} onClick={() => onNavigate(id)} aria-current={activePage === id ? "page" : undefined}>
          <span className="nav-dot" aria-hidden="true"></span><span>{label}</span>
        </button>
      ))}
    </nav>
  );
}

export default Sidebar;
