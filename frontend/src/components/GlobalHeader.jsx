function initials(name) {
  return name.split(/\s+/).filter(Boolean).slice(0, 2).map((part) => part[0]).join("").toUpperCase();
}

function GlobalHeader({ user, title, onLogout }) {
  return (
    <header className="global-header">
      <div className="header-left">
        <div className="header-logo" aria-hidden="true">LL</div>
        <div><span className="header-title">Leaflight</span><span className="header-page-title">{title}</span></div>
      </div>
      <div className="header-right">
        <div className="user-profile" aria-label={`Signed in as ${user.name}`}>
          {user.profile_picture ? <img className="user-avatar" src={user.profile_picture} alt="" referrerPolicy="no-referrer" /> : <div className="user-avatar" aria-hidden="true">{initials(user.name)}</div>}
          <div className="user-info"><span className="user-name">{user.name}</span><span className="user-role">{user.email}</span></div>
        </div>
        <button className="logout-button" type="button" onClick={onLogout}>Sign out</button>
      </div>
    </header>
  );
}

export default GlobalHeader;
