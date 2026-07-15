function Profile({ user, sessionExpiresAt }) {
  return (
    <main className="main-content">
      <div className="page-heading"><div><p className="eyebrow">Account</p><h1>Your profile</h1><p>This identity comes from your verified Google account.</p></div></div>
      <section className="profile-card">
        {user.profile_picture ? <img src={user.profile_picture} alt={`${user.name} profile`} referrerPolicy="no-referrer" /> : <div className="profile-placeholder" aria-hidden="true">{user.name.slice(0, 1).toUpperCase()}</div>}
        <dl>
          <div><dt>Name</dt><dd>{user.name}</dd></div>
          <div><dt>Email</dt><dd>{user.email}</dd></div>
          <div><dt>Provider</dt><dd>{user.auth_provider}</dd></div>
          <div><dt>Member since</dt><dd>{new Date(user.created_at).toLocaleString()}</dd></div>
          <div><dt>Last sign-in</dt><dd>{new Date(user.last_login_at).toLocaleString()}</dd></div>
          <div><dt>Session expires</dt><dd>{new Date(sessionExpiresAt).toLocaleString()}</dd></div>
        </dl>
      </section>
    </main>
  );
}

export default Profile;
