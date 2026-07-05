import { useAuth } from "../context/AuthContext";

export function ProfilePage() {
  const { user } = useAuth();
  if (!user) return null;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Profile</h1>
          <p className="muted">Your administrator account</p>
        </div>
      </header>

      <section className="panel profile-panel">
        <div className="profile-avatar">{user.first_name[0]}{user.last_name[0]}</div>
        <div className="profile-details">
          <h2>
            {user.first_name} {user.last_name}
          </h2>
          <p className="muted">{user.job_title}</p>
          <ul className="detail-list">
            <li>
              <span>Email</span>
              <strong>{user.email}</strong>
            </li>
            <li>
              <span>Phone</span>
              <strong>{user.phone}</strong>
            </li>
            <li>
              <span>Health facility</span>
              <strong>{user.health_facility_name ?? "—"}</strong>
            </li>
            <li>
              <span>Role</span>
              <strong>{user.user_type}</strong>
            </li>
            <li>
              <span>Member since</span>
              <strong>{new Date(user.created_at).toLocaleString()}</strong>
            </li>
          </ul>
        </div>
      </section>
    </div>
  );
}
