import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchDashboardStats } from "../api";
import { useAuth } from "../context/AuthContext";
import type { DashboardStats } from "../types";

export function DashboardOverview() {
  const { token } = useAuth();
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!token) return;
    fetchDashboardStats(token)
      .then(setStats)
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load stats"));
  }, [token]);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Overview</h1>
          <p className="muted">District-wide malaria intake and model status</p>
        </div>
      </header>

      {error && <div className="alert alert-error">{error}</div>}

      {stats && (
        <>
          <section className="stat-grid">
            <article className="stat-card">
              <span className="stat-label">Total patient visits</span>
              <strong className="stat-value">{stats.total_patients}</strong>
            </article>
            <article className="stat-card stat-danger">
              <span className="stat-label">Malaria confirmed</span>
              <strong className="stat-value">{stats.malaria_confirmed}</strong>
            </article>
            <article className="stat-card stat-success">
              <span className="stat-label">Not malaria</span>
              <strong className="stat-value">{stats.not_malaria}</strong>
            </article>
            <article className="stat-card stat-warning">
              <span className="stat-label">Pending diagnosis</span>
              <strong className="stat-value">{stats.pending_diagnosis}</strong>
            </article>
          </section>

          <section className="panel-grid">
            <article className="panel">
              <h2>Staff accounts</h2>
              <ul className="detail-list">
                <li>
                  <span>Medical personnel</span>
                  <strong>{stats.total_medical_personnel}</strong>
                </li>
                <li>
                  <span>Administrators</span>
                  <strong>{stats.total_admins}</strong>
                </li>
              </ul>
              <Link to="/dashboard/users" className="text-link">
                Manage users →
              </Link>
            </article>
            <article className="panel">
              <h2>ML model status</h2>
              <ul className="detail-list">
                <li>
                  <span>Loaded</span>
                  <strong className={stats.model_loaded ? "badge badge-ok" : "badge badge-warn"}>
                    {stats.model_loaded ? "Yes" : "No"}
                  </strong>
                </li>
                <li>
                  <span>Deployment model</span>
                  <strong>{stats.model_name ?? "—"}</strong>
                </li>
              </ul>
            </article>
            <article className="panel">
              <h2>Quick actions</h2>
              <div className="action-links">
                <Link to="/dashboard/patients" className="btn btn-secondary">
                  View all patients
                </Link>
              </div>
            </article>
          </section>
        </>
      )}

      {!stats && !error && <p className="muted">Loading dashboard…</p>}
    </div>
  );
}
