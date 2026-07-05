import { type FormEvent, useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export function RegisterPage() {
  const { register, token, loading } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState({
    first_name: "",
    last_name: "",
    email: "",
    phone: "",
    job_title: "",
    health_facility_name: "",
    password: "",
    admin_registration_secret: "",
  });
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  if (!loading && token) return <Navigate to="/dashboard" replace />;

  function updateField(field: string, value: string) {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      await register({
        ...form,
        health_facility_name: form.health_facility_name || undefined,
      });
      navigate("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-card auth-card-wide">
        <div className="auth-header">
          <span className="brand-mark large">M</span>
          <h1>Register administrator</h1>
          <p className="muted">Requires the server admin registration secret</p>
        </div>
        <form onSubmit={handleSubmit} className="form-grid">
          {error && (
            <div className="alert alert-error full-width">{error}</div>
          )}
          <label>
            First name
            <input
              value={form.first_name}
              onChange={(e) => updateField("first_name", e.target.value)}
              required
            />
          </label>
          <label>
            Last name
            <input
              value={form.last_name}
              onChange={(e) => updateField("last_name", e.target.value)}
              required
            />
          </label>
          <label>
            Email
            <input
              type="email"
              value={form.email}
              onChange={(e) => updateField("email", e.target.value)}
              required
            />
          </label>
          <label>
            Phone
            <input
              value={form.phone}
              onChange={(e) => updateField("phone", e.target.value)}
              required
            />
          </label>
          <label>
            Job title
            <input
              value={form.job_title}
              onChange={(e) => updateField("job_title", e.target.value)}
              required
            />
          </label>
          <label>
            Health facility (optional)
            <input
              value={form.health_facility_name}
              onChange={(e) => updateField("health_facility_name", e.target.value)}
            />
          </label>
          <label className="full-width">
            Password
            <input
              type="password"
              minLength={8}
              value={form.password}
              onChange={(e) => updateField("password", e.target.value)}
              required
            />
          </label>
          <label className="full-width">
            Admin registration secret
            <input
              type="password"
              value={form.admin_registration_secret}
              onChange={(e) => updateField("admin_registration_secret", e.target.value)}
              required
            />
          </label>
          <button type="submit" className="btn btn-primary full-width" disabled={submitting}>
            {submitting ? "Creating account…" : "Create admin account"}
          </button>
        </form>
        <p className="auth-footer">
          Already registered? <Link to="/login">Sign in</Link>
        </p>
      </div>
    </div>
  );
}
