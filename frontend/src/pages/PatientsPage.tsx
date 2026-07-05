import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchPatients } from "../api";
import { useAuth } from "../context/AuthContext";
import type { Patient } from "../types";

function diagnosisBadge(value: string | null) {
  if (!value) return <span className="badge badge-muted">Pending</span>;
  if (value.toLowerCase().includes("not")) {
    return <span className="badge badge-ok">Not malaria</span>;
  }
  return <span className="badge badge-danger">Malaria</span>;
}

export function PatientsPage() {
  const { token } = useAuth();
  const [patients, setPatients] = useState<Patient[]>([]);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");

  useEffect(() => {
    if (!token) return;
    fetchPatients(token)
      .then(setPatients)
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load patients"));
  }, [token]);

  const filtered = patients.filter((p) => {
    const haystack = [
      p.patient_id_anonymous_code,
      p.health_facility_name,
      p.geographical_zone,
      p.created_by_name,
      p.final_confirmed_diagnosis,
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return haystack.includes(query.toLowerCase());
  });

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Patient records</h1>
          <p className="muted">All visits submitted via the mobile app</p>
        </div>
        <input
          className="search-input"
          placeholder="Search by ID, facility, zone…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </header>

      {error && <div className="alert alert-error">{error}</div>}

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Patient code</th>
              <th>Visit date</th>
              <th>Age / Gender</th>
              <th>Facility</th>
              <th>Diagnosis</th>
              <th>Submitted by</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((patient) => (
              <tr key={patient.id}>
                <td>{patient.id}</td>
                <td>{patient.patient_id_anonymous_code ?? "—"}</td>
                <td>{patient.date_of_visit ?? "—"}</td>
                <td>
                  {patient.age ?? "—"} / {patient.gender ?? "—"}
                </td>
                <td>{patient.health_facility_name ?? "—"}</td>
                <td>{diagnosisBadge(patient.final_confirmed_diagnosis)}</td>
                <td>{patient.created_by_name ?? "—"}</td>
                <td>
                  <Link to={`/dashboard/patients/${patient.id}`} className="text-link">
                    View
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {filtered.length === 0 && !error && (
          <p className="empty-state">No patient records found.</p>
        )}
      </div>
    </div>
  );
}
