import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { fetchPatient } from "../api";
import { useAuth } from "../context/AuthContext";
import type { Patient } from "../types";

function DetailRow({ label, value }: { label: string; value: string | number | null | undefined }) {
  return (
    <div className="detail-row">
      <span>{label}</span>
      <strong>{value ?? "—"}</strong>
    </div>
  );
}

export function PatientDetailPage() {
  const { id } = useParams();
  const { token } = useAuth();
  const [patient, setPatient] = useState<Patient | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!token || !id) return;
    fetchPatient(token, Number(id))
      .then(setPatient)
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load patient"));
  }, [token, id]);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <Link to="/dashboard/patients" className="text-link back-link">
            ← Back to patients
          </Link>
          <h1>Patient #{id}</h1>
        </div>
      </header>

      {error && <div className="alert alert-error">{error}</div>}

      {patient && (
        <div className="detail-grid">
          <section className="panel">
            <h2>Demographics</h2>
            <DetailRow label="Anonymous code" value={patient.patient_id_anonymous_code} />
            <DetailRow label="Age" value={patient.age} />
            <DetailRow label="Gender" value={patient.gender} />
            <DetailRow label="Visit date" value={patient.date_of_visit} />
            <DetailRow label="Facility" value={patient.health_facility_name} />
            <DetailRow label="Geographical zone" value={patient.geographical_zone} />
            <DetailRow label="Season" value={patient.season_of_visit} />
          </section>

          <section className="panel">
            <h2>Symptoms</h2>
            <DetailRow label="Fever" value={patient.fever} />
            <DetailRow label="Headache" value={patient.headache} />
            <DetailRow label="Chills" value={patient.chills} />
            <DetailRow label="Vomiting" value={patient.vomiting} />
            <DetailRow label="Fatigue" value={patient.fatigue} />
            <DetailRow label="Fever duration (days)" value={patient.fever_duration_days} />
            <DetailRow label="Anemia signs" value={patient.anemia_signs} />
            <DetailRow label="Other symptoms" value={patient.other_symptoms_specify} />
          </section>

          <section className="panel">
            <h2>Diagnostics & treatment</h2>
            <DetailRow label="RDT" value={patient.rdt_result} />
            <DetailRow label="Microscopy" value={patient.microscopy_result} />
            <DetailRow label="Parasite density" value={patient.parasite_density} />
            <DetailRow label="Final diagnosis" value={patient.final_confirmed_diagnosis} />
            <DetailRow label="Treatment" value={patient.treatment_given} />
            <DetailRow label="Notes" value={patient.additional_clinical_notes} />
          </section>

          <section className="panel">
            <h2>Risk factors & location</h2>
            <DetailRow label="Recent travel" value={patient.recent_travel} />
            <DetailRow label="Exposure risk" value={patient.exposure_risk} />
            <DetailRow label="Household malaria history" value={patient.household_malaria_history} />
            <DetailRow label="Latitude" value={patient.latitude} />
            <DetailRow label="Longitude" value={patient.longitude} />
          </section>

          <section className="panel">
            <h2>Submission</h2>
            <DetailRow label="Submitted by" value={patient.created_by_name} />
            <DetailRow label="Staff email" value={patient.created_by_email} />
            <DetailRow label="Staff facility" value={patient.created_by_facility} />
            <DetailRow label="Created at" value={patient.created_at} />
          </section>
        </div>
      )}
    </div>
  );
}
