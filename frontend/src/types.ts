export type UserType = "ADMIN" | "MEDICAL_PERSONNEL";

export interface User {
  id: number;
  first_name: string;
  last_name: string;
  email: string;
  phone: string;
  job_title: string;
  health_facility_name: string | null;
  user_type: UserType;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  remember_me: boolean;
  user: User;
}

export interface DashboardStats {
  total_patients: number;
  malaria_confirmed: number;
  not_malaria: number;
  pending_diagnosis: number;
  total_medical_personnel: number;
  total_admins: number;
  model_loaded: boolean;
  model_name: string | null;
}

export interface Patient {
  id: number;
  created_by_user_id: number;
  patient_id_anonymous_code: string | null;
  health_facility_name: string | null;
  date_of_visit: string | null;
  age: number | null;
  gender: string | null;
  geographical_zone: string | null;
  fever: string | null;
  headache: string | null;
  chills: string | null;
  vomiting: string | null;
  fatigue: string | null;
  other_symptoms_specify: string | null;
  fever_duration_days: number | null;
  anemia_signs: string | null;
  season_of_visit: string | null;
  recent_travel: string | null;
  exposure_risk: string | null;
  household_malaria_history: string | null;
  rdt_result: string | null;
  microscopy_result: string | null;
  parasite_density: string | null;
  final_confirmed_diagnosis: string | null;
  treatment_given: string | null;
  additional_clinical_notes: string | null;
  latitude: number | null;
  longitude: number | null;
  created_at: string;
  updated_at: string;
  created_by_name?: string;
  created_by_email?: string;
  created_by_facility?: string | null;
}

export interface AdminRegisterPayload {
  first_name: string;
  last_name: string;
  email: string;
  phone: string;
  job_title: string;
  health_facility_name?: string;
  password: string;
  admin_registration_secret: string;
}

export interface LoginPayload {
  email: string;
  password: string;
  remember_me: boolean;
}
