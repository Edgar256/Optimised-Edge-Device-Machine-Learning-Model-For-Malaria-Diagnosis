import type {
  AdminRegisterPayload,
  DashboardStats,
  LoginPayload,
  Patient,
  TokenResponse,
  User,
} from "./types";

const API_BASE = import.meta.env.VITE_API_URL?.replace(/\/$/, "") ?? "";

function authHeaders(token: string): HeadersInit {
  return {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  };
}

async function parseError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg: string }) => d.msg).join(", ");
  } catch {
    /* ignore */
  }
  return `Request failed (${response.status})`;
}

export async function adminLogin(payload: LoginPayload): Promise<TokenResponse> {
  const response = await fetch(`${API_BASE}/v1/auth/admin/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function adminRegister(payload: AdminRegisterPayload): Promise<TokenResponse> {
  const response = await fetch(`${API_BASE}/v1/auth/admin/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function fetchAdminProfile(token: string): Promise<User> {
  const response = await fetch(`${API_BASE}/v1/auth/admin/me`, {
    headers: authHeaders(token),
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function fetchDashboardStats(token: string): Promise<DashboardStats> {
  const response = await fetch(`${API_BASE}/v1/admin/stats`, {
    headers: authHeaders(token),
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function fetchPatients(token: string): Promise<Patient[]> {
  const response = await fetch(`${API_BASE}/v1/admin/patients`, {
    headers: authHeaders(token),
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function fetchPatient(token: string, id: number): Promise<Patient> {
  const response = await fetch(`${API_BASE}/v1/admin/patients/${id}`, {
    headers: authHeaders(token),
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function fetchUsers(token: string, userType?: string): Promise<User[]> {
  const query = userType ? `?user_type=${userType}` : "";
  const response = await fetch(`${API_BASE}/v1/admin/users${query}`, {
    headers: authHeaders(token),
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

export async function fetchHealth(): Promise<{ model_loaded: boolean; model_name: string | null }> {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) throw new Error("API unreachable");
  return response.json();
}
