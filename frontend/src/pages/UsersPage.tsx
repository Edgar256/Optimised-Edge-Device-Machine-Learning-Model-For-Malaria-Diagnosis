import { useEffect, useState } from "react";
import { fetchUsers } from "../api";
import { useAuth } from "../context/AuthContext";
import type { User } from "../types";

export function UsersPage() {
  const { token } = useAuth();
  const [users, setUsers] = useState<User[]>([]);
  const [filter, setFilter] = useState<"" | "MEDICAL_PERSONNEL" | "ADMIN">("");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!token) return;
    fetchUsers(token, filter || undefined)
      .then(setUsers)
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load users"));
  }, [token, filter]);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Users</h1>
          <p className="muted">Medical personnel and administrators</p>
        </div>
        <select value={filter} onChange={(e) => setFilter(e.target.value as typeof filter)}>
          <option value="">All roles</option>
          <option value="MEDICAL_PERSONNEL">Medical personnel</option>
          <option value="ADMIN">Administrators</option>
        </select>
      </header>

      {error && <div className="alert alert-error">{error}</div>}

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Email</th>
              <th>Phone</th>
              <th>Job title</th>
              <th>Facility</th>
              <th>Role</th>
              <th>Joined</th>
            </tr>
          </thead>
          <tbody>
            {users.map((user) => (
              <tr key={user.id}>
                <td>
                  {user.first_name} {user.last_name}
                </td>
                <td>{user.email}</td>
                <td>{user.phone}</td>
                <td>{user.job_title}</td>
                <td>{user.health_facility_name ?? "—"}</td>
                <td>
                  <span className={user.user_type === "ADMIN" ? "badge badge-admin" : "badge badge-staff"}>
                    {user.user_type === "ADMIN" ? "Admin" : "Medical"}
                  </span>
                </td>
                <td>{new Date(user.created_at).toLocaleDateString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {users.length === 0 && !error && <p className="empty-state">No users found.</p>}
      </div>
    </div>
  );
}
