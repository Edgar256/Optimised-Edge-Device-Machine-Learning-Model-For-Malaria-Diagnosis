import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { adminLogin, adminRegister, fetchAdminProfile } from "../api";
import type { AdminRegisterPayload, LoginPayload, User } from "../types";

const STORAGE_KEY = "malaria_admin_auth";

interface StoredAuth {
  token: string;
  user: User;
  remember_me: boolean;
}

interface AuthContextValue {
  token: string | null;
  user: User | null;
  loading: boolean;
  login: (payload: LoginPayload) => Promise<void>;
  register: (payload: AdminRegisterPayload) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function loadStoredAuth(): StoredAuth | null {
  const session = sessionStorage.getItem(STORAGE_KEY);
  const local = localStorage.getItem(STORAGE_KEY);
  const raw = local ?? session;
  if (!raw) return null;
  try {
    return JSON.parse(raw) as StoredAuth;
  } catch {
    return null;
  }
}

function persistAuth(data: StoredAuth): void {
  const target = data.remember_me ? localStorage : sessionStorage;
  const other = data.remember_me ? sessionStorage : localStorage;
  other.removeItem(STORAGE_KEY);
  target.setItem(STORAGE_KEY, JSON.stringify(data));
}

function clearAuth(): void {
  localStorage.removeItem(STORAGE_KEY);
  sessionStorage.removeItem(STORAGE_KEY);
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const stored = loadStoredAuth();
    if (!stored) {
      setLoading(false);
      return;
    }
    fetchAdminProfile(stored.token)
      .then((profile: User) => {
        setToken(stored.token);
        setUser(profile);
      })
      .catch(() => clearAuth())
      .finally(() => setLoading(false));
  }, []);

  const login = useCallback(async (payload: LoginPayload) => {
    const result = await adminLogin(payload);
    persistAuth({
      token: result.access_token,
      user: result.user,
      remember_me: payload.remember_me,
    });
    setToken(result.access_token);
    setUser(result.user);
  }, []);

  const register = useCallback(async (payload: AdminRegisterPayload) => {
    const result = await adminRegister(payload);
    persistAuth({
      token: result.access_token,
      user: result.user,
      remember_me: true,
    });
    setToken(result.access_token);
    setUser(result.user);
  }, []);

  const logout = useCallback(() => {
    clearAuth();
    setToken(null);
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({ token, user, loading, login, register, logout }),
    [token, user, loading, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
