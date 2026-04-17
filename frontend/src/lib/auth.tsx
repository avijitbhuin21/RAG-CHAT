import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import { Navigate, useLocation } from 'react-router-dom';

import { ApiError, api } from './api';

export type UserSession = {
  user_id: string;
  email: string;
  name: string | null;
  picture_url: string | null;
};

export type AdminSession = { type: 'admin' };

// A single browser can hold a user session AND an admin session at the same
// time (each in its own cookie). `null` in either slot means not signed in
// for that role.
export type Session = {
  user: UserSession | null;
  admin: AdminSession | null;
};

const EMPTY_SESSION: Session = { user: null, admin: null };

type Ctx = {
  session: Session;
  loading: boolean;
  refresh: () => Promise<void>;
  logoutUser: () => Promise<void>;
  logoutAdmin: () => Promise<void>;
  logoutAll: () => Promise<void>;
};

const SessionContext = createContext<Ctx | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session>(EMPTY_SESSION);
  const [loading, setLoading] = useState(true);

  const refresh = async () => {
    try {
      const data = await api<Session>('/auth/me');
      setSession({ user: data.user ?? null, admin: data.admin ?? null });
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) setSession(EMPTY_SESSION);
      else throw e;
    } finally {
      setLoading(false);
    }
  };

  const logoutUser = async () => {
    await api('/auth/logout?which=user', { method: 'POST' });
    setSession((s) => ({ ...s, user: null }));
  };
  const logoutAdmin = async () => {
    await api('/auth/logout?which=admin', { method: 'POST' });
    setSession((s) => ({ ...s, admin: null }));
  };
  const logoutAll = async () => {
    await api('/auth/logout', { method: 'POST' });
    setSession(EMPTY_SESSION);
  };

  useEffect(() => {
    refresh();
  }, []);

  return (
    <SessionContext.Provider
      value={{ session, loading, refresh, logoutUser, logoutAdmin, logoutAll }}
    >
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error('useSession must be used inside SessionProvider');
  return ctx;
}

export function RequireUser({ children }: { children: ReactNode }) {
  const { session, loading } = useSession();
  const loc = useLocation();
  if (loading) return <FullScreenLoader />;
  if (!session.user) return <Navigate to="/login" state={{ from: loc }} replace />;
  return <>{children}</>;
}

export function RequireAdmin({ children }: { children: ReactNode }) {
  const { session, loading } = useSession();
  const loc = useLocation();
  if (loading) return <FullScreenLoader />;
  if (!session.admin) return <Navigate to="/admin/login" state={{ from: loc }} replace />;
  return <>{children}</>;
}

function FullScreenLoader() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary/30 border-t-primary" />
    </div>
  );
}
