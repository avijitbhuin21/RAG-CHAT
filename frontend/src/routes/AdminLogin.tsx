import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

import { api, ApiError } from '../lib/api';
import { useSession } from '../lib/auth';

export default function AdminLogin() {
  const [username, setUsername] = useState('admin');
  const [password, setPassword] = useState('');
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const nav = useNavigate();
  const { refresh } = useSession();

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    setBusy(true);
    try {
      await api('/auth/admin/login', { method: 'POST', json: { username, password } });
      await refresh();
      nav('/admin', { replace: true });
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) setErr('Invalid credentials.');
      else setErr('Something went wrong. Try again.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-muted/40 p-6">
      <form
        onSubmit={submit}
        className="w-full max-w-sm rounded-lg border border-border bg-background p-8 shadow-sm"
      >
        <h1 className="text-2xl font-semibold tracking-tight text-foreground">Admin sign in</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Enter the admin credentials to manage the knowledge base.
        </p>

        <label className="mt-6 block text-xs font-medium text-muted-foreground">Username</label>
        <input
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          autoComplete="username"
          className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
        />

        <label className="mt-4 block text-xs font-medium text-muted-foreground">Password</label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoComplete="current-password"
          className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
        />

        {err && <p className="mt-4 text-xs text-red-600">{err}</p>}

        <button
          type="submit"
          disabled={busy}
          className="mt-6 inline-flex w-full items-center justify-center rounded-md bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground transition hover:opacity-90 disabled:opacity-50"
        >
          {busy ? 'Signing in…' : 'Sign in'}
        </button>
      </form>
    </main>
  );
}
