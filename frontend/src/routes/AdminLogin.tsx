import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

import { api, ApiError } from '../lib/api';
import { InfoTip } from '../components/ui/InfoTip';
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
    <main className="flex min-h-[100dvh] items-center justify-center bg-background p-6">
      <form onSubmit={submit} className="panel w-full max-w-sm p-8">
        <div className="text-center">
          <img src="/logo-short.png" alt="" className="mx-auto h-16 w-16 object-contain" />
          <div className="mt-3 font-serif text-2xl font-semibold tracking-tight text-accent">1stAId4SME</div>
          <div className="eyebrow mt-1">Knowledge base admin</div>
        </div>
        <h1 className="mt-8 flex items-center gap-1.5 font-serif text-xl font-semibold text-text-100">
          Admin sign in
          <InfoTip side="bottom">Enter the admin credentials to manage the knowledge base.</InfoTip>
        </h1>

        <label className="mt-6 block text-xs font-medium text-text-300">Username</label>
        <input
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          autoComplete="username"
          className="field mt-1.5"
        />

        <label className="mt-4 block text-xs font-medium text-text-300">Password</label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoComplete="current-password"
          className="field mt-1.5"
        />

        {err && <p className="mt-4 text-xs text-danger">{err}</p>}

        <button type="submit" disabled={busy} className="btn-primary mt-6 w-full py-2.5">
          {busy ? 'Signing in…' : 'Sign in'}
        </button>
      </form>
    </main>
  );
}
