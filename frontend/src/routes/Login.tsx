import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { API_BASE } from '@/lib/apiBase';
import { useSession } from '@/lib/auth';
import { InfoTip } from '@/components/ui/InfoTip';

export default function Login() {
  const { startGuest } = useSession();
  const navigate = useNavigate();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  /** Start a guest session and open the chat. */
  async function continueAsGuest() {
    setBusy(true);
    setErr(null);
    try {
      await startGuest();
      navigate('/chat', { replace: true });
    } catch {
      setErr('Could not start a guest session. Please try again.');
      setBusy(false);
    }
  }

  return (
    <main className="flex min-h-[100dvh] items-center justify-center bg-background p-6">
      <div className="panel w-full max-w-sm p-8 text-center">
        <img src="/logo-short.png" alt="" className="mx-auto h-16 w-16 object-contain" />
        <div className="mt-3 font-serif text-2xl font-semibold tracking-tight text-accent">1stAId4SME</div>
        <div className="eyebrow mt-1">AI for SMEs</div>
        <h1 className="mt-8 flex items-center justify-center gap-1.5 font-serif text-xl font-semibold text-text-100">
          Sign in
          <InfoTip side="bottom">Continue with your Google account to chat with the knowledge base.</InfoTip>
        </h1>
        <a href={`${API_BASE}/auth/google/login`} className="btn-primary mt-6 w-full py-2.5">
          Continue with Google
        </a>
        <div className="my-4 flex items-center gap-3 text-[11px] uppercase tracking-wider text-text-500">
          <span className="h-px flex-1 bg-border" />
          or
          <span className="h-px flex-1 bg-border" />
        </div>
        <button
          type="button"
          onClick={continueAsGuest}
          disabled={busy}
          className="btn-ghost w-full border border-border py-2.5 disabled:opacity-60"
        >
          {busy ? 'Starting…' : 'Continue as guest'}
        </button>
        <p className="mt-2 text-[11.5px] text-text-400">
          Guests get 50 messages per day. Guest chats aren't saved.
        </p>
        {err && <p className="mt-2 text-xs text-danger">{err}</p>}
        <div className="mt-6">
          <Link
            to="/admin/login"
            className="text-xs font-medium text-text-400 transition hover:text-accent"
          >
            Admin sign in
          </Link>
        </div>
      </div>
    </main>
  );
}
