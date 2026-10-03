import { Link } from 'react-router-dom';
import { API_BASE } from '@/lib/apiBase';
import { InfoTip } from '@/components/ui/InfoTip';

export default function Login() {
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
