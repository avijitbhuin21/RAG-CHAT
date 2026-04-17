import { Link } from 'react-router-dom';

export default function Login() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-muted/40 p-6">
      <div className="w-full max-w-sm rounded-lg border border-border bg-background p-8 shadow-sm">
        <h1 className="text-2xl font-semibold tracking-tight text-foreground">Sign in</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Continue with your Google account to chat with the knowledge base.
        </p>
        <a
          href="/api/auth/google/login"
          className="mt-6 inline-flex w-full items-center justify-center rounded-md bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground transition hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2"
        >
          Continue with Google
        </a>
        <div className="mt-6 text-center">
          <Link
            to="/admin/login"
            className="text-xs font-medium text-muted-foreground transition hover:text-foreground"
          >
            Admin sign in
          </Link>
        </div>
      </div>
    </main>
  );
}
