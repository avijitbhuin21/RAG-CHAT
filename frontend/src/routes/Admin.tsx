import { useEffect, useRef, useState } from 'react';

import { api } from '../lib/api';
import { useSession } from '../lib/auth';
import { uploadWithProgress } from '../lib/stream';

type FileRow = {
  id: string;
  filename: string;
  content_hash: string;
  size_bytes: number;
  mime_type: string;
  status: string;
  stage_current: number;
  stage_total: number;
  error_message: string | null;
  created_at: string;
  updated_at: string;
};

type ActiveUpload = {
  key: string;
  filename: string;
  phase: 'uploading' | 'done' | 'error' | 'duplicate' | 'conflict';
  loaded: number;
  total: number;
  message?: string;
  existingFileId?: string;
};

type ReplacePrompt = {
  existingFileId: string;
  existingStatus: string;
  file: File;
};

function humanSize(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

const STATUS_LABEL: Record<string, string> = {
  pending_ingest: 'Queued',
  parsing: 'Parsing',
  chunking: 'Chunking',
  embedding: 'Embedding',
  ready: 'Ready',
  failed: 'Failed',
  delete_failed: 'Delete failed',
};

const STATUS_TONE: Record<string, string> = {
  ready: 'bg-accent text-accent-foreground',
  failed: 'bg-red-100 text-red-700',
  delete_failed: 'bg-red-100 text-red-700',
  pending_ingest: 'bg-muted text-muted-foreground',
  parsing: 'bg-primary/10 text-primary',
  chunking: 'bg-primary/10 text-primary',
  embedding: 'bg-primary/10 text-primary',
};

type Toast = { id: number; tone: 'error' | 'info'; message: string };

export default function Admin() {
  const { logoutAdmin } = useSession();
  const [files, setFiles] = useState<FileRow[]>([]);
  const [uploads, setUploads] = useState<ActiveUpload[]>([]);
  const [replacePrompt, setReplacePrompt] = useState<ReplacePrompt | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const fileInput = useRef<HTMLInputElement>(null);

  function pushToast(tone: Toast['tone'], message: string) {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, tone, message }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 6000);
  }

  async function refresh() {
    const rows = await api<FileRow[]>('/admin/files');
    setFiles(rows);
  }

  useEffect(() => {
    refresh();
    const es = new EventSource('/api/admin/files/events', { withCredentials: true });
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        setFiles((prev) =>
          prev.map((f) =>
            f.id === data.file_id
              ? {
                  ...f,
                  status: data.status ?? f.status,
                  stage_current: data.stage_current ?? f.stage_current,
                  stage_total: data.stage_total ?? f.stage_total,
                  error_message: data.error ?? f.error_message,
                }
              : f,
          ),
        );
        if (data.status === 'delete_failed') {
          pushToast(
            'error',
            `Failed to delete file — ${data.error ?? 'unknown stage'}. It's back in the list.`,
          );
          refresh();
          return;
        }
        if (
          data.status === 'ready' ||
          data.status === 'pending_ingest' ||
          data.status === 'deleted'
        ) {
          refresh();
        }
      } catch {
        /* ignore */
      }
    };
    return () => es.close();
  }, []);

  async function handleFiles(list: FileList | null) {
    if (!list || list.length === 0) return;
    const items = Array.from(list);
    const active: ActiveUpload[] = items.map((f) => ({
      key: `${f.name}-${f.size}-${Date.now()}-${Math.random()}`,
      filename: f.name,
      phase: 'uploading',
      loaded: 0,
      total: f.size,
    }));
    setUploads((u) => [...active, ...u]);

    for (const item of active) {
      const file = items.find((f) => f.name === item.filename && f.size === item.total)!;
      const form = new FormData();
      form.append('files', file);
      try {
        const { status, body } = await uploadWithProgress('/admin/files', form, (p) => {
          setUploads((prev) =>
            prev.map((u) => (u.key === item.key ? { ...u, loaded: p.loaded, total: p.total } : u)),
          );
        });
        if (status !== 200) {
          setUploads((prev) =>
            prev.map((u) =>
              u.key === item.key ? { ...u, phase: 'error', message: `HTTP ${status}` } : u,
            ),
          );
          continue;
        }
        const result = (body as { results: any[] }).results?.[0];
        if (!result) continue;
        if (result.status === 'queued') {
          setUploads((prev) =>
            prev.map((u) =>
              u.key === item.key ? { ...u, phase: 'done', loaded: u.total } : u,
            ),
          );
          setTimeout(() => setUploads((prev) => prev.filter((u) => u.key !== item.key)), 1500);
        } else if (result.status === 'exact_duplicate') {
          setUploads((prev) =>
            prev.map((u) =>
              u.key === item.key
                ? {
                    ...u,
                    phase: 'duplicate',
                    message: `identical to "${result.existing_filename}"`,
                  }
                : u,
            ),
          );
        } else if (result.status === 'name_conflict') {
          setUploads((prev) =>
            prev.map((u) =>
              u.key === item.key
                ? { ...u, phase: 'conflict', existingFileId: result.existing_file_id }
                : u,
            ),
          );
          setReplacePrompt({
            existingFileId: result.existing_file_id,
            existingStatus: result.existing_status ?? 'unknown',
            file,
          });
        }
      } catch (e) {
        setUploads((prev) =>
          prev.map((u) =>
            u.key === item.key ? { ...u, phase: 'error', message: String(e) } : u,
          ),
        );
      }
    }
    refresh();
  }

  async function confirmReplace() {
    if (!replacePrompt) return;
    const form = new FormData();
    form.append('file', replacePrompt.file);
    try {
      await uploadWithProgress(`/admin/files/${replacePrompt.existingFileId}/replace`, form, () => {});
      setUploads((prev) => prev.filter((u) => u.existingFileId !== replacePrompt.existingFileId));
      setReplacePrompt(null);
      refresh();
    } catch (e) {
      alert(`Replace failed: ${e}`);
    }
  }

  async function del(id: string) {
    if (!confirm('Delete this file and all of its vectors?')) return;
    // Optimistic: remove from UI immediately. Background task handles the
    // actual Qdrant + S3 + DB work. If it fails, the SSE handler pushes a
    // toast and refresh() re-adds the row with status=delete_failed.
    setFiles((prev) => prev.filter((f) => f.id !== id));
    try {
      await api(`/admin/files/${id}`, { method: 'DELETE' });
    } catch (e) {
      pushToast('error', `Delete request failed: ${e}. Refreshing.`);
      refresh();
    }
  }

  return (
    <main className="mx-auto max-w-5xl p-4 sm:p-6 md:p-8">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-foreground sm:text-2xl">
            Admin · Knowledge base
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Upload, list, and manage the documents users can chat with.
          </p>
        </div>
        <button
          type="button"
          onClick={logoutAdmin}
          className="self-start rounded-md border border-border px-3 py-1.5 text-xs font-medium text-muted-foreground transition hover:bg-muted sm:self-auto"
        >
          Sign out
        </button>
      </header>

      <section className="mt-6">
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            handleFiles(e.dataTransfer.files);
          }}
          onClick={() => fileInput.current?.click()}
          className="cursor-pointer rounded-lg border-2 border-dashed border-border bg-background p-6 text-center transition hover:border-primary/50 hover:bg-accent/20 sm:p-10"
        >
          <p className="text-sm font-medium text-foreground">
            Drop files here or click to browse
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            PDF, DOCX, PPTX, XLSX, TXT — multi-select supported
          </p>
          <input
            ref={fileInput}
            type="file"
            multiple
            hidden
            onChange={(e) => handleFiles(e.target.files)}
          />
        </div>
      </section>

      {uploads.length > 0 && (
        <section className="mt-4 rounded-lg border border-border bg-background p-4">
          <h3 className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Uploads
          </h3>
          <ul className="mt-3 space-y-3">
            {uploads.map((u) => (
              <li key={u.key} className="text-sm">
                <div className="flex items-center justify-between">
                  <span className="truncate font-medium text-foreground">{u.filename}</span>
                  <span className="ml-3 text-xs text-muted-foreground">
                    {u.phase === 'uploading' && `${Math.round((u.loaded / u.total) * 100)}%`}
                    {u.phase === 'done' && 'Uploaded — queued for ingest'}
                    {u.phase === 'duplicate' && `Skipped — ${u.message}`}
                    {u.phase === 'conflict' && 'Name conflict — awaiting decision'}
                    {u.phase === 'error' && `Error — ${u.message}`}
                  </span>
                </div>
                <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-muted">
                  <div
                    className={`h-full transition-all ${
                      u.phase === 'error' || u.phase === 'duplicate'
                        ? 'bg-red-400'
                        : 'bg-primary'
                    }`}
                    style={{
                      width:
                        u.phase === 'done'
                          ? '100%'
                          : `${Math.min(100, (u.loaded / Math.max(1, u.total)) * 100)}%`,
                    }}
                  />
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="mt-6 rounded-lg border border-border bg-background">
        <header className="border-b border-border px-4 py-3">
          <h3 className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Files ({files.length})
          </h3>
        </header>
        {files.length === 0 ? (
          <div className="p-8 text-center text-sm text-muted-foreground">
            No files yet. Drop one above to get started.
          </div>
        ) : (
          <ul className="divide-y divide-border">
            {files.map((f) => {
              const inFlight = ['parsing', 'chunking', 'embedding'].includes(f.status);
              const pct =
                f.stage_total > 0
                  ? Math.round((f.stage_current / f.stage_total) * 100)
                  : 0;
              return (
                <li key={f.id} className="flex items-center gap-3 px-4 py-3 sm:gap-4">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                      <span className="min-w-0 truncate text-sm font-medium text-foreground">
                        {f.filename}
                      </span>
                      <span
                        className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ${
                          STATUS_TONE[f.status] ?? 'bg-muted text-muted-foreground'
                        }`}
                      >
                        {STATUS_LABEL[f.status] ?? f.status}
                      </span>
                    </div>
                    <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
                      <span>{humanSize(f.size_bytes)}</span>
                      {inFlight && f.stage_total > 0 && (
                        <span>
                          {f.stage_current}/{f.stage_total} chunks
                        </span>
                      )}
                      {f.status === 'failed' && f.error_message && (
                        <span className="break-all text-red-600">{f.error_message}</span>
                      )}
                    </div>
                    {inFlight && (
                      <div className="mt-2 h-1 overflow-hidden rounded-full bg-muted">
                        <div
                          className="h-full bg-primary transition-all"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    )}
                  </div>
                  <button
                    type="button"
                    onClick={() => del(f.id)}
                    className="shrink-0 rounded-md border border-border px-2.5 py-1 text-xs font-medium text-muted-foreground transition hover:border-red-200 hover:bg-red-50 hover:text-red-700"
                  >
                    Delete
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      <div className="pointer-events-none fixed inset-x-4 bottom-4 z-40 flex flex-col gap-2 sm:inset-x-auto sm:bottom-6 sm:right-6 sm:w-full sm:max-w-sm">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`pointer-events-auto rounded-md border px-4 py-3 text-sm shadow-sm ${
              t.tone === 'error'
                ? 'border-red-200 bg-red-50 text-red-800'
                : 'border-border bg-background text-foreground'
            }`}
          >
            {t.message}
          </div>
        ))}
      </div>

      {replacePrompt && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-sm rounded-lg border border-border bg-background p-6 shadow-lg">
            <h3 className="text-base font-semibold text-foreground">File already exists</h3>
            <p className="mt-2 text-sm text-muted-foreground">
              A file named <span className="font-medium">{replacePrompt.file.name}</span> already
              exists (status: {replacePrompt.existingStatus}). Replace it with the new version?
              The old content and all of its embeddings will be dropped.
            </p>
            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => {
                  setUploads((prev) =>
                    prev.filter((u) => u.existingFileId !== replacePrompt.existingFileId),
                  );
                  setReplacePrompt(null);
                }}
                className="rounded-md border border-border px-3 py-1.5 text-sm font-medium text-muted-foreground transition hover:bg-muted"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={confirmReplace}
                className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground transition hover:opacity-90"
              >
                Replace
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
