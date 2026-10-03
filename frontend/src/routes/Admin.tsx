import {
  AlertCircle,
  BookOpen,
  CheckCircle2,
  FileText,
  Loader,
  LogOut,
  Play,
  Trash2,
  UploadCloud,
} from 'lucide-react';
import { useEffect, useRef, useState, type ReactNode } from 'react';

import { InfoTip } from '../components/ui/InfoTip';
import { ThemeToggle } from '../components/ui/ThemeToggle';
import { api } from '../lib/api';
import { API_BASE } from '../lib/apiBase';
import { useSession } from '../lib/auth';
import { uploadWithProgress } from '../lib/stream';

// Matches the backend ingest routing in backend/services/ingest.py — PDFs go
// via pymupdf/Docling, the rest via Docling. Anything outside this set is
// silently skipped at upload time so we don't stage files the pipeline can't
// parse.
const SUPPORTED_EXTENSIONS = new Set(['pdf', 'docx', 'pptx', 'xlsx', 'txt']);

function isSupportedFile(name: string): boolean {
  const ext = name.toLowerCase().split('.').pop();
  return !!ext && SUPPORTED_EXTENSIONS.has(ext);
}

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
  staged: 'Staged',
  queued: 'Queued',
  pending_ingest: 'Starting',
  parsing: 'Parsing',
  chunking: 'Chunking',
  embedding: 'Embedding',
  ready: 'Ready',
  failed: 'Failed',
  delete_failed: 'Delete failed',
};

const STATUS_TONE: Record<string, string> = {
  ready: 'bg-accent/10 text-accent',
  failed: 'bg-danger-soft text-danger',
  delete_failed: 'bg-danger-soft text-danger',
  staged: 'bg-warn-soft text-warn',
  queued: 'bg-bg-200 text-text-300',
  pending_ingest: 'bg-bg-200 text-text-300',
  parsing: 'bg-sand text-text-200',
  chunking: 'bg-sand text-text-200',
  embedding: 'bg-sand text-text-200',
};

const IN_FLIGHT = new Set(['queued', 'pending_ingest', 'parsing', 'chunking', 'embedding']);

const FILE_BADGE: Record<string, string> = {
  pdf: 'bg-[#C0472F]',
  docx: 'bg-[#2B579A]',
  pptx: 'bg-[#C4652A]',
  xlsx: 'bg-[#1E7145]',
  txt: 'bg-[#4A5555]',
};

/** Renders a small coloured file-type badge based on the filename extension. */
function FileBadge({ name }: { name: string }) {
  const ext = (name.toLowerCase().split('.').pop() ?? '').slice(0, 4);
  return (
    <span
      className={`inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-[9px] font-bold uppercase tracking-wide text-white ${
        FILE_BADGE[ext] ?? 'bg-text-400'
      }`}
    >
      {ext || 'file'}
    </span>
  );
}

/** Formats an ISO timestamp as a short locale date and time. */
function shortDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/** Displays one summary metric card on the admin dashboard. */
function StatCard({
  icon,
  label,
  value,
  hint,
  hintTone = 'text-text-400',
  iconTone = 'text-accent',
}: {
  icon: ReactNode;
  label: string;
  value: number;
  hint: string;
  hintTone?: string;
  iconTone?: string;
}) {
  return (
    <div className="panel flex items-center gap-4 p-5">
      <div
        className={`flex h-14 w-14 shrink-0 items-center justify-center rounded-full border border-border bg-bg-0 shadow-sm ${iconTone}`}
      >
        {icon}
      </div>
      <div className="min-w-0">
        <div className="text-sm text-text-300">{label}</div>
        <div className="font-serif text-3xl font-semibold leading-tight text-text-100">{value}</div>
        <div className={`text-xs ${hintTone}`}>{hint}</div>
      </div>
    </div>
  );
}

type Toast = { id: number; tone: 'error' | 'info'; message: string };

export default function Admin() {
  const { logoutAdmin } = useSession();
  const [files, setFiles] = useState<FileRow[]>([]);
  const [uploads, setUploads] = useState<ActiveUpload[]>([]);
  const [replacePrompt, setReplacePrompt] = useState<ReplacePrompt | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [dragOver, setDragOver] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  function pushToast(tone: Toast['tone'], message: string) {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, tone, message }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 6000);
  }

  async function refresh() {
    const rows = await api<FileRow[]>('/admin/files');
    setFiles(rows);
    setSelected((prev) => {
      const valid = new Set(rows.map((r) => r.id));
      const next = new Set<string>();
      for (const id of prev) if (valid.has(id)) next.add(id);
      return next.size === prev.size ? prev : next;
    });
  }

  useEffect(() => {
    refresh();
    const es = new EventSource(`${API_BASE}/admin/files/events`, { withCredentials: true });
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
          data.status === 'staged' ||
          data.status === 'queued' ||
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
    const all = Array.from(list);
    const items = all.filter((f) => isSupportedFile(f.name));
    const skipped = all.length - items.length;
    if (skipped > 0) {
      const names = all
        .filter((f) => !isSupportedFile(f.name))
        .map((f) => f.name)
        .join(', ');
      pushToast(
        'info',
        `Skipped ${skipped} unsupported file${skipped === 1 ? '' : 's'}: ${names}. Supported: PDF, DOCX, PPTX, XLSX, TXT.`,
      );
    }
    if (items.length === 0) return;
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
        if (result.status === 'staged' || result.status === 'queued') {
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
    setSelected((prev) => {
      if (!prev.has(id)) return prev;
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
    try {
      await api(`/admin/files/${id}`, { method: 'DELETE' });
    } catch (e) {
      pushToast('error', `Delete request failed: ${e}. Refreshing.`);
      refresh();
    }
  }

  function toggleSelect(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function toggleSelectAll() {
    setSelected((prev) => (prev.size === files.length ? new Set() : new Set(files.map((f) => f.id))));
  }

  async function startIngestion() {
    try {
      const res = await api<{ queued: number }>('/admin/files/start-ingestion', {
        method: 'POST',
      });
      pushToast('info', `Queued ${res.queued} file${res.queued === 1 ? '' : 's'} for ingestion.`);
      refresh();
    } catch (e) {
      pushToast('error', `Failed to start ingestion: ${e}`);
    }
  }

  async function delSelected() {
    if (selected.size === 0) return;
    const ids = Array.from(selected);
    if (!confirm(`Delete ${ids.length} file${ids.length === 1 ? '' : 's'} and all of their vectors?`)) return;
    setSelected(new Set());

    let failed = 0;
    for (const id of ids) {
      try {
        await api(`/admin/files/${id}`, { method: 'DELETE' });
        setFiles((prev) => prev.filter((f) => f.id !== id));
      } catch {
        failed += 1;
      }
    }

    if (failed > 0) {
      pushToast('error', `${failed} delete request${failed === 1 ? '' : 's'} failed. Refreshing.`);
      refresh();
    }
  }

  const stagedCount = files.filter((f) => f.status === 'staged').length;
  const readyCount = files.filter((f) => f.status === 'ready').length;
  const processingCount = files.filter((f) => IN_FLIGHT.has(f.status)).length;
  const failedCount = files.filter((f) => f.status === 'failed' || f.status === 'delete_failed').length;
  const pctOf = (n: number) => (files.length ? `${((n / files.length) * 100).toFixed(1)}% of total` : '—');

  return (
    <div className="flex min-h-[100dvh] gap-3 bg-background p-3">
      <aside className="panel sticky top-3 hidden h-[calc(100dvh-1.5rem)] w-64 shrink-0 flex-col lg:flex">
        <div className="relative flex flex-col items-center px-6 pb-6 pt-8 text-center">
          <ThemeToggle className="absolute right-3 top-3" />
          <img src="/logo-short.png" alt="" className="h-20 w-20 object-contain" />
          <div className="mt-3 font-serif text-2xl font-semibold tracking-tight text-accent">1stAId4SME</div>
          <div className="eyebrow mt-1">Knowledge base admin</div>
        </div>
        <nav className="px-3">
          <div className="relative flex items-center gap-3 rounded-xl bg-sand px-4 py-3 text-sm font-medium text-accent">
            <span className="absolute bottom-2 left-0 top-2 w-[3px] rounded-full bg-accent" />
            <BookOpen className="h-4 w-4" />
            Documents
          </div>
        </nav>
        <div className="mt-auto p-3">
          <div className="flex items-center gap-3 rounded-2xl border border-border bg-bg-100 px-3 py-2.5">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-accent text-xs font-semibold text-white">
              KB
            </div>
            <div className="min-w-0 flex-1 truncate text-sm font-medium text-text-100">Administrator</div>
            <button
              type="button"
              onClick={logoutAdmin}
              title="Sign out"
              className="rounded-lg p-1.5 text-text-400 transition hover:bg-bg-200 hover:text-accent"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        </div>
      </aside>

      <main className="min-w-0 flex-1 space-y-4 sm:px-2 sm:py-1">
        <header className="flex items-center justify-between gap-3 px-1 pt-1">
          <div className="flex items-center gap-3">
            <img src="/logo-short.png" alt="" className="h-10 w-10 object-contain lg:hidden" />
            <div>
              <h1 className="font-serif text-2xl font-semibold tracking-tight text-text-100 sm:text-3xl">
                Documents
                <InfoTip side="bottom" className="ml-2">
                  Upload, track and manage the documents users can chat with.
                </InfoTip>
              </h1>
            </div>
          </div>
          <button type="button" onClick={logoutAdmin} className="btn-ghost lg:hidden">
            <LogOut className="h-4 w-4" />
            <span className="hidden sm:inline">Sign out</span>
          </button>
        </header>

        <section className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <StatCard icon={<FileText className="h-6 w-6" />} label="Total documents" value={files.length} hint="In the knowledge base" iconTone="text-text-200" />
          <StatCard icon={<CheckCircle2 className="h-6 w-6" />} label="Ready" value={readyCount} hint={pctOf(readyCount)} hintTone="text-accent" />
          <StatCard
            icon={<Loader className={`h-6 w-6 ${processingCount > 0 ? 'animate-spin [animation-duration:2.5s]' : ''}`} />}
            label="Processing"
            value={processingCount}
            hint={pctOf(processingCount)}
            iconTone="text-text-200"
          />
          <StatCard icon={<AlertCircle className="h-6 w-6" />} label="Failed" value={failedCount} hint={pctOf(failedCount)} hintTone="text-danger" iconTone="text-danger" />
        </section>

        <section className="panel p-3">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              handleFiles(e.dataTransfer.files);
            }}
            onClick={() => fileInput.current?.click()}
            className={`flex cursor-pointer flex-col items-center rounded-2xl border-2 border-dashed px-6 py-8 text-center transition sm:py-10 ${
              dragOver ? 'border-accent bg-accent/5' : 'border-bg-300 hover:border-accent/50 hover:bg-bg-0'
            }`}
          >
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-accent text-white shadow-[0_10px_24px_-10px_rgba(15,94,94,0.7)]">
              <UploadCloud className="h-7 w-7" />
            </div>
            <p className="mt-4 flex items-center gap-1.5 font-serif text-xl font-semibold text-accent">
              Upload documents
              <span onClick={(e) => e.stopPropagation()}>
                <InfoTip side="top">
                  Drag and drop files here, or click to browse. Supports PDF, DOCX, PPTX, XLSX and TXT, multi-select allowed.
                </InfoTip>
              </span>
            </p>
            <input
              ref={fileInput}
              type="file"
              multiple
              hidden
              accept=".pdf,.docx,.pptx,.xlsx,.txt"
              onChange={(e) => handleFiles(e.target.files)}
            />
          </div>
        </section>

      {stagedCount > 0 && (
        <section className="panel flex flex-col gap-3 border-warn/30 bg-warn-soft/60 p-4 sm:flex-row sm:items-center sm:justify-between sm:px-5">
          <div>
            <p className="text-sm font-semibold text-warn">
              {stagedCount} file{stagedCount === 1 ? '' : 's'} staged
              <InfoTip side="bottom" className="ml-1.5">
                Files are uploaded but not ingested yet. Start ingestion to begin parsing &amp; embedding.
              </InfoTip>
            </p>
          </div>
          <button type="button" onClick={startIngestion} className="btn-primary shrink-0">
            <Play className="h-4 w-4" />
            Start ingestion ({stagedCount})
          </button>
        </section>
      )}

      {uploads.length > 0 && (
        <section className="panel p-5">
          <h3 className="eyebrow">Uploads</h3>
          <ul className="mt-3 space-y-3">
            {uploads.map((u) => (
              <li key={u.key} className="text-sm">
                <div className="flex items-center gap-3">
                  <FileBadge name={u.filename} />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between">
                      <span className="truncate font-medium text-text-100">{u.filename}</span>
                      <span className="ml-3 shrink-0 text-xs text-text-400">
                        {u.phase === 'uploading' && `${Math.round((u.loaded / u.total) * 100)}%`}
                        {u.phase === 'done' && 'Uploaded — staged'}
                        {u.phase === 'duplicate' && `Skipped — ${u.message}`}
                        {u.phase === 'conflict' && 'Name conflict — awaiting decision'}
                        {u.phase === 'error' && `Error — ${u.message}`}
                      </span>
                    </div>
                    <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-bg-200">
                      <div
                        className={`h-full rounded-full transition-all ${
                          u.phase === 'error' || u.phase === 'duplicate' ? 'bg-danger' : 'bg-accent'
                        }`}
                        style={{
                          width:
                            u.phase === 'done'
                              ? '100%'
                              : `${Math.min(100, (u.loaded / Math.max(1, u.total)) * 100)}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="panel overflow-hidden">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-4">
          <h3 className="font-serif text-lg font-semibold text-text-100">
            Library <span className="font-sans text-sm font-normal text-text-400">({files.length})</span>
            {selected.size > 0 && (
              <span className="ml-2 font-sans text-sm font-medium text-accent">
                · {selected.size} selected
              </span>
            )}
          </h3>
          {selected.size > 0 && (
            <button
              type="button"
              onClick={delSelected}
              className="inline-flex items-center gap-1.5 rounded-xl border border-danger/30 bg-danger-soft px-3 py-1.5 text-xs font-medium text-danger transition hover:opacity-90"
            >
              <Trash2 className="h-3.5 w-3.5" />
              Delete selected ({selected.size})
            </button>
          )}
        </header>
        {files.length === 0 ? (
          <div className="p-10 text-center text-sm text-text-400">
            No files yet. Drop one above to get started.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs font-medium text-text-400">
                  <th className="w-12 py-3 pl-5 pr-2">
                    <input
                      type="checkbox"
                      aria-label="Select all files"
                      checked={selected.size === files.length && files.length > 0}
                      ref={(el) => {
                        if (el) el.indeterminate = selected.size > 0 && selected.size < files.length;
                      }}
                      onChange={toggleSelectAll}
                      className="h-4 w-4 cursor-pointer accent-[#0F5E5E]"
                    />
                  </th>
                  <th className="px-3 py-3 font-medium">File name</th>
                  <th className="px-3 py-3 font-medium">Size</th>
                  <th className="px-3 py-3 font-medium">Status</th>
                  <th className="px-3 py-3 font-medium">Progress</th>
                  <th className="px-3 py-3 font-medium">Updated</th>
                  <th className="py-3 pl-3 pr-5 text-right font-medium">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {files.map((f) => {
                  const inFlight = ['parsing', 'chunking', 'embedding'].includes(f.status);
                  const pct =
                    f.stage_total > 0 ? Math.round((f.stage_current / f.stage_total) * 100) : 0;
                  const isSel = selected.has(f.id);
                  return (
                    <tr key={f.id} className={`transition ${isSel ? 'bg-sand/60' : 'hover:bg-bg-0'}`}>
                      <td className="py-3 pl-5 pr-2">
                        <input
                          type="checkbox"
                          aria-label={`Select ${f.filename}`}
                          checked={isSel}
                          onChange={() => toggleSelect(f.id)}
                          className="h-4 w-4 cursor-pointer accent-[#0F5E5E]"
                        />
                      </td>
                      <td className="max-w-[320px] px-3 py-3">
                        <div className="flex items-center gap-3">
                          <FileBadge name={f.filename} />
                          <div className="min-w-0">
                            <div className="truncate font-medium text-text-100" title={f.filename}>
                              {f.filename}
                            </div>
                            {f.status === 'failed' && f.error_message && (
                              <div className="truncate text-xs text-danger" title={f.error_message}>
                                {f.error_message}
                              </div>
                            )}
                          </div>
                        </div>
                      </td>
                      <td className="whitespace-nowrap px-3 py-3 text-text-300">{humanSize(f.size_bytes)}</td>
                      <td className="px-3 py-3">
                        <span
                          className={`inline-flex whitespace-nowrap rounded-full px-2.5 py-1 text-xs font-medium ${
                            STATUS_TONE[f.status] ?? 'bg-bg-200 text-text-300'
                          }`}
                        >
                          {STATUS_LABEL[f.status] ?? f.status}
                        </span>
                      </td>
                      <td className="px-3 py-3">
                        {inFlight ? (
                          <div className="flex items-center gap-2.5">
                            <span className="w-9 text-right text-xs tabular-nums text-text-300">{pct}%</span>
                            <div className="h-1.5 w-28 overflow-hidden rounded-full bg-bg-200">
                              <div
                                className="h-full rounded-full bg-accent transition-all"
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                          </div>
                        ) : (
                          <span className="text-text-500">—</span>
                        )}
                      </td>
                      <td className="whitespace-nowrap px-3 py-3 text-text-300">{shortDate(f.updated_at)}</td>
                      <td className="py-3 pl-3 pr-5 text-right">
                        <button
                          type="button"
                          onClick={() => del(f.id)}
                          title="Delete file"
                          className="rounded-lg p-2 text-text-400 transition hover:bg-danger-soft hover:text-danger"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <div className="pointer-events-none fixed inset-x-4 bottom-4 z-40 flex flex-col gap-2 sm:inset-x-auto sm:bottom-6 sm:right-6 sm:w-full sm:max-w-sm">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`pointer-events-auto rounded-2xl border px-4 py-3 text-sm shadow-[0_10px_30px_-12px_rgba(70,55,25,0.3)] ${
              t.tone === 'error'
                ? 'border-danger/30 bg-danger-soft text-danger'
                : 'border-border bg-bg-100 text-text-100'
            }`}
          >
            {t.message}
          </div>
        ))}
      </div>

      {replacePrompt && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-scrim p-4 backdrop-blur-sm">
          <div className="panel w-full max-w-sm p-6">
            <h3 className="font-serif text-xl font-semibold text-text-100">File already exists</h3>
            <p className="mt-2 text-sm text-text-300">
              A file named <span className="font-medium text-text-100">{replacePrompt.file.name}</span> already
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
                className="btn-ghost"
              >
                Cancel
              </button>
              <button type="button" onClick={confirmReplace} className="btn-primary">
                Replace
              </button>
            </div>
          </div>
        </div>
      )}
      </main>
    </div>
  );
}
