import { useCallback, useEffect, useRef, useState } from 'react';

import { API_BASE } from '@/lib/apiBase';
import { uploadWithProgress } from '@/lib/stream';

export type AttachmentMeta = {
  id: string;
  kind: 'image' | 'document';
  filename: string;
  mime: string;
  size: number;
  chars?: number;
  truncated?: boolean;
  previewUrl?: string;
};

export type PendingAttachment = {
  key: string;
  kind: 'image' | 'document';
  name: string;
  size: number;
  previewUrl: string | null;
  status: 'uploading' | 'ready' | 'error';
  progress: number;
  meta?: AttachmentMeta;
  error?: string;
};

export const MAX_IMAGES = 4;
export const MAX_DOCS = 3;
export const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
export const MAX_DOC_BYTES = 10 * 1024 * 1024;

const IMAGE_EXTS = ['png', 'jpg', 'jpeg', 'webp', 'gif'];
const DOC_EXTS = [
  'pdf', 'docx', 'xlsx', 'txt', 'md', 'markdown', 'csv', 'tsv', 'json', 'xml', 'html', 'htm',
  'log', 'yaml', 'yml', 'ini', 'rtf', 'sql',
];

export const IMAGE_ACCEPT = IMAGE_EXTS.map((e) => `.${e}`).join(',');
export const DOC_ACCEPT = DOC_EXTS.map((e) => `.${e}`).join(',');

/** Classifies a file as image/document by extension, or null if unsupported. */
export function classifyFile(name: string): 'image' | 'document' | null {
  const ext = name.toLowerCase().split('.').pop() ?? '';
  if (IMAGE_EXTS.includes(ext)) return 'image';
  if (DOC_EXTS.includes(ext)) return 'document';
  return null;
}

/** Formats a byte count as a short human-readable size. */
export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/** URL for fetching a stored attachment's original bytes. */
export function attachmentUrl(id: string): string {
  return `${API_BASE}/chat/attachments/${id}`;
}

/** Manages composer attachments: validation, upload with progress, and removal. */
export function useAttachments() {
  const [items, setItems] = useState<PendingAttachment[]>([]);
  const itemsRef = useRef(items);
  itemsRef.current = items;

  const patch = useCallback((key: string, p: Partial<PendingAttachment>) => {
    setItems((prev) => prev.map((it) => (it.key === key ? { ...it, ...p } : it)));
  }, []);

  const upload = useCallback(
    async (key: string, file: File) => {
      const form = new FormData();
      form.append('file', file, file.name);
      try {
        const { status, body } = await uploadWithProgress('/chat/attachments', form, (p) =>
          patch(key, { progress: p.total ? p.loaded / p.total : 0 }),
        );
        if (status !== 200) {
          const detail =
            body && typeof body === 'object' && 'detail' in body ? String((body as { detail: unknown }).detail) : `Upload failed (${status})`;
          patch(key, { status: 'error', error: detail });
          return;
        }
        patch(key, { status: 'ready', progress: 1, meta: body as AttachmentMeta });
      } catch (e) {
        patch(key, { status: 'error', error: `Upload failed: ${String(e)}` });
      }
    },
    [patch],
  );

  const add = useCallback(
    (files: File[]) => {
      const current = itemsRef.current.filter((i) => i.status !== 'error');
      let images = current.filter((i) => i.kind === 'image').length;
      let docs = current.filter((i) => i.kind === 'document').length;
      const next: PendingAttachment[] = [];
      const toUpload: [string, File][] = [];
      for (const file of files) {
        const name = file.name || `pasted-${Date.now()}.png`;
        const kind = classifyFile(name);
        const key = `${name}-${file.size}-${Math.random().toString(36).slice(2, 8)}`;
        const base = { key, name, size: file.size, progress: 0, previewUrl: null };
        if (!kind) {
          next.push({ ...base, kind: 'document', status: 'error', error: 'Unsupported file type' });
          continue;
        }
        const limit = kind === 'image' ? MAX_IMAGE_BYTES : MAX_DOC_BYTES;
        if (file.size > limit) {
          next.push({ ...base, kind, status: 'error', error: `Too large (max ${formatBytes(limit)})` });
          continue;
        }
        if (kind === 'image' && images >= MAX_IMAGES) {
          next.push({ ...base, kind, status: 'error', error: `Max ${MAX_IMAGES} images per message` });
          continue;
        }
        if (kind === 'document' && docs >= MAX_DOCS) {
          next.push({ ...base, kind, status: 'error', error: `Max ${MAX_DOCS} documents per message` });
          continue;
        }
        if (kind === 'image') images += 1;
        else docs += 1;
        const named = file.name ? file : new File([file], name, { type: file.type });
        next.push({
          ...base,
          kind,
          status: 'uploading',
          previewUrl: kind === 'image' ? URL.createObjectURL(named) : null,
        });
        toUpload.push([key, named]);
      }
      setItems((prev) => [...prev, ...next]);
      for (const [key, file] of toUpload) void upload(key, file);
    },
    [upload],
  );

  const remove = useCallback((key: string) => {
    setItems((prev) => {
      const it = prev.find((i) => i.key === key);
      if (it?.previewUrl) URL.revokeObjectURL(it.previewUrl);
      return prev.filter((i) => i.key !== key);
    });
  }, []);

  const clear = useCallback(() => setItems([]), []);

  useEffect(
    () => () => {
      for (const it of itemsRef.current) if (it.previewUrl) URL.revokeObjectURL(it.previewUrl);
    },
    [],
  );

  const ready: AttachmentMeta[] = items
    .filter((i) => i.status === 'ready' && i.meta)
    .map((i) => ({ ...(i.meta as AttachmentMeta), previewUrl: i.previewUrl ?? undefined }));

  return {
    items,
    ready,
    uploading: items.some((i) => i.status === 'uploading'),
    add,
    remove,
    clear,
  };
}
