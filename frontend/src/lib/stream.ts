import { API_BASE } from './apiBase';

export type UploadProgress = { loaded: number; total: number };

export function uploadWithProgress(
  path: string,
  formData: FormData,
  onProgress: (p: UploadProgress) => void,
): Promise<{ status: number; body: unknown }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}${path}`);
    xhr.withCredentials = true;
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) onProgress({ loaded: e.loaded, total: e.total });
    });
    xhr.onload = () => {
      let body: unknown = xhr.responseText;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {}
      resolve({ status: xhr.status, body });
    };
    xhr.onerror = () => reject(new Error('network error'));
    xhr.send(formData);
  });
}

export async function streamSSE(
  path: string,
  init: RequestInit,
  onEvent: (event: any) => void,
): Promise<void> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    credentials: 'include',
    headers: { Accept: 'text/event-stream', ...(init.headers ?? {}) },
  });
  if (!res.ok || !res.body) {
    throw new Error(`stream failed: ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';

  const dispatch = (frame: string) => {
    const data: string[] = [];
    for (const raw of frame.split('\n')) {
      if (!raw || raw.startsWith(':')) continue;
      if (raw.startsWith('data:')) data.push(raw.slice(5).replace(/^ /, ''));
    }
    if (data.length === 0) return;
    try {
      onEvent(JSON.parse(data.join('\n')));
    } catch {
      /* ignore malformed */
    }
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true }).replace(/\r\n?/g, '\n');
      let split = buf.indexOf('\n\n');
      while (split !== -1) {
        dispatch(buf.slice(0, split));
        buf = buf.slice(split + 2);
        split = buf.indexOf('\n\n');
      }
    }
    buf += decoder.decode();
    if (buf.trim()) dispatch(buf);
  } finally {
    reader.releaseLock();
  }
}
