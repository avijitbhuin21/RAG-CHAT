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
  const res = await fetch(`${API_BASE}${path}`, { ...init, credentials: 'include' });
  if (!res.ok || !res.body) {
    throw new Error(`stream failed: ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split('\n\n');
    buf = parts.pop() ?? '';
    for (const part of parts) {
      const line = part.trim();
      if (line.startsWith('data: ')) {
        try {
          onEvent(JSON.parse(line.slice(6)));
        } catch {
          /* ignore malformed */
        }
      }
    }
  }
}
