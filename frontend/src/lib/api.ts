export class ApiError extends Error {
  constructor(public status: number, message: string, public body?: unknown) {
    super(message);
  }
}

type Options = Omit<RequestInit, 'body'> & { json?: unknown; body?: BodyInit };

export async function api<T = unknown>(path: string, opts: Options = {}): Promise<T> {
  const { json, headers, ...rest } = opts;
  const init: RequestInit = {
    ...rest,
    credentials: 'include',
    headers: {
      ...(json !== undefined ? { 'Content-Type': 'application/json' } : {}),
      ...(headers ?? {}),
    },
    body: json !== undefined ? JSON.stringify(json) : opts.body,
  };
  const res = await fetch(`/api${path}`, init);
  // Read the body as text exactly once. Reading it twice (e.g. res.json()
  // followed by res.text() in a catch block) throws "body stream already
  // read" because fetch locks the stream after the first read attempt --
  // even if that attempt threw a parse error.
  const raw = await res.text();
  const ct = res.headers.get('content-type') ?? '';
  const isJson = ct.includes('application/json');
  const parse = (): unknown => {
    if (!raw) return undefined;
    if (!isJson) return raw;
    try {
      return JSON.parse(raw);
    } catch {
      return raw;
    }
  };
  if (!res.ok) {
    throw new ApiError(res.status, `API ${res.status} on ${path}`, parse());
  }
  return parse() as T;
}
