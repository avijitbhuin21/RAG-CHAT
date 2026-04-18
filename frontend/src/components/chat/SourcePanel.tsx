import { FileText, Loader2, X } from 'lucide-react';
import {
  Fragment,
  ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import type { PDFDocumentProxy, TextItem } from 'pdfjs-dist/types/src/display/api';
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import mammoth from 'mammoth';
import { API_BASE } from '@/lib/apiBase';

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

export type SourcePanelTarget = {
  fileId: string;
  filename: string;
  chunkTexts: string[];
};

type Kind = 'pdf' | 'docx' | 'text' | 'unknown';

function kindOf(filename: string): Kind {
  const ext = filename.toLowerCase().split('.').pop() ?? '';
  if (ext === 'pdf') return 'pdf';
  if (ext === 'docx') return 'docx';
  if (['txt', 'md', 'markdown', 'csv', 'log', 'json', 'xml', 'html', 'htm'].includes(ext))
    return 'text';
  return 'unknown';
}

// Build a normalized projection of `source` plus a map from each index in the
// normalized string back to its origin index in `source`. Used so we can do a
// tolerant substring match against extracted PDF/DOCX text, then remap the
// match back to the original positions for highlighting.
function normalize(source: string): { norm: string; map: number[] } {
  const nfkc = source.normalize('NFKC');
  const out: string[] = [];
  const map: number[] = [];
  let lastWasSpace = false;
  for (let i = 0; i < nfkc.length; i++) {
    const c = nfkc[i];
    // De-hyphenate line breaks: "informa-\ninformation" → "information"
    if (c === '-' && (nfkc[i + 1] === '\n' || nfkc[i + 1] === '\r')) {
      let j = i + 1;
      while (j < nfkc.length && (nfkc[j] === '\n' || nfkc[j] === '\r' || nfkc[j] === ' '))
        j++;
      i = j - 1;
      lastWasSpace = false;
      continue;
    }
    if (/\s/.test(c)) {
      if (lastWasSpace || out.length === 0) continue;
      out.push(' ');
      map.push(i);
      lastWasSpace = true;
      continue;
    }
    out.push(c.toLowerCase());
    map.push(i);
    lastWasSpace = false;
  }
  // Trim trailing space
  while (out.length && out[out.length - 1] === ' ') {
    out.pop();
    map.pop();
  }
  return { norm: out.join(''), map };
}

// Resolve every chunk into an original-coordinate [start, end) range in the
// given haystack, drop the ones that didn't match, then merge overlapping or
// touching ranges so downstream code can apply marks sequentially without
// double-wrapping shared text.
function findAllChunkRanges(
  haystack: string,
  needles: string[],
): [number, number][] {
  const found = needles
    .map((n) => findChunkRange(haystack, n))
    .filter((r): r is [number, number] => !!r)
    .sort((a, b) => a[0] - b[0]);
  const merged: [number, number][] = [];
  for (const r of found) {
    const last = merged[merged.length - 1];
    if (last && r[0] <= last[1]) {
      last[1] = Math.max(last[1], r[1]);
    } else {
      merged.push([r[0], r[1]]);
    }
  }
  return merged;
}

// Find the chunk inside a larger text. Returns [start, end) on the ORIGINAL
// string coordinates, or null if no match.
function findChunkRange(haystack: string, needle: string): [number, number] | null {
  if (!needle.trim()) return null;
  const h = normalize(haystack);
  const n = normalize(needle);
  if (!n.norm) return null;

  let idx = h.norm.indexOf(n.norm);
  // If the full chunk doesn't match, try progressively shorter prefixes so
  // we still highlight something useful when extraction drifts near the end.
  if (idx === -1) {
    const minLen = Math.max(40, Math.floor(n.norm.length * 0.4));
    for (let len = n.norm.length - 20; len >= minLen; len -= 20) {
      const slice = n.norm.slice(0, len);
      idx = h.norm.indexOf(slice);
      if (idx !== -1) {
        const start = h.map[idx];
        const endNormIdx = idx + slice.length - 1;
        const end = h.map[endNormIdx] + 1;
        return [start, end];
      }
    }
    return null;
  }
  const start = h.map[idx];
  const endNormIdx = idx + n.norm.length - 1;
  const end = h.map[endNormIdx] + 1;
  return [start, end];
}

export function SourcePanel({
  target,
  onClose,
}: {
  target: SourcePanelTarget | null;
  onClose: () => void;
}) {
  const [blob, setBlob] = useState<Blob | null>(null);
  const [contentType, setContentType] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const panelRef = useRef<HTMLElement>(null);

  // Close on mousedown anywhere outside the panel. Using mousedown (not click)
  // means opening a *different* citation still works: the citation's click
  // handler sets a new target in the same tick React closes + reopens in one
  // render, so there's no flicker. Also close on Escape for keyboard users.
  useEffect(() => {
    if (!target) return;
    function onDocMouseDown(e: MouseEvent) {
      if (!panelRef.current) return;
      if (e.target instanceof Node && panelRef.current.contains(e.target)) return;
      onClose();
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
    }
    document.addEventListener('mousedown', onDocMouseDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDocMouseDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [target, onClose]);

  useEffect(() => {
    if (!target) return;
    setBlob(null);
    setContentType('');
    setError(null);
    setLoading(true);
    const ac = new AbortController();
    fetch(
      `${API_BASE}/chat/files/${target.fileId}?name=${encodeURIComponent(target.filename)}`,
      { credentials: 'include', signal: ac.signal },
    )
      .then(async (r) => {
        if (!r.ok) {
          const body = await r.text().catch(() => '');
          throw new Error(`HTTP ${r.status}${body ? ` — ${body}` : ''}`);
        }
        const ct = r.headers.get('content-type') ?? '';
        const b = await r.blob();
        setBlob(b);
        setContentType(ct);
      })
      .catch((e) => {
        if (e.name !== 'AbortError') setError(String(e.message ?? e));
      })
      .finally(() => setLoading(false));
    return () => ac.abort();
  }, [target?.fileId]);

  if (!target) return null;

  const kind = kindOf(target.filename);

  return (
    <aside
      ref={panelRef}
      className="pointer-events-auto fixed right-0 top-0 z-50 flex h-[100dvh] w-full flex-col border-l border-border bg-bg-0 shadow-2xl animate-slide-in-right sm:max-w-[640px]"
    >
      <header className="flex items-center gap-2 border-b border-border bg-bg-100 px-4 py-3">
        <FileText className="h-4 w-4 shrink-0 text-accent" />
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-semibold text-text-100" title={target.filename}>
            {target.filename}
          </div>
          <div className="text-xs text-text-400">Source document</div>
        </div>
        <button
          type="button"
          onClick={onClose}
          title="Close"
          className="rounded-md p-1.5 text-text-300 transition hover:bg-bg-200 hover:text-text-100"
        >
          <X className="h-4 w-4" />
        </button>
      </header>

      <div className="flex-1 overflow-y-auto bg-bg-0">
        {loading && (
          <div className="flex h-full items-center justify-center text-text-400">
            <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading document…
          </div>
        )}
        {error && (
          <div className="p-6 text-sm text-red-600">Failed to load document: {error}</div>
        )}
        {!loading && !error && blob && (
          <ViewerSwitch
            kind={kind}
            blob={blob}
            contentType={contentType}
            chunkTexts={target.chunkTexts}
          />
        )}
      </div>
    </aside>
  );
}

function ViewerSwitch({
  kind,
  blob,
  contentType,
  chunkTexts,
}: {
  kind: Kind;
  blob: Blob;
  contentType: string;
  chunkTexts: string[];
}) {
  // Fall back on MIME sniffing when the extension doesn't tell us enough.
  let effective = kind;
  if (effective === 'unknown') {
    if (contentType.includes('pdf')) effective = 'pdf';
    else if (contentType.includes('wordprocessingml')) effective = 'docx';
    else if (contentType.startsWith('text/')) effective = 'text';
  }

  if (effective === 'pdf') return <PdfViewer blob={blob} chunkTexts={chunkTexts} />;
  if (effective === 'docx') return <DocxViewer blob={blob} chunkTexts={chunkTexts} />;
  if (effective === 'text') return <TextViewer blob={blob} chunkTexts={chunkTexts} />;
  return (
    <div className="p-6 text-sm text-text-300">
      Preview isn't available for this file type yet.
    </div>
  );
}

function TextViewer({
  blob,
  chunkTexts,
}: {
  blob: Blob;
  chunkTexts: string[];
}) {
  const [text, setText] = useState<string>('');
  useEffect(() => {
    blob.text().then(setText);
  }, [blob]);

  const ranges = useMemo(
    () => findAllChunkRanges(text, chunkTexts),
    [text, chunkTexts],
  );
  const firstMarkRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    firstMarkRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, [ranges]);

  if (!text) return null;
  if (ranges.length === 0) {
    return (
      <pre className="whitespace-pre-wrap p-4 font-mono text-xs text-text-200 sm:p-6">
        {text}
      </pre>
    );
  }

  const segments: ReactNode[] = [];
  let cursor = 0;
  ranges.forEach(([s, e], i) => {
    if (cursor < s) segments.push(<Fragment key={`t-${i}`}>{text.slice(cursor, s)}</Fragment>);
    segments.push(
      <mark
        key={`m-${i}`}
        ref={i === 0 ? firstMarkRef : undefined}
        className="rounded bg-amber-200/70 px-0.5 text-text-100 ring-1 ring-amber-400/60"
      >
        {text.slice(s, e)}
      </mark>,
    );
    cursor = e;
  });
  if (cursor < text.length) {
    segments.push(<Fragment key="t-end">{text.slice(cursor)}</Fragment>);
  }

  return (
    <pre className="whitespace-pre-wrap p-4 font-mono text-xs leading-relaxed text-text-200 sm:p-6">
      {segments}
    </pre>
  );
}

function DocxViewer({
  blob,
  chunkTexts,
}: {
  blob: Blob;
  chunkTexts: string[];
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [html, setHtml] = useState<string>('');

  useEffect(() => {
    blob
      .arrayBuffer()
      .then((buf) => mammoth.convertToHtml({ arrayBuffer: buf }))
      .then((res) => setHtml(res.value));
  }, [blob]);

  useEffect(() => {
    if (!html || !containerRef.current) return;
    const root = containerRef.current;
    root.innerHTML = html;
    highlightInDom(root, chunkTexts);
  }, [html, chunkTexts]);

  return (
    <div
      ref={containerRef}
      className="docx-content max-w-none p-4 text-sm leading-relaxed text-text-200 sm:p-6 [&_h1]:mb-2 [&_h1]:mt-4 [&_h1]:text-xl [&_h1]:font-semibold [&_h2]:mb-2 [&_h2]:mt-3 [&_h2]:text-lg [&_h2]:font-semibold [&_h3]:mb-1 [&_h3]:mt-3 [&_h3]:text-base [&_h3]:font-semibold [&_p]:my-2 [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-6 [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-6 [&_li]:my-0.5 [&_strong]:font-semibold [&_em]:italic [&_table]:my-3 [&_table]:border-collapse [&_td]:border [&_td]:border-bg-300 [&_td]:px-2 [&_td]:py-1 [&_th]:border [&_th]:border-bg-300 [&_th]:bg-bg-100 [&_th]:px-2 [&_th]:py-1"
    />
  );
}

// Walks text nodes in `root` in document order, concatenates their text,
// resolves every chunk to a range, merges overlaps, then maps each range
// back to individual text nodes and wraps the matched portions with <mark>.
// Ranges are applied in REVERSE document order so earlier positions aren't
// invalidated by later DOM splices.
function highlightInDom(root: HTMLElement, chunkTexts: string[]) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (n) =>
      n.nodeValue && n.nodeValue.length > 0
        ? NodeFilter.FILTER_ACCEPT
        : NodeFilter.FILTER_REJECT,
  });
  type NodeSpan = { node: Text; start: number; end: number };
  const spans: NodeSpan[] = [];
  let full = '';
  let node: Node | null;
  while ((node = walker.nextNode())) {
    const t = node as Text;
    const len = t.nodeValue!.length;
    spans.push({ node: t, start: full.length, end: full.length + len });
    full += t.nodeValue!;
  }
  const ranges = findAllChunkRanges(full, chunkTexts);
  if (ranges.length === 0) return;

  for (let r = ranges.length - 1; r >= 0; r--) {
    const [s, e] = ranges[r];
    const hit: NodeSpan[] = spans.filter((sp) => sp.end > s && sp.start < e);
    for (let i = hit.length - 1; i >= 0; i--) {
      const sp = hit[i];
      const localStart = Math.max(0, s - sp.start);
      const localEnd = Math.min(sp.end - sp.start, e - sp.start);
      const text = sp.node.nodeValue!;
      const before = text.slice(0, localStart);
      const mid = text.slice(localStart, localEnd);
      const after = text.slice(localEnd);
      const parent = sp.node.parentNode;
      if (!parent) continue;
      const mark = document.createElement('mark');
      mark.className =
        'rounded bg-amber-200/70 px-0.5 text-text-100 ring-1 ring-amber-400/60';
      mark.textContent = mid;
      const beforeNode = document.createTextNode(before);
      const afterNode = document.createTextNode(after);
      parent.replaceChild(afterNode, sp.node);
      parent.insertBefore(mark, afterNode);
      parent.insertBefore(beforeNode, mark);
    }
  }
  // First <mark> in document order now exists; scroll it into view.
  const firstMark = root.querySelector('mark');
  if (firstMark) {
    window.setTimeout(
      () => firstMark.scrollIntoView({ behavior: 'smooth', block: 'center' }),
      50,
    );
  }
}

function PdfViewer({
  blob,
  chunkTexts,
}: {
  blob: Blob;
  chunkTexts: string[];
}) {
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [cssWidth, setCssWidth] = useState<number>(608);

  useEffect(() => {
    let cancelled = false;
    let doc: PDFDocumentProxy | null = null;
    blob.arrayBuffer().then(async (buf) => {
      const task = pdfjsLib.getDocument({ data: buf });
      doc = await task.promise;
      if (cancelled) {
        doc.destroy();
        return;
      }
      setPdf(doc);
    });
    return () => {
      cancelled = true;
      doc?.destroy();
    };
  }, [blob]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = Math.floor(entry.contentRect.width);
        if (w > 0) setCssWidth(Math.min(608, w));
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  if (!pdf) {
    return (
      <div className="flex h-full items-center justify-center text-text-400">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Preparing PDF…
      </div>
    );
  }
  const pages = Array.from({ length: pdf.numPages }, (_, i) => i + 1);
  return (
    <div ref={wrapRef} className="space-y-4 p-2 sm:p-4">
      {pages.map((p) => (
        <PdfPage
          key={p}
          pdf={pdf}
          pageNumber={p}
          chunkTexts={chunkTexts}
          cssWidth={cssWidth}
        />
      ))}
    </div>
  );
}

function PdfPage({
  pdf,
  pageNumber,
  chunkTexts,
  cssWidth,
}: {
  pdf: PDFDocumentProxy;
  pageNumber: number;
  chunkTexts: string[];
  cssWidth: number;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const layerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const page = await pdf.getPage(pageNumber);
      if (cancelled) return;
      const unscaled = page.getViewport({ scale: 1 });
      const cssScale = cssWidth / unscaled.width;
      // Render the canvas backing store at devicePixelRatio so high-DPI
      // displays (retina, 125/150% Windows scaling) get crisp output instead
      // of a blurry upscale. Anything above 2x is wasted pixels; clamp.
      const dpr = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
      const renderScale = cssScale * dpr;
      const renderViewport = page.getViewport({ scale: renderScale });
      const cssViewport = page.getViewport({ scale: cssScale });

      const canvas = canvasRef.current!;
      const ctx = canvas.getContext('2d')!;
      canvas.width = Math.floor(renderViewport.width);
      canvas.height = Math.floor(renderViewport.height);
      canvas.style.width = `${cssViewport.width}px`;
      canvas.style.height = `${cssViewport.height}px`;
      const container = containerRef.current!;
      container.style.width = `${cssViewport.width}px`;
      container.style.height = `${cssViewport.height}px`;

      await page.render({
        canvasContext: ctx,
        viewport: renderViewport,
        canvas,
      }).promise;
      if (cancelled) return;
      // Highlight math below expects CSS-coordinate positions.
      const viewport = cssViewport;

      const textContent = await page.getTextContent();
      const items = textContent.items.filter(
        (it): it is TextItem => 'str' in it && 'transform' in it,
      );
      // Build the page's concatenated text with per-item offsets.
      type ItemSpan = { item: TextItem; start: number; end: number };
      const spans: ItemSpan[] = [];
      let full = '';
      for (const it of items) {
        spans.push({ item: it, start: full.length, end: full.length + it.str.length });
        full += it.str;
        // pdf.js splits items roughly by glyph run; add a space between items
        // when the next item clearly begins a new visual token. Simple default:
        // add space if the item has a hasEOL marker or if lengths suggest so.
        if ((it as TextItem & { hasEOL?: boolean }).hasEOL) {
          full += '\n';
          // Don't add a span for the synthetic newline — it won't be highlighted.
        } else {
          full += ' ';
        }
      }

      const ranges = findAllChunkRanges(full, chunkTexts);
      const layer = layerRef.current!;
      layer.innerHTML = '';
      if (ranges.length === 0) return;

      for (const [s, e] of ranges) {
        for (const sp of spans) {
          if (sp.end <= s || sp.start >= e) continue;
          const t = sp.item.transform;
          // Transform (a, b, c, d, tx, ty) from pdf.js gives font size +
          // position at baseline. Convert with the viewport's default
          // transform to get CSS-pixel coordinates.
          const [, , , , tx, ty] = t;
          const fontHeight = Math.hypot(t[2], t[3]);
          const x = tx * viewport.transform[0] + viewport.transform[4];
          const yBaseline = ty * viewport.transform[3] + viewport.transform[5];
          const h = fontHeight * Math.abs(viewport.transform[3]);
          const w = sp.item.width * viewport.transform[0];
          const top = yBaseline - h;

          const hl = document.createElement('div');
          hl.style.position = 'absolute';
          hl.style.left = `${x}px`;
          hl.style.top = `${top}px`;
          hl.style.width = `${w}px`;
          hl.style.height = `${h}px`;
          hl.style.background = 'rgba(251, 191, 36, 0.42)';
          hl.style.borderRadius = '2px';
          hl.style.pointerEvents = 'none';
          layer.appendChild(hl);
        }
      }

      const first = layer.querySelector<HTMLDivElement>('div');
      if (first) {
        window.setTimeout(
          () => first.scrollIntoView({ behavior: 'smooth', block: 'center' }),
          60,
        );
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [pdf, pageNumber, chunkTexts, cssWidth]);

  return (
    <div
      ref={containerRef}
      className="relative mx-auto overflow-hidden rounded-md border border-bg-300 bg-white shadow-sm"
    >
      <canvas ref={canvasRef} />
      <div
        ref={layerRef}
        className="pointer-events-none absolute inset-0"
        aria-hidden
      />
    </div>
  );
}
