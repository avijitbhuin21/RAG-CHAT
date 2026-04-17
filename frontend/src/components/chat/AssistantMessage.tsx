import { ChevronDown, FileText, Search } from 'lucide-react';
import {
  cloneElement,
  Fragment,
  isValidElement,
  ReactNode,
  useEffect,
  useRef,
  useState,
} from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export type Citation = {
  index: number;
  filename: string;
  file_id: string | null;
  snippet: string;
};

export type ToolStatus = 'searching' | 'done' | null;

export type AssistantMsg = {
  id: string;
  role: 'assistant';
  content: string;
  thinking: string | null;
  citations: Citation[] | null;
  streaming?: boolean;
  thinkingStartedAt?: number | null;
  thinkingEndedAt?: number | null;
  toolStatus?: ToolStatus;
  toolQuery?: string | null;
};

export function ShimmerText({ text }: { text: string }) {
  const chars = text.split('');
  const stagger = 0.07;
  return (
    <span className="shimmer-wave inline-flex select-none text-sm font-medium text-text-400">
      {chars.map((c, i) =>
        c === ' ' ? (
          <span key={i} className="space" aria-hidden />
        ) : (
          <span key={i} style={{ animationDelay: `${i * stagger}s` }}>
            {c}
          </span>
        ),
      )}
    </span>
  );
}

function useElapsedSeconds(startedAt: number | null | undefined, frozen: number | null) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (frozen != null || !startedAt) return;
    const id = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(id);
  }, [startedAt, frozen]);
  if (frozen != null) return frozen;
  if (!startedAt) return 0;
  return Math.max(0, (now - startedAt) / 1000);
}

/**
 * Progressively reveal `target` one character-batch at a time while `enabled`
 * is true, so the user sees a visible typing motion even when the upstream
 * provider delivers content in large chunks (Bifrost currently hands us the
 * full tail of a Sonnet response in 1-2 deltas). The moment `enabled` flips
 * to false (stream done) we snap to the full string — no content ever gets
 * hidden by the animation.
 */
function useTypewriter(target: string, enabled: boolean): string {
  const [visibleLen, setVisibleLen] = useState(target.length);

  // When streaming ends OR the target shrank (component reused), snap.
  useEffect(() => {
    if (!enabled) setVisibleLen(target.length);
    else if (visibleLen > target.length) setVisibleLen(target.length);
  }, [enabled, target.length, visibleLen]);

  useEffect(() => {
    if (!enabled) return;
    if (visibleLen >= target.length) return;
    const remaining = target.length - visibleLen;
    const step = Math.min(30, Math.max(1, Math.floor(remaining / 30)));
    const t = setTimeout(
      () => setVisibleLen((n) => Math.min(target.length, n + step)),
      18,
    );
    return () => clearTimeout(t);
  }, [enabled, target, visibleLen]);

  // DIAGNOSTIC: log every render so we can confirm in DevTools whether the
  // hook is actually running and advancing during streaming.
  // eslint-disable-next-line no-console
  console.debug('[typewriter]', { enabled, visibleLen, targetLen: target.length });

  return target.slice(0, visibleLen);
}

function ThinkingPanel({
  thinking,
  startedAt,
  endedAt,
  active,
}: {
  thinking: string;
  startedAt: number | null | undefined;
  endedAt: number | null | undefined;
  active: boolean;
}) {
  const frozen = endedAt && startedAt ? (endedAt - startedAt) / 1000 : null;
  const elapsed = useElapsedSeconds(startedAt, frozen);
  const seconds = Math.max(0, Math.round(elapsed));
  const hasTiming = !!startedAt;
  // Auto-expand while the model is actively thinking, auto-collapse once done.
  // Track user override so they can manually toggle either way.
  const [userOverride, setUserOverride] = useState<boolean | null>(null);
  const open = userOverride != null ? userOverride : active;
  const bodyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open && active && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [thinking, open, active]);

  const headerLabel = active ? (
    <ShimmerText text={hasTiming ? `Thinking ${seconds}s` : 'Thinking…'} />
  ) : (
    <span className="text-text-300">
      {hasTiming ? `Thought for ${seconds}s` : 'Thinking'}
    </span>
  );

  return (
    <div className="rounded-lg border border-bg-300 bg-bg-100/70 backdrop-blur-sm">
      <button
        type="button"
        onClick={() => setUserOverride(!open)}
        className="flex w-full items-center justify-between px-3 py-2 text-xs"
      >
        {headerLabel}
        <ChevronDown
          className={`h-3.5 w-3.5 text-text-400 transition-transform ${open ? '' : '-rotate-90'}`}
        />
      </button>
      {open && (
        <div
          ref={bodyRef}
          className="custom-scrollbar max-h-72 overflow-y-auto whitespace-pre-wrap border-t border-bg-300 px-3 py-2 text-xs leading-relaxed text-text-300"
        >
          {thinking || (active ? '…' : '')}
        </div>
      )}
    </div>
  );
}

function ToolCallPill({ status, query }: { status: ToolStatus; query: string | null }) {
  if (!status) return null;
  return (
    <div className="inline-flex items-center gap-2 rounded-full border border-bg-300 bg-bg-100 px-3 py-1 text-xs text-text-300">
      <Search className="h-3 w-3 text-accent" />
      {status === 'searching' ? (
        <ShimmerText text={`Searching knowledge base${query ? `: "${truncate(query, 40)}"` : '…'}`} />
      ) : (
        <span>
          Searched knowledge base
          {query ? <>: <span className="text-text-200">"{truncate(query, 40)}"</span></> : ''}
        </span>
      )}
    </div>
  );
}

function truncate(s: string, n: number) {
  return s.length <= n ? s : s.slice(0, n - 1) + '…';
}

function CitationSup({
  index,
  onClick,
}: {
  index: number;
  onClick: (index: number) => void;
}) {
  return (
    <a
      href={`#cite-${index}`}
      className="citation-sup"
      onClick={(e) => {
        e.preventDefault();
        onClick(index);
      }}
    >
      {index}
    </a>
  );
}

function replaceCitations(
  node: ReactNode,
  onClick: (index: number) => void,
  keyPrefix = 'c',
): ReactNode {
  if (typeof node === 'string' || typeof node === 'number') {
    const str = String(node);
    if (!/\[\d+\]/.test(str)) return str;
    const parts = str.split(/(\[\d+\])/g);
    return parts.map((p, i) => {
      const m = p.match(/^\[(\d+)\]$/);
      if (m) {
        return (
          <CitationSup
            key={`${keyPrefix}-${i}`}
            index={parseInt(m[1], 10)}
            onClick={onClick}
          />
        );
      }
      return p ? <Fragment key={`${keyPrefix}-${i}`}>{p}</Fragment> : null;
    });
  }
  if (Array.isArray(node)) {
    return node.map((c, i) => (
      <Fragment key={`${keyPrefix}-${i}`}>
        {replaceCitations(c, onClick, `${keyPrefix}-${i}`)}
      </Fragment>
    ));
  }
  if (isValidElement<{ children?: ReactNode }>(node)) {
    return cloneElement(node, {
      ...node.props,
      children: replaceCitations(node.props.children, onClick, keyPrefix),
    });
  }
  return node;
}

function makeMarkdownComponents(onCite: (i: number) => void) {
  const wrap = (Tag: keyof JSX.IntrinsicElements) =>
    ({ children, ...rest }: any) => {
      const Element = Tag as any;
      return <Element {...rest}>{replaceCitations(children, onCite)}</Element>;
    };
  return {
    p: wrap('p'),
    li: wrap('li'),
    h1: wrap('h1'),
    h2: wrap('h2'),
    h3: wrap('h3'),
    h4: wrap('h4'),
    h5: wrap('h5'),
    h6: wrap('h6'),
    td: wrap('td'),
    th: wrap('th'),
    strong: wrap('strong'),
    em: wrap('em'),
    blockquote: wrap('blockquote'),
  };
}

function dedupeByFile(citations: Citation[]) {
  const seen = new Map<string, { citation: Citation; indices: number[] }>();
  for (const c of citations) {
    const key = c.file_id || c.filename;
    const existing = seen.get(key);
    if (existing) existing.indices.push(c.index);
    else seen.set(key, { citation: c, indices: [c.index] });
  }
  return Array.from(seen.values());
}

function CitationsList({
  citations,
  highlightIndex,
}: {
  citations: Citation[];
  highlightIndex: number | null;
}) {
  const grouped = dedupeByFile(citations);
  return (
    <div className="mt-3 rounded-lg border border-bg-300 bg-bg-100/70 p-3">
      <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-text-400">
        Sources
      </div>
      <ol className="space-y-1.5">
        {grouped.map(({ citation, indices }) => {
          const isHighlight =
            highlightIndex != null && indices.includes(highlightIndex);
          return (
            <li
              key={citation.file_id || citation.filename}
              id={`cite-${indices[0]}`}
              className={`flex items-start gap-2 rounded-md px-2 py-1 text-xs transition ${
                isHighlight ? 'bg-accent/10 ring-1 ring-accent/40' : ''
              }`}
            >
              <span className="mt-0.5 inline-flex shrink-0 items-center gap-1">
                {indices.map((i) => (
                  <span
                    key={i}
                    className="inline-flex h-5 min-w-[20px] items-center justify-center rounded bg-accent/15 px-1 font-semibold text-accent"
                  >
                    {i}
                  </span>
                ))}
              </span>
              <FileText className="mt-0.5 h-3.5 w-3.5 shrink-0 text-text-400" />
              <span className="flex-1 truncate text-text-200" title={citation.filename}>
                {citation.filename}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

export function AssistantMessage({ msg }: { msg: AssistantMsg }) {
  const [highlight, setHighlight] = useState<number | null>(null);

  const isStreaming = !!msg.streaming;
  const displayedContent = useTypewriter(msg.content, isStreaming);
  // hasContent follows the real buffer (not the typewriter output) so the
  // content div mounts the instant text arrives from the stream; the
  // typewriter just controls how much of it is visible inside that div.
  const hasContent = msg.content.length > 0;
  const hasThinking = (msg.thinking ?? '').length > 0;
  const thinkingActive = isStreaming && hasThinking && !hasContent;

  const showInitialShimmer =
    isStreaming && !hasContent && !hasThinking && !msg.toolStatus;

  function handleCite(index: number) {
    setHighlight(index);
    const el = document.getElementById(`cite-${index}`);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    window.setTimeout(() => setHighlight(null), 1600);
  }

  const mdComponents = makeMarkdownComponents(handleCite);

  return (
    <div className="flex gap-3">
      <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center overflow-hidden rounded-full border border-bg-300 bg-bg-100">
        <img
          src="/logo-short.png"
          alt="Agent"
          className="h-full w-full object-contain p-1"
        />
      </div>
      <div className="min-w-0 flex-1 space-y-2">
        {showInitialShimmer && <ShimmerText text="Thinking..." />}

        {(hasThinking || (isStreaming && !hasContent && !msg.toolStatus)) &&
          (hasThinking || !showInitialShimmer) && (
            <ThinkingPanel
              thinking={msg.thinking ?? ''}
              startedAt={msg.thinkingStartedAt}
              endedAt={msg.thinkingEndedAt}
              active={thinkingActive}
            />
          )}

        <ToolCallPill status={msg.toolStatus ?? null} query={msg.toolQuery ?? null} />

        {hasContent && (
          <div className="md-content">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>
              {displayedContent}
            </ReactMarkdown>
          </div>
        )}

        {msg.citations && msg.citations.length > 0 && hasContent && (
          <CitationsList citations={msg.citations} highlightIndex={highlight} />
        )}
      </div>
    </div>
  );
}
