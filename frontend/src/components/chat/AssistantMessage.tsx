import { ChevronDown, Search } from 'lucide-react';
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
  // Every retrieved chunk from this source file, used client-side to locate
  // and highlight the cited passages inside the document in the side panel.
  // `chunk_text` (single string) and `snippet` (truncated) are older formats
  // kept as fallbacks when reading messages saved before the per-file change.
  chunk_texts?: string[];
  chunk_text?: string;
  snippet?: string;
};

export type OpenSourceFn = (c: Citation) => void;

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
    <button
      type="button"
      className="citation-sup"
      onClick={(e) => {
        e.preventDefault();
        onClick(index);
      }}
    >
      {index}
    </button>
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

export function AssistantMessage({
  msg,
  onOpenSource,
}: {
  msg: AssistantMsg;
  onOpenSource: OpenSourceFn;
}) {
  const isStreaming = !!msg.streaming;
  const hasContent = msg.content.length > 0;
  const hasThinking = (msg.thinking ?? '').length > 0;
  const thinkingActive = isStreaming && hasThinking && !hasContent;

  const showInitialShimmer =
    isStreaming && !hasContent && !hasThinking && !msg.toolStatus;

  function handleCite(index: number) {
    const c = msg.citations?.find((x) => x.index === index);
    if (c) onOpenSource(c);
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
              {msg.content}
            </ReactMarkdown>
          </div>
        )}

      </div>
    </div>
  );
}
