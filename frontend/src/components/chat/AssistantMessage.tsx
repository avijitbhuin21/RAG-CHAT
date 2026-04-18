import { ChevronDown, Search } from 'lucide-react';
import {
  cloneElement,
  Fragment,
  isValidElement,
  ReactNode,
  useEffect,
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

export type ToolCall = {
  name?: string;
  query: string | null;
  // Live-streaming state only. Persisted calls loaded from history are always
  // treated as done (omitted → done).
  done?: boolean;
};

export type AssistantMsg = {
  id: string;
  role: 'assistant';
  content: string;
  thinking: string | null;
  citations: Citation[] | null;
  tool_calls: ToolCall[] | null;
  streaming?: boolean;
  thinkingStartedAt?: number | null;
  thinkingEndedAt?: number | null;
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

// A single row in the vertical "reasoning timeline" that sits above the
// answer. Renders a dot in the left gutter and a connecting vertical line
// down to the next row (suppressed on the last row). Keeps alignment stable
// regardless of whether the row's content is one line or expands into a
// multi-line dropdown body.
function TimelineRow({
  done,
  active,
  isLast,
  showMarker,
  children,
}: {
  done: boolean;
  active: boolean;
  isLast: boolean;
  showMarker: boolean;
  children: ReactNode;
}) {
  if (!showMarker) {
    return <div className="pb-3 last:pb-0">{children}</div>;
  }
  return (
    <div className="relative flex gap-3 pb-3 last:pb-0">
      {!isLast && (
        <span
          className="absolute left-[5px] top-[18px] w-px bg-bg-300"
          style={{ bottom: 0 }}
          aria-hidden
        />
      )}
      <span className="relative z-10 mt-[7px] flex h-2.5 w-2.5 shrink-0 items-center justify-center">
        <span
          className={`h-2.5 w-2.5 rounded-full ${
            done
              ? 'bg-accent'
              : active
                ? 'bg-accent/40 ring-2 ring-accent/30 animate-pulse'
                : 'border border-accent bg-bg-0'
          }`}
        />
      </span>
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  );
}

function ThoughtStep({
  thinking,
  startedAt,
  endedAt,
  active,
  showChevron,
}: {
  thinking: string;
  startedAt: number | null | undefined;
  endedAt: number | null | undefined;
  active: boolean;
  showChevron: boolean;
}) {
  const frozen = endedAt && startedAt ? (endedAt - startedAt) / 1000 : null;
  const elapsed = useElapsedSeconds(startedAt, frozen);
  // Floor at 1 — sub-second counters read as glitchy ("Thought for 0s").
  const seconds = Math.max(1, Math.round(elapsed));
  const hasTiming = !!startedAt;
  // Auto-expand while actively thinking, auto-collapse once done; user click
  // overrides in either direction.
  const [userOverride, setUserOverride] = useState<boolean | null>(null);
  const open = userOverride != null ? userOverride : active;

  const label = active ? (
    <ShimmerText text={hasTiming ? `Thinking ${seconds}s` : 'Thinking…'} />
  ) : (
    <span className="text-sm text-text-300">
      {hasTiming ? `Thought for ${seconds}s` : 'Thinking'}
    </span>
  );

  return (
    <div>
      {showChevron ? (
        <button
          type="button"
          onClick={() => setUserOverride(!open)}
          className="flex items-center gap-1 text-left transition hover:text-text-100"
        >
          {label}
          <ChevronDown
            className={`h-3.5 w-3.5 text-text-400 transition-transform ${
              open ? '' : '-rotate-90'
            }`}
          />
        </button>
      ) : (
        <div className="flex items-center gap-1">{label}</div>
      )}
      {open && thinking && (
        <div className="mt-1.5 whitespace-pre-wrap text-xs italic leading-relaxed text-text-400">
          {thinking}
        </div>
      )}
    </div>
  );
}

function ToolStep({
  call,
}: {
  call: ToolCall;
}) {
  const { query, done } = call;
  const active = done === false;
  return (
    <div className="flex items-center gap-2 py-[2px] text-sm text-text-300">
      <Search className="h-3 w-3 shrink-0 text-accent" />
      {active ? (
        <ShimmerText
          text={`Searching knowledge base${query ? `: "${truncate(query, 40)}"` : '…'}`}
        />
      ) : (
        <span>
          Searched knowledge base
          {query ? (
            <>
              : <span className="text-text-200">"{truncate(query, 40)}"</span>
            </>
          ) : (
            ''
          )}
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
    table: ({ children, ...rest }: any) => (
      <div className="md-table-wrap">
        <table {...rest}>{children}</table>
      </div>
    ),
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
  const toolCalls = msg.tool_calls ?? [];
  const hasTools = toolCalls.length > 0;

  // Show the Thought row whenever we have any thinking content OR we're
  // still waiting on the first token of anything else (so there's something
  // on screen while the model warms up). Active = not yet finished its work.
  const showThoughtRow =
    hasThinking || (isStreaming && !hasContent && !hasTools);
  const thoughtActive = isStreaming && !hasContent;
  const thoughtDone = !thoughtActive;

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
      <div className="min-w-0 flex-1 space-y-2 pt-2">
        {(showThoughtRow || hasTools) && (
          <div className="space-y-0">
            {showThoughtRow && (
              <TimelineRow
                done={thoughtDone}
                active={thoughtActive && !hasThinking}
                isLast={!hasTools && !hasContent}
                showMarker={hasContent}
              >
                <ThoughtStep
                  thinking={msg.thinking ?? ''}
                  startedAt={msg.thinkingStartedAt}
                  endedAt={msg.thinkingEndedAt}
                  active={thoughtActive}
                  showChevron={hasContent}
                />
              </TimelineRow>
            )}
            {toolCalls.map((call, i) => {
              const isLastTool = i === toolCalls.length - 1;
              const done = call.done !== false;
              return (
                <TimelineRow
                  key={i}
                  done={done}
                  active={!done}
                  isLast={isLastTool && !hasContent}
                  showMarker={hasContent}
                >
                  <ToolStep call={call} />
                </TimelineRow>
              );
            })}
          </div>
        )}

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
