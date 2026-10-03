import { Check, ChevronDown, Database, Sparkles } from 'lucide-react';
import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from 'react';

import { EASE_OUT_CSS } from '@/lib/motion';
import { cn } from '@/lib/utils';

export type ChainTool = { name?: string; query: string | null; done?: boolean; offset?: number };

export type ChainStep =
  | { kind: 'thinking'; text: string }
  | { kind: 'tool'; index: number; query: string | null; done: boolean };

/** Appends streamed reasoning to the trailing thinking step, or opens a new one after a tool call. */
export function appendThinking(steps: ChainStep[], text: string): ChainStep[] {
  if (!text) return steps;
  const last = steps[steps.length - 1];
  if (last && last.kind === 'thinking') {
    return [...steps.slice(0, -1), { kind: 'thinking', text: last.text + text }];
  }
  return [...steps, { kind: 'thinking', text }];
}

/** Rebuilds the chronological chain for a saved message by splitting reasoning at each tool call's offset. */
export function stepsFromHistory(thinking: string | null, tools: ChainTool[] | null): ChainStep[] {
  const text = thinking ?? '';
  const calls = (tools ?? []).filter(Boolean).map((t, i) => ({
    query: t.query,
    index: i,
    offset: typeof t.offset === 'number' ? Math.min(Math.max(0, t.offset), text.length) : text.length,
  }));
  calls.sort((a, b) => a.offset - b.offset);
  const steps: ChainStep[] = [];
  let cursor = 0;
  for (const c of calls) {
    if (c.offset > cursor) steps.push({ kind: 'thinking', text: text.slice(cursor, c.offset) });
    steps.push({ kind: 'tool', index: c.index, query: c.query, done: true });
    cursor = Math.max(cursor, c.offset);
  }
  if (cursor < text.length) steps.push({ kind: 'thinking', text: text.slice(cursor) });
  return steps.filter((s) => s.kind === 'tool' || s.text.trim());
}

/** Gradient sweep across a label; used for the live "Thinking" / "Searching" state. */
export function Shimmer({ text, className }: { text: string; className?: string }) {
  return (
    <span
      className={cn('whitespace-nowrap bg-clip-text text-transparent', className)}
      style={{
        backgroundImage: 'linear-gradient(90deg, var(--text-400) 35%, var(--accent) 50%, var(--text-400) 65%)',
        backgroundSize: '200% 100%',
        animation: 'shimmer-text 1.6s linear infinite',
      }}
    >
      {text}
    </span>
  );
}

/** Builds the collapsed summary line for a finished reasoning chain. */
function summary(toolCount: number, seconds: number | null) {
  const searched = toolCount === 0 ? '' : `searched ${toolCount} ${toolCount === 1 ? 'time' : 'times'}`;
  if (seconds == null) {
    return toolCount === 0
      ? 'Show reasoning'
      : `Reasoned and searched the knowledge base ${toolCount} ${toolCount === 1 ? 'time' : 'times'}`;
  }
  const secs = Math.max(1, Math.round(seconds));
  const time = `Thought for ${secs} second${secs === 1 ? '' : 's'}`;
  return searched ? `${time}, ${searched}` : time;
}

/** Collapsible, chronological trace of reasoning stretches and knowledge-base searches. */
export function ThinkingChain({ steps, pending }: { steps: ChainStep[]; pending: boolean }) {
  const [manual, setManual] = useState<boolean | null>(null);
  const liveRef = useRef(pending);
  const startedAt = useRef(Date.now());
  const [seconds, setSeconds] = useState<number | null>(pending ? 0 : null);
  const traceRef = useRef<HTMLDivElement>(null);
  const [lineHeight, setLineHeight] = useState(0);

  const expanded = manual ?? pending;
  const toolCount = steps.filter((s) => s.kind === 'tool').length;
  const last = steps[steps.length - 1];
  const activeLabel = last?.kind === 'tool' && !last.done ? 'Searching the knowledge base' : 'Thinking';

  useEffect(() => {
    if (!liveRef.current) return;
    if (!pending) {
      setSeconds((Date.now() - startedAt.current) / 1000);
      return;
    }
    const t = window.setInterval(() => setSeconds((Date.now() - startedAt.current) / 1000), 1000);
    return () => window.clearInterval(t);
  }, [pending]);

  useEffect(() => {
    if (!pending) setManual(null);
  }, [pending]);

  useLayoutEffect(() => {
    if (traceRef.current) setLineHeight(traceRef.current.offsetHeight);
  }, [steps, expanded, pending]);

  if (!pending && steps.length === 0) return null;

  const rowAnim = (i: number): CSSProperties => ({
    animation: `fade-up 320ms ${EASE_OUT_CSS} ${pending ? 0 : Math.min(i, 6) * 60}ms both`,
  });

  return (
    <div className="mb-3 flex w-full flex-col">
      <button
        type="button"
        aria-expanded={expanded}
        onClick={() => setManual(!expanded)}
        className="-mx-2 flex w-fit items-center gap-2 rounded-lg px-2 py-1 transition-colors duration-100 hover:bg-bg-200"
      >
        <Sparkles className={cn('h-3.5 w-3.5', pending ? 'text-accent' : 'text-text-400')} fill="currentColor" />
        {pending ? (
          <Shimmer
            text={seconds ? `${activeLabel} · ${Math.max(1, Math.round(seconds))}s` : activeLabel}
            className="text-[13px] font-medium"
          />
        ) : (
          <span
            className="whitespace-nowrap text-[13px] font-medium text-text-400"
            style={{ animation: 'fade-in-soft 350ms ease-out both' }}
          >
            {summary(toolCount, seconds)}
          </span>
        )}
        <ChevronDown
          className={cn('h-3.5 w-3.5 text-text-400 transition-transform duration-300', expanded && 'rotate-180')}
        />
      </button>

      <div
        className="grid transition-[grid-template-rows,opacity] duration-[400ms]"
        style={{
          gridTemplateRows: expanded ? '1fr' : '0fr',
          opacity: expanded ? 1 : 0,
          transitionTimingFunction: EASE_OUT_CSS,
        }}
      >
        <div className="overflow-hidden">
          <div className="relative ml-[5px] mt-1 pl-4">
            <span
              aria-hidden
              className="absolute left-[3px] w-px bg-bg-300"
              style={{ top: -8, height: lineHeight ? lineHeight - 2 : 0, transition: `height 500ms ${EASE_OUT_CSS}` }}
            />
            <div ref={traceRef} className="flex flex-col gap-1.5 py-1">
              {steps.length === 0 && pending && (
                <div
                  className="flex min-h-7 items-center gap-2 px-1.5 text-[12.5px] text-text-400"
                  style={rowAnim(0)}
                >
                  <Spinner />
                  Reading the question
                </div>
              )}
              {steps.map((step, i) => {
                if (step.kind === 'thinking') {
                  return (
                    <ThinkingRow
                      key={`t-${i}`}
                      text={step.text}
                      live={pending && i === steps.length - 1}
                      style={rowAnim(i)}
                    />
                  );
                }
                return (
                  <div
                    key={`c-${i}`}
                    style={rowAnim(i)}
                    className="flex min-h-7 w-full items-center gap-2 rounded-md px-1.5 py-0.5"
                  >
                    {step.done ? (
                      <Check className="h-3.5 w-3.5 shrink-0 text-accent" strokeWidth={2.5} />
                    ) : (
                      <Spinner />
                    )}
                    <Database className="h-3 w-3 shrink-0 text-text-400" />
                    <span className="shrink-0 text-[12.5px] font-medium text-text-200">
                      {step.done ? 'Searched' : 'Searching'} the knowledge base
                    </span>
                    {step.query && (
                      <span className="min-w-0 truncate text-[11.5px] text-text-400">“{step.query}”</span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/** Small ring spinner for in-flight steps. */
function Spinner() {
  return (
    <span
      className="h-3 w-3 shrink-0 rounded-full border-[1.5px] border-bg-300 border-t-accent"
      style={{ animation: 'spin-ring 700ms linear infinite' }}
    />
  );
}

/** Reasoning text that auto-scrolls to the newest token while live. */
function ThinkingRow({ text, live, style }: { text: string; live: boolean; style: CSSProperties }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (live && ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [text, live]);
  if (!text.trim()) return null;
  return (
    <div style={style} className="w-full rounded-md px-1.5 py-0.5">
      <div
        ref={ref}
        className={cn(
          'scrollbar-none overflow-y-auto whitespace-pre-wrap font-mono text-[12px] italic leading-relaxed text-text-400',
          live ? 'max-h-40' : 'max-h-72',
        )}
      >
        {text.trim()}
      </div>
    </div>
  );
}
