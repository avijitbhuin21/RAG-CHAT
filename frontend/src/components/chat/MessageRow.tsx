import { motion } from 'motion/react';
import { Check, Copy, FileSpreadsheet, FileText, FileType, Presentation } from 'lucide-react';
import { cloneElement, Fragment, isValidElement, memo, useState, type ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

import { MESSAGE_IN } from '@/lib/motion';
import { cn } from '@/lib/utils';
import { ThinkingChain, stepsFromHistory, type ChainStep, type ChainTool } from './ThinkingChain';
import { attachmentUrl, formatBytes, type AttachmentMeta } from './attachments';

export type Citation = {
  index: number;
  filename: string;
  file_id: string | null;
  chunk_texts?: string[];
  chunk_text?: string;
  snippet?: string;
};

export type ToolCall = ChainTool;

export type ChatMsg = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  thinking: string | null;
  citations: Citation[] | null;
  tool_calls: ToolCall[] | null;
  steps?: ChainStep[];
  attachments?: AttachmentMeta[] | null;
  streaming?: boolean;
  error?: string | null;
  startedAt?: number;
  endedAt?: number;
};

/** One transcript turn: a user bubble or a streamed, cited assistant answer. */
export function MessageRow({
  msg,
  onOpenSource,
  activeFileId,
}: {
  msg: ChatMsg;
  onOpenSource: (c: Citation) => void;
  activeFileId: string | null;
}) {
  if (msg.role === 'user') {
    const atts = msg.attachments ?? [];
    return (
      <motion.div {...MESSAGE_IN} className="flex flex-col items-end gap-2">
        {atts.length > 0 && <UserAttachments items={atts} />}
        {msg.content && (
          <div className="max-w-[80%] whitespace-pre-wrap break-words rounded-2xl rounded-br-md bg-accent px-4 py-3 font-mono text-[13.5px] leading-relaxed text-white shadow-[0_8px_20px_-12px_rgba(15,94,94,0.7)]">
            {msg.content}
          </div>
        )}
      </motion.div>
    );
  }

  const streaming = !!msg.streaming;
  const hasContent = msg.content.length > 0;
  const citations = msg.citations ?? [];

  return (
    <motion.div {...MESSAGE_IN} className="min-w-0">
      <ThinkingChain
        steps={msg.steps ?? stepsFromHistory(msg.thinking, msg.tool_calls)}
        pending={streaming && !hasContent}
      />

      {hasContent && (
        <div className="md-content">
          <AnswerBody content={msg.content} citations={citations} onCite={onOpenSource} />
          {streaming && (
            <span
              className="ml-0.5 inline-block h-4 w-[3px] rounded-full bg-accent align-middle"
              style={{ animation: 'caret-blink 1s ease-in-out infinite' }}
            />
          )}
        </div>
      )}

      {msg.error && (
        <div className="mt-3 rounded-xl border border-danger/20 bg-danger-soft px-3 py-2.5 text-[13px] text-danger">
          {msg.error}
        </div>
      )}

      {!streaming && hasContent && (
        <div style={{ animation: 'fade-up 360ms cubic-bezier(0.23, 1, 0.32, 1) both' }}>
          <SourceStrip citations={citations} onSelect={onOpenSource} activeFileId={activeFileId} />
          <AnswerActions msg={msg} />
        </div>
      )}
    </motion.div>
  );
}

const CitationPill = ({ index, onClick }: { index: number; onClick: (i: number) => void }) => (
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

/** Recursively swaps `[n]` markers inside rendered markdown children for citation pills. */
function replaceCitations(node: ReactNode, onClick: (i: number) => void, key = 'c'): ReactNode {
  if (typeof node === 'string' || typeof node === 'number') {
    const str = String(node);
    if (!/\[\d+\]/.test(str)) return str;
    return str.split(/(\[\d+\])/g).map((p, i) => {
      const m = p.match(/^\[(\d+)\]$/);
      if (m) return <CitationPill key={`${key}-${i}`} index={parseInt(m[1], 10)} onClick={onClick} />;
      return p ? <Fragment key={`${key}-${i}`}>{p}</Fragment> : null;
    });
  }
  if (Array.isArray(node)) {
    return node.map((c, i) => <Fragment key={`${key}-${i}`}>{replaceCitations(c, onClick, `${key}-${i}`)}</Fragment>);
  }
  if (isValidElement<{ children?: ReactNode }>(node)) {
    return cloneElement(node, { ...node.props, children: replaceCitations(node.props.children, onClick, key) });
  }
  return node;
}

/** Builds react-markdown component overrides that inject citation pills. */
function makeComponents(onCite: (i: number) => void) {
  const wrap = (Tag: keyof JSX.IntrinsicElements) =>
    ({ children, node: _node, ...rest }: any) => {
      const El = Tag as any;
      return <El {...rest}>{replaceCitations(children, onCite)}</El>;
    };
  return {
    p: wrap('p'),
    li: wrap('li'),
    h1: wrap('h1'),
    h2: wrap('h2'),
    h3: wrap('h3'),
    h4: wrap('h4'),
    td: wrap('td'),
    th: wrap('th'),
    strong: wrap('strong'),
    em: wrap('em'),
    blockquote: wrap('blockquote'),
    table: ({ children, node: _node, ...rest }: any) => (
      <div className="md-table-wrap">
        <table {...rest}>{children}</table>
      </div>
    ),
  };
}

/** Markdown answer body; memoised so unchanged answers skip re-parsing. */
const AnswerBody = memo(function AnswerBody({
  content,
  citations,
  onCite,
}: {
  content: string;
  citations: Citation[];
  onCite: (c: Citation) => void;
}) {
  const handle = (i: number) => {
    const c = citations.find((x) => x.index === i);
    if (c) onCite(c);
  };
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={makeComponents(handle)}>
      {content}
    </ReactMarkdown>
  );
});

const FILE_STYLES: Record<string, { label: string; tone: string; Icon: typeof FileText }> = {
  pdf: { label: 'PDF', tone: 'bg-[#C0472F]/10 text-[#C0472F]', Icon: FileText },
  docx: { label: 'Word', tone: 'bg-[#2B579A]/10 text-[#2B579A]', Icon: FileText },
  pptx: { label: 'PowerPoint', tone: 'bg-[#C4652A]/10 text-[#C4652A]', Icon: Presentation },
  xlsx: { label: 'Excel', tone: 'bg-[#1E7145]/10 text-[#1E7145]', Icon: FileSpreadsheet },
  txt: { label: 'Text', tone: 'bg-text-100/5 text-text-300', Icon: FileType },
};

/** Compact row of file-type source icons; hovering one reveals its filename. */
function SourceStrip({
  citations,
  onSelect,
  activeFileId,
}: {
  citations: Citation[];
  onSelect: (c: Citation) => void;
  activeFileId: string | null;
}) {
  if (citations.length === 0) return null;
  const sorted = [...citations].sort((a, b) => a.index - b.index);
  return (
    <div className="mt-5 flex flex-wrap items-center gap-1.5">
      <span className="eyebrow mr-1.5">Sources</span>
      {sorted.map((c, i) => {
        const ext = (c.filename.split('.').pop() ?? '').toLowerCase();
        const style = FILE_STYLES[ext] ?? { label: ext.toUpperCase() || 'File', tone: 'bg-text-100/5 text-text-300', Icon: FileText };
        const active = !!c.file_id && c.file_id === activeFileId;
        return (
          <div
            key={c.index}
            className="group relative"
            style={{ animation: `fade-up 320ms cubic-bezier(0.23, 1, 0.32, 1) ${Math.min(i, 8) * 45}ms both` }}
          >
            <button
              type="button"
              disabled={!c.file_id}
              onClick={() => onSelect(c)}
              aria-label={`Source ${c.index}: ${c.filename}`}
              className={cn(
                'relative flex h-8 w-8 items-center justify-center rounded-lg transition-all duration-150 disabled:cursor-default',
                style.tone,
                active ? 'ring-2 ring-accent/50 ring-offset-1 ring-offset-bg-0' : 'hover:-translate-y-0.5 hover:shadow-[0_6px_14px_-8px_rgba(70,55,25,0.5)]',
              )}
            >
              <style.Icon className="h-4 w-4" />
              <span className="absolute -right-1 -top-1 flex h-3.5 min-w-3.5 items-center justify-center rounded-full bg-bg-100 px-0.5 font-sans text-[9px] font-semibold text-text-300 ring-1 ring-border">
                {c.index}
              </span>
            </button>
            <div className="pointer-events-none absolute bottom-full left-1/2 z-20 mb-2 w-max max-w-[260px] -translate-x-1/2 translate-y-1 rounded-lg bg-inverse px-2.5 py-1.5 text-left opacity-0 shadow-lg transition-all duration-150 group-hover:translate-y-0 group-hover:opacity-100">
              <div className="truncate font-sans text-[11.5px] font-medium text-inverse-fg">{c.filename}</div>
              <div className="font-sans text-[10px] text-inverse-fg/60">{style.label} · source {c.index}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/** Copy button plus source count and elapsed time for a finished answer. */
function AnswerActions({ msg }: { msg: ChatMsg }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    void navigator.clipboard.writeText(msg.content).catch(() => undefined);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  };
  const count = msg.citations?.length ?? 0;
  const elapsed = msg.startedAt && msg.endedAt ? Math.max(1, Math.round((msg.endedAt - msg.startedAt) / 1000)) : null;
  return (
    <div className="mt-4 flex items-center gap-1">
      <button
        type="button"
        onClick={copy}
        title="Copy answer"
        className="flex h-7 w-7 items-center justify-center rounded-full text-text-400 transition-colors hover:bg-bg-200 hover:text-accent"
      >
        {copied ? <Check className="h-3.5 w-3.5 text-accent" /> : <Copy className="h-3.5 w-3.5" />}
      </button>
      {(count > 0 || elapsed) && (
        <span className="ml-auto text-[11px] tabular-nums text-text-400">
          {count > 0 && `${count} source${count === 1 ? '' : 's'}`}
          {count > 0 && elapsed && ' · '}
          {elapsed && `${elapsed}s`}
        </span>
      )}
    </div>
  );
}

/** Image thumbnails and document cards shown above a user's message. */
function UserAttachments({ items }: { items: AttachmentMeta[] }) {
  const images = items.filter((a) => a.kind === 'image');
  const docs = items.filter((a) => a.kind === 'document');
  return (
    <div className="flex max-w-[80%] flex-col items-end gap-2">
      {images.length > 0 && (
        <div className="flex flex-wrap justify-end gap-2">
          {images.map((a, i) => {
            const src = a.previewUrl ?? attachmentUrl(a.id);
            return (
              <a
                key={a.id}
                href={attachmentUrl(a.id)}
                target="_blank"
                rel="noreferrer"
                title={a.filename}
                style={{ animation: `fade-up 320ms cubic-bezier(0.23, 1, 0.32, 1) ${i * 50}ms both` }}
                className="block overflow-hidden rounded-2xl border border-border bg-bg-200 shadow-[0_8px_20px_-14px_rgba(70,55,25,0.5)] transition-transform duration-200 hover:-translate-y-0.5"
              >
                <img
                  src={src}
                  alt={a.filename}
                  loading="lazy"
                  className={cn('object-cover', images.length === 1 ? 'max-h-64 max-w-[320px]' : 'h-28 w-28')}
                />
              </a>
            );
          })}
        </div>
      )}
      {docs.length > 0 && (
        <div className="flex flex-wrap justify-end gap-2">
          {docs.map((a, i) => {
            const ext = (a.filename.split('.').pop() ?? '').toLowerCase();
            const style = FILE_STYLES[ext] ?? { label: ext.toUpperCase() || 'File', tone: 'bg-text-100/5 text-text-300', Icon: FileText };
            return (
              <a
                key={a.id}
                href={attachmentUrl(a.id)}
                title={`Download ${a.filename}`}
                style={{ animation: `fade-up 320ms cubic-bezier(0.23, 1, 0.32, 1) ${i * 50}ms both` }}
                className="flex max-w-[260px] items-center gap-2.5 rounded-xl border border-border bg-bg-100 px-2.5 py-2 transition-all duration-150 hover:-translate-y-0.5 hover:border-accent/30"
              >
                <span className={cn('flex h-8 w-8 shrink-0 items-center justify-center rounded-lg', style.tone)}>
                  <style.Icon className="h-4 w-4" />
                </span>
                <span className="min-w-0">
                  <span className="block truncate text-[12px] font-medium text-text-100">{a.filename}</span>
                  <span className="block text-[10.5px] text-text-400">
                    {style.label} · {formatBytes(a.size)}
                    {a.truncated ? ' · truncated' : ''}
                  </span>
                </span>
              </a>
            );
          })}
        </div>
      )}
    </div>
  );
}