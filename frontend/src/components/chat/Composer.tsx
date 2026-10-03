import { AnimatePresence, motion } from 'motion/react';
import { AlertCircle, ArrowUp, FileText, ImagePlus, Paperclip, Plus, Square, Upload, X } from 'lucide-react';
import { useEffect, useRef, useState, type DragEvent, type ReactNode } from 'react';

import { InfoTip } from '@/components/ui/InfoTip';
import { cn } from '@/lib/utils';
import { DOC_ACCEPT, IMAGE_ACCEPT, MAX_DOCS, MAX_IMAGES, formatBytes, type PendingAttachment } from './attachments';

/** Auto-growing composer with attachments (menu, paste, drop) and a send/stop button. */
export function Composer({
  value,
  onChange,
  onSubmit,
  onStop,
  streaming,
  attachments,
  onAddFiles,
  onRemoveAttachment,
  placeholder = 'Ask anything about your documents…',
  autoFocus,
}: {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  streaming: boolean;
  attachments: PendingAttachment[];
  onAddFiles: (files: File[]) => void;
  onRemoveAttachment: (key: string) => void;
  placeholder?: string;
  autoFocus?: boolean;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);
  const imageInput = useRef<HTMLInputElement>(null);
  const docInput = useRef<HTMLInputElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [dragging, setDragging] = useState(false);
  const dragDepth = useRef(0);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  useEffect(() => {
    if (autoFocus) ref.current?.focus();
  }, [autoFocus]);

  useEffect(() => {
    if (!menuOpen) return;
    const onDown = (e: MouseEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) setMenuOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [menuOpen]);

  const uploading = attachments.some((a) => a.status === 'uploading');
  const readyCount = attachments.filter((a) => a.status === 'ready').length;
  const canSend = (value.trim().length > 0 || readyCount > 0) && !streaming && !uploading;

  const pick = (input: HTMLInputElement | null) => {
    setMenuOpen(false);
    input?.click();
  };

  const onDrag = (e: DragEvent, kind: 'enter' | 'leave' | 'over' | 'drop') => {
    if (!e.dataTransfer.types.includes('Files')) return;
    e.preventDefault();
    if (kind === 'enter') {
      dragDepth.current += 1;
      setDragging(true);
    } else if (kind === 'leave') {
      dragDepth.current = Math.max(0, dragDepth.current - 1);
      if (dragDepth.current === 0) setDragging(false);
    } else if (kind === 'drop') {
      dragDepth.current = 0;
      setDragging(false);
      const files = Array.from(e.dataTransfer.files);
      if (files.length) onAddFiles(files);
    }
  };

  return (
    <div
      onDragEnter={(e) => onDrag(e, 'enter')}
      onDragLeave={(e) => onDrag(e, 'leave')}
      onDragOver={(e) => onDrag(e, 'over')}
      onDrop={(e) => onDrag(e, 'drop')}
      className="relative rounded-[26px] border border-border bg-bg-100 p-1 shadow-[0_10px_30px_-16px_rgba(70,55,25,0.25)] transition-shadow duration-200 focus-within:border-accent/30 focus-within:shadow-[0_14px_34px_-14px_rgba(15,94,94,0.28)]"
    >
      <div className="rounded-[22px] bg-bg-0">
        <AnimatePresence initial={false}>
          {attachments.length > 0 && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.22, ease: [0.23, 1, 0.32, 1] }}
              className="overflow-hidden"
            >
              <div className="scrollbar-none flex gap-2 overflow-x-auto px-3 pb-1 pt-3">
                <AnimatePresence initial={false}>
                  {attachments.map((a) => (
                    <AttachmentChip key={a.key} item={a} onRemove={() => onRemoveAttachment(a.key)} />
                  ))}
                </AnimatePresence>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <textarea
          ref={ref}
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onPaste={(e) => {
            const files = Array.from(e.clipboardData.items)
              .filter((i) => i.kind === 'file')
              .map((i) => i.getAsFile())
              .filter((f): f is File => !!f);
            if (files.length) {
              e.preventDefault();
              onAddFiles(files);
            }
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              if (canSend) onSubmit();
            }
          }}
          placeholder={placeholder}
          className="block w-full resize-none bg-transparent px-4 pb-2 pt-3.5 text-[14.5px] leading-relaxed text-text-100 outline-none placeholder:text-text-500"
        />

        <div className="flex items-center gap-1 px-2.5 pb-2.5">
          <div ref={menuRef} className="relative">
            <button
              type="button"
              onClick={() => setMenuOpen((v) => !v)}
              title="Attach"
              className={cn(
                'flex h-9 w-9 items-center justify-center rounded-full text-text-300 transition-all duration-200 hover:bg-bg-200 hover:text-accent',
                menuOpen && 'bg-bg-200 text-accent',
              )}
            >
              <Plus className={cn('h-4 w-4 transition-transform duration-200', menuOpen && 'rotate-45')} />
            </button>
            <AnimatePresence>
              {menuOpen && (
                <motion.div
                  initial={{ opacity: 0, y: 6, scale: 0.97 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 4, scale: 0.98 }}
                  transition={{ duration: 0.16, ease: [0.23, 1, 0.32, 1] }}
                  className="panel absolute bottom-full left-0 z-30 mb-2 w-60 origin-bottom-left p-1.5"
                >
                  <MenuItem
                    icon={<ImagePlus className="h-4 w-4" />}
                    title="Upload image"
                    hint={`PNG, JPG, WEBP, GIF · up to ${MAX_IMAGES}, 5 MB each`}
                    onClick={() => pick(imageInput.current)}
                  />
                  <MenuItem
                    icon={<Paperclip className="h-4 w-4" />}
                    title="Upload document"
                    hint={`PDF, DOCX, XLSX, TXT, MD, CSV… · up to ${MAX_DOCS}, 10 MB each`}
                    onClick={() => pick(docInput.current)}
                  />
                </motion.div>
              )}
            </AnimatePresence>
            <input
              ref={imageInput}
              type="file"
              multiple
              accept={IMAGE_ACCEPT}
              className="hidden"
              onChange={(e) => {
                if (e.target.files) onAddFiles(Array.from(e.target.files));
                e.target.value = '';
              }}
            />
            <input
              ref={docInput}
              type="file"
              multiple
              accept={DOC_ACCEPT}
              className="hidden"
              onChange={(e) => {
                if (e.target.files) onAddFiles(Array.from(e.target.files));
                e.target.value = '';
              }}
            />
          </div>

          <InfoTip side="top" width="w-72" className="ml-0.5">
            Enter to send · Shift+Enter for a new line. Paste images with Ctrl+V or drop files onto this box to attach them.
            <span className="mt-1.5 block opacity-70">AI can make mistakes. Check important information against the cited sources.</span>
          </InfoTip>
          <AnimatePresence>
            {(streaming || uploading) && (
              <motion.span
                initial={{ opacity: 0, x: -4 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.16 }}
                className="ml-1 text-[11px] text-text-500"
              >
                {streaming ? 'Generating answer…' : 'Uploading attachments…'}
              </motion.span>
            )}
          </AnimatePresence>
          <button
            type="button"
            onClick={() => (streaming ? onStop() : canSend && onSubmit())}
            disabled={!streaming && !canSend}
            title={streaming ? 'Stop' : 'Send'}
            className={cn(
              'ml-auto flex h-9 w-9 items-center justify-center rounded-full transition-all duration-200 active:scale-95',
              streaming
                ? 'bg-inverse text-inverse-fg hover:opacity-90'
                : canSend
                  ? 'bg-accent text-white shadow-[0_6px_14px_-6px_rgba(15,94,94,0.8)] hover:bg-accent-hover'
                  : 'cursor-default bg-accent/20 text-white',
            )}
          >
            <AnimatePresence mode="wait" initial={false}>
              <motion.span
                key={streaming ? 'stop' : 'send'}
                initial={{ opacity: 0, scale: 0.6, rotate: -45 }}
                animate={{ opacity: 1, scale: 1, rotate: 0 }}
                exit={{ opacity: 0, scale: 0.6, rotate: 45 }}
                transition={{ duration: 0.16 }}
                className="flex"
              >
                {streaming ? <Square className="h-3 w-3 fill-current" /> : <ArrowUp className="h-4 w-4" />}
              </motion.span>
            </AnimatePresence>
          </button>
        </div>
      </div>

      <AnimatePresence>
        {dragging && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="pointer-events-none absolute inset-0 z-20 flex flex-col items-center justify-center rounded-[26px] border-2 border-dashed border-accent bg-bg-0/90 backdrop-blur-sm"
          >
            <Upload className="mb-1.5 h-6 w-6 text-accent" />
            <p className="text-[13px] font-medium text-accent">Drop images or documents to attach</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/** One row in the attach popover menu. */
function MenuItem({
  icon,
  title,
  hint,
  onClick,
}: {
  icon: ReactNode;
  title: string;
  hint: string;
  onClick: () => void;
}) {
  return (
    <div className="flex w-full items-center gap-1 rounded-xl pr-1.5 transition-colors hover:bg-bg-200">
      <button type="button" onClick={onClick} className="flex min-w-0 flex-1 items-center gap-3 px-2.5 py-2 text-left">
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-accent">
          {icon}
        </span>
        <span className="truncate text-[13px] font-medium text-text-100">{title}</span>
      </button>
      <InfoTip side="right" width="w-56">
        {hint}
      </InfoTip>
    </div>
  );
}

/** Preview chip for one pending attachment: thumbnail or document card with progress / error state. */
function AttachmentChip({ item, onRemove }: { item: PendingAttachment; onRemove: () => void }) {
  const uploading = item.status === 'uploading';
  const failed = item.status === 'error';
  const pct = Math.round(item.progress * 100);
  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.9, y: 6 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.9 }}
      transition={{ duration: 0.2, ease: [0.23, 1, 0.32, 1] }}
      title={failed ? `${item.name}: ${item.error}` : item.name}
      className="group relative shrink-0"
    >
      {item.kind === 'image' && item.previewUrl && !failed ? (
        <div className="relative h-16 w-16 overflow-hidden rounded-xl border border-border bg-bg-200">
          <img src={item.previewUrl} alt={item.name} className="h-full w-full object-cover" />
          {uploading && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/40">
              <ProgressRing value={item.progress} />
            </div>
          )}
        </div>
      ) : (
        <div
          className={cn(
            'flex h-16 w-56 items-center gap-2.5 rounded-xl border px-2.5',
            failed ? 'border-danger/30 bg-danger-soft' : 'border-border bg-bg-100',
          )}
        >
          <span
            className={cn(
              'flex h-9 w-9 shrink-0 items-center justify-center rounded-lg',
              failed ? 'bg-danger/10 text-danger' : 'bg-accent/10 text-accent',
            )}
          >
            {failed ? <AlertCircle className="h-4 w-4" /> : <FileText className="h-4 w-4" />}
          </span>
          <span className="min-w-0 flex-1">
            <span className={cn('block truncate text-[12px] font-medium', failed ? 'text-danger' : 'text-text-100')}>
              {item.name}
            </span>
            {failed ? (
              <span className="block truncate text-[10.5px] text-danger/80">{item.error}</span>
            ) : uploading ? (
              <span className="mt-1 block h-1 overflow-hidden rounded-full bg-bg-200">
                <span className="block h-full rounded-full bg-accent transition-all duration-200" style={{ width: `${pct}%` }} />
              </span>
            ) : (
              <span className="block text-[10.5px] text-text-400">
                {item.name.split('.').pop()?.toUpperCase()} · {formatBytes(item.size)}
                {item.meta?.truncated ? ' · truncated' : ''}
              </span>
            )}
          </span>
        </div>
      )}
      <button
        type="button"
        onClick={onRemove}
        title="Remove"
        className="absolute -right-1.5 -top-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-inverse text-inverse-fg opacity-0 shadow transition-opacity duration-150 hover:bg-danger hover:text-white group-hover:opacity-100"
      >
        <X className="h-3 w-3" />
      </button>
    </motion.div>
  );
}

/** Circular upload progress indicator. */
function ProgressRing({ value }: { value: number }) {
  const r = 11;
  const c = 2 * Math.PI * r;
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" className="-rotate-90">
      <circle cx="14" cy="14" r={r} fill="none" stroke="rgba(255,255,255,0.35)" strokeWidth="2.5" />
      <circle
        cx="14"
        cy="14"
        r={r}
        fill="none"
        stroke="white"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeDasharray={c}
        strokeDashoffset={c * (1 - Math.max(0.04, value))}
        style={{ transition: 'stroke-dashoffset 200ms ease-out' }}
      />
    </svg>
  );
}