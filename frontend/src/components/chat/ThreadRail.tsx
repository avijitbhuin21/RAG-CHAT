import { AnimatePresence, motion } from 'motion/react';
import { LogOut, MessageSquarePlus, Search, Trash2, X } from 'lucide-react';
import { useEffect, useRef, useState, type ReactNode } from 'react';

import { ThemeToggle } from '@/components/ui/ThemeToggle';
import { SLIDE } from '@/lib/motion';
import { cn, groupByAge, relativeTime } from '@/lib/utils';

export type RailThread = { id: string; title: string; updated_at: string };

export type RailAccount = { name: string; email: string; initials: string; photo: string | null };

/** Left rail with new-chat, expanding search, age-grouped history and the account row. */
export function ThreadRail({
  threads,
  activeId,
  account,
  onOpen,
  onNew,
  onDelete,
  onSignOut,
}: {
  threads: RailThread[];
  activeId: string | null;
  account: RailAccount;
  onOpen: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onSignOut: () => void;
}) {
  const [query, setQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const rowRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (searching) inputRef.current?.focus();
  }, [searching]);

  useEffect(() => {
    if (!searching) return;
    const onDown = (e: MouseEvent) => {
      if (!rowRef.current?.contains(e.target as Node) && !query) setSearching(false);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [searching, query]);

  const closeSearch = () => {
    setSearching(false);
    setQuery('');
  };

  const q = query.trim().toLowerCase();
  const filtered = threads.filter((t) => (t.title || 'New chat').toLowerCase().includes(q));
  const groups = groupByAge(filtered, (t) => t.updated_at);

  return (
    <aside className="flex h-full w-[280px] shrink-0 flex-col gap-2">
      <div className="panel flex min-h-0 flex-1 flex-col">
        <div className="flex items-center gap-2.5 px-4 pb-1 pt-4">
          <img src="/logo-short.png" alt="" className="h-9 w-9 shrink-0 object-contain" />
          <div className="min-w-0 flex-1 leading-tight">
            <div className="font-serif text-[17px] font-semibold tracking-tight text-accent">1stAId4SME</div>
            <div className="text-[11px] text-text-400">AI for SMEs</div>
          </div>
          <ThemeToggle />
        </div>

        <div ref={rowRef} className="flex items-center gap-1.5 px-3 pt-4">
          <motion.button
            type="button"
            animate={{ flexGrow: searching ? 0 : 1 }}
            transition={SLIDE}
            style={{ flexBasis: 38, flexShrink: 0 }}
            onClick={() => {
              closeSearch();
              onNew();
            }}
            title="New chat"
            className="flex h-[38px] items-center overflow-hidden rounded-full bg-accent text-white shadow-[0_8px_18px_-10px_rgba(15,94,94,0.8)] transition-colors hover:bg-accent-hover"
          >
            <span className="flex h-[38px] w-[38px] shrink-0 items-center justify-center">
              <MessageSquarePlus className="h-4 w-4" />
            </span>
            <motion.span
              animate={{ opacity: searching ? 0 : 1 }}
              transition={SLIDE}
              className="whitespace-nowrap pr-4 text-[13px] font-medium"
            >
              New chat
            </motion.span>
          </motion.button>

          <motion.div
            animate={{ flexGrow: searching ? 1 : 0 }}
            transition={SLIDE}
            style={{ flexBasis: 38, flexShrink: 0 }}
            className="flex h-[38px] items-center overflow-hidden rounded-full border border-border bg-bg-0"
          >
            <button
              type="button"
              onClick={() => setSearching(true)}
              title="Search chats"
              disabled={searching}
              className={cn(
                'flex h-[36px] w-[36px] shrink-0 items-center justify-center text-text-400 transition-colors',
                !searching && 'hover:text-accent',
              )}
            >
              <Search className="h-4 w-4" />
            </button>
            <input
              ref={inputRef}
              value={query}
              tabIndex={searching ? 0 : -1}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Escape') closeSearch();
              }}
              placeholder="Search chats"
              className={cn(
                'min-w-0 flex-1 bg-transparent text-[13px] text-text-100 outline-none transition-opacity duration-200 placeholder:text-text-500',
                searching ? 'opacity-100' : 'pointer-events-none opacity-0',
              )}
            />
            <button
              type="button"
              onClick={closeSearch}
              title="Close search"
              tabIndex={searching ? 0 : -1}
              className={cn(
                'flex h-[36px] w-[36px] shrink-0 items-center justify-center text-text-400 transition-opacity duration-200 hover:text-text-100',
                searching ? 'opacity-100' : 'pointer-events-none opacity-0',
              )}
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </motion.div>
        </div>

        <div className="scrollbar-none mt-4 min-h-0 flex-1 overflow-y-auto px-2.5 pb-3">
          {searching && q && (
            <p className="px-2 pb-2 text-[11px] tabular-nums text-text-400">
              {filtered.length} of {threads.length} chats
            </p>
          )}
          {groups.map((g) => (
            <Group key={g.label} label={g.label}>
              {g.items.map((t) => (
                <ThreadItem
                  key={t.id}
                  thread={t}
                  active={t.id === activeId}
                  onOpen={onOpen}
                  onDelete={onDelete}
                />
              ))}
            </Group>
          ))}
          {threads.length === 0 && (
            <p className="px-2 py-6 text-center text-xs text-text-400">No chats yet</p>
          )}
          {threads.length > 0 && filtered.length === 0 && (
            <p className="px-2 py-6 text-center text-xs text-text-400">Nothing matches “{query}”.</p>
          )}
        </div>
      </div>

      <button
        type="button"
        onClick={onSignOut}
        title="Sign out"
        className="panel group flex shrink-0 items-center gap-3 p-3 text-left transition-colors hover:bg-bg-200"
      >
        <span className="flex h-9 w-9 shrink-0 items-center justify-center overflow-hidden rounded-full bg-accent text-xs font-semibold text-white">
          {account.photo ? (
            <img src={account.photo} alt="" className="h-full w-full object-cover" />
          ) : (
            account.initials
          )}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-[13px] font-medium text-text-100">{account.name}</span>
          <span className="block truncate text-[11px] text-text-400">{account.email}</span>
        </span>
        <LogOut className="h-4 w-4 shrink-0 text-text-400 transition-colors group-hover:text-accent" />
      </button>
    </aside>
  );
}

/** Labelled section of thread rows. */
function Group({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="mb-4">
      <div className="eyebrow px-2 pb-1.5">{label}</div>
      <div className="flex flex-col gap-0.5">{children}</div>
    </div>
  );
}

/** One history row with a shared-layout active highlight and fade-in delete action. */
function ThreadItem({
  thread,
  active,
  onOpen,
  onDelete,
}: {
  thread: RailThread;
  active: boolean;
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  const [hover, setHover] = useState(false);
  const title = thread.title || 'New chat';

  return (
    <div className="relative" onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
      <button
        type="button"
        onClick={() => onOpen(thread.id)}
        title={title}
        className={cn(
          'relative flex w-full items-center rounded-xl px-3 py-2 text-left transition-colors',
          active ? 'text-text-100' : 'text-text-300 hover:text-text-100',
          !active && hover && 'bg-bg-200/70',
        )}
      >
        {active && (
          <motion.span
            layoutId="thread-active"
            transition={SLIDE}
            className="absolute inset-0 rounded-xl bg-sand ring-1 ring-inset ring-accent/10"
          />
        )}
        <span className={cn('relative min-w-0 flex-1 truncate text-[13px] italic leading-snug', hover ? 'pr-8' : 'pr-2', active && 'font-medium')}>
          {title}
        </span>
        {!hover && (
          <span className="relative shrink-0 text-[10.5px] tabular-nums text-text-500">
            {relativeTime(thread.updated_at)}
          </span>
        )}
      </button>
      <div className="pointer-events-none absolute inset-y-0 right-1.5 flex items-center">
        <AnimatePresence>
          {hover && (
            <motion.button
              type="button"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.12 }}
              onClick={(e) => {
                e.stopPropagation();
                onDelete(thread.id);
              }}
              title="Delete chat"
              className="pointer-events-auto flex h-6 w-6 items-center justify-center rounded-md text-text-400 transition-colors hover:bg-danger-soft hover:text-danger"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </motion.button>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}