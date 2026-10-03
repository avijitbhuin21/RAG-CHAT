import { AnimatePresence, motion } from 'motion/react';
import { LogOut, Menu, PanelRightClose, Sparkles, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';

import { Composer } from '../components/chat/Composer';
import { MessageRow, type ChatMsg, type Citation } from '../components/chat/MessageRow';
import { SourcePanel, type SourcePanelTarget } from '../components/chat/SourcePanel';
import { appendThinking, type ChainStep } from '../components/chat/ThinkingChain';
import { useAttachments } from '../components/chat/attachments';
import { InfoTip } from '../components/ui/InfoTip';
import { ThreadRail } from '../components/chat/ThreadRail';
import { api } from '../lib/api';
import { useSession } from '../lib/auth';
import { EASE, RISE, SLIDE } from '../lib/motion';
import { streamSSE } from '../lib/stream';

type ChatSummary = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

const STARTERS = [
  'What is the 1stAId4SME project about?',
  'Summarise the key points of the project manual',
  'How can SMEs start using generative AI?',
  'Which training modules are covered?',
];

const PIN_THRESHOLD = 90;

export default function Chat() {
  const { session, logoutUser } = useSession();
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [draft, setDraft] = useState('');
  const [sending, setSending] = useState(false);
  const [showSignOut, setShowSignOut] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const [sourceTarget, setSourceTarget] = useState<SourcePanelTarget | null>(null);
  const att = useAttachments();

  const scrollRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);
  const messageCache = useRef<Map<string, ChatMsg[]>>(new Map());
  const messagesRef = useRef<ChatMsg[]>(messages);
  const activeIdRef = useRef<string | null>(activeId);
  const inlineCreatedRef = useRef<Set<string>>(new Set());
  const abortRef = useRef<AbortController | null>(null);
  const pendingContent = useRef('');
  const pendingThinking = useRef('');
  const frameRef = useRef<number | null>(null);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);
  useEffect(() => {
    activeIdRef.current = activeId;
  }, [activeId]);

  const openSource = useCallback((c: Citation) => {
    if (!c.file_id) return;
    const chunkTexts =
      c.chunk_texts && c.chunk_texts.length > 0
        ? c.chunk_texts
        : c.chunk_text
          ? [c.chunk_text]
          : c.snippet
            ? [c.snippet]
            : [];
    setSourceTarget({ fileId: c.file_id, filename: c.filename, chunkTexts });
  }, []);

  const closeSource = useCallback(() => setSourceTarget(null), []);

  async function refreshChats() {
    const list = await api<ChatSummary[]>('/chat/chats');
    setChats(list);
    return list;
  }

  async function loadMessages(chatId: string) {
    const cached = messageCache.current.get(chatId);
    if (cached) setMessages(cached);
    const rows = await api<ChatMsg[]>(`/chat/chats/${chatId}/messages`);
    const fresh = rows.map((m) => ({ ...m, streaming: false }));
    messageCache.current.set(chatId, fresh);
    if (activeIdRef.current === chatId) setMessages(fresh);
  }

  useEffect(() => {
    refreshChats().then((list) => {
      if (list.length > 0 && !activeIdRef.current) setActiveId(list[0].id);
    });
  }, []);

  useEffect(() => {
    const chatId = activeId;
    pinnedRef.current = true;
    if (chatId) {
      if (!inlineCreatedRef.current.has(chatId)) loadMessages(chatId);
    } else {
      setMessages([]);
    }
    return () => {
      if (chatId) messageCache.current.set(chatId, messagesRef.current);
    };
  }, [activeId]);

  useEffect(() => {
    const el = scrollRef.current;
    if (el && pinnedRef.current) el.scrollTop = el.scrollHeight;
  }, [messages]);

  function onScroll() {
    const el = scrollRef.current;
    if (!el) return;
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < PIN_THRESHOLD;
  }

  function updateLast(fn: (m: ChatMsg) => ChatMsg) {
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (!last || last.role !== 'assistant') return prev;
      return [...prev.slice(0, -1), fn(last)];
    });
  }

  function flush() {
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }
    const c = pendingContent.current;
    const t = pendingThinking.current;
    if (!c && !t) return;
    pendingContent.current = '';
    pendingThinking.current = '';
    updateLast((m) => ({
      ...m,
      content: m.content + c,
      thinking: (m.thinking ?? '') + t,
      steps: t ? appendThinking(m.steps ?? [], t) : m.steps,
    }));
  }

  function schedule() {
    if (frameRef.current === null) {
      frameRef.current = requestAnimationFrame(() => {
        frameRef.current = null;
        flush();
      });
    }
  }

  function newChat() {
    abortRef.current?.abort();
    setActiveId(null);
    setMessages([]);
    setDrawerOpen(false);
    setSourceTarget(null);
  }

  function selectChat(id: string) {
    setActiveId(id);
    setDrawerOpen(false);
    setSourceTarget(null);
  }

  async function deleteChat(id: string) {
    const remaining = chats.filter((c) => c.id !== id);
    setChats(remaining);
    messageCache.current.delete(id);
    if (activeId === id) setActiveId(remaining[0]?.id ?? null);
    try {
      await api(`/chat/chats/${id}`, { method: 'DELETE' });
    } catch {
      refreshChats();
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  async function send(rawText: string) {
    const userText = rawText.trim();
    const metas = att.ready;
    if ((!userText && metas.length === 0) || sending || att.uploading) return;
    setSending(true);
    setDraft('');
    att.clear();
    pinnedRef.current = true;
    pendingContent.current = '';
    pendingThinking.current = '';

    const now = Date.now();
    const userMsg: ChatMsg = {
      id: `tmp-user-${now}`,
      role: 'user',
      content: userText,
      thinking: null,
      citations: null,
      tool_calls: null,
      attachments: metas,
    };
    const assistantMsg: ChatMsg = {
      id: `tmp-asst-${now}`,
      role: 'assistant',
      content: '',
      thinking: '',
      citations: null,
      tool_calls: [],
      steps: [],
      streaming: true,
      startedAt: now,
    };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    const controller = new AbortController();
    abortRef.current = controller;

    let chatId = activeId;
    try {
      if (!chatId) {
        const chat = await api<ChatSummary>('/chat/chats', { method: 'POST' });
        chatId = chat.id;
        setChats((prev) => [chat, ...prev]);
        inlineCreatedRef.current.add(chatId);
        setActiveId(chatId);
      }

      await streamSSE(
        `/chat/chats/${chatId}/messages`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: userText, attachment_ids: metas.map((m) => m.id) }),
          signal: controller.signal,
        },
        (event) => {
          switch (event.type) {
            case 'thinking_delta':
              pendingThinking.current += event.content ?? '';
              schedule();
              return;
            case 'content_delta':
              pendingContent.current += event.content ?? '';
              schedule();
              return;
            case 'content_reset':
              pendingContent.current = '';
              if (event.reason === 'retry') pendingThinking.current = '';
              flush();
              updateLast((m) =>
                event.reason === 'retry'
                  ? { ...m, content: '', thinking: '', steps: [], tool_calls: [] }
                  : { ...m, content: '' },
              );
              return;
            case 'citations':
              flush();
              updateLast((m) => ({ ...m, citations: event.citations }));
              return;
            case 'tool_call_start':
              flush();
              updateLast((m) => {
                const copy = (m.tool_calls ?? []).slice();
                const slot = typeof event.index === 'number' ? event.index : copy.length;
                copy[slot] = { name: event.name, query: event.query ?? null, done: false, offset: event.offset };
                const steps: ChainStep[] = [
                  ...(m.steps ?? []),
                  { kind: 'tool', index: slot, query: event.query ?? null, done: false },
                ];
                return { ...m, tool_calls: copy, steps };
              });
              return;
            case 'tool_call_done':
              flush();
              updateLast((m) => {
                const copy = (m.tool_calls ?? []).slice();
                let slot = typeof event.index === 'number' ? event.index : -1;
                if (slot < 0 || slot >= copy.length) slot = copy.findIndex((c) => c && c.done === false);
                if (slot >= 0 && copy[slot]) copy[slot] = { ...copy[slot], done: true };
                const steps = (m.steps ?? []).map((s) =>
                  s.kind === 'tool' && s.index === slot ? { ...s, done: true } : s,
                );
                return { ...m, tool_calls: copy, steps };
              });
              return;
            case 'error':
              flush();
              updateLast((m) => ({ ...m, error: event.message ?? 'Something went wrong.' }));
              return;
            case 'done':
              flush();
              updateLast((m) => ({ ...m, id: event.message_id, streaming: false, endedAt: Date.now() }));
              return;
          }
        },
      );
    } catch (e) {
      flush();
      if (controller.signal.aborted) {
        updateLast((m) => ({ ...m, error: m.content ? null : 'Stopped before an answer was generated.' }));
      } else {
        updateLast((m) => ({ ...m, error: `Something went wrong: ${String(e)}` }));
      }
    } finally {
      flush();
      updateLast((m) => (m.streaming ? { ...m, streaming: false, endedAt: Date.now() } : m));
      setSending(false);
      abortRef.current = null;
      if (chatId) inlineCreatedRef.current.delete(chatId);
      refreshChats();
    }
  }

  const isEmpty = messages.length === 0;
  const userName = session.user
    ? session.user.email.split('@')[0].replace(/[._]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
    : 'there';
  const firstName = userName.split(' ')[0];
  const initials =
    userName
      .split(' ')
      .filter(Boolean)
      .slice(0, 2)
      .map((w) => w[0])
      .join('')
      .toUpperCase() || 'U';
  const hour = new Date().getHours();
  const greeting = hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening';
  const activeChat = chats.find((c) => c.id === activeId);
  const pendingDeleteChat = chats.find((c) => c.id === pendingDeleteId);
  const sourceCount = new Set(
    messages.flatMap((m) => (m.citations ?? []).map((c) => c.file_id ?? c.filename)),
  ).size;

  const rail = (
    <ThreadRail
      threads={chats}
      activeId={activeId}
      account={{
        name: userName,
        email: session.user?.email ?? '',
        initials,
        photo: session.user?.picture_url ?? null,
      }}
      onOpen={selectChat}
      onNew={newChat}
      onDelete={setPendingDeleteId}
      onSignOut={() => setShowSignOut(true)}
    />
  );

  return (
    <div className="flex h-[100dvh] gap-3 overflow-hidden bg-background lg:p-3">
      <div className="hidden lg:flex">{rail}</div>

      <AnimatePresence>
        {drawerOpen && (
          <>
            <motion.div
              key="scrim"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="fixed inset-0 z-30 bg-scrim backdrop-blur-[2px] lg:hidden"
              onClick={() => setDrawerOpen(false)}
            />
            <motion.div
              key="drawer"
              initial={{ x: '-105%' }}
              animate={{ x: 0 }}
              exit={{ x: '-105%' }}
              transition={SLIDE}
              className="fixed inset-y-0 left-0 z-40 flex p-3 lg:hidden"
            >
              {rail}
            </motion.div>
          </>
        )}
      </AnimatePresence>

      <main className="relative flex min-w-0 flex-1 flex-col overflow-hidden">
        <header className="flex h-14 shrink-0 items-center gap-2 px-3 lg:px-4">
          <button
            type="button"
            onClick={() => setDrawerOpen(true)}
            title="Open menu"
            className="rounded-lg p-2 text-text-300 transition hover:bg-bg-200 lg:hidden"
          >
            <Menu className="h-5 w-5" />
          </button>
          <div className="min-w-0 flex-1">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={activeId ?? 'new'}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: 0.18 }}
              >
                <div className="flex min-w-0 items-center gap-1.5">
                  <div className="truncate font-serif text-[16px] font-semibold italic text-text-100">
                    {activeChat?.title && activeChat.title !== 'New chat' ? activeChat.title : 'New chat'}
                  </div>
                  <InfoTip side="bottom">
                    {sourceCount > 0
                      ? `${sourceCount} source document${sourceCount === 1 ? '' : 's'} referenced in this chat. Answers are grounded in the 1stAId4SME knowledge base.`
                      : 'Answers are grounded in the 1stAId4SME knowledge base, with sources you can open.'}
                  </InfoTip>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>
          <AnimatePresence>
            {sourceTarget && (
              <motion.button
                type="button"
                initial={{ opacity: 0, scale: 0.92 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.92 }}
                transition={{ duration: 0.16 }}
                onClick={closeSource}
                className="hidden items-center gap-1.5 rounded-full border border-border bg-bg-100 px-3 py-1.5 text-[12px] font-medium text-text-300 transition-colors hover:text-accent lg:flex"
              >
                <PanelRightClose className="h-3.5 w-3.5" />
                Hide source
              </motion.button>
            )}
          </AnimatePresence>
        </header>

        <div ref={scrollRef} onScroll={onScroll} className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-[720px] px-4 sm:px-6">
            {isEmpty ? (
              <motion.div {...RISE} key="empty" className="flex flex-col items-center pb-10 pt-[11vh] text-center">
                <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-border bg-bg-100 shadow-[0_10px_24px_-14px_rgba(70,55,25,0.35)]">
                  <img src="/logo-short.png" alt="" className="h-10 w-10 object-contain" />
                </div>
                <h1 className="mt-6 flex items-center gap-2 font-serif text-3xl font-semibold tracking-tight text-accent sm:text-[2.2rem]">
                  {greeting}, {firstName}
                  <InfoTip side="bottom" width="w-72">
                    Ask anything about the knowledge base. Answers are grounded in the uploaded documents, with sources you can open. You can also attach images and documents.
                  </InfoTip>
                </h1>
                <div className="mt-9 grid w-full grid-cols-1 gap-2.5 sm:grid-cols-2">
                  {STARTERS.map((s, i) => (
                    <motion.button
                      key={s}
                      type="button"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.36, ease: EASE, delay: 0.12 + i * 0.06 }}
                      onClick={() => send(s)}
                      className="group flex items-start gap-3 rounded-2xl border border-border bg-bg-100 px-4 py-3.5 text-left text-[13.5px] text-text-200 shadow-[0_6px_18px_-14px_rgba(70,55,25,0.4)] transition-all duration-200 hover:-translate-y-0.5 hover:border-accent/30 hover:text-text-100 hover:shadow-[0_12px_24px_-14px_rgba(15,94,94,0.35)]"
                    >
                      <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-accent/60 transition-colors group-hover:text-accent" />
                      {s}
                    </motion.button>
                  ))}
                </div>
              </motion.div>
            ) : (
              <div className="flex flex-col gap-9 pb-8 pt-4">
                {messages.map((m) => (
                  <MessageRow
                    key={m.id}
                    msg={m}
                    onOpenSource={openSource}
                    activeFileId={sourceTarget?.fileId ?? null}
                  />
                ))}
              </div>
            )}
          </div>
        </div>

        <footer className="shrink-0 px-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] pt-1 sm:px-6">
          <div className="mx-auto w-full max-w-[720px]">
            <Composer
              value={draft}
              onChange={setDraft}
              onSubmit={() => send(draft)}
              onStop={stop}
              streaming={sending}
              attachments={att.items}
              onAddFiles={att.add}
              onRemoveAttachment={att.remove}
              autoFocus
              placeholder={isEmpty ? 'Ask a question…' : 'Ask a follow-up question…'}
            />
          </div>
        </footer>
      </main>

      <SourcePanel target={sourceTarget} onClose={closeSource} />

      <Dialog open={!!pendingDeleteChat} onClose={() => setPendingDeleteId(null)}>
        <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-danger-soft">
          <Trash2 className="h-6 w-6 text-danger" />
        </div>
        <h2 className="font-serif text-xl font-semibold text-text-100">Delete chat?</h2>
        <p className="mt-1 text-sm text-text-300">
          "{pendingDeleteChat?.title || 'New chat'}" and all of its messages will be permanently deleted. This cannot be undone.
        </p>
        <div className="mt-6 flex justify-end gap-2">
          <button type="button" onClick={() => setPendingDeleteId(null)} className="btn-ghost">
            Cancel
          </button>
          <button
            type="button"
            onClick={() => {
              const id = pendingDeleteId;
              setPendingDeleteId(null);
              if (id) deleteChat(id);
            }}
            className="btn-danger"
          >
            Delete
          </button>
        </div>
      </Dialog>

      <Dialog open={showSignOut} onClose={() => setShowSignOut(false)}>
        <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-accent/10">
          <LogOut className="h-6 w-6 text-accent" />
        </div>
        <h2 className="font-serif text-xl font-semibold text-text-100">Sign out?</h2>
        <p className="mt-1 text-sm text-text-300">You'll need to sign in again to return to your chats.</p>
        <div className="mt-6 flex justify-end gap-2">
          <button type="button" onClick={() => setShowSignOut(false)} className="btn-ghost">
            Cancel
          </button>
          <button
            type="button"
            onClick={() => {
              setShowSignOut(false);
              logoutUser();
            }}
            className="btn-primary"
          >
            Sign out
          </button>
        </div>
      </Dialog>
    </div>
  );
}

/** Animated modal shell with scrim and scale-in card. */
function Dialog({ open, onClose, children }: { open: boolean; onClose: () => void; children: ReactNode }) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18 }}
          className="fixed inset-0 z-[60] flex items-center justify-center bg-scrim backdrop-blur-sm"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.97, y: 4 }}
            transition={{ duration: 0.24, ease: [0.23, 1, 0.32, 1] }}
            className="panel w-[92%] max-w-sm p-6"
            onClick={(e) => e.stopPropagation()}
          >
            {children}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}