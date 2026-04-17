import { ChevronDown, LogOut, Plus, Trash2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

import ClaudeChatInput from '../components/ui/claude-style-chat-input';
import { AssistantMessage, type AssistantMsg, type Citation, type ToolCall } from '../components/chat/AssistantMessage';
import { SourcePanel, type SourcePanelTarget } from '../components/chat/SourcePanel';
import { api } from '../lib/api';
import { useSession } from '../lib/auth';
import { streamSSE } from '../lib/stream';

type ChatSummary = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

type Msg = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  thinking: string | null;
  citations: Citation[] | null;
  tool_calls: ToolCall[] | null;
  streaming?: boolean;
  thinkingStartedAt?: number | null;
  thinkingEndedAt?: number | null;
};

export default function Chat() {
  const { session, logoutUser } = useSession();
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [sending, setSending] = useState(false);
  const [showSignOut, setShowSignOut] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(true);
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const [sourceTarget, setSourceTarget] = useState<SourcePanelTarget | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  function openSource(c: Citation) {
    if (!c.file_id) return;
    // Prefer the per-file chunk_texts[] emitted by the new backend. Fall back
    // to the single-chunk `chunk_text` or truncated `snippet` so citations on
    // messages saved before those format changes still open the panel.
    const chunkTexts =
      c.chunk_texts && c.chunk_texts.length > 0
        ? c.chunk_texts
        : c.chunk_text
          ? [c.chunk_text]
          : c.snippet
            ? [c.snippet]
            : [];
    setSourceTarget({
      fileId: c.file_id,
      filename: c.filename,
      chunkTexts,
    });
  }

  const messageCache = useRef<Map<string, Msg[]>>(new Map());
  const messagesRef = useRef<Msg[]>(messages);
  const activeIdRef = useRef<string | null>(activeId);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);
  useEffect(() => {
    activeIdRef.current = activeId;
  }, [activeId]);

  async function refreshChats() {
    const list = await api<ChatSummary[]>('/chat/chats');
    setChats(list);
    return list;
  }

  async function loadMessages(chatId: string) {
    const cached = messageCache.current.get(chatId);
    if (cached) setMessages(cached);
    const t0 = performance.now();
    const rows = await api<Msg[]>(`/chat/chats/${chatId}/messages`);
    console.debug(
      '[loadMessages]',
      chatId,
      `${(performance.now() - t0).toFixed(0)}ms`,
    );
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
    if (chatId) {
      loadMessages(chatId);
    } else {
      setMessages([]);
    }
    return () => {
      if (chatId) messageCache.current.set(chatId, messagesRef.current);
    };
  }, [activeId]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  function newChat() {
    setActiveId(null);
    setMessages([]);
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

  const pendingDeleteChat = chats.find((c) => c.id === pendingDeleteId);

  async function send(rawText: string) {
    const userText = rawText.trim();
    if (!userText || sending) return;
    setSending(true);

    // Optimistic: place the user message + a streaming assistant placeholder
    // BEFORE awaiting the (possibly slow) chat-creation roundtrip. This flips
    // the view out of the welcome screen immediately and, critically, prevents
    // the `activeId` effect from overwriting these messages with an empty GET
    // response when we create a new chat inline.
    const userMsg: Msg = {
      id: `tmp-user-${Date.now()}`,
      role: 'user',
      content: userText,
      thinking: null,
      citations: null,
      tool_calls: null,
    };
    const assistantMsg: Msg = {
      id: `tmp-asst-${Date.now()}`,
      role: 'assistant',
      content: '',
      thinking: '',
      citations: null,
      tool_calls: [],
      streaming: true,
      thinkingStartedAt: null,
      thinkingEndedAt: null,
    };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    let chatId = activeId;
    if (!chatId) {
      const chat = await api<ChatSummary>('/chat/chats', { method: 'POST' });
      chatId = chat.id;
      setChats((prev) => [chat, ...prev]);
      messageCache.current.set(chatId, []);
      setActiveId(chatId);
    }

    try {
      await streamSSE(
        `/chat/chats/${chatId}/messages`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: userText }),
        },
        (event) => {
          // eslint-disable-next-line no-console
          console.debug('[chat sse]', event.type, event);
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (!last || last.role !== 'assistant') return prev;
            const updated: Msg = { ...last };
            if (event.type === 'citations') {
              updated.citations = event.citations;
            } else if (event.type === 'thinking_start') {
              // marker only; actual start time captured on first thinking_delta
            } else if (event.type === 'thinking_delta') {
              if (!updated.thinkingStartedAt) updated.thinkingStartedAt = Date.now();
              updated.thinking = (updated.thinking ?? '') + event.content;
            } else if (event.type === 'tool_call_start') {
              if (updated.thinkingStartedAt && !updated.thinkingEndedAt) {
                updated.thinkingEndedAt = Date.now();
              }
              // Use `index` from the event to place this entry at the right
              // slot. Parallel tool calls all start before any finishes, and
              // done events may arrive out of order — matching on index
              // prevents a "searching forever" sibling entry.
              const existing = updated.tool_calls ?? [];
              const copy = existing.slice();
              const slot = typeof event.index === 'number' ? event.index : copy.length;
              copy[slot] = { name: event.name, query: event.query ?? null, done: false };
              updated.tool_calls = copy;
            } else if (event.type === 'tool_call_done') {
              const calls = updated.tool_calls ?? [];
              if (calls.length > 0) {
                const copy = calls.slice();
                // Prefer the explicit index; fall back to first not-yet-done
                // entry (for any legacy events without it).
                let slot =
                  typeof event.index === 'number' ? event.index : -1;
                if (slot < 0 || slot >= copy.length) {
                  slot = copy.findIndex((c) => c && c.done === false);
                }
                if (slot >= 0 && copy[slot]) {
                  copy[slot] = { ...copy[slot], done: true };
                  updated.tool_calls = copy;
                }
              }
            } else if (event.type === 'content_delta') {
              if (updated.thinkingStartedAt && !updated.thinkingEndedAt) {
                updated.thinkingEndedAt = Date.now();
              }
              updated.content = updated.content + event.content;
            } else if (event.type === 'content_reset') {
              // Backend hit a transient upstream error mid-stream and is
              // retrying. Drop the partial answer so the retried output
              // doesn't stack on top of the aborted one.
              updated.content = '';
            } else if (event.type === 'done') {
              updated.id = event.message_id;
              updated.streaming = false;
              if (updated.thinkingStartedAt && !updated.thinkingEndedAt) {
                updated.thinkingEndedAt = Date.now();
              }
            }
            return [...prev.slice(0, -1), updated];
          });
        },
      );
    } catch (e) {
      setMessages((prev) => {
        const copy = [...prev];
        const last = copy[copy.length - 1];
        if (last && last.role === 'assistant') {
          last.content = last.content || `_Error: ${String(e)}_`;
          last.streaming = false;
        }
        return copy;
      });
    } finally {
      setSending(false);
      refreshChats();
    }
  }

  const isEmpty = messages.length === 0;
  const userName = session.user
    ? session.user.email.split('@')[0].replace(/[._]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
    : 'there';
  const hour = new Date().getHours();
  const greeting = hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening';

  return (
    <div className="flex h-screen">
      <aside className="flex w-64 flex-col border-r border-border bg-muted/40">
        <div className="border-b border-border px-4 py-2 pl-16">
          <img src="/logo.png" alt="1stAId4SME" className="h-8 w-auto" />
        </div>

        <div className="px-3 pt-3">
          <button
            type="button"
            onClick={newChat}
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-white/70 bg-gradient-to-b from-sky-300/25 to-sky-500/20 px-3 py-2 text-sm font-semibold text-sky-800 backdrop-blur-xl transition hover:from-sky-300/40 hover:to-sky-500/35 shadow-[inset_0_1px_0_rgba(255,255,255,0.8)]"
          >
            <Plus className="h-4 w-4" />
            New chat
          </button>
        </div>

        <nav className="flex-1 overflow-y-auto px-3 pt-3 pb-2">
          <button
            type="button"
            onClick={() => setHistoryOpen((v) => !v)}
            className="flex w-full items-center justify-between rounded-xl border border-white/60 bg-white/30 px-3 py-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground backdrop-blur-xl transition hover:bg-white/50 hover:text-foreground shadow-[inset_0_1px_0_rgba(255,255,255,0.8)]"
          >
            <span>History {chats.length > 0 && <span className="ml-1 font-normal normal-case tracking-normal">({chats.length})</span>}</span>
            <ChevronDown className={`h-4 w-4 transition-transform ${historyOpen ? '' : '-rotate-90'}`} />
          </button>

          {historyOpen && (
            <div className="mt-1 space-y-0.5">
              {chats.map((c) => {
                const isActive = c.id === activeId;
                return (
                  <div
                    key={c.id}
                    onClick={() => setActiveId(c.id)}
                    className={`group relative flex items-center gap-2 overflow-hidden rounded-md px-3 py-2 text-sm cursor-pointer transition ${
                      isActive
                        ? 'text-foreground font-semibold'
                        : 'text-foreground hover:bg-muted'
                    }`}
                  >
                    {isActive && (
                      <>
                        <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-sky-500/30 via-teal-400/15 to-white/0" />
                      </>
                    )}
                    <span className="relative flex-1 truncate">{c.title || 'New chat'}</span>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setPendingDeleteId(c.id);
                      }}
                      title="Delete chat"
                      className="relative rounded p-1 text-muted-foreground opacity-0 transition hover:bg-bg-200 hover:text-red-600 group-hover:opacity-100"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                );
              })}
              {chats.length === 0 && (
                <div className="px-3 py-2 text-xs text-muted-foreground">No chats yet</div>
              )}
            </div>
          )}
        </nav>
        <div className="border-t border-border p-2">
          <button
            type="button"
            onClick={() => setShowSignOut(true)}
            title="Sign out"
            className="group flex w-full items-center gap-2 rounded-md px-2 py-2 text-sm text-foreground transition hover:bg-muted"
          >
            <div className="h-7 w-7 shrink-0 overflow-hidden rounded-full bg-primary/20">
              {session.user?.picture_url ? (
                <img src={session.user.picture_url} alt="" className="h-full w-full object-cover" />
              ) : null}
            </div>
            <span className="flex-1 truncate text-left text-xs font-medium">
              {session.user ? session.user.email.split('@')[0] : 'user'}
            </span>
            <LogOut className="h-4 w-4 text-muted-foreground transition group-hover:text-foreground" />
          </button>
        </div>
      </aside>

      <main className="flex flex-1 flex-col bg-bg-0">
        {isEmpty ? (
          <div className="flex flex-1 flex-col items-center justify-center px-4 py-8">
            <div className="mb-8 text-center animate-fade-in">
              <img
                src="/logo-short.png"
                alt=""
                className="h-20 w-auto mx-auto mb-6 object-contain"
              />
              <h1 className="text-3xl sm:text-4xl font-serif text-text-200 mb-3 tracking-tight font-[500]">
                {greeting},{' '}
                <span className="relative inline-block pb-2">
                  {userName}
                  <svg
                    className="absolute w-[140%] h-[20px] -bottom-1 -left-[5%] text-accent"
                    viewBox="0 0 140 24"
                    fill="none"
                    preserveAspectRatio="none"
                    aria-hidden="true"
                  >
                    <path d="M6 16 Q 70 24, 134 14" stroke="currentColor" strokeWidth="3" strokeLinecap="round" fill="none" />
                  </svg>
                </span>
              </h1>
              <p className="text-sm text-text-300">
                Ask anything about the knowledge base — answers are grounded in uploaded documents.
              </p>
            </div>

            <ClaudeChatInput
              onSendMessage={({ message }) => send(message)}
              disabled={sending}
              placeholder="Ask anything about the knowledge base…"
            />
          </div>
        ) : (
          <>
            <div ref={scrollRef} className="flex-1 overflow-y-auto bg-bg-0">
              <div className="mx-auto max-w-3xl space-y-6 px-6 py-8">
                {messages.map((m) => (
                  <MessageBubble key={m.id} msg={m} onOpenSource={openSource} />
                ))}
              </div>
            </div>

            <footer className="bg-bg-0 pb-4 pt-2">
              <ClaudeChatInput
                onSendMessage={({ message }) => send(message)}
                disabled={sending}
                placeholder="Ask anything about the knowledge base…"
                showFooterNote={false}
              />
            </footer>
          </>
        )}
      </main>

      <SourcePanel target={sourceTarget} onClose={() => setSourceTarget(null)} />

      {pendingDeleteChat && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm animate-fade-in"
          onClick={() => setPendingDeleteId(null)}
        >
          <div
            className="w-[92%] max-w-sm rounded-2xl border border-bg-300 bg-bg-100 p-6 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-red-50">
              <Trash2 className="h-6 w-6 text-red-600" />
            </div>
            <h2 className="text-lg font-semibold text-text-100">Delete chat?</h2>
            <p className="mt-1 text-sm text-text-300">
              "{pendingDeleteChat.title || 'New chat'}" and all of its messages will be
              permanently deleted. This cannot be undone.
            </p>
            <div className="mt-6 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setPendingDeleteId(null)}
                className="rounded-md border border-bg-300 px-4 py-2 text-sm font-medium text-text-200 transition hover:bg-bg-200"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => {
                  const id = pendingDeleteChat.id;
                  setPendingDeleteId(null);
                  deleteChat(id);
                }}
                className="rounded-md bg-red-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-red-700"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {showSignOut && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm animate-fade-in"
          onClick={() => setShowSignOut(false)}
        >
          <div
            className="w-[92%] max-w-sm rounded-2xl border border-bg-300 bg-bg-100 p-6 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-accent/10">
              <LogOut className="h-6 w-6 text-accent" />
            </div>
            <h2 className="text-lg font-semibold text-text-100">Sign out?</h2>
            <p className="mt-1 text-sm text-text-300">
              You'll need to sign in again to return to your chats.
            </p>
            <div className="mt-6 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setShowSignOut(false)}
                className="rounded-md border border-bg-300 px-4 py-2 text-sm font-medium text-text-200 transition hover:bg-bg-200"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowSignOut(false);
                  logoutUser();
                }}
                className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white transition hover:bg-accent-hover"
              >
                Sign out
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function MessageBubble({
  msg,
  onOpenSource,
}: {
  msg: Msg;
  onOpenSource: (c: Citation) => void;
}) {
  if (msg.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-lg bg-primary px-4 py-2.5 text-sm text-primary-foreground">
          {msg.content}
        </div>
      </div>
    );
  }
  return <AssistantMessage msg={msg as AssistantMsg} onOpenSource={onOpenSource} />;
}
