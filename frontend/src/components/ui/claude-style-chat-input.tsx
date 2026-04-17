import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Plus, ChevronDown, ArrowUp, X, FileText, Loader2, Check, Archive } from 'lucide-react';

export const Icons = {
  Plus,
  SelectArrow: ChevronDown,
  ArrowUp,
  X,
  FileText,
  Loader2,
  Check,
  Archive,
  Thinking: (props: React.SVGProps<SVGSVGElement>) => (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path d="M10.3857 2.50977C14.3486 2.71054 17.5 5.98724 17.5 10C17.5 14.1421 14.1421 17.5 10 17.5C5.85786 17.5 2.5 14.1421 2.5 10C2.5 9.72386 2.72386 9.5 3 9.5C3.27614 9.5 3.5 9.72386 3.5 10C3.5 13.5899 6.41015 16.5 10 16.5C13.5899 16.5 16.5 13.5899 16.5 10C16.5 6.5225 13.7691 3.68312 10.335 3.50879L10 3.5L9.89941 3.49023C9.67145 3.44371 9.5 3.24171 9.5 3C9.5 2.72386 9.72386 2.5 10 2.5L10.3857 2.50977ZM10 5.5C10.2761 5.5 10.5 5.72386 10.5 6V9.69043L13.2236 11.0527C13.4706 11.1762 13.5708 11.4766 13.4473 11.7236C13.3392 11.9397 13.0957 12.0435 12.8711 11.9834L12.7764 11.9473L9.77637 10.4473C9.60698 10.3626 9.5 10.1894 9.5 10V6C9.5 5.72386 9.72386 5.5 10 5.5ZM3.66211 6.94141C4.0273 6.94159 4.32303 7.23735 4.32324 7.60254C4.32324 7.96791 4.02743 8.26446 3.66211 8.26465C3.29663 8.26465 3 7.96802 3 7.60254C3.00021 7.23723 3.29676 6.94141 3.66211 6.94141ZM4.95605 4.29395C5.32146 4.29404 5.61719 4.59063 5.61719 4.95605C5.6171 5.3214 5.3214 5.61709 4.95605 5.61719C4.59063 5.61719 4.29403 5.32146 4.29395 4.95605C4.29395 4.59057 4.59057 4.29395 4.95605 4.29395ZM7.60254 3C7.96802 3 8.26465 3.29663 8.26465 3.66211C8.26446 4.02743 7.96791 4.32324 7.60254 4.32324C7.23736 4.32302 6.94159 4.0273 6.94141 3.66211C6.94141 3.29676 7.23724 3.00022 7.60254 3Z" />
    </svg>
  ),
};

const formatFileSize = (bytes: number) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export interface AttachedFile {
  id: string;
  file: File;
  type: string;
  preview: string | null;
  uploadStatus: string;
  content?: string;
}

export interface PastedSnippet {
  id: string;
  content: string;
  timestamp: Date;
}

interface FilePreviewCardProps {
  file: AttachedFile;
  onRemove: (id: string) => void;
}

const FilePreviewCard: React.FC<FilePreviewCardProps> = ({ file, onRemove }) => {
  const isImage = file.type.startsWith('image/') && file.preview;
  return (
    <div className="relative group flex-shrink-0 w-24 h-24 rounded-xl overflow-hidden border border-bg-300 bg-bg-200 animate-fade-in transition-all hover:border-text-400">
      {isImage ? (
        <div className="w-full h-full relative">
          <img src={file.preview!} alt={file.file.name} className="w-full h-full object-cover" />
          <div className="absolute inset-0 bg-black/20 group-hover:bg-black/0 transition-colors" />
        </div>
      ) : (
        <div className="w-full h-full p-3 flex flex-col justify-between">
          <div className="flex items-center gap-2">
            <div className="p-1.5 bg-bg-300 rounded">
              <Icons.FileText className="w-4 h-4 text-text-300" />
            </div>
            <span className="text-[10px] font-medium text-text-400 uppercase tracking-wider truncate">
              {file.file.name.split('.').pop()}
            </span>
          </div>
          <div className="space-y-0.5">
            <p className="text-xs font-medium text-text-200 truncate" title={file.file.name}>
              {file.file.name}
            </p>
            <p className="text-[10px] text-text-500">{formatFileSize(file.file.size)}</p>
          </div>
        </div>
      )}
      <button
        onClick={() => onRemove(file.id)}
        className="absolute top-1 right-1 p-1 bg-black/50 hover:bg-black/70 rounded-full text-white opacity-0 group-hover:opacity-100 transition-opacity"
      >
        <Icons.X className="w-3 h-3" />
      </button>
      {file.uploadStatus === 'uploading' && (
        <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
          <Icons.Loader2 className="w-5 h-5 text-white animate-spin" />
        </div>
      )}
    </div>
  );
};

interface PastedContentCardProps {
  content: PastedSnippet;
  onRemove: (id: string) => void;
}

const PastedContentCard: React.FC<PastedContentCardProps> = ({ content, onRemove }) => (
  <div className="relative group flex-shrink-0 w-28 h-28 rounded-2xl overflow-hidden border border-bg-300 bg-bg-100 animate-fade-in p-3 flex flex-col justify-between shadow-[0_1px_2px_rgba(0,0,0,0.05)]">
    <div className="overflow-hidden w-full">
      <p className="text-[10px] text-text-400 leading-[1.4] font-mono break-words whitespace-pre-wrap line-clamp-4 select-none">
        {content.content}
      </p>
    </div>
    <div className="flex items-center justify-between w-full mt-2">
      <div className="inline-flex items-center justify-center px-1.5 py-[2px] rounded border border-bg-300 bg-bg-100">
        <span className="text-[9px] font-bold text-text-300 uppercase tracking-wider font-sans">PASTED</span>
      </div>
    </div>
    <button
      onClick={() => onRemove(content.id)}
      className="absolute top-2 right-2 p-[3px] bg-bg-100 border border-bg-300 rounded-full text-text-400 hover:text-text-200 transition-colors shadow-sm opacity-0 group-hover:opacity-100"
    >
      <Icons.X className="w-2 h-2" />
    </button>
  </div>
);

interface Model {
  id: string;
  name: string;
  description: string;
  badge?: string;
}

export interface ChatInputPayload {
  message: string;
  files: AttachedFile[];
  pastedContent: PastedSnippet[];
  model: string;
  isThinkingEnabled: boolean;
}

interface ClaudeChatInputProps {
  onSendMessage: (data: ChatInputPayload) => void;
  disabled?: boolean;
  placeholder?: string;
  models?: Model[];
  showFooterNote?: boolean;
  autoFocus?: boolean;
}

const DEFAULT_MODELS: Model[] = [
  { id: 'sonnet-4.6', name: 'Sonnet 4.6', description: 'Best for everyday tasks' },
];

export const ClaudeChatInput: React.FC<ClaudeChatInputProps> = ({
  onSendMessage,
  disabled = false,
  placeholder = 'How can I help you today?',
  models = DEFAULT_MODELS,
  showFooterNote = true,
  autoFocus = true,
}) => {
  const [message, setMessage] = useState('');
  const [files, setFiles] = useState<AttachedFile[]>([]);
  const [pastedContent, setPastedContent] = useState<PastedSnippet[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedModel] = useState(models[0]?.id ?? '');
  const [isThinkingEnabled] = useState(false);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 384) + 'px';
    }
  }, [message]);

  const handleFiles = useCallback((newFilesList: FileList | File[]) => {
    const newFiles: AttachedFile[] = Array.from(newFilesList).map((file) => {
      const isImage = file.type.startsWith('image/') || /\.(jpg|jpeg|png|gif|webp|svg)$/i.test(file.name);
      return {
        id: Math.random().toString(36).slice(2, 11),
        file,
        type: isImage ? 'image/unknown' : file.type || 'application/octet-stream',
        preview: isImage ? URL.createObjectURL(file) : null,
        uploadStatus: 'pending',
      };
    });
    setFiles((prev) => [...prev, ...newFiles]);
    newFiles.forEach((f) => {
      setTimeout(() => {
        setFiles((prev) => prev.map((p) => (p.id === f.id ? { ...p, uploadStatus: 'complete' } : p)));
      }, 800 + Math.random() * 1000);
    });
  }, []);

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };
  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files) handleFiles(e.dataTransfer.files);
  };

  const handlePaste = (e: React.ClipboardEvent) => {
    const items = e.clipboardData.items;
    const pastedFiles: File[] = [];
    for (let i = 0; i < items.length; i++) {
      if (items[i].kind === 'file') {
        const file = items[i].getAsFile();
        if (file) pastedFiles.push(file);
      }
    }
    if (pastedFiles.length > 0) {
      e.preventDefault();
      handleFiles(pastedFiles);
      return;
    }
    const text = e.clipboardData.getData('text');
    if (text.length > 300) {
      e.preventDefault();
      setPastedContent((prev) => [
        ...prev,
        { id: Math.random().toString(36).slice(2, 11), content: text, timestamp: new Date() },
      ]);
    }
  };

  const handleSend = () => {
    if (disabled) return;
    if (!message.trim() && files.length === 0 && pastedContent.length === 0) return;
    onSendMessage({ message, files, pastedContent, model: selectedModel, isThinkingEnabled });
    setMessage('');
    setFiles([]);
    setPastedContent([]);
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const hasContent = message.trim() || files.length > 0 || pastedContent.length > 0;

  return (
    <div
      className="relative w-full max-w-2xl mx-auto transition-all duration-300 font-sans"
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <div
        className="!box-content flex flex-col mx-2 md:mx-0 items-stretch transition-all duration-200 relative z-10 rounded-2xl cursor-text border border-bg-300 shadow-[0_0_15px_rgba(0,0,0,0.06)] hover:shadow-[0_0_20px_rgba(0,0,0,0.1)] focus-within:shadow-[0_0_25px_rgba(14,165,233,0.15)] bg-bg-100 font-sans antialiased"
      >
        <div className="flex flex-col px-3 pt-3 gap-2">
          {(files.length > 0 || pastedContent.length > 0) && (
            <div className="flex gap-3 overflow-x-auto custom-scrollbar pb-2 px-1">
              {pastedContent.map((content) => (
                <PastedContentCard
                  key={content.id}
                  content={content}
                  onRemove={(id) => setPastedContent((prev) => prev.filter((c) => c.id !== id))}
                />
              ))}
              {files.map((file) => (
                <FilePreviewCard
                  key={file.id}
                  file={file}
                  onRemove={(id) => setFiles((prev) => prev.filter((f) => f.id !== id))}
                />
              ))}
            </div>
          )}

          <div className="relative">
            <div className="max-h-96 w-full overflow-y-auto custom-scrollbar font-sans break-words transition-opacity duration-200 min-h-[2.5rem] pl-1 pr-12">
              <textarea
                ref={textareaRef}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onPaste={handlePaste}
                onKeyDown={handleKeyDown}
                placeholder={placeholder}
                disabled={disabled}
                className="w-full bg-transparent border-0 outline-none text-text-100 text-[16px] placeholder:text-text-400 resize-none overflow-hidden py-0 leading-relaxed block font-normal antialiased disabled:opacity-60"
                rows={1}
                autoFocus={autoFocus}
                style={{ minHeight: '1.5em' }}
              />
            </div>
            <button
              onClick={handleSend}
              disabled={!hasContent || disabled}
              className={`absolute right-0 bottom-[0.65rem] inline-flex items-center justify-center h-8 w-8 rounded-xl transition-colors active:scale-95 ${
                hasContent && !disabled
                  ? 'bg-accent text-white hover:bg-accent-hover shadow-md'
                  : 'bg-accent/30 text-white/70 cursor-default'
              }`}
              type="button"
              aria-label="Send message"
            >
              <Icons.ArrowUp className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {isDragging && (
        <div className="absolute inset-0 bg-bg-200/90 border-2 border-dashed border-accent rounded-2xl z-50 flex flex-col items-center justify-center backdrop-blur-sm pointer-events-none">
          <Icons.Archive className="w-10 h-10 text-accent mb-2 animate-bounce" />
          <p className="text-accent font-medium">Drop files to upload</p>
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        multiple
        className="hidden"
        onChange={(e) => {
          if (e.target.files) handleFiles(e.target.files);
          e.target.value = '';
        }}
      />

      {showFooterNote && (
        <div className="text-center mt-4">
          <p className="text-xs text-text-500">AI can make mistakes. Please check important information.</p>
        </div>
      )}
    </div>
  );
};

export default ClaudeChatInput;
