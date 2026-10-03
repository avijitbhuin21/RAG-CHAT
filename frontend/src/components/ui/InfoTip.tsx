import { Info } from 'lucide-react';
import type { ReactNode } from 'react';

import { cn } from '@/lib/utils';

const SIDE: Record<'top' | 'bottom' | 'right', string> = {
  top: 'bottom-full left-1/2 mb-2 -translate-x-1/2 translate-y-1 group-hover/info:translate-y-0 group-focus-within/info:translate-y-0',
  bottom: 'top-full left-1/2 mt-2 -translate-x-1/2 -translate-y-1 group-hover/info:translate-y-0 group-focus-within/info:translate-y-0',
  right: 'left-full top-1/2 ml-2 -translate-y-1/2 -translate-x-1 group-hover/info:translate-x-0 group-focus-within/info:translate-x-0',
};

/** Small "i" icon that reveals a description tooltip on hover or keyboard focus. */
export function InfoTip({
  children,
  side = 'bottom',
  className,
  width = 'w-64',
}: {
  children: ReactNode;
  side?: 'top' | 'bottom' | 'right';
  className?: string;
  width?: string;
}) {
  return (
    <span className={cn('group/info relative inline-flex align-middle', className)}>
      <button
        type="button"
        aria-label="More info"
        className="flex h-5 w-5 items-center justify-center rounded-full text-text-400 transition-colors hover:text-accent focus-visible:text-accent focus-visible:outline-none"
      >
        <Info className="h-3.5 w-3.5" />
      </button>
      <span
        role="tooltip"
        className={cn(
          'pointer-events-none absolute z-50 rounded-xl bg-inverse px-3 py-2 text-left text-[11.5px] font-normal normal-case leading-relaxed tracking-normal text-inverse-fg opacity-0 shadow-[0_12px_28px_-10px_rgba(0,0,0,0.45)] transition-all duration-150 group-hover/info:opacity-100 group-focus-within/info:opacity-100',
          width,
          SIDE[side],
        )}
      >
        {children}
      </span>
    </span>
  );
}
