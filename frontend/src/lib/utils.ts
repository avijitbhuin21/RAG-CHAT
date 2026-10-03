import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Buckets items into Today / Yesterday / This week / Earlier by a date accessor. */
export function groupByAge<T>(items: T[], getDate: (t: T) => string): { label: string; items: T[] }[] {
  const startOfToday = new Date();
  startOfToday.setHours(0, 0, 0, 0);
  const day = 86_400_000;
  const buckets: Record<string, T[]> = { Today: [], Yesterday: [], 'This week': [], Earlier: [] };
  for (const item of items) {
    const t = new Date(getDate(item)).getTime();
    if (t >= startOfToday.getTime()) buckets.Today.push(item);
    else if (t >= startOfToday.getTime() - day) buckets.Yesterday.push(item);
    else if (t >= startOfToday.getTime() - 6 * day) buckets['This week'].push(item);
    else buckets.Earlier.push(item);
  }
  return Object.entries(buckets)
    .filter(([, v]) => v.length > 0)
    .map(([label, v]) => ({ label, items: v }));
}

/** Formats a timestamp as a compact relative label like "now", "5m", "3h", "2d" or a date. */
export function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  if (Number.isNaN(diff)) return '';
  const m = Math.floor(diff / 60_000);
  if (m < 1) return 'now';
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h`;
  const d = Math.floor(h / 24);
  if (d < 7) return `${d}d`;
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}
