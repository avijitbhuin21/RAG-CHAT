import { useCallback, useEffect, useState } from 'react';

export type Theme = 'light' | 'dark';

const STORAGE_KEY = 'theme';

/** Reads the theme currently applied to the document root. */
function currentTheme(): Theme {
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
}

/** Applies a theme to <html>, briefly enabling colour transitions so the switch fades smoothly. */
export function applyTheme(theme: Theme) {
  const root = document.documentElement;
  root.classList.add('theme-transition');
  root.classList.toggle('dark', theme === 'dark');
  window.setTimeout(() => root.classList.remove('theme-transition'), 380);
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    /* storage unavailable */
  }
  window.dispatchEvent(new CustomEvent('themechange', { detail: theme }));
}

/** Returns the active theme and a toggle that persists the choice. */
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(currentTheme);

  useEffect(() => {
    const on = () => setTheme(currentTheme());
    window.addEventListener('themechange', on);
    return () => window.removeEventListener('themechange', on);
  }, []);

  const toggle = useCallback(() => applyTheme(currentTheme() === 'dark' ? 'light' : 'dark'), []);

  return { theme, toggle };
}
