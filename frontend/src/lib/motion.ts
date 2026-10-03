export const EASE: [number, number, number, number] = [0.83, 0, 0.17, 1];
export const EASE_OUT_CSS = 'cubic-bezier(0.23, 1, 0.32, 1)';

export const MORPH = { duration: 0.42, ease: EASE };
export const SLIDE = { duration: 0.28, ease: EASE };

export const FADE = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit: { opacity: 0 },
  transition: { duration: 0.18, ease: 'easeOut' as const },
};

export const RISE = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.32, ease: EASE },
};

export const MESSAGE_IN = {
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.36, ease: [0.23, 1, 0.32, 1] as [number, number, number, number] },
};
