import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        border: 'hsl(var(--border))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        accent: {
          DEFAULT: 'rgb(var(--accent-rgb) / <alpha-value>)',
          foreground: 'var(--accent-foreground)',
          hover: 'var(--accent-hover)',
        },
        'accent-hover': 'var(--accent-hover)',
        danger: {
          DEFAULT: 'rgb(var(--danger-rgb) / <alpha-value>)',
          soft: 'rgb(var(--danger-soft-rgb) / <alpha-value>)',
        },
        warn: {
          DEFAULT: 'rgb(var(--warn-rgb) / <alpha-value>)',
          soft: 'rgb(var(--warn-soft-rgb) / <alpha-value>)',
        },
        sand: 'rgb(var(--sand-rgb) / <alpha-value>)',
        inverse: {
          DEFAULT: 'var(--inverse)',
          fg: 'var(--inverse-fg)',
        },
        scrim: 'var(--scrim)',
        bg: {
          0: 'var(--bg-0)',
          100: 'var(--bg-100)',
          200: 'var(--bg-200)',
          300: 'var(--bg-300)',
        },
        text: {
          100: 'rgb(var(--text-100-rgb) / <alpha-value>)',
          200: 'var(--text-200)',
          300: 'var(--text-300)',
          400: 'var(--text-400)',
          500: 'var(--text-500)',
        },
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },
      fontFamily: {
        sans: ['"Victor Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
        serif: ['"Victor Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
        mono: ['"Victor Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      keyframes: {
        'fade-in': {
          '0%': { opacity: '0', transform: 'translateY(8px) scale(0.98)', filter: 'blur(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0) scale(1)', filter: 'blur(0)' },
        },
        'slide-in-right': {
          '0%': { transform: 'translateX(100%)' },
          '100%': { transform: 'translateX(0)' },
        },
      },
      animation: {
        'fade-in': 'fade-in 0.3s cubic-bezier(0.2, 0.0, 0, 1.0) both',
        'slide-in-right': 'slide-in-right 0.25s cubic-bezier(0.2, 0.0, 0, 1.0) both',
      },
    },
  },
  plugins: [],
} satisfies Config;
