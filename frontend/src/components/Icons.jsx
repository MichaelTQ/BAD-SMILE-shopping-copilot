export function CartIcon({ size = 22 }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path d="M2.4 3.9h1.75a1.2 1.2 0 0 1 1.17.94l.34 1.56" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M5.66 6.4H21.1l-1.83 7.35a1.7 1.7 0 0 1-1.65 1.3H9.05a1.7 1.7 0 0 1-1.66-1.33L5.66 6.4Z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round"/>
    <path d="M7.5 9.9h12.1" stroke="currentColor" strokeWidth="1.1" opacity=".45"/>
    <circle cx="9.9" cy="19.3" r="1.55" stroke="currentColor" strokeWidth="1.6"/>
    <circle cx="17.4" cy="19.3" r="1.55" stroke="currentColor" strokeWidth="1.6"/>
  </svg>
}

export function SparkIcon({ size = 20 }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="m12 2 1.75 6.25L20 10l-6.25 1.75L12 18l-1.75-6.25L4 10l6.25-1.75L12 2ZM19 16l.7 2.3L22 19l-2.3.7L19 22l-.7-2.3L16 19l2.3-.7L19 16Z" fill="currentColor"/></svg>
}

export function ArrowIcon() {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M5 12h13M13 6l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
}

export function ResetIcon() {
  return <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M20 11a8 8 0 1 0 1.1 4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/><path d="M20 4v7h-7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>
}

export function ChevronIcon() {
  return <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="m7 10 5 5 5-5" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round"/></svg>
}
