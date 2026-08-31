import { CartIcon, ResetIcon } from './Icons'

export function AppHeader({ onReset, resetting }) {
  return <header className="app-header">
    <a className="brand" href="#conversation" aria-label="BAD SMILE Shopping Copilot home">
      <span className="brand-mark"><CartIcon size={19} /></span>
      <span><b>BAD SMILE</b><em> / </em>Shopping Copilot</span>
    </a>
    <div className="header-actions">
      <button className="reset-button" type="button" onClick={onReset} disabled={resetting}>
        <ResetIcon /> {resetting ? 'Resetting…' : 'Start over'}
      </button>
    </div>
  </header>
}
