import { createRoot } from 'react-dom/client'
import './styles.css'
import { AppHeader } from './components/AppHeader'
import { ConversationPanel } from './components/ConversationPanel'
import { RecommendationPanel } from './components/RecommendationPanel'
import { useShoppingSession } from './hooks/useShoppingSession'

function App() {
  const session = useShoppingSession()
  return <main className="app-shell">
    <AppHeader onReset={session.resetSession} resetting={session.status === 'resetting'} />
    <div className="workspace">
      <ConversationPanel {...session} onSend={session.sendMessage} onRetry={session.retryLastMessage} />
      <RecommendationPanel {...session} />
    </div>
  </main>
}

createRoot(document.getElementById('root')).render(<App />)
