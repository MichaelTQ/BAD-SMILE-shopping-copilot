import { useEffect, useRef, useState } from 'react'
import { ArrowIcon, BagIcon, SparkIcon } from './Icons'

const examples = [
  'Waterproof hiking boots under $150',
  'A black leather belt for work',
  'Something comfortable for summer travel',
]

function WelcomeScreen({ onExample }) {
  return <section className="welcome-screen">
    <span className="welcome-icon"><BagIcon size={31} /></span>
    <p className="kicker">YOUR CATALOG GUIDE</p>
    <h1>Shop with more clarity.</h1>
    <p>Tell BAD SMILE what you need. It will ask focused follow-up questions and surface the closest catalog matches.</p>
    <div className="example-list" aria-label="Try a shopping example">
      {examples.map((example) => <button key={example} type="button" onClick={() => onExample(example)}>{example}<ArrowIcon /></button>)}
    </div>
  </section>
}

function MessageList({ messages, loading }) {
  const endRef = useRef(null)
  useEffect(() => {
    const endNode = endRef.current
    if (endNode && typeof endNode.scrollIntoView === 'function') {
      endNode.scrollIntoView({ block: 'end', behavior: 'smooth' })
    }
  }, [messages, loading])

  return <div className="message-list" aria-live="polite">
    {messages.map((message) => <article className={`message message-${message.role}`} key={message.id}>
      {message.role === 'assistant' && <span className="assistant-avatar"><SparkIcon size={16} /></span>}
      <div className="message-bubble">
        <p>{message.text}</p>
        {message.detail && <small>{message.detail}</small>}
      </div>
    </article>)}
    {loading && <article className="message message-assistant" aria-label="The assistant is looking for catalog matches"><span className="assistant-avatar"><SparkIcon size={16} /></span><div className="typing-indicator"><i /><i /><i /><span>Updating matches</span></div></article>}
    <span ref={endRef} />
  </div>
}

function Composer({ onSend, loading, complete }) {
  const [input, setInput] = useState('')
  const submitInput = () => {
    if (!input.trim()) return
    onSend(input)
    setInput('')
  }
  const handleSubmit = (event) => {
    event.preventDefault()
    submitInput()
  }
  return <div className="composer-wrap">
    {complete && <p className="turn-limit">Ready to explore something new? Start a fresh search whenever you like.</p>}
    <form className="composer" onSubmit={handleSubmit}>
      <label className="sr-only" htmlFor="shopping-request">Describe what you are shopping for</label>
      <input id="shopping-request" value={input} disabled={loading || complete} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); submitInput() } }} placeholder={complete ? 'Start a new session to continue' : 'Describe what you are looking for…'} />
      <button className="send-button" type="submit" disabled={!input.trim() || loading || complete} aria-label="Send shopping request"><ArrowIcon /></button>
    </form>
    <p>Press Enter to send · Recommendations update as you share what matters.</p>
  </div>
}

export function ConversationPanel({ messages, isStarted, turn, status, error, onSend, onRetry }) {
  const loading = status === 'loading'
  const complete = status === 'complete'
  return <section className="conversation-panel" id="conversation" aria-labelledby="conversation-title">
    <div className="panel-heading">
      <div><p className="kicker">YOUR SHOPPING ASSISTANT</p><h2 id="conversation-title">{isStarted ? 'Your recommendations, updated.' : 'Tell us what you need.'}</h2></div>
    </div>
    <div className="conversation-body">
      {isStarted ? <MessageList messages={messages} loading={loading} /> : <WelcomeScreen onExample={onSend} />}
      {error && <div className="error-banner" role="alert"><div><b>We could not update the demo catalog.</b><span>{error}</span></div><button type="button" onClick={onRetry}>Try again</button></div>}
    </div>
    <Composer onSend={onSend} loading={loading} complete={complete} />
  </section>
}
