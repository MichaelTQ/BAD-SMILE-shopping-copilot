import { useCallback, useState } from 'react'
import { shoppingClient } from '../api/shoppingClient'

const openingMessage = {
  id: 'opening',
  role: 'assistant',
  text: 'What are you shopping for today?',
  detail: 'Tell me about the item, the occasion, or the details that matter most.',
}

function createSessionId() {
  return globalThis.crypto?.randomUUID?.() || `demo-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export function useShoppingSession() {
  const [sessionId, setSessionId] = useState(createSessionId)
  const [messages, setMessages] = useState([openingMessage])
  const [preferences, setPreferences] = useState([])
  const [recommendations, setRecommendations] = useState([])
  const [turn, setTurn] = useState(0)
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState('')
  const [lastMessage, setLastMessage] = useState('')

  const requestResponse = useCallback(async (text, appendUser) => {
    const nextTurn = turn + 1
    setStatus('loading')
    setError('')
    setLastMessage(text)
    if (appendUser) {
      setMessages((current) => [...current, { id: `user-${Date.now()}`, role: 'user', text }])
    }

    try {
      const response = await shoppingClient.respond({ sessionId, message: text, turn: nextTurn })
      setPreferences(response.preferences)
      setRecommendations(response.recommendations)
      setTurn(nextTurn)
      setMessages((current) => [...current, {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        text: response.message,
        askAttribute: response.askAttribute,
      }])
      setStatus(nextTurn >= 10 ? 'complete' : 'idle')
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Something went wrong while updating the catalog.')
      setStatus('error')
    }
  }, [preferences, sessionId, turn])

  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || status === 'loading' || status === 'complete') return
    await requestResponse(text.trim(), true)
  }, [requestResponse, status])

  const retryLastMessage = useCallback(async () => {
    if (!lastMessage || status !== 'error') return
    await requestResponse(lastMessage, false)
  }, [lastMessage, requestResponse, status])

  const resetSession = useCallback(async () => {
    setStatus('resetting')
    setError('')
    const nextSessionId = createSessionId()
    try {
      const response = await shoppingClient.reset({ sessionId: nextSessionId })
      setSessionId(nextSessionId)
      setMessages([{ ...openingMessage, text: response.message }])
      setPreferences([])
      setRecommendations([])
      setTurn(0)
      setLastMessage('')
      setStatus('idle')
    } catch {
      setStatus('idle')
    }
  }, [])

  return {
    messages, preferences, recommendations, turn, status, error,
    isStarted: turn > 0 || messages.length > 1,
    sendMessage, retryLastMessage, resetSession,
  }
}
