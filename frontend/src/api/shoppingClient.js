const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

async function post(path, body) {
  let response
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  } catch {
    throw new Error(`Can't reach the agent at ${BASE_URL}. Start it with: python3 -m scripts.serve`)
  }
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}))
    throw new Error(detail.error || `The agent returned ${response.status}.`)
  }
  return response.json()
}

export const shoppingClient = {
  async reset({ sessionId }) {
    return post('/api/reset', { sessionId })
  },

  async respond({ sessionId, message, turn }) {
    return post('/api/respond', { sessionId, message, turn })
  },
}
