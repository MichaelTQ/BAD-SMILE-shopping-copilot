import { mockCatalog } from '../data/mockCatalog'
import { matchLevelFrom, preferenceLabel } from '../utils/matchPresentation'

const questions = [
  ['category', 'What kind of item are you shopping for exactly?'],
  ['feature', 'Which feature matters most for this purchase?'],
  ['material', 'Do you have a material preference?'],
  ['color', 'Is there a color you would like to prioritize?'],
  ['useCase', 'Where or when will you use it most?'],
  ['budget', 'What budget would you like me to stay within?'],
]

const vocabularies = {
  category: ['boots', 'boot', 'shoes', 'shoe', 'sneakers', 'sneaker', 'shirt', 'tee', 'jacket', 'coat', 'bag', 'backpack', 'belt', 'dress', 'leggings', 'scarf'],
  material: ['leather', 'cotton', 'linen', 'wool', 'mesh', 'canvas', 'spandex', 'polyester'],
  color: ['black', 'white', 'blue', 'brown', 'gray', 'grey'],
  feature: ['waterproof', 'rain', 'lightweight', 'comfortable', 'breathable', 'warm', 'stretch', 'pockets', 'packable'],
  useCase: ['hiking', 'running', 'work', 'office', 'travel', 'gym', 'yoga', 'winter', 'summer', 'outdoor', 'walking'],
}

function wait(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds))
}

function formatBudget(amount) {
  return `Under $${Math.round(amount)}`
}

function parsePreferences(message, existing) {
  const lower = message.toLowerCase()
  const found = new Map(existing.map((item) => [item.attribute + item.label, item]))

  Object.entries(vocabularies).forEach(([attribute, words]) => {
    const match = words.find((word) => new RegExp(`\\b${word}\\b`, 'i').test(lower))
    if (match) found.set(attribute + match, preferenceLabel(attribute, match.replace(/^./, (letter) => letter.toUpperCase())))
  })

  const budget = lower.match(/(?:under|below|up to|max(?:imum)?|budget(?: of)?)\s*\$?\s*(\d+(?:\.\d+)?)/i) || lower.match(/\$\s*(\d+(?:\.\d+)?)/)
  if (budget) found.set('budget', preferenceLabel('budget', formatBudget(Number(budget[1]))))

  return [...found.values()]
}

function productSignals(product, preferences) {
  const termsFor = (attribute) => preferences
    .filter((item) => item.attribute === attribute)
    .flatMap((item) => item.label.toLowerCase().split(/\s+/))
  const productText = [...product.tags, product.title, product.store, ...product.category].join(' ').toLowerCase()
  const signals = []

  if (termsFor('category').some((term) => product.tags.includes(term))) signals.push('Relevant product category')
  if (termsFor('material').some((term) => product.tags.includes(term))) signals.push('Matches material preference')
  if (termsFor('color').some((term) => product.tags.includes(term))) signals.push('Matches preferred color')
  if (termsFor('feature').some((term) => product.tags.includes(term))) signals.push('Matches requested feature')
  if (termsFor('useCase').some((term) => product.tags.includes(term))) signals.push('Fits your planned use')

  const budget = preferences.find((item) => item.attribute === 'budget')
  if (budget) {
    const amount = Number(budget.label.replace(/[^\d.]/g, ''))
    if (product.price <= amount) signals.push('Within your stated budget')
  }

  const tokenMatches = preferences
    .flatMap((item) => item.label.toLowerCase().split(/\s+/))
    .filter((term) => term.length > 2 && productText.includes(term)).length
  return { signals: [...new Set(signals)], score: signals.length * 2 + tokenMatches + product.averageRating / 10 }
}

function recommendationSet(preferences) {
  return mockCatalog
    .map((product) => ({ ...product, ...productSignals(product, preferences) }))
    .sort((left, right) => right.score - left.score || right.ratingNumber - left.ratingNumber)
    .slice(0, 12)
    .map((product, index) => ({
      ...product,
      rank: index + 1,
      inTopTen: index < 10,
      matchLevel: matchLevelFrom(product.score),
      matchSignals: product.signals.length ? product.signals : ['Relevant catalog result', 'Strong customer feedback'],
    }))
}

function nextQuestion(preferences) {
  const asked = new Set(preferences.map((item) => item.attribute))
  return questions.find(([attribute]) => !asked.has(attribute)) || ['other', 'Would you like to refine the results in another way?']
}

export const mockShoppingClient = {
  async reset() {
    await wait(240)
    return { message: 'What are you shopping for today?', preferences: [], recommendations: [] }
  },

  async respond({ message, preferences, turn }) {
    await wait(520)
    if (message.includes('__network_error__')) throw new Error('The demo catalog is temporarily unavailable.')

    const nextPreferences = parsePreferences(message, preferences)
    const [askAttribute, question] = nextQuestion(nextPreferences)
    const recommendations = recommendationSet(nextPreferences)
    return {
      message: `I found a few promising options based on what you shared. ${question}`,
      askAttribute,
      preferences: nextPreferences,
      recommendations,
      turn,
    }
  },
}
