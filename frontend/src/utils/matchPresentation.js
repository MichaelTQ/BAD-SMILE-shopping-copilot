const displayNames = {
  category: 'Category', material: 'Material', color: 'Color', size: 'Size or fit',
  style: 'Style', brand: 'Brand', budget: 'Budget', feature: 'Feature', useCase: 'Use case',
}

export function preferenceLabel(attribute, value) {
  return { attribute, label: value, displayName: displayNames[attribute] || attribute }
}

export function matchLevelFrom(score) {
  if (score >= 5) return 'strong'
  if (score >= 3) return 'good'
  if (score >= 1) return 'partial'
  return 'consider'
}

export const matchLabels = {
  strong: 'Great fit',
  good: 'Good fit',
  partial: 'Possible fit',
  consider: 'More to explore',
}
