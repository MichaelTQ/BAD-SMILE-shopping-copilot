import { useState } from 'react'
import { BagIcon, ChevronIcon } from './Icons'
import { matchLabels } from '../utils/matchPresentation'

const categorySymbols = { boot: '⌁', shoe: '◒', jacket: '◇', shirt: '▤', bag: '◫', belt: '⎯', leggings: '∿', dress: '♢', scarf: '≈' }

function ProductVisual({ visual, title }) {
  return <div className={`catalog-tile tone-${visual.tone} texture-${visual.texture}`} aria-label={`${title} catalog preview`} role="img">
    <span aria-hidden="true">{categorySymbols[visual.icon] || '✦'}</span>
    <small>CATALOG</small>
  </div>
}

function ProductCard({ product }) {
  const [expanded, setExpanded] = useState(false)
  const categoryText = product.category.join(' / ')
  return <article className={`product-card ${expanded ? 'is-expanded' : ''}`}>
    <button className="product-trigger" type="button" onClick={() => setExpanded((current) => !current)} aria-expanded={expanded} aria-controls={`product-${product.parent_asin}`}>
      <span className="rank-block"><b>{product.rank}</b></span>
      <ProductVisual visual={product.visual} title={product.title} />
      <span className="product-summary">
        <span className={`match-level level-${product.matchLevel}`}>{matchLabels[product.matchLevel]}</span>
        <strong>{product.title}</strong>
        <span className="product-store">{product.store} · {categoryText}</span>
        <span className="product-metrics"><b>★ {product.averageRating.toFixed(1)}</b><span>{product.ratingNumber.toLocaleString()} reviews</span><em>${product.price.toFixed(2)}</em></span>
      </span>
      <span className="chevron"><ChevronIcon /></span>
    </button>
    <div className="match-signals" aria-label="Why this product matches">
      {product.matchSignals.slice(0, 2).map((signal) => <span key={signal}>{signal}</span>)}
    </div>
    {expanded && <div className="product-detail" id={`product-${product.parent_asin}`}>
      <p>{product.description}</p>
      <dl><div><dt>Catalog ID</dt><dd>{product.parent_asin}</dd></div><div><dt>Catalog category</dt><dd>{categoryText}</dd></div><div><dt>All match signals</dt><dd>{product.matchSignals.join(' · ')}</dd></div></dl>
    </div>}
  </article>
}

function PreferenceSummary({ preferences }) {
  return <section className="preference-summary" aria-labelledby="preference-title">
    <div><p className="kicker">YOUR PREFERENCES</p><h3 id="preference-title">Based on what you’ve told us</h3></div>
    <div className="preference-chips">
      {preferences.length ? preferences.map((preference) => <span className="preference-chip" key={`${preference.attribute}-${preference.label}`}><small>{preference.displayName}</small>{preference.label}</span>) : <span className="preference-placeholder">Your preferences will appear here as you chat.</span>}
    </div>
  </section>
}

function EmptyResults() {
  return <div className="results-empty"><span><BagIcon size={34} /></span><h3>Catalog matches will appear here.</h3><p>Start with a product, occasion, material, color, budget, or feature.</p></div>
}

function LoadingResults() {
  return <div className="skeleton-list" aria-label="Loading product recommendations"><i /><i /><i /><i /></div>
}

export function RecommendationPanel({ preferences, recommendations, status, isStarted }) {
  const loading = status === 'loading'
  return <aside className="recommendation-panel" aria-labelledby="recommendation-title">
    <div className="recommendation-heading"><div><p className="kicker">RECOMMENDATIONS</p><h2 id="recommendation-title">{isStarted ? 'Recommended for you' : 'Suggestions will appear here.'}</h2></div>{isStarted && <span className="count-badge">12 suggestions</span>}</div>
    <PreferenceSummary preferences={preferences} />
    <div className="results-body">
      {loading && !recommendations.length ? <LoadingResults /> : !recommendations.length ? <EmptyResults /> : <><p className="result-explainer">Each suggestion highlights the details that fit your request.</p><div className="product-list">{recommendations.map((product) => <ProductCard product={product} key={product.parent_asin} />)}</div></>}
    </div>
  </aside>
}
