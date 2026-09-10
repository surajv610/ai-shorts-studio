// The single "what to do next" card for a project (deterministic, from backend).
export default function NextActionCard({ nextAction = {}, onContinue, busy }) {
  if (!nextAction || !nextAction.action) return null
  const { label, reason, blocking } = nextAction
  return (
    <div className={`next-action next-action-${blocking ? 'review' : 'go'}`}>
      <div className="next-action-body">
        <span className="next-action-eyebrow">
          {blocking ? 'Needs your review' : 'Next step'}
        </span>
        <h3>{label}</h3>
        {reason && <p className="muted">{reason}</p>}
      </div>
      {onContinue && (
        <button
          className="btn btn-primary next-action-btn"
          disabled={busy}
          onClick={onContinue}
        >
          {busy ? 'Working…' : blocking ? 'Continue to Review' : 'Continue'}
        </button>
      )}
    </div>
  )
}