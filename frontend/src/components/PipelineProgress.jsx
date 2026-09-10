// The 8-stage production pipeline bar, driven entirely by the backend-derived
// summary (no local state machine — the backend remains the source of truth).
const GLYPHS = {
  completed: '✓',
  current: '',
  waiting: '',
  failed: '✕',
  stale: '!',
}

export default function PipelineProgress({ stages = [], progress = {}, onSelect }) {
  const percent = progress.percent ?? 0
  const count = stages.length

  return (
    <div className="pipeline">
      <div className="pipeline-bar" role="progressbar" aria-valuenow={percent}>
        <span style={{ width: `${percent}%` }} />
      </div>
      <div className="pipeline-label">
        <strong>{percent}%</strong>
        <span className="muted">
          {progress.done ?? 0} / {progress.total ?? count} stages
        </span>
      </div>
      <div className="pipeline-stages">
        {stages.map((stage, i) => (
          <button
            key={stage.key}
            type="button"
            className={`pipe-stage pipe-${stage.status}`}
            onClick={onSelect ? () => onSelect(stage.key) : undefined}
            data-testid={`pipeline-${stage.key}`}
          >
            <span className="pipe-glyph">
              {i + 1}
              {GLYPHS[stage.status] ? ` ${GLYPHS[stage.status]}` : ''}
            </span>
            <span className="pipe-name">{stage.label}</span>
          </button>
        ))}
      </div>
    </div>
  )
}