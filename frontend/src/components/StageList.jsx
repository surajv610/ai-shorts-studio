import { Link } from 'react-router-dom'
import { PIPELINE_SHORT, stageRoute, stageStatusText } from '../labels'

// Full vertical stage list: status chip + one-line detail + jump link.
export default function StageList({ stages = [], id }) {
  return (
    <ul className="stage-list">
      {stages.map((stage) => {
        const route = stageRoute(stage.key)
        return (
          <li
            key={stage.key}
            className={`stage-row stage-${stage.status}`}
            data-testid={`stage-${stage.key}`}
          >
            <span className={`stage-chip chip-${stage.status}`}>{
              stageStatusText(stage.status)
            }</span>
            <div className="stage-body">
              <span className="stage-name">{stage.label}</span>
              {stage.detail && (
                <span className="stage-detail">
                  {stage.detail.startsWith('Warning') || stage.status === 'stale'
                    ? '⚠ '
                    : ''}
                  {stage.detail}
                </span>
              )}
            </div>
            {route && (
              <Link
                className="stage-jump"
                to={`/projects/${id}/${route}`}
              >
                Open →
              </Link>
            )}
          </li>
        )
      })}
    </ul>
  )
}

export { PIPELINE_SHORT }