import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { stageLabel, formatDate } from '../labels'

function goRoute(nextAction) {
  // "" means the story review screen of the workspace.
  const go = (nextAction && nextAction.go) || 'story'
  return go
}

function Thumb({ card }) {
  if (!card.thumbnail) return null
  if (/\.(mp4|mov|webm)$/i.test(card.thumbnail)) {
    return (
      <video className="card-thumb" src={card.thumbnail} muted preload="metadata" />
    )
  }
  return <img className="card-thumb" src={card.thumbnail} alt="" />
}

export default function Projects() {
  const navigate = useNavigate()
  const [projects, setProjects] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [confirmId, setConfirmId] = useState(null)

  async function load() {
    try {
      setError('')
      const list = await api.listProjects()
      setProjects(list)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  async function handleDelete(id) {
    try {
      await api.deleteProject(id)
      setConfirmId(null)
      await load()
    } catch (e) {
      setError(e.message)
    }
  }

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Projects</h1>
          <p className="muted">
            One tool — Story to Images, Videos, Assembly, QC, and Metadata.
          </p>
        </div>
        <button className="btn btn-primary" onClick={() => navigate('/new')}>
          New Project
        </button>
      </div>

      {error && <div className="alert error">{error}</div>}

      {loading ? (
        <p className="muted">Loading projects…</p>
      ) : projects.length === 0 ? (
        <div className="empty">
          <p>No projects yet. Describe your first short and start production.</p>
          <button className="btn btn-primary" onClick={() => navigate('/new')}>
            Create your first project
          </button>
        </div>
      ) : (
        <div className="project-grid unified">
          {projects.map((p) => {
            const blocking = p.next_action?.blocking
            return (
              <div key={p.id} className="project-card">
                <Thumb card={p} />
                <div className="project-card-head">
                  <h3>{p.name || p.idea}</h3>
                  <span
                    className={`status-chip paused-${isPaused(p.status)} ${
                      p.stale ? 'chip-stale' : ''
                    }`}
                  >
                    {p.stale ? 'Update needed' : stageLabel(p.status)}
                  </span>
                </div>

                <div className="card-progress">
                  <div className="mini-bar" aria-hidden="true">
                    <span style={{ width: `${p.progress?.percent ?? 0}%` }} />
                  </div>
                  <span className="mini-bar-text">
                    {p.progress?.percent ?? 0}%
                  </span>
                </div>

                <p className="project-idea" title={p.idea}>
                  {p.idea}
                </p>
                <p className="card-step">
                  <strong>Next:</strong>{' '}
                  {p.next_action?.label || p.current_step || '—'}
                </p>

                <div className="project-meta">
                  <span>Updated {formatDate(p.updated_at)}</span>
                </div>

                <div className="project-actions">
                  {blocking ? (
                    <>
                      <button
                        className="btn"
                        onClick={() =>
                          navigate(`/projects/${p.id}/${goRoute(p.next_action)}`)
                        }
                        data-testid={`review-${p.id}`}
                      >
                        Review
                      </button>
                      <button
                        className="btn btn-primary btn-sm"
                        onClick={() => navigate(`/projects/${p.id}`)}
                      >
                        Open
                      </button>
                    </>
                  ) : (
                    <button
                      className="btn btn-primary btn-sm"
                      onClick={() => navigate(`/projects/${p.id}`)}
                      data-testid={`continue-${p.id}`}
                    >
                      Continue
                    </button>
                  )}
                  <button
                    className="btn btn-danger btn-sm"
                    onClick={() => setConfirmId(p.id)}
                  >
                    Delete
                  </button>
                </div>

                {confirmId === p.id && (
                  <div className="confirm">
                    <p>Delete this project? This cannot be undone.</p>
                    <div className="confirm-actions">
                      <button
                        className="btn btn-danger btn-sm"
                        onClick={() => handleDelete(p.id)}
                      >
                        Delete
                      </button>
                      <button
                        className="btn btn-sm"
                        onClick={() => setConfirmId(null)}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function isPaused(status) {
  return new Set([
    'WAITING_FOR_STORY_APPROVAL',
    'WAITING_FOR_IMAGE_SELECTION',
    'WAITING_FOR_FINAL_APPROVAL',
    'READY_FOR_REVIEW',
  ]).has(status)
}