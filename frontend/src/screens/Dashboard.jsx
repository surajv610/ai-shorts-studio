import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api'
import { stageLabel, isPaused, formatDate } from '../labels'

export default function Dashboard() {
  const [projects, setProjects] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [confirmId, setConfirmId] = useState(null)
  const navigate = useNavigate()

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
        <h1>Projects</h1>
        <button className="btn btn-primary" onClick={() => navigate('/new')}>
          New Project
        </button>
      </div>

      {error && <div className="alert error">{error}</div>}

      {loading ? (
        <p className="muted">Loading projects…</p>
      ) : projects.length === 0 ? (
        <div className="empty">
          <p>No projects yet.</p>
          <button className="btn btn-primary" onClick={() => navigate('/new')}>
            Create your first project
          </button>
        </div>
      ) : (
        <div className="project-grid">
          {projects.map((p) => (
            <div key={p.id} className="project-card">
              <div className="project-card-head">
                <h3>{p.name || p.idea}</h3>
                <span className={`status-chip paused-${isPaused(p.status)}`}>
                  {stageLabel(p.status)}
                </span>
              </div>
              <p className="project-idea" title={p.idea}>
                {p.idea}
              </p>
              <div className="project-meta">
                <span>Created {formatDate(p.created_at)}</span>
                <span>Updated {formatDate(p.updated_at)}</span>
              </div>
              <div className="project-actions">
                <Link className="btn" to={`/projects/${p.id}`}>
                  {isPaused(p.status) ? 'Review & Approve' : 'Open'}
                </Link>
                <button
                  className="btn btn-danger"
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
                      className="btn btn-danger"
                      onClick={() => handleDelete(p.id)}
                    >
                      Delete
                    </button>
                    <button className="btn" onClick={() => setConfirmId(null)}>
                      Cancel
                    </button>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
