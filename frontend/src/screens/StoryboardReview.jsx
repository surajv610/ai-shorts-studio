import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api'
import { stageLabel, isPaused, formatDate } from '../labels'

const BIBLE_LABELS = {
  subject: 'Subject',
  environment: 'Environment',
  time_lighting: 'Time / Lighting',
  camera: 'Camera',
  lens_look: 'Lens / Look',
  visual_style: 'Visual style',
  color_appearance: 'Color / appearance',
  physical_characteristics: 'Physical characteristics',
  continuity_rules: 'Continuity rules',
  negative_constraints: 'Negative constraints',
  aspect_ratio: 'Aspect ratio',
  target_duration: 'Target duration',
}

export default function StoryboardReview() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [project, setProject] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [feedback, setFeedback] = useState('')
  const [editingId, setEditingId] = useState(null)
  const [editText, setEditText] = useState('')

  async function load() {
    try {
      setError('')
      const p = await api.getProject(id)
      setProject(p)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [id])

  const story = project?.storyboard || null
  const bible = project?.project_bible
  const hasStory = !!story
  const paused = project ? isPaused(project.status) : false

  async function run(action) {
    setBusy(true)
    setError('')
    try {
      await action()
      await load()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const beginStory = () =>
    run(async () => {
      const p = await api.getProject(id)
      if (p.status === 'DRAFT') await api.startPlanning(id)
      await api.planStory(id)
    })

  const approve = () => run(() => api.approveStory(id))
  const regenerate = () => run(() => api.regenerateStory(id, feedback))
  const reject = () => run(() => api.rejectStory(id, feedback))

  function startEdit(scene) {
    setEditingId(scene.scene_number)
    setEditText(scene.visual_description || '')
  }

  function saveEdit(scene) {
    const updated = story.scenes.map((s) =>
      s.scene_number === scene.scene_number
        ? { ...s, visual_description: editText }
        : s
    )
    setProject({
      ...project,
      storyboard: { ...story, scenes: updated },
    })
    setEditingId(null)
  }

  if (loading) return <div className="page"><p className="muted">Loading…</p></div>

  if (!project) {
    return (
      <div className="page">
        <div className="alert error">{error || 'Project not found.'}</div>
        <button className="btn link" onClick={() => navigate('/')}>
          ← Back to projects
        </button>
      </div>
    )
  }

  const totalDuration = story
    ? story.scenes.reduce((sum, s) => sum + (s.estimated_duration || 0), 0)
    : 0

  return (
    <div className="page">
      <button className="btn link" onClick={() => navigate('/')}>
        ← Back to projects
      </button>

      <div className="project-header">
        <div>
          <h1>{project.name || project.idea}</h1>
          <p className="muted idea-line">{project.idea}</p>
          <div className="project-meta">
            <span>Updated {formatDate(project.updated_at)}</span>
            <span>{project.settings?.content_type}</span>
            <span>{project.settings?.aspect_ratio}</span>
          </div>
        </div>
        <span className={`status-chip paused-${paused}`}>
          {stageLabel(project.status)}
        </span>
      </div>

      {error && <div className="alert error">{error}</div>}

      {paused && hasStory && (
        <div className="alert waiting">
          <strong>Waiting for your approval.</strong> The storyboard is ready —
          review it below, then approve, regenerate, or reject.
        </div>
      )}

      {project.status === 'GENERATING_IMAGES' ||
      project.status === 'WAITING_FOR_IMAGE_SELECTION' ? (
        <div className="alert waiting image-banner">
          <span>
            <strong>The storyboard is approved.</strong> Next: generate and pick
            the two image candidates per scene.
          </span>
          <Link className="btn btn-primary btn-sm" to={`/projects/${id}/images`}>
            Go to image selection →
          </Link>
        </div>
      ) : null}

      {project.status === 'GENERATING_VIDEOS' ? (
        <div className="alert waiting image-banner">
          <span>
            <strong>Images are locked.</strong> Next: animate each scene into a
            short vertical clip.
          </span>
          <Link className="btn btn-primary btn-sm" to={`/projects/${id}/videos`}>
            Go to video generation →
          </Link>
        </div>
      ) : null}

      {!hasStory ? (
        <div className="empty">
          <p>
            This project is about to generate its storyboard and Project Bible.
          </p>
          <button
            className="btn btn-primary"
            disabled={busy}
            onClick={beginStory}
          >
            {busy ? 'Working…' : 'Generate Storyboard'}
          </button>
        </div>
      ) : (
        <>
          <h2>Storyboard</h2>
          <p className="muted">{story.summary}</p>

          <div className="scene-list">
            {story.scenes.map((scene) => (
              <div key={scene.scene_number} className="scene-card">
                <div className="scene-head">
                  <span className="scene-num">{scene.scene_number}</span>
                  <h4>{scene.title}</h4>
                  <span className="scene-dur">
                    {scene.estimated_duration}s
                  </span>
                </div>
                {editingId === scene.scene_number ? (
                  <div className="edit-box">
                    <textarea
                      rows={2}
                      value={editText}
                      onChange={(e) => setEditText(e.target.value)}
                    />
                    <div className="edit-actions">
                      <button
                        className="btn btn-primary btn-sm"
                        onClick={() => saveEdit(scene)}
                      >
                        Save
                      </button>
                      <button
                        className="btn btn-sm"
                        onClick={() => setEditingId(null)}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    <p className="scene-desc">{scene.visual_description}</p>
                    <button
                      className="btn btn-sm link"
                      onClick={() => startEdit(scene)}
                    >
                      Edit description
                    </button>
                  </>
                )}
              </div>
            ))}
          </div>

          <div className="total-duration">
            Total estimated duration:{' '}
            <strong>{totalDuration.toFixed(1)}s</strong>
          </div>

          <h2>Project Bible</h2>
          <div className="bible">
            {Object.entries(bible || {}).map(([key, value]) => (
              <div className="bible-row" key={key}>
                <span className="bible-key">{BIBLE_LABELS[key] || key}</span>
                <span className="bible-value">{String(value)}</span>
              </div>
            ))}
          </div>

          {paused && (
            <div className="review-actions">
              <button
                className="btn btn-primary"
                disabled={busy}
                onClick={approve}
              >
                Approve Storyboard
              </button>
              <button className="btn" disabled={busy} onClick={reject}>
                Reject
              </button>
              <div className="feedback">
                <input
                  placeholder="Optional feedback for regeneration"
                  value={feedback}
                  onChange={(e) => setFeedback(e.target.value)}
                />
                <button className="btn" disabled={busy} onClick={regenerate}>
                  Regenerate Storyboard
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
