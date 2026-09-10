import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { friendlyError } from '../errors'

const VIDEO_TYPES = ['Timelapse', 'Transformation']
const VISUAL_STYLES = ['Photorealistic', 'Cinematic', 'Minimal', '3D Animation']
const ASPECT_RATIOS = ['9:16', '1:1', '16:9', '4:5']

export default function NewProject() {
  const navigate = useNavigate()
  const [form, setForm] = useState({
    name: '',
    idea: 'Seed growing from seed to flowering plant',
    content_type: 'Timelapse',
    visual_style: 'Photorealistic',
    duration_seconds: 30,
    aspect_ratio: '9:16',
  })
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  function set(field, value) {
    setForm((f) => ({ ...f, [field]: value }))
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!form.idea.trim()) {
      setError('Please enter a video idea.')
      return
    }
    setError('')
    setBusy(true)
    try {
      const project = await api.createProject(form)
      if (project.status === 'DRAFT') await api.startPlanning(project.id)
      await api.planStory(project.id)
      navigate(`/projects/${project.id}`)
    } catch (err) {
      setError(friendlyError(err))
      setBusy(false)
    }
  }

  return (
    <div className="page narrow">
      <button className="btn link" onClick={() => navigate('/')}>
        ← Back to projects
      </button>
      <h1>New Project</h1>
      <p className="muted">
        Describe the short you want to produce. Starting production creates the
        project and generates the first storyboard draft.
      </p>

      {error && <div className="alert error">{error}</div>}

      <form onSubmit={handleSubmit} className="form">
        <label className="field">
          <span>Project name <small>(optional)</small></span>
          <input
            type="text"
            value={form.name}
            placeholder="My Seed Timelapse"
            onChange={(e) => set('name', e.target.value)}
          />
        </label>

        <label className="field">
          <span>Video idea</span>
          <textarea
            rows={3}
            value={form.idea}
            onChange={(e) => set('idea', e.target.value)}
          />
        </label>

        <div className="form-row">
          <label className="field">
            <span>Video type</span>
            <select
              value={form.content_type}
              onChange={(e) => set('content_type', e.target.value)}
            >
              {VIDEO_TYPES.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>Visual style</span>
            <select
              value={form.visual_style}
              onChange={(e) => set('visual_style', e.target.value)}
            >
              {VISUAL_STYLES.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="form-row">
          <label className="field">
            <span>Duration (seconds)</span>
            <input
              type="number"
              min={1}
              max={120}
              value={form.duration_seconds}
              onChange={(e) =>
                set('duration_seconds', Number(e.target.value))
              }
            />
          </label>

          <label className="field">
            <span>Aspect ratio</span>
            <select
              value={form.aspect_ratio}
              onChange={(e) => set('aspect_ratio', e.target.value)}
            >
              {ASPECT_RATIOS.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="form-actions">
          <button className="btn" type="button" onClick={() => navigate('/')}>
            Cancel
          </button>
          <button className="btn btn-primary" type="submit" disabled={busy}>
            {busy ? 'Starting…' : 'Start Production'}
          </button>
        </div>
      </form>
    </div>
  )
}
