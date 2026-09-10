import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api'
import { stageLabel } from '../labels'

function assetUrl(path) {
  if (!path) return null
  const p = String(path)
  if (p.startsWith('/assets/')) return p
  const marker = 'storage/projects/'
  const idx = p.lastIndexOf(marker)
  if (idx !== -1) return '/assets/' + p.slice(idx + marker.length)
  if (p.startsWith('/')) return p
  return '/assets/' + p
}

function latestGen(scene) {
  const gens = scene.video_generations || []
  return gens.length ? gens[gens.length - 1] : null
}

function videoState(scene) {
  const gen = latestGen(scene)
  if (!gen) return 'pending'
  if (gen.status === 'SUCCEEDED') return 'done'
  if (gen.status === 'FAILED' || gen.status === 'CANCELLED') return 'failed'
  return 'progress'
}

const STATE_LABEL = {
  pending: 'waiting',
  progress: 'processing',
  done: 'animated ✓',
  failed: 'failed',
}

const STATE_ICON = {
  pending: '⏳',
  progress: '⏳',
  done: '✓',
  failed: '❌',
}

export default function VideoGeneration() {
  const { id } = useParams()
  const [project, setProject] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [busyScene, setBusyScene] = useState(null)
  const [busyAction, setBusyAction] = useState(null)
  const [openScene, setOpenScene] = useState(null)

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

  async function run(action, sceneId = null, actionName = null) {
    setBusy(true)
    setError('')
    if (sceneId) {
      setBusyScene(sceneId)
      setBusyAction(actionName)
    }
    try {
      await action()
      await load()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
      setBusyScene(null)
      setBusyAction(null)
    }
  }

  const generateAll = () => run(() => api.generateVideos(id))
  const generateScene = (scene) =>
    run(() => api.generateVideoScene(id, scene.id), scene.id, 'generate')
  const regenerateClip = (scene) =>
    run(() => api.regenerateVideo(id, scene.id), scene.id, 'regenerate')
  const regeneratePrompt = (scene) =>
    run(() => api.regenerateVideoPrompt(id, scene.id), scene.id, 'regenerate-prompt')
  const retry = (scene) =>
    run(() => api.retryVideo(id, scene.id), scene.id, 'retry')

  if (loading) return <div className="page"><p className="muted">Loading…</p></div>

  if (!project) {
    return (
      <div className="page">
        <div className="alert error">{error || 'Project not found.'}</div>
        <Link className="btn link" to="/">← Back to projects</Link>
      </div>
    )
  }

  const scenes = project.scenes || []
  const generating = project.status === 'GENERATING_VIDEOS'
  const allDone = scenes.every((s) => videoState(s) === 'done')
  const finished = project.status !== 'GENERATING_VIDEOS'

  return (
    <div className="page">
      <Link className="btn link" to="/">← Back to projects</Link>

      <div className="project-header">
        <div>
          <h1>{project.name || project.idea}</h1>
          <p className="muted idea-line">{project.idea}</p>
          <div className="project-meta">
            <span>{scenes.length} scenes</span>
            <span>{project.settings?.aspect_ratio}</span>
          </div>
        </div>
        <span className={`status-chip paused-${generating}`}>
          {stageLabel(project.status)}
        </span>
      </div>

      <h2>Which scenes have been animated successfully?</h2>

      {error && <div className="alert error">{error}</div>}

      {finished && (
        <div className="alert waiting">
          <strong>Video generation is complete.</strong> Every scene now has an
          animated clip. Generation history below is kept for reference.
        </div>
      )}
      {generating && allDone && (
        <div className="alert waiting">
          <strong>Every scene is animated.</strong> The project moves forward
          (Assembly) once all scenes are confirmed.
        </div>
      )}

      <div className="img-toolbar">
        <button
          className="btn btn-primary"
          disabled={busy || !generating || allDone}
          onClick={generateAll}
        >
          {busy ? 'Working…' : 'Generate Remaining Clips'}
        </button>
        {['ASSEMBLING', 'READY_FOR_REVIEW', 'ASSEMBLY_FAILED'].includes(project.status) && (
          <Link className="btn btn-primary" to={`/projects/${id}/assembly`}>
            {project.status === 'ASSEMBLY_FAILED' ? 'Retry Assembly →' : 'Go to Assembly →'}
          </Link>
        )}
        <Link className="btn" to={`/projects/${id}/images`}>
          ← Back to image selection
        </Link>
      </div>

      <div className="scene-progress">
        <span className="sp-label">Progress</span>
        {scenes.map((scene, i) => {
          const st = videoState(scene)
          return (
            <span
              key={scene.id}
              className={`sp-chip sp-${st === 'done' ? 'locked' : st === 'failed' ? 'pending' : 'ready'} ${openScene === scene.id ? 'sp-active' : ''}`}
              title={`Scene ${i + 1}: ${STATE_LABEL[st]}`}
              onClick={() => setOpenScene(scene.id)}
            >
              {i + 1} {STATE_ICON[st]}
            </span>
          )
        })}
        <span className="sp-muted">
          {scenes.filter((s) => videoState(s) === 'done').length}/{scenes.length} animated
        </span>
      </div>

      <div className="scene-list">
        {scenes.map((scene, i) => {
          const st = videoState(scene)
          const gen = latestGen(scene)
          const storyScene = project.storyboard?.scenes?.[i]
          const sourceImg = assetUrl(scene.selected_image)
          const clipUrl = gen?.output_path ? assetUrl(gen.output_path) : null
          return (
            <div key={scene.id} className="scene-card img-scene">
              <div className="scene-head">
                <span className="scene-num">{i + 1}</span>
                <h4>{storyScene?.title || `Scene ${i + 1}`}</h4>
                <span className={`status-chip chip-${st}`}>
                  {STATE_ICON[st]} {STATE_LABEL[st]}
                </span>
              </div>
              {storyScene?.visual_description && (
                <p className="scene-desc">{storyScene.visual_description}</p>
              )}

              <div className="vid-panel">
                <div className="vid-thumb">
                  {sourceImg ? (
                    <img
                      className="vid-thumb-img"
                      src={sourceImg}
                      alt={`Scene ${i + 1} source image`}
                      loading="lazy"
                    />
                  ) : (
                    <div className="cand-empty">Locked image not available</div>
                  )}
                  <span className="vid-thumb-label">Source image</span>
                </div>

                <div className="vid-result">
                  {!gen && st === 'pending' && (
                    <div className="cand-empty">
                      No clip yet — generate it (or all scenes) above.
                    </div>
                  )}
                  {gen && st === 'progress' && (
                    <div className="cand-empty">
                      <div className="vid-spinner" />
                      <div>Processing: {gen.status}</div>
                    </div>
                  )}
                  {gen && st === 'failed' && (
                    <div className="cand-empty">
                      <div className="vid-err">{gen.error || 'Generation failed.'}</div>
                      <button
                        className="btn btn-sm btn-primary"
                        disabled={busy || !generating}
                        onClick={() => retry(scene)}
                      >
                        Retry
                      </button>
                    </div>
                  )}
                  {gen && st === 'done' && clipUrl && (
                    <video
                      className="vid-player"
                      src={clipUrl}
                      controls
                      preload="metadata"
                    />
                  )}
                  <span className="vid-thumb-label">
                    {st === 'done'
                      ? `Clip${gen.duration_seconds ? ` · ${gen.duration_seconds}s` : ''}`
                      : 'Animated clip'}
                  </span>
                </div>
              </div>

              {gen && (
                <div className="cand-meta muted">
                  <div className="cand-line">
                    {gen.provider || 'provider'} · {gen.model || 'model'}
                    {gen.job_id ? ` · job ${gen.job_id.slice(0, 12)}` : ''}
                    {gen.attempt_id ? ` · attempt ${gen.attempt_id.slice(0, 8)}` : ''}
                  </div>
                  {(gen.prompt || scene.video_prompt) && (
                    <details className="cand-details">
                      <summary>Animation prompt (details)</summary>
                      <p>{gen.prompt || scene.video_prompt}</p>
                    </details>
                  )}
                  {gen.error && <div className="vid-err small">Error: {gen.error}</div>}
                </div>
              )}

              <div className="cand-actions">
                <button
                  className="btn btn-sm"
                  disabled={busy || !generating || !gen}
                  onClick={() => regenerateClip(scene)}
                >
                  {busyScene === scene.id && busyAction === 'regenerate'
                    ? 'Working…'
                    : 'Regenerate clip'}
                </button>
                <button
                  className="btn btn-sm"
                  disabled={busy || !generating}
                  onClick={() => regeneratePrompt(scene)}
                >
                  {busyScene === scene.id && busyAction === 'regenerate-prompt'
                    ? 'Working…'
                    : 'Regenerate prompt + clip'}
                </button>
                <button
                  className="btn btn-sm"
                  disabled={busy || !generating || st !== 'pending'}
                  onClick={() => generateScene(scene)}
                >
                  Generate clip
                </button>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}