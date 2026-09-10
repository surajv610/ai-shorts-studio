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

function sceneState(scene) {
  if (scene.selected_image_id || scene.selected_image) return 'locked'
  const ok = (scene.image_candidates || []).filter((c) => c.status === 'SUCCEEDED')
  if (ok.length) return 'ready'
  return 'pending'
}

const STATE_LABEL = {
  pending: 'waiting',
  ready: 'ready to select',
  locked: 'locked',
}

export default function ImageSelection() {
  const { id } = useParams()
  const [project, setProject] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [regenTarget, setRegenTarget] = useState(null)
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

  const generate = () =>
    run(async () => {
      await api.generateImages(id)
      setOpenScene(project?.scenes?.[0]?.id ?? 0)
    })

  const selectCandidate = (scene, candidate) =>
    run(() => api.selectOneImage(id, scene.id, candidate))

  const confirmRegenerate = (scene, candidate) => setRegenTarget({ scene, candidate })
  const doRegenerate = () => {
    const { scene, candidate } = regenTarget
    setRegenTarget(null)
    run(() => api.regenerateImage(id, scene.id, candidate.candidate_id))
  }

  const finishSelection = () => {
    const selections = {}
    for (const scene of project.scenes) {
      selections[scene.id] = scene.selected_image_id || scene.selected_image
    }
    return run(() => api.selectImages(id, selections))
  }

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
  const allLocked = scenes.every((s) => s.selected_image_id || s.selected_image)
  const generating = project.status === 'GENERATING_IMAGES'
  const selecting = project.status === 'WAITING_FOR_IMAGE_SELECTION'
  const pastSelection = !generating && !selecting

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
        <span className={`status-chip paused-${selecting || generating}`}>
          {stageLabel(project.status)}
        </span>
      </div>

      {error && <div className="alert error">{error}</div>}

      {generating && (
        <div className="alert waiting">
          <strong>Images are being generated.</strong> Candidates appear below as
          each scene finishes — you&apos;ll pick the best of A/B per scene.
        </div>
      )}
      {selecting && !allLocked && (
        <div className="alert waiting">
          <strong>Select one image per scene.</strong> Regenerate only regenerates
          the individual candidate you ask for — never your locked choices.
        </div>
      )}

      {pastSelection && (
        <div className="alert">
          Image selection is complete.
          {project.status === 'GENERATING_VIDEOS' ? (
            <>
              {' '}Every scene is locked. Next step: animate each into a clip.{' '}
              <Link className="btn btn-primary btn-sm" to={`/projects/${id}/videos`}>
                Go to video generation →
              </Link>
            </>
          ) : (
            ' The project has moved past the video generation phase.'
          )}
        </div>
      )}

      <div className="img-toolbar">
        <button
          className="btn btn-primary"
          disabled={busy || !(generating || selecting)}
          onClick={generate}
        >
          {busy ? 'Working…' : 'Generate Candidates'}
        </button>
        <button
          className="btn"
          disabled={busy || !selecting || !allLocked}
          onClick={finishSelection}
        >
          Continue to videos →
        </button>
      </div>

      <div className="scene-progress">
        <span className="sp-label">Progress</span>
        {scenes.map((scene, i) => {
          const st = sceneState(scene)
          return (
            <span
              key={scene.id}
              className={`sp-chip sp-${st} ${openScene === scene.id ? 'sp-active' : ''}`}
              title={`Scene ${i + 1}: ${STATE_LABEL[st]}`}
              onClick={() => setOpenScene(scene.id)}
            >
              {i + 1}
              {st === 'locked' ? ' ✓' : ''}
            </span>
          )
        })}
        <span className="sp-muted">
          {scenes.filter((s) => sceneState(s) === 'locked').length}/{scenes.length} locked
        </span>
      </div>

      <div className="scene-list">
        {scenes.map((scene, i) => {
          const st = sceneState(scene)
          const candidates = scene.image_candidates || []
          const storyScene = project.storyboard?.scenes?.[i]
          return (
            <div key={scene.id} className="scene-card img-scene">
              <div className="scene-head">
                <span className="scene-num">{i + 1}</span>
                <h4>{storyScene?.title || `Scene ${i + 1}`}</h4>
                <span className={`status-chip chip-${st}`}>
                  {STATE_LABEL[st]}
                </span>
              </div>
              {storyScene?.visual_description && (
                <p className="scene-desc">{storyScene.visual_description}</p>
              )}

              {candidates.length === 0 && st === 'pending' ? (
                <div className="empty mini">
                  <p className="muted">No candidates yet — generate them above.</p>
                </div>
              ) : (
                <div className="cand-grid">
                  {candidates.map((c) => {
                    const locked = scene.selected_image_id === c.candidate_id
                    const failed = c.status === 'FAILED'
                    return (
                      <div
                        key={c.candidate_id}
                        className={`cand ${locked ? 'cand-selected' : ''} ${failed ? 'cand-failed' : ''}`}
                      >
                        <div className="cand-head">
                          <span className="cand-tag">Candidate {c.candidate_id}</span>
                          {locked && <span className="cand-lock">Selected ✓</span>}
                          {failed && <span className="cand-err">Failed</span>}
                        </div>
                        {!failed && c.storage_path ? (
                          <img
                            className="cand-image"
                            src={assetUrl(c.storage_path)}
                            alt={`Scene ${i + 1} candidate ${c.candidate_id}`}
                            loading="lazy"
                          />
                        ) : (
                          <div className="cand-empty">
                            {failed && c.error ? c.error : 'Unavailable'}
                          </div>
                        )}
                        <div className="cand-meta muted">
                          {c.prompt && (
                            <details className="cand-details">
                              <summary>Prompt</summary>
                              <p>{c.prompt}</p>
                            </details>
                          )}
                          <div className="cand-line">
                            {c.provider} · {c.model}
                            {c.generation_id ? ` · ${c.generation_id.slice(0, 8)}` : ''}
                          </div>
                        </div>
                        <div className="cand-actions">
                          <button
                            className="btn btn-primary btn-sm"
                            disabled={busy || selecting === false}
                            onClick={() => selectCandidate(scene, c.candidate_id)}
                          >
                            {locked ? 'Selected' : 'Use this'}
                          </button>
                          <button
                            className="btn btn-sm"
                            disabled={busy || generating}
                            onClick={() => confirmRegenerate(scene, c)}
                          >
                            Regenerate
                          </button>
                        </div>
                        {regenTarget &&
                          regenTarget.scene.id === scene.id &&
                          regenTarget.candidate.candidate_id === c.candidate_id && (
                            <div className="confirm">
                              <p>
                                Regenerate candidate {c.candidate_id}? This replaces
                                only this image — your selections stay untouched.
                              </p>
                              <div className="confirm-actions">
                                <button className="btn btn-primary btn-sm" onClick={doRegenerate}>
                                  {busy ? 'Working…' : 'Regenerate'}
                                </button>
                                <button className="btn btn-sm" onClick={() => setRegenTarget(null)}>
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

              {st === 'locked' &&
                candidates.length === 0 &&
                (scene.selected_image || scene.selected_image_id) && (
                  <p className="scene-desc">
                    Selected:{' '}
                    {scene.selected_image_id
                      ? `candidate ${scene.selected_image_id}`
                      : assetUrl(scene.selected_image)}
                  </p>
                )}
            </div>
          )
        })}
      </div>

      {allLocked && selecting && (
        <div className="alert waiting">
          <strong>All scenes locked.</strong> Click Continue to videos to move the
          project to the next phase.
        </div>
      )}
    </div>
  )
}