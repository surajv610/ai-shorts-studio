import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api'
import { qcLabel, QC_CHECK_GLYPHS, stageLabel } from '../labels'

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

function formatDuration(secs) {
  if (!secs) return '—'
  return `${Math.round(secs)}s`
}

function formatSize(bytes) {
  if (!bytes) return '—'
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function sceneReady(scene) {
  const gens = scene.video_generations || []
  const gen = gens[gens.length - 1]
  return Boolean(scene.selected_image && scene.video_clip && gen && gen.status === 'SUCCEEDED')
}

export default function Assembly() {
  const { id } = useParams()
  const [project, setProject] = useState(null)
  const [info, setInfo] = useState(null)
  const [result, setResult] = useState(null)
  const [qcInfo, setQcInfo] = useState(null)
  const [qcResult, setQcResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)

  async function load() {
    try {
      setError('')
      const p = await api.getProject(id)
      const s = await api.assemblyStatus(id)
      setProject(p)
      setInfo(s)
      try {
        const q = await api.qcStatus(id)
        setQcInfo(q)
      } catch (_) {
        setQcInfo(null)
      }
      try {
        const qr = await api.qcResult(id)
        setQcResult(qr)
      } catch (_) {
        setQcResult(null)
      }
      try {
        const r = await api.assemblyResult(id)
        setResult(r)
      } catch (_) {
        setResult(null)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    setLoading(true)
    load()
  }, [id])

  async function assemble(reassemble) {
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const res = await (reassemble ? api.reassembleVideo(id) : api.assembleVideo(id))
      setNotice(res.message)
      await load()
    } catch (e) {
      setError(e.message)
      await load()
    } finally {
      setBusy(false)
    }
  }

  async function runQc() {
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const res = await api.runQC(id)
      setNotice(`Quality check: ${qcLabel(res.overall)}`)
      await load()
    } catch (e) {
      setError(e.message)
      await load()
    } finally {
      setBusy(false)
    }
  }

  if (loading) return <div className="page"><p className="muted">Loading…</p></div>

  if (!project || !info) {
    return (
      <div className="page">
        <div className="alert error">{error || 'Project not found.'}</div>
        <Link className="btn link" to="/">← Back to projects</Link>
      </div>
    )
  }

  const scenes = project.scenes || []
  const status = info.status
  const readyCount = scenes.filter(sceneReady).length
  const allReady = readyCount === scenes.length
  const record = info.assembly
  const hasRecord = Boolean(record)
  const assembled = record?.status === 'SUCCEEDED'
  const isStale = info.stale === true
  const failed = status === 'ASSEMBLY_FAILED'
  const review = status === 'READY_FOR_REVIEW'
  const assembling = status === 'ASSEMBLING' && !hasRecord
  const canAssemble = info.can_assemble && !busy

  const previewUrl = result?.final_asset_url
    ? result.final_asset_url
    : project.final_video
      ? assetUrl(project.final_video)
      : null

  const qcReport = qcResult?.report || qcInfo?.qc || null

  const showButton = !assembled || isStale || failed
  const buttonLabel = isStale
    ? 'Assemble Updated Video'
    : failed
      ? 'Retry Assembly'
      : 'Assemble Video'

  const assemblyStage = busy
    ? 'busy'
    : assembled && !isStale
      ? 'done'
      : isStale
        ? 'stale'
        : failed
          ? 'failed'
          : 'pending'

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
        <span className="status-chip">{stageLabel(status)}</span>
      </div>

      <h2>Video Assembly</h2>

      <div className="asm-stages">
        <span className={`asm-stage ${scenes.every((s) => s.selected_image) ? 'done' : ''}`}>
          ✓ Images
        </span>
        <span className={`asm-stage ${allReady ? 'done' : ''}`}>✓ Videos</span>
        <span className={`asm-stage asm-${assemblyStage}`}>
          {assemblyStage === 'done' ? '✓ Assembly' : assemblyStage === 'busy' ? 'Assembly' : assemblyStage === 'stale' ? '! Assembly' : assemblyStage === 'failed' ? '✕ Assembly' : 'Assembly'}
        </span>
        <span className={`asm-stage ${review ? 'done' : ''}`}>Ready for review</span>
      </div>

      {error && <div className="alert error">{error}</div>}
      {notice && <div className="alert">{notice}</div>}

      {isStale && (
        <div className="alert waiting">
          <strong>Your final video is out of date.</strong> One of the scene clips
          changed after assembly. Reassemble to update the final MP4 and rerun
          the quality check.
        </div>
      )}
      {failed && (
        <div className="alert error">
          <strong>Assembly failed.</strong>{' '}
          {(record?.error || '').split('.')[0]} You can retry without regenerating
          any images or clips.
        </div>
      )}

      {busy && (
        <div className="alert waiting">
          <strong>Assembling…</strong>
          <div className="progress-bar"><span /></div>
        </div>
      )}

      <div className="scene-progress">
        <span className="sp-label">Scenes</span>
        {scenes.map((scene, i) => {
          const ok = sceneReady(scene)
          return (
            <span
              key={scene.id}
              className={`sp-chip ${ok ? 'sp-locked' : 'sp-pending'}`}
              title={`Scene ${i + 1}: ${ok ? 'clip ready' : 'clip missing'}`}
            >
              {i + 1} {ok ? '✓' : ''}
            </span>
          )
        })}
        <span className="sp-muted">
          {readyCount}/{scenes.length} clips ready
        </span>
      </div>

      {!allReady && (
        <div className="alert error">
          Not every scene has a usable clip yet. Finish video generation first.
        </div>
      )}

      {showButton && (
        <div className="img-toolbar">
          <button
            className="btn btn-primary"
            disabled={!canAssemble || !allReady}
            onClick={() => assemble(isStale || failed)}
          >
            {busy ? 'Working…' : buttonLabel}
          </button>
          {assembled && (
            <button
              className="btn"
              disabled={busy}
              onClick={() => assemble(true)}
            >
              Reassemble
            </button>
          )}
        </div>
      )}

      {review && assembled && !isStale && (
        <div className="alert waiting">
          <strong>✓ Final video ready.</strong> Review it below before moving on.
          The final stage of V1 is a manual review before uploading to YouTube.
        </div>
      )}

      {hasRecord && (
        <div className="asm-result">
          <div className="asm-result-head">
            <h3>
              {assembled && !isStale
                ? 'Final video'
                : isStale
                  ? 'Final video (out of date)'
                  : record.status === 'FAILED'
                    ? 'Assembly attempt failed'
                    : 'Assembly'}
            </h3>
            <span className={`status-chip chip-${assemblyStage === 'done' ? 'locked' : 'pending'}`}>
              {record.status}
            </span>
          </div>
          <div className="asm-metrics">
            <span>{formatDuration(record.output_duration)} duration</span>
            <span>{record.output_width}×{record.output_height}</span>
            <span>{formatSize(record.output_size_bytes)}</span>
            <span>{record.scene_count} scenes</span>
          </div>
          {previewUrl && (
            <video
              className="vid-player asm-player"
              src={previewUrl}
              controls
              preload="metadata"
            />
          )}
          {record.error && <div className="vid-err small">{record.error}</div>}
        </div>
      )}

      {review && assembled && !isStale && (
        <div className="qc-panel">
          <div className="qc-head">
            <div>
              <h3>Quality check</h3>
              <p className="muted small">
                Automated review of the final video. QC reports findings — it never
                edits your video, and it is not a substitute for your own review.
              </p>
            </div>
            {qcReport && (
              <span className={`status-chip qc-chip qc-chip-${String(qcReport.status).toLowerCase()}`}>
                QC {qcLabel(qcReport.status)}
              </span>
            )}
          </div>

          {qcInfo?.stale ? (
            <div className="alert waiting">
              <strong>QC can’t run yet.</strong> The final video is out of date.
              Reassemble it, then rerun the quality check.
            </div>
          ) : qcReport ? (
            <div
              className={`alert ${qcReport.status === 'PASSED' ? '' : qcReport.status === 'WARNINGS' ? 'waiting' : 'error'}`}
            >
              <strong>QC {qcLabel(qcReport.status)}.</strong> {qcReport.summary}
              {qcReport.status === 'PASSED' &&
                ' Review the video below; this report does not approve anything by itself.'}
            </div>
          ) : (
            <div className="alert waiting">
              <strong>No quality check yet.</strong> Run QC to verify the final
              video before your review.
            </div>
          )}

          {qcReport && (qcReport.checks?.length > 0 || qcReport.findings?.length > 0) && (
            <details className="qc-details">
              <summary>
                Show technical details ({qcReport.checks?.length || 0} checks)
              </summary>
              <ul className="qc-checks">
                {(qcReport.checks || []).map((c) => (
                  <li key={c.check_name} className={`qc-check qc-check-${String(c.status).toLowerCase()}`}>
                    <span className="qc-check-name">
                      {QC_CHECK_GLYPHS[c.status] || '·'} {c.check_name}
                    </span>
                    <span className="qc-check-msg">{c.message}</span>
                    {c.measured_value !== null && c.measured_value !== undefined && (
                      <span className="qc-check-vals">
                        measured: {String(c.measured_value)} · expected:{' '}
                        {String(c.expected_value ?? '—')}
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            </details>
          )}

          {qcInfo?.can_run && (
            <div className="img-toolbar">
              <button
                className="btn btn-primary"
                disabled={busy || !qcInfo.can_run}
                onClick={runQc}
              >
                {busy ? 'Running QC…' : qcReport ? 'Run QC Again' : 'Run QC'}
              </button>
            </div>
          )}

          <div className="img-toolbar">
            <Link className="btn btn-secondary" to={`/projects/${id}/metadata`}>
              Next: YouTube Shorts package →
            </Link>
          </div>
        </div>
      )}

      <div className="img-toolbar">
        <Link className="btn" to={`/projects/${id}/videos`}>
          ← Back to video generation
        </Link>
      </div>
    </div>
  )
}