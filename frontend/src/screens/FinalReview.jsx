import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api'
import { qcLabel, formatDate } from '../labels'
import { friendlyError, qcFriendlyMessage } from '../errors'
import PipelineProgress from '../components/PipelineProgress'

function formatDuration(secs) {
  if (!secs) return '—'
  return `${Math.round(secs)}s`
}

function formatSize(bytes) {
  if (!bytes) return '—'
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function listFrom(value) {
  return String(value || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
}

const EMPTY = { title: '', description: '', hashtags: [], keywords: [] }

export default function FinalReview() {
  const { id } = useParams()
  const [summary, setSummary] = useState(null)
  const [qcReport, setQcReport] = useState(null)
  const [form, setForm] = useState(EMPTY)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)
  const [copied, setCopied] = useState('')
  const [checkedReview, setCheckedReview] = useState(false)

  async function load() {
    try {
      setError('')
      const s = await api.projectSummary(id)
      setSummary(s)
      const meta = s.metadata || null
      setForm(
        meta
          ? {
              title: meta.title || '',
              description: meta.description || '',
              hashtags: meta.hashtags || [],
              keywords: meta.keywords || [],
            }
          : EMPTY
      )
      try {
        const q = await api.qcResult(id)
        setQcReport(q.report || null)
      } catch (_) {
        setQcReport(null)
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

  async function save() {
    const meta = summary?.metadata
    if (!meta) return
    setBusy(true)
    setError('')
    setNotice('')
    try {
      await api.metadataPatch(id, meta.metadata_id, {
        title: form.title,
        description: form.description,
        hashtags: form.hashtags,
        keywords: form.keywords,
      })
      setNotice('Saved — your edits are the active version.')
      await load()
    } catch (e) {
      setError(friendlyError(e))
    } finally {
      setBusy(false)
    }
  }

  async function regenerate() {
    setBusy(true)
    setError('')
    setNotice('')
    try {
      await api.metadataRegenerate(id)
      setNotice('New package generated. Review and save your edits.')
      await load()
    } catch (e) {
      setError(friendlyError(e))
      await load()
    } finally {
      setBusy(false)
    }
  }

  async function copy(which) {
    const meta = summary?.metadata
    if (!meta) return
    const clip =
      which === 'title'
        ? meta.title
        : which === 'description'
          ? meta.description
          : which === 'hashtags'
            ? (meta.hashtags || []).join(' ')
            : [
                `TITLE\n${meta.title}`,
                `DESCRIPTION\n${meta.description}`,
                `HASHTAGS\n${(meta.hashtags || []).join(' ')}`,
                `KEYWORDS\n${(meta.keywords || []).join(', ')}`,
              ].join('\n\n')
    try {
      await navigator.clipboard.writeText(clip || '')
      setCopied(which)
      setTimeout(() => setCopied(''), 1800)
    } catch (_) {
      setError('Copy failed — your browser blocked clipboard access.')
    }
  }

  if (loading) return <div className="page"><p className="muted">Loading…</p></div>

  if (!summary) {
    return (
      <div className="page">
        <div className="alert error">{error || 'Project not found.'}</div>
        <Link className="btn link" to="/">← Back to projects</Link>
      </div>
    )
  }

  const ready = summary.next_action?.action === 'READY_TO_UPLOAD'
  const video = summary.final_video
  const qc = summary.qc
  const meta = summary.metadata
  const videoReady = Boolean(video && video.exists)
  const qcReady = Boolean(qc && qc.current && qc.status !== 'FAILED')
  const metaReady = Boolean(meta && meta.current)

  const missing = []
  if (!videoReady) missing.push('a final video has not been produced yet')
  if (!qcReady) {
    if (!qc || !qc.current) missing.push('QC has not been run on the current final video')
    else if (qc.status === 'FAILED') missing.push('QC failed — fix the video and rerun QC')
  }
  if (!metaReady) missing.push('the upload package is not current for the final video')

  return (
    <div className="page">
      <div className="workspace-top">
        <Link className="btn link" to={`/projects/${id}`}>← Back to project</Link>
      </div>

      <div className="project-header">
        <div>
          <h1>Final Review</h1>
          <p className="muted idea-line">{summary.idea}</p>
          <div className="project-meta">
            <span>Updated {formatDate(summary.updated_at)}</span>
          </div>
        </div>
        <span className={`status-chip ${ready ? 'stale-off' : ''}`}>
          {ready ? 'Ready to Upload' : 'In review'}
        </span>
      </div>

      <PipelineProgress stages={summary.stages} progress={summary.progress} />

      {error && <div className="alert error">{error}</div>}
      {notice && <div className="alert">{notice}</div>}

      {qcFriendlyMessage(qc) && qc.status === 'FAILED' && (
        <div className="alert error">
          <strong>{qcFriendlyMessage(qc)}</strong>
        </div>
      )}
      {qc && qc.status === 'WARNINGS' && (
        <div className="alert waiting">
          <strong>{qcFriendlyMessage(qc)}</strong>
        </div>
      )}

      {ready && (
        <div className="ready-banner" data-testid="ready-banner">
          <div>
            <h2>✓ Ready to Upload</h2>
            <p>
              Your video and metadata are ready.{' '}
              <strong>Upload them manually to YouTube.</strong> Nothing is
              published or scheduled automatically.
            </p>
          </div>
          <div className="img-toolbar">
            {video?.url && (
              <a
                className="btn btn-primary"
                href={video.url}
                target="_blank"
                rel="noreferrer"
              >
                Open Final Video
              </a>
            )}
            <button className="btn" onClick={() => copy('all')} disabled={!meta}>
              {copied === 'all' ? 'Copied ✓' : 'Copy Upload Package'}
            </button>
          </div>
        </div>
      )}

      {!ready && missing.length > 0 && (
        <div className="alert waiting">
          <strong>Not ready yet.</strong> {missing.join(', ')}.
        </div>
      )}

      <div className="review-col">
        <h2>Final video</h2>
        {video?.url ? (
          <video
            className="vid-player asm-player"
            src={video.url}
            controls
            preload="metadata"
          />
        ) : (
          <p className="muted">No final video to preview yet.</p>
        )}
        {video && (
          <div className="asm-metrics">
            <span>{formatDuration(video.duration)} duration</span>
            {video.width && video.height && (
              <span>{video.width}×{video.height}</span>
            )}
            <span>{video.scene_count} scenes</span>
            <span>{formatSize(video.size_bytes)}</span>
          </div>
        )}
      </div>

      <div className="review-grid">
        <div className="review-col">
          <h2>Quality check</h2>
          {qc ? (
            <>
              <div className="qc-head">
                <span className={`status-chip qc-chip qc-chip-${String(qc.status).toLowerCase()}`}>
                  QC {qcLabel(qc.status)}
                </span>
                <span className="muted small">
                  {qc.check_count} checks · {qc.finding_count} findings
                </span>
              </div>
              {qc.summary && <p className="muted">{qc.summary}</p>}
              {qcReport && (qcReport.checks?.length > 0 || qcReport.findings?.length > 0) && (
                <details className="qc-details">
                  <summary>Show technical details ({qcReport.checks?.length || 0} checks)</summary>
                  <ul className="qc-checks">
                    {qcReport.checks.map((c) => (
                      <li key={c.check_name} className={`qc-check qc-check-${String(c.status).toLowerCase()}`}>
                        <span className="qc-check-name">{c.check_name}</span>
                        <span className="qc-check-msg">{c.message}</span>
                      </li>
                    ))}
                    {qcReport.findings.map((f, i) => (
                      <li key={i} className="qc-check qc-check-fail">
                        <span className="qc-check-msg">{f}</span>
                      </li>
                    ))}
                  </ul>
                </details>
              )}
              {qc.current && qc.status === 'PASSED' && (
                <p className="muted small">✓ QC passed for the current final video.</p>
              )}
            </>
          ) : (
            <div className="alert waiting">
              <strong>No QC yet.</strong> Run the quality check from the workspace or assembly screen.
            </div>
          )}
        </div>

        <div className="review-col">
          <h2>Upload package</h2>
          {meta ? (
            <>
              <label className="field-label">
                <span className="field-label-head">
                  Title
                  <button type="button" className="copy-btn" onClick={() => copy('title')}>
                    {copied === 'title' ? 'Copied ✓' : 'Copy'}
                  </button>
                </span>
                <input
                  className="text-input"
                  value={form.title}
                  maxLength={100}
                  onChange={(e) => setForm({ ...form, title: e.target.value })}
                />
              </label>
              <label className="field-label">
                <span className="field-label-head">
                  Description
                  <button type="button" className="copy-btn" onClick={() => copy('description')}>
                    {copied === 'description' ? 'Copied ✓' : 'Copy'}
                  </button>
                </span>
                <textarea
                  className="text-input textarea"
                  rows={5}
                  value={form.description}
                  maxLength={2000}
                  onChange={(e) => setForm({ ...form, description: e.target.value })}
                />
              </label>
              <label className="field-label">
                <span className="field-label-head">
                  Hashtags (comma separated)
                  <button type="button" className="copy-btn" onClick={() => copy('hashtags')}>
                    {copied === 'hashtags' ? 'Copied ✓' : 'Copy'}
                  </button>
                </span>
                <input
                  className="text-input"
                  value={form.hashtags.join(', ')}
                  onChange={(e) => setForm({ ...form, hashtags: listFrom(e.target.value) })}
                />
              </label>
              <label className="field-label">
                Keywords (comma separated)
                <input
                  className="text-input"
                  value={form.keywords.join(', ')}
                  onChange={(e) => setForm({ ...form, keywords: listFrom(e.target.value) })}
                />
              </label>

              <div className="img-toolbar">
                <button className="btn btn-primary btn-sm" disabled={busy} onClick={save}>
                  {busy ? 'Working…' : 'Save Changes'}
                </button>
                <button className="btn btn-sm" disabled={busy} onClick={regenerate}>
                  Regenerate Metadata
                </button>
                <button className="btn btn-sm" disabled={busy} onClick={() => copy('all')}>
                  {copied === 'all' ? 'Copied ✓' : 'Copy All'}
                </button>
              </div>
              {meta.is_edited && (
                <p className="muted small">Edited by you — the untouched AI draft is preserved.</p>
              )}
            </>
          ) : (
            <div className="alert waiting">
              <strong>No upload package yet.</strong> Generate metadata for the final video on the workspace.
            </div>
          )}
        </div>
      </div>

      <h2>Final checklist</h2>
      <ul className="checklist">
        <li className={videoReady ? 'done' : ''}>
          {videoReady ? '✓' : '○'} Video — final MP4 exists
        </li>
        <li className={ready ? 'done' : ''}>
          {ready ? '✓' : '○'} Video — final assembly is current
        </li>
        <li className={qcReady ? 'done' : ''}>
          {qcReady ? '✓' : '○'} Quality — QC is current and passed
        </li>
        <li className={metaReady ? 'done' : ''}>
          {metaReady ? '✓' : '○'} Metadata — title, description, and hashtags ready
        </li>
        <li className={checkedReview ? 'done' : ''}>
          {checkedReview ? '✓' : '○'} You — review the video and package
        </li>
      </ul>
      <label className="checklist-check">
        <input
          type="checkbox"
          checked={checkedReview}
          onChange={(e) => setCheckedReview(e.target.checked)}
        />
        <span>
          I reviewed the final video, the QC result, and the upload package.
          (This only marks your own review — it triggers nothing.)
        </span>
      </label>

      <p className="muted small paper">
        Nothing here uploads, schedules, or publishes to YouTube. The final
        upload is always manual and always yours.
      </p>
    </div>
  )
}