import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api'
import { qcLabel, stageLabel, metadataLabel, formatDate } from '../labels'

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

const EMPTY = {
  title: '',
  description: '',
  hashtags: [],
  keywords: [],
  category: '',
  content_summary: '',
}

function listFrom(value) {
  return String(value || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
}

export default function Metadata() {
  const { id } = useParams()
  const [project, setProject] = useState(null)
  const [status, setStatus] = useState(null)
  const [result, setResult] = useState(null)
  const [form, setForm] = useState(EMPTY)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)
  const [copied, setCopied] = useState('')

  async function load() {
    try {
      setError('')
      const p = await api.getProject(id)
      const s = await api.metadataStatus(id)
      setProject(p)
      setStatus(s)
      try {
        const r = await api.metadataResult(id)
        setResult(r)
      } catch (_) {
        setResult(null)
      }
      const rec = s?.metadata || null
      setForm(
        rec
          ? {
              title: rec.title || '',
              description: rec.description || '',
              hashtags: rec.hashtags || [],
              keywords: rec.keywords || [],
              category: rec.category || '',
              content_summary: rec.content_summary || '',
            }
          : EMPTY
      )
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

  async function generate(regenerate) {
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const res = await (regenerate
        ? api.metadataRegenerate(id)
        : api.metadataGenerate(id))
      setNotice(
        res.qc_warning
          ? `Metadata generated. Note: the last QC was ${res.qc_overall}.`
          : 'Metadata generated — review and edit it below.'
      )
      await load()
    } catch (e) {
      setError(e.message)
      await load()
    } finally {
      setBusy(false)
    }
  }

  async function save() {
    const rec = status?.metadata
    if (!rec) return
    setBusy(true)
    setError('')
    setNotice('')
    try {
      await api.metadataPatch(id, rec.metadata_id, {
        title: form.title,
        description: form.description,
        hashtags: form.hashtags,
        keywords: form.keywords,
        category: form.category,
        content_summary: form.content_summary,
      })
      setNotice('Saved — your edits are the active version now.')
      await load()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  async function copy(which) {
    const rec = status?.metadata
    if (!rec) return
    const map = {
      title: rec.title,
      description: rec.description,
      hashtags: (rec.hashtags || []).join(' '),
      all:
        `${rec.title}\n\n${rec.description}\n\n` +
        (rec.hashtags || []).join(' ') +
        ((rec.keywords || []).length ? `\n\nKeywords: ${rec.keywords.join(', ')}` : ''),
    }
    try {
      await navigator.clipboard.writeText(map[which] || '')
      setCopied(which)
      setTimeout(() => setCopied(''), 1600)
    } catch (_) {
      setError('Copy failed — your browser blocked clipboard access.')
    }
  }

  if (loading) return <div className="page"><p className="muted">Loading…</p></div>

  if (!project || !status) {
    return (
      <div className="page">
        <div className="alert error">{error || 'Project not found.'}</div>
        <Link className="btn link" to="/">← Back to projects</Link>
      </div>
    )
  }

  const rec = status.metadata || null
  const hasMetadata = status.metadata_current === true && rec
  const stale = status.stale === true
  const noAssembly = !status.assembly
  const qcBlocked = status.qc_blocked === true
  const qcWarn = status.qc_warning === true
  const canGenerate = status.can_generate === true && !busy && !hasMetadata
  const canRegenerate = status.can_generate === true && !busy && hasMetadata

  const records = status.metadata_records || []
  const currentIsStale = rec && (status.stale === true || (status.assembly && rec.assembly_id !== status.assembly.assembly_id))
  const previewUrl = result?.final_asset_url
    ? result.final_asset_url
    : project.final_video
      ? assetUrl(project.final_video)
      : null

  return (
    <div className="page">
      <Link className="btn link" to={`/projects/${id}/assembly`}>← Back to assembly</Link>

      <div className="project-header">
        <div>
          <h1>YouTube Shorts Package</h1>
          <p className="muted idea-line">{project.idea}</p>
        </div>
        <span className="status-chip">{stageLabel(status.status)}</span>
      </div>

      {error && <div className="alert error">{error}</div>}
      {notice && <div className="alert">{notice}</div>}

      {noAssembly && (
        <div className="alert error">
          <strong>No final video yet.</strong> Assemble the video before generating
          metadata — this package describes the exact final MP4.
        </div>
      )}
      {stale && (
        <div className="alert error">
          <strong>Your final video is out of date.</strong> A scene clip changed
          after assembly. Reassemble and rerun QC first, then regenerate metadata
          for the new video.
        </div>
      )}
      {!noAssembly && !stale && qcBlocked && (
        <div className="alert error">
          <strong>QC FAILED.</strong> Metadata generation is blocked until the
          video passes the quality check. Fix the issues, rerun QC, then generate.
        </div>
      )}
      {!noAssembly && !stale && !qcBlocked && qcWarn && (
        <div className="alert waiting">
          <strong>QC completed with findings ({status.qc_overall}).</strong>{' '}
          You may proceed, but review the QC report on the assembly screen before
          uploading.
        </div>
      )}
      {!noAssembly && !stale && !qcWarn && !qcBlocked && (
        <div className="alert waiting">
          <strong>Metadata never uploads anything.</strong> This package is a draft
          for <em>your</em> manual review and upload to YouTube. Nothing is
          published automatically.
        </div>
      )}

      {busy && (
        <div className="alert waiting">
          <strong>{canGenerate || canRegenerate ? 'Generating metadata…' : 'Saving…'}</strong>
          <div className="progress-bar"><span /></div>
        </div>
      )}

      <div className="metadata-layout">
        <div className="metadata-col">
          <h2>Final video</h2>
          {previewUrl ? (
            <video
              className="vid-player asm-player"
              src={previewUrl}
              controls
              preload="metadata"
            />
          ) : (
            <p className="muted">No final video to preview.</p>
          )}

          <div className="qc-panel">
            <div className="qc-head">
              <div>
                <h3>Quality check</h3>
                <p className="muted small">
                  Metadata uses the latest QC result as context. QC is advisory —
                  it does not approve anything by itself.
                </p>
              </div>
              {status.qc_overall && (
                <span className={`status-chip qc-chip qc-chip-${String(status.qc_overall).toLowerCase()}`}>
                  QC {qcLabel(status.qc_overall)}
                </span>
              )}
            </div>
            {!status.qc_overall && (
              <p className="muted small">
                QC not run yet — not required, but recommended before generating.
              </p>
            )}
          </div>

          <div className="img-toolbar">
            <button
              className="btn btn-primary"
              disabled={!canGenerate}
              onClick={() => generate(false)}
            >
              {busy ? 'Working…' : 'Generate Metadata'}
            </button>
            <button
              className="btn"
              disabled={!canRegenerate}
              onClick={() => generate(true)}
            >
              Regenerate
            </button>
          </div>
        </div>

        <div className="metadata-col">
          <h2>Upload package</h2>

          {!hasMetadata && (
            <div className="alert waiting">
              <strong>No metadata yet.</strong> Generate it from the final video,
              then edit anything here before you upload.
            </div>
          )}

          {hasMetadata && (
            <>
              {currentIsStale && (
                <div className="alert waiting">
                  This version describes an older final video. Regenerate for the
                  current assembly.
                </div>
              )}

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
                  placeholder="Short, accurate, curiosity-driven (max 100 chars)"
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
                  value={form.description}
                  maxLength={2000}
                  rows={6}
                  onChange={(e) => setForm({ ...form, description: e.target.value })}
                  placeholder="What the viewer is seeing. No invented links, handles, or claims."
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
                  onChange={(e) =>
                    setForm({ ...form, hashtags: listFrom(e.target.value) })
                  }
                  placeholder="#Shorts, #Timelapse, …"
                />
              </label>

              <label className="field-label">
                Keywords (comma separated)
                <input
                  className="text-input"
                  value={form.keywords.join(', ')}
                  onChange={(e) =>
                    setForm({ ...form, keywords: listFrom(e.target.value) })
                  }
                  placeholder="optional search keywords"
                />
              </label>

              <label className="field-label">
                Category
                <input
                  className="text-input"
                  value={form.category}
                  onChange={(e) => setForm({ ...form, category: e.target.value })}
                  placeholder="optional YouTube category"
                />
              </label>

              <label className="field-label">
                One-line summary
                <textarea
                  className="text-input textarea"
                  value={form.content_summary}
                  rows={2}
                  onChange={(e) =>
                    setForm({ ...form, content_summary: e.target.value })
                  }
                  placeholder="optional summary"
                />
              </label>

              <div className="img-toolbar">
                <button className="btn btn-primary" disabled={busy} onClick={save}>
                  Save edits
                </button>
                <button
                  type="button"
                  className="btn"
                  disabled={busy}
                  onClick={() => copy('all')}
                >
                  {copied === 'all' ? 'Copied ✓' : 'Copy All'}
                </button>
              </div>

              {rec.is_edited && (
                <p className="muted small">
                  Edited by you — the untouched AI draft is preserved in the
                  version history below.
                </p>
              )}
            </>
          )}

          <h2>Version history</h2>
          {records.length === 0 ? (
            <p className="muted small">No versions yet.</p>
          ) : (
            <ul className="metadata-history">
              {records.map((r) => (
                <li
                  key={r.metadata_id}
                  className={`metadata-version metadata-version-${String(r.status).toLowerCase()}`}
                >
                  <div className="metadata-version-top">
                    <span className={`status-chip qc-chip`}>
                      {metadataLabel(r.status)}
                    </span>
                    <span className="muted small">{formatDate(r.created_at)}</span>
                  </div>
                  <p className="metadata-version-title">{r.title || '—'}</p>
                  <p className="muted small">
                    {r.provider} / {r.model} · assembly {r.assembly_id.slice(0, 8)}
                    {r.is_edited ? ' · edited' : ''}
                  </p>
                  {r.ai_original && (
                    <details className="qc-details">
                      <summary>Show untouched AI draft</summary>
                      <pre className="metadata-ai-original">
                        {r.ai_original.title}
                        {'\n\n'}
                        {r.ai_original.description}
                        {'\n\n'}
                        {(r.ai_original.hashtags || []).join(' ')}
                      </pre>
                    </details>
                  )}
                </li>
              ))}
            </ul>
          )}

          <div className="img-toolbar">
            <Link className="btn" to={`/projects/${id}/assembly`}>
              ← Back to assembly
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}