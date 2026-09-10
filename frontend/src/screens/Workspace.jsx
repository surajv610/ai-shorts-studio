import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api'
import { stageLabel, formatDate, stageRoute } from '../labels'
import { friendlyError } from '../errors'
import PipelineProgress from '../components/PipelineProgress'
import StageList from '../components/StageList'
import NextActionCard from '../components/NextActionCard'
import GenerationConfirm from '../components/GenerationConfirm'

const SCREEN_LINKS = [
  ['story', 'Story'],
  ['images', 'Images'],
  ['videos', 'Videos'],
  ['assembly', 'Assembly & QC'],
  ['metadata', 'Metadata'],
  ['final', 'Final Review'],
]

export default function Workspace() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [summary, setSummary] = useState(null)
  const [settings, setSettings] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)
  const [confirm, setConfirm] = useState(null)

  async function load() {
    try {
      setError('')
      const s = await api.projectSummary(id)
      setSummary(s)
      try {
        setSettings(await api.settings())
      } catch (_) {}
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

  const routeFor = (goVal) => {
    const go = (goVal && goVal !== '' ? goVal : 'story')
    return `/projects/${id}/${go}`
  }

  function onStageSelect(key) {
    navigate(`/projects/${id}/${stageRoute(key)}`)
  }

  // Decode the backend next-action into an explicit API call or navigation.
  async function dispatch(na) {
    setBusy(true)
    setError('')
    setNotice('')
    try {
      switch (na.action) {
        case 'START_PRODUCTION': {
          const p = await api.getProject(id)
          if (p.status === 'DRAFT') await api.startPlanning(id)
          const res = await api.planStory(id)
          _afterStoryRun(res)
          break
        }
        case 'GENERATE_STORY': {
          const res = await api.planStory(id)
          _afterStoryRun(res)
          break
        }
        case 'ASSEMBLE_VIDEO': {
          const res = await api.assembleVideo(id)
          _afterAssembly(res)
          break
        }
        case 'REASSEMBLE_VIDEO': {
          const res = await api.reassembleVideo(id)
          _afterAssembly(res)
          break
        }
        case 'RUN_QC': {
          const res = await api.runQC(id)
          setNotice(
            `Quality check complete: ${res.overall}. Review the findings before uploading.`
          )
          break
        }
        case 'GENERATE_METADATA': {
          await api.metadataGenerate(id)
          setNotice('Upload package generated. Review and edit it on the Final Review screen.')
          break
        }
        default:
          break
      }
      await load()
    } catch (e) {
      const ctx = /ASSEMBL/.test(na.action) ? 'assembly' : ''
      setError(friendlyError(e, ctx))
      await load()
    } finally {
      setBusy(false)
    }
  }

  function _afterStoryRun(res) {
    if (res && res.errors && res.errors.length) {
      setNotice('The storyboard was generated but reported issues. Review it and regenerate if needed.')
    } else {
      setNotice('Storyboard ready. Review it on the next screen.')
    }
  }

  function _afterAssembly(res) {
    if (res && res.error) {
      setError(friendlyError(res.error, 'assembly'))
    } else if (res && res.reused) {
      setNotice('The final video is already up to date.')
    } else if (res && res.message) {
      setNotice(res.message)
    }
  }

  async function startUI() {
    const na = summary?.next_action
    if (!na) return
    if (na.blocking) {
      navigate(routeFor(na.go))
      return
    }
    if (na.operation === 'images' || na.operation === 'videos') {
      setConfirm(na.operation)
      return
    }
    await dispatch(na)
  }

  async function confirmGeneration(operation) {
    const na = summary?.next_action
    setConfirm(null)
    setBusy(true)
    setError('')
    setNotice('')
    try {
      let res
      if (operation === 'images') {
        res = await api.generateImages(id)
        const errMsg = res?.errors?.[res.errors.length - 1] || ''
        if (errMsg.includes('Image generation failed')) {
          setNotice('Some images could not be generated. No candidates were charged for failed scenes — review and retry.')
        } else {
          setNotice('Images generated. Pick A or B for every scene.')
        }
      } else {
        res = await api.generateVideos(id)
        const errMsg = res?.errors?.[res.errors.length - 1] || ''
        if (errMsg.includes('Video generation failed')) {
          setNotice('Video generation could not be started. Your existing completed clips were preserved.')
        } else {
          setNotice('Clips generated. Review them before assembling.')
        }
      }
      await load()
    } catch (e) {
      setError(friendlyError(e, operation === 'videos' ? 'videos' : ''))
      await load()
    } finally {
      setBusy(false)
    }
  }

  if (loading) return <div className="page"><p className="muted">Loading project…</p></div>

  if (!summary) {
    return (
      <div className="page">
        <div className="alert error">{error || 'Project not found.'}</div>
        <Link className="btn link" to="/">← Back to projects</Link>
      </div>
    )
  }

  const next = summary.next_action || {}

  return (
    <div className="page">
      <div className="workspace-top">
        <Link className="btn link" to="/">← All projects</Link>
        {summary.status !== 'DRAFT' && (
          <span className={`status-chip ${summary.status === 'FAILED' ? 'chip-stale' : ''}`}>
            {stageLabel(summary.status)}
          </span>
        )}
      </div>

      {confirm && (
        <GenerationConfirm
          operation={confirm}
          estimate={
            confirm === 'images'
              ? summary.image_generation_estimate
              : summary.video_generation_estimate
          }
          settings={settings}
          busy={busy}
          onCancel={() => setConfirm(null)}
          onConfirm={() => confirmGeneration(confirm)}
        />
      )}

      <div className="project-header">
        <div>
          <h1>{summary.name || summary.idea}</h1>
          <p className="muted idea-line">{summary.idea}</p>
          <div className="project-meta">
            <span>{summary.scenes?.length || 0} scenes</span>
            <span>Updated {formatDate(summary.updated_at)}</span>
          </div>
        </div>
      </div>

      {error && <div className="alert error">{error}</div>}
      {notice && <div className="alert">{notice}</div>}

      {next.action && (
        <NextActionCard
          nextAction={next}
          busy={busy}
          onContinue={startUI}
        />
      )}

      {busy && (
        <div className="alert waiting">
          <strong>Working…</strong>
          <div className="progress-bar"><span /></div>
        </div>
      )}

      <h2>Production pipeline</h2>
      <PipelineProgress
        stages={summary.stages}
        progress={summary.progress}
        onSelect={onStageSelect}
      />

      <div className="workspace-grid">
        <div>
          <h2>Stage status</h2>
          <StageList stages={summary.stages} id={id} />
        </div>

        <div className="side-panel">
          <h2>Open a screen</h2>
          <ul className="screen-links">
            {SCREEN_LINKS.map(([route, label]) => (
              <li key={route}>
                <Link to={`/projects/${id}/${route}`}>{label} →</Link>
              </li>
            ))}
          </ul>
          <p className="muted small">
            The pipeline above always shows the true, current state.
          </p>
        </div>
      </div>
    </div>
  )
}