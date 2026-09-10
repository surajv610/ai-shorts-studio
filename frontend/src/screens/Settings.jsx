import { useEffect, useState } from 'react'
import { api } from '../api'

const STATUS_LABEL = {
  READY: 'Ready',
  CONFIGURED: 'Configured',
  NOT_CONFIGURED: 'Not configured',
  ERROR: 'Error',
}

const DEP_LABELS = {
  database: 'Database',
  storage: 'Storage',
  ffmpeg: 'FFmpeg',
  llm: 'LLM provider',
  image_provider: 'Image provider',
  video_provider: 'Video provider',
}

export default function Settings() {
  const [health, setHealth] = useState(null)
  const [config, setConfig] = useState(null)
  const [error, setError] = useState('')

  async function load() {
    try {
      setError('')
      const [h, c] = await Promise.all([api.health(), api.settings()])
      setHealth(h)
      setConfig(c)
    } catch (e) {
      setError(e.message)
    }
  }

  useEffect(() => {
    load()
  }, [])

  return (
    <div className="page">
      <div className="page-head">
        <h1>Settings &amp; System Health</h1>
        <button className="btn" onClick={load}>
          Refresh
        </button>
      </div>

      {error && <div className="alert error">{error}</div>}

      {!health ? (
        <p className="muted">Loading…</p>
      ) : (
        <>
          <div
            className={`alert ${
              health.status === 'READY' ? '' : 'waiting'
            }`}
          >
            <strong>Overall status: {statusWord(health.status)}</strong>
          </div>

          <h2>Dependencies</h2>
          <div className="bible">
            {Object.entries(health.dependencies || {}).map(([key, value]) => (
              <div className="bible-row" key={key}>
                <span className="bible-key">{DEP_LABELS[key] || key}</span>
                <span
                  className={`dep-status dep-${String(value).toLowerCase()}`}
                >
                  {STATUS_LABEL[value] || value}
                </span>
              </div>
            ))}
          </div>

          <h2>Provider Configuration</h2>
          {config ? (
            <div className="bible">
              {cfgRows(config).map(([key, value]) => (
                <div className="bible-row" key={key}>
                  <span className="bible-key">{key}</span>
                  <span className="bible-value">{String(value)}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted">No configuration info.</p>
          )}
        </>
      )}
    </div>
  )
}

function statusWord(status) {
  return STATUS_LABEL[status] || status
}

function cfgRows(config) {
  const rows = [
    ['LLM provider', providerDisplay(config.llm_provider)],
    ['LLM model', config.llm_model || '—'],
    ['LLM credential', config.llm_credentialed ? 'Configured' : 'Missing'],
    ['LLM status', statusWord(config.llm_status)],
    ['Image provider', providerDisplay(config.image_provider)],
    ['Image model', config.image_model || '—'],
    ['Image configured', yesNo(config.image_configured)],
    ['Video provider', providerDisplay(config.video_provider)],
    ['Video model', config.video_model || '—'],
    ['Video configured', yesNo(config.video_configured)],
    ['Database set', yesNo(config.database_url_set)],
  ]
  return rows
}

function providerDisplay(name) {
  const map = {
    gemini: 'Google Gemini',
    openai: 'OpenAI',
    mock: 'Mock (development)',
    google: 'Google (Gemini)',
  }
  return map[name] || name || '—'
}

function yesNo(v) {
  return v ? 'Yes' : 'No'
}
