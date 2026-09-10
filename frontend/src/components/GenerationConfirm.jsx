// Confirmation dialog before bulk image / video generation. Shows the exact
// operation estimate (from the backend summary) plus provider/model status from
// /settings — never a price or secrecy claim.
function ProviderStatus({ settings, kind }) {
  if (!settings) return null
  const isImages = kind === 'images'
  const provider = isImages ? settings.image_provider : settings.video_provider
  const model = isImages ? settings.image_model : settings.video_model
  const configured = isImages
    ? settings.image_configured
    : settings.video_configured
  return (
    <li>
      <span className="gen-key">{isImages ? 'Image provider' : 'Video provider'}</span>
      <span className="gen-val">
        {provider} {model ? `· ${model}` : ''} ·{' '}
        {configured ? (
          <span className="ok">Configured</span>
        ) : (
          <span className="muted">Not configured (offline mock)</span>
        )}
      </span>
    </li>
  )
}

const IMAGE_ROWS = {
  scenes: 'Scenes',
  candidates_per_scene: 'Candidates per scene',
  total_candidates: 'Total candidates',
}

const VIDEO_ROWS = {
  scenes_to_generate: 'Scenes requiring generation',
  estimated_clips: 'Estimated clips',
}

export default function GenerationConfirm({
  operation,
  estimate,
  settings,
  onCancel,
  onConfirm,
  busy,
}) {
  const rows = operation === 'images' ? IMAGE_ROWS : VIDEO_ROWS
  const title =
    operation === 'images' ? 'Generate Images' : 'Generate Videos'
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true">
      <div className="modal">
        <h3>{title}</h3>
        <p className="muted">
          This operation generates new assets using your configured providers.
          Review the estimate before continuing.
        </p>

        <ul className="gen-estimate">
          {Object.entries(rows).map(([key, label]) => (
            <li key={key}>
              <span className="gen-key">{label}</span>
              <span className="gen-val">{estimate?.[key] ?? 0}</span>
            </li>
          ))}
          <ProviderStatus settings={settings} kind={operation} />
        </ul>

        <div className="modal-actions">
          <button className="btn" onClick={onCancel} disabled={busy}>
            Cancel
          </button>
          <button className="btn btn-primary" onClick={onConfirm} disabled={busy}>
            {busy ? 'Working…' : title}
          </button>
        </div>
      </div>
    </div>
  )
}