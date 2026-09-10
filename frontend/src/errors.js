// Friendly, human-centered error messages for the production workflow.
//
// The backend already sends reasonable detail strings; these mappings only kick
// in for well-known failure families (provider quota / unavailability, assembly
// failures) so the user never sees a raw internal message. No secrets or auth
// headers ever reach these strings.
const QUOTA_PATTERNS = [
  'quota',
  'resource exhausted',
  'resource_exhausted',
  '429',
  'insufficient',
  'spend limit',
  'billing',
]
const UNCONFIGURED_PATTERNS = ['not configured', 'no api key', 'missing api key']
const ASSEMBLY_PATTERNS = ['assembly failed', 'failed to assemble', 'ffmpeg']

export function friendlyError(error, context = '') {
  const raw = String((error && error.message) || error || '')
  const lower = raw.toLowerCase()

  if (QUOTA_PATTERNS.some((p) => lower.includes(p))) {
    return 'Gemini quota is currently unavailable. No paid fallback was used.'
  }

  if (lower.includes('video') && (UNCONFIGURED_PATTERNS.some((p) => lower.includes(p)) || lower.includes('unavailable'))) {
    return 'Video generation could not be started. Your existing completed clips were preserved.'
  }
  if (context === 'assembly' && ASSEMBLY_PATTERNS.some((p) => lower.includes(p))) {
    return 'Final video assembly failed. Your scene clips are safe. Retry assembly.'
  }

  return raw
}

export function qcFriendlyMessage(qcSummary) {
  if (!qcSummary || !qcSummary.status) return ''
  if (qcSummary.status === 'FAILED') {
    return 'QC found problems with the current final video. Review the findings before uploading.'
  }
  if (qcSummary.status === 'WARNINGS') {
    return `QC completed with warnings. Review the findings before uploading (QC ${qcSummary.status}).`
  }
  return ''
}