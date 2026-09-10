export const WORKFLOW = [
  'DRAFT',
  'PLANNING',
  'WAITING_FOR_STORY_APPROVAL',
  'GENERATING_IMAGES',
  'WAITING_FOR_IMAGE_SELECTION',
  'GENERATING_VIDEOS',
  'ASSEMBLING',
  'READY_FOR_REVIEW',
  'ASSEMBLY_FAILED',
  'QUALITY_CHECK',
  'GENERATING_METADATA',
  'WAITING_FOR_FINAL_APPROVAL',
  'COMPLETED',
]

export const STAGE_LABELS = {
  DRAFT: 'Draft',
  PLANNING: 'Planning',
  WAITING_FOR_STORY_APPROVAL: 'Waiting for story approval',
  GENERATING_IMAGES: 'Generating images',
  WAITING_FOR_IMAGE_SELECTION: 'Waiting for image selection',
  GENERATING_VIDEOS: 'Generating videos',
  ASSEMBLING: 'Assembling',
  READY_FOR_REVIEW: 'Ready for review',
  ASSEMBLY_FAILED: 'Assembly failed',
  QUALITY_CHECK: 'Quality check',
  GENERATING_METADATA: 'Generating metadata',
  WAITING_FOR_FINAL_APPROVAL: 'Waiting for final approval',
  COMPLETED: 'Completed',
  FAILED: 'Failed',
  CANCELLED: 'Cancelled',
}

export const PAUSE_STATUSES = new Set([
  'WAITING_FOR_STORY_APPROVAL',
  'WAITING_FOR_IMAGE_SELECTION',
  'WAITING_FOR_FINAL_APPROVAL',
  'READY_FOR_REVIEW',
])

export const QC_LABELS = {
  PENDING: 'Pending',
  RUNNING: 'Checking…',
  PASSED: 'Passed',
  WARNINGS: 'Warnings',
  FAILED: 'Failed',
  INVALID: 'Out of date',
}

export const QC_CHECK_GLYPHS = {
  PASS: '✓',
  WARN: '⚠',
  FAIL: '✕',
}

export const METADATA_LABELS = {
  DRAFT: 'Draft',
  CURRENT: 'Current',
  STALE: 'Out of date',
}

export function metadataLabel(status) {
  return METADATA_LABELS[status] || status || '—'
}

export function qcLabel(status) {
  return QC_LABELS[status] || status || '—'
}

export function stageLabel(status) {
  return STAGE_LABELS[status] || status
}

export function stepIndex(status) {
  return Math.max(0, WORKFLOW.indexOf(status))
}

export function isPaused(status) {
  return PAUSE_STATUSES.has(status)
}

export function formatDate(iso) {
  if (!iso) return '—'
  return new Date(iso).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

// --- Unified pipeline (deterministic, derived on the backend) ---

export const PIPELINE = [
  'story',
  'images',
  'videos',
  'assembly',
  'qc',
  'metadata',
  'final_review',
  'ready',
]

export const PIPELINE_SHORT = {
  story: 'Story',
  images: 'Images',
  videos: 'Videos',
  assembly: 'Assembly',
  qc: 'QC',
  metadata: 'Metadata',
  final_review: 'Final Review',
  ready: 'Ready to Upload',
}

// Storyboard review lives at /projects/:id/story; images/videos/assembly/
// metadata/final have their own screens. "" from the backend means the story
// review screen.
export function stageRoute(stageKey) {
  const map = {
    story: 'story',
    images: 'images',
    videos: 'videos',
    assembly: 'assembly',
    qc: 'assembly',
    metadata: 'metadata',
    final_review: 'final',
    ready: 'final',
  }
  return map[stageKey]
}

export const STAGE_STATUS_TEXT = {
  completed: 'Completed',
  current: 'In progress',
  waiting: 'Waiting',
  failed: 'Failed',
  stale: 'Out of date',
}

export function stageStatusText(status) {
  return STAGE_STATUS_TEXT[status] || status || '—'
}
