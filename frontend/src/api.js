const BASE = '/api'

async function request(path, options = {}) {
  const resp = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!resp.ok) {
    let detail = resp.statusText
    try {
      const body = await resp.json()
      detail = body.detail || detail
    } catch (_) {}
    throw new Error(detail || `Request failed (${resp.status})`)
  }
  if (resp.status === 204) return null
  return resp.json()
}

export const api = {
  health: () => request('/health'),
  settings: () => request('/settings'),
  listProjects: () => request('/projects'),
  getProject: (id) => request(`/projects/${id}`),
  createProject: (data) =>
    request('/projects', { method: 'POST', body: JSON.stringify(data) }),
  deleteProject: (id) => request(`/projects/${id}`, { method: 'DELETE' }),
  startPlanning: (id) => request(`/projects/${id}/planning`, { method: 'POST' }),
  planStory: (id) => request(`/projects/${id}/plan`, { method: 'POST' }),
  approveStory: (id) =>
    request(`/projects/${id}/story/approve`, { method: 'POST' }),
  rejectStory: (id, reason = '') =>
    request(`/projects/${id}/story/reject`, {
      method: 'POST',
      body: JSON.stringify({ reason }),
    }),
  regenerateStory: (id, reason = '') =>
    request(`/projects/${id}/story/regenerate`, {
      method: 'POST',
      body: JSON.stringify({ reason }),
    }),
  generateImages: (id) =>
    request(`/projects/${id}/images/generate`, { method: 'POST' }),
  regenerateImage: (id, scene_id, candidate_id) =>
    request(`/projects/${id}/images/regenerate`, {
      method: 'POST',
      body: JSON.stringify({ scene_id, candidate_id }),
    }),
  selectOneImage: (id, scene_id, candidate) =>
    request(`/projects/${id}/images/select-one`, {
      method: 'POST',
      body: JSON.stringify({ scene_id, candidate }),
    }),
  selectImages: (id, selections) =>
    request(`/projects/${id}/images/select`, {
      method: 'POST',
      body: JSON.stringify({ selections }),
    }),
  generateVideos: (id) =>
    request(`/projects/${id}/videos/generate`, { method: 'POST' }),
  generateVideoScene: (id, scene_id) =>
    request(`/projects/${id}/videos/generate-scene`, {
      method: 'POST',
      body: JSON.stringify({ scene_id }),
    }),
  regenerateVideo: (id, scene_id) =>
    request(`/projects/${id}/videos/regenerate`, {
      method: 'POST',
      body: JSON.stringify({ scene_id }),
    }),
  regenerateVideoPrompt: (id, scene_id) =>
    request(`/projects/${id}/videos/regenerate-prompt`, {
      method: 'POST',
      body: JSON.stringify({ scene_id }),
    }),
  retryVideo: (id, scene_id) =>
    request(`/projects/${id}/videos/retry`, {
      method: 'POST',
      body: JSON.stringify({ scene_id }),
    }),
  videoSceneStatus: (id, scene_id) =>
    request(`/projects/${id}/videos/${scene_id}/status`),
  assembleVideo: (id) =>
    request(`/projects/${id}/assembly/assemble`, { method: 'POST' }),
  reassembleVideo: (id) =>
    request(`/projects/${id}/assembly/reassemble`, { method: 'POST' }),
  assemblyStatus: (id) => request(`/projects/${id}/assembly/status`),
  assemblyResult: (id) => request(`/projects/${id}/assembly/result`),
  runQC: (id) => request(`/projects/${id}/qc/run`, { method: 'POST' }),
  qcStatus: (id) => request(`/projects/${id}/qc/status`),
  qcResult: (id) => request(`/projects/${id}/qc/result`),
  metadataStart: (id) =>
    request(`/projects/${id}/metadata/start`, { method: 'POST' }),
  metadataGenerate: (id) =>
    request(`/projects/${id}/metadata/generate`, { method: 'POST' }),
  metadataRegenerate: (id) =>
    request(`/projects/${id}/metadata/regenerate`, { method: 'POST' }),
  metadataStatus: (id) => request(`/projects/${id}/metadata/status`),
  metadataResult: (id) => request(`/projects/${id}/metadata/result`),
  metadataPatch: (id, metadata_id, data) =>
    request(`/projects/${id}/metadata/${metadata_id}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    }),
  projectSummary: (id) => request(`/projects/${id}/summary`),
  projectNextAction: (id) => request(`/projects/${id}/next-action`),
}
