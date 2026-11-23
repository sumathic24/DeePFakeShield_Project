import axios from 'axios'
const BASE_URL = import.meta.env.VITE_API_URL || ''

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
})

export const getAnalysisErrorMessage = (error, modality) => {
  const detail = error?.response?.data?.detail
  if (typeof detail === 'string' && detail.trim()) return detail
  if (error?.code === 'ECONNABORTED') {
    return `${modality} analysis timed out. Check the backend terminal and try again.`
  }
  if (!error?.response) {
    return `Cannot reach the backend. Run start.bat from the project folder and wait for the app to open.`
  }
  return `${modality} analysis failed (HTTP ${error.response.status}). Check the backend terminal for details.`
}

// ── Text Analysis ─────────────────────────────────────────────
export const analyzeText = async (text) => {
  const res = await api.post('/api/text/analyze', { text }, { timeout: 120000 })
  return res.data
}

export const analyzeAIAuthorship = async (text) => {
  const res = await api.post('/api/text/ai-authorship', { text }, { timeout: 120000 })
  return res.data
}

export const analyzeFakeNews = async (text) => {
  const res = await api.post('/api/text/fake-news', { text }, { timeout: 120000 })
  return res.data
}

// ── Image Analysis ────────────────────────────────────────────
export const analyzeImage = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  const res = await api.post('/api/image/analyze', formData, {
    timeout: 120000,
  })
  return res.data
}

export const analyzeVideo = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  const res = await api.post('/api/video/analyze', formData, {
    timeout: 180000,
  })
  return res.data
}

// ── Dashboard Stats ───────────────────────────────────────────
export const getDashboardStats = async () => {
  const [stats, recent] = await Promise.all([
    api.get('/api/stats'),
    api.get('/api/stats/recent'),
  ])
  return { stats: stats.data, recent: recent.data }
}

export default api
