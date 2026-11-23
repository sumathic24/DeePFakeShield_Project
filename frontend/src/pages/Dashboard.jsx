import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { FileText, Image, Film, TrendingUp, Activity, AlertTriangle, CheckCircle } from 'lucide-react'
import { AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { Link } from 'react-router-dom'
import { getDashboardStats } from '../utils/api'

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  show: (i = 0) => ({ opacity: 1, y: 0, transition: { duration: 0.5, delay: i * 0.08 } }),
}

const statCards = [
  { label: 'All Analyses', icon: Activity, color: '#a7b79a', getValue: data => data.all_time_analyses },
  { label: 'AI-Generated Content (7 days)', icon: AlertTriangle, color: '#e18773', getValue: data => data.ai_content_detected },
  { label: 'Mean Confidence (7 days)', icon: TrendingUp, color: '#d0af7b', getValue: data => data.avg_confidence, suffix: '%' },
  { label: 'Human / Real Content (7 days)', icon: CheckCircle, color: '#a7b79a', getValue: data => data.verified_real },
]

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-strong rounded-xl px-4 py-3 text-xs">
        <p className="text-slate-400 mb-1">{label}</p>
        {payload.map((p) => (
          <p key={p.dataKey} style={{ color: p.color }}>{p.dataKey}: <strong>{p.value}</strong></p>
        ))}
      </div>
    )
  }
  return null
}

export default function Dashboard() {
  const [stats, setStats] = useState(null)
  const [recentAnalyses, setRecentAnalyses] = useState([])
  const [error, setError] = useState(null)

  useEffect(() => {
    let active = true
    getDashboardStats().then(data => {
      if (!active) return
      setStats(data.stats)
      setRecentAnalyses(data.recent.items)
    }).catch(() => {
      if (active) setError('Dashboard activity is unavailable. Check that the backend is running.')
    })
    return () => { active = false }
  }, [])

  const areaData = stats?.daily || []
  const breakdown = stats?.breakdown || {}
  const barData = [
    { name: 'AI Text', value: breakdown.ai_text || 0, color: '#e18773' },
    { name: 'Human Text', value: breakdown.human_text || 0, color: '#a7b79a' },
    { name: 'AI Image', value: breakdown.ai_image || 0, color: '#d0af7b' },
    { name: 'Real Image', value: breakdown.real_image || 0, color: '#758574' },
    { name: 'AI Video', value: breakdown.ai_video || 0, color: '#e18773' },
    { name: 'Real Video', value: breakdown.real_video || 0, color: '#849481' },
  ]
  const pieData = [
    { name: 'AI Text', value: breakdown.ai_text || 0, color: '#e18773' },
    { name: 'Human Text', value: breakdown.human_text || 0, color: '#a7b79a' },
    { name: 'AI Image', value: breakdown.ai_image || 0, color: '#d0af7b' },
    { name: 'Real Image', value: breakdown.real_image || 0, color: '#758574' },
    { name: 'AI Video', value: breakdown.ai_video || 0, color: '#e18773' },
    { name: 'Real Video', value: breakdown.real_video || 0, color: '#849481' },
  ].filter(item => item.value > 0)

  return (
    <main className="relative z-10 pt-28 pb-20 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div variants={fadeUp} initial="hidden" animate="show" custom={0} className="mb-10">
          <p className="section-label mb-2">Analytics Overview</p>
          <h1 className="text-4xl font-bold text-white mb-2">
            Detection <span className="accent-text">Dashboard</span>
          </h1>
          <p className="text-slate-400">Recorded activity and classification trends across all three modalities.</p>
        </motion.div>
        {error && <p role="alert" className="mb-6 text-sm text-coral-400">{error}</p>}

        {/* Top stats */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {statCards.map((s, i) => (
            <motion.div key={s.label} variants={fadeUp} initial="hidden" animate="show" custom={i * 0.1}
              className="card group">
              <div className="flex items-start justify-between mb-3">
                <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                  style={{ background: `${s.color}18`, border: `1px solid ${s.color}30` }}>
                  <s.icon size={18} style={{ color: s.color }} />
                </div>
              </div>
              <p className="text-2xl font-bold text-white">
                {!stats ? '—' : s.getValue(stats) == null ? 'N/A' : `${Number(s.getValue(stats)).toLocaleString()}${s.suffix || ''}`}
              </p>
              <p className="text-xs text-slate-500 mt-1">{s.label}</p>
            </motion.div>
          ))}
        </div>

        {/* Charts row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Area chart */}
          <motion.div variants={fadeUp} initial="hidden" animate="show" custom={0.5}
            className="card lg:col-span-2">
            <div className="flex items-center justify-between mb-6">
              <div>
                <p className="text-white font-semibold">Analyses This Week</p>
                <p className="text-xs text-slate-500 mt-1">Recorded, classified text, image, and video analyses</p>
              </div>
              <div className="flex items-center gap-4 text-xs text-slate-400">
                <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-sage-400 inline-block rounded" /> Text</span>
                <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-brass-400 inline-block rounded" /> Image</span>
                <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-coral-400 inline-block rounded" /> Video</span>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={areaData}>
                <defs>
                  <linearGradient id="colorText" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#a7b79a" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#a7b79a" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorImage" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#d0af7b" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#d0af7b" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="day" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="text" stroke="#a7b79a" strokeWidth={2} fill="url(#colorText)" />
                <Area type="monotone" dataKey="image" stroke="#d0af7b" strokeWidth={2} fill="url(#colorImage)" />
                <Area type="monotone" dataKey="video" stroke="#e18773" strokeWidth={2} fill="none" />
              </AreaChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Pie chart */}
          <motion.div variants={fadeUp} initial="hidden" animate="show" custom={0.6} className="card">
            <p className="text-white font-semibold mb-1">Content Breakdown</p>
            <p className="text-xs text-slate-500 mb-4">Recorded classified results</p>
            {pieData.length ? <ResponsiveContainer width="100%" height={160}>
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" innerRadius={45} outerRadius={70} paddingAngle={4} dataKey="value">
                  {pieData.map((entry, index) => <Cell key={index} fill={entry.color} />)}
                </Pie><Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer> : <div className="h-40 flex items-center justify-center text-xs text-slate-500">No classified results yet</div>}
            <div className="space-y-2 mt-2">
              {pieData.map(d => (
                <div key={d.name} className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full" style={{ background: d.color }} />
                    <span className="text-slate-400">{d.name}</span>
                  </div>
                  <span className="text-white font-medium">{d.value}</span>
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Bar chart + recent */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <motion.div variants={fadeUp} initial="hidden" animate="show" custom={0.7} className="card">
            <p className="text-white font-semibold mb-1">Recorded Classifications</p>
            <p className="text-xs text-slate-500 mb-6">Counts exclude uncertain results</p>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={barData} barSize={28}>
                <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                  {barData.map((entry, index) => (
                    <Cell key={index} fill={entry.color} fillOpacity={0.8} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Recent analyses */}
          <motion.div variants={fadeUp} initial="hidden" animate="show" custom={0.8} className="card">
            <div className="flex items-center justify-between mb-6">
              <p className="text-white font-semibold">Recent Analyses</p>
              <Link to="/text-analysis" className="text-xs text-sage-400 hover:text-sage-300 transition-colors">
                New analysis →
              </Link>
            </div>
            <div className="space-y-3">
              {recentAnalyses.map((a, i) => (
                <motion.div key={i} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.9 + i * 0.08 }}
                  className="flex items-center gap-3 p-3 rounded-xl transition-colors hover:bg-white/3">
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                    style={{ background: `${a.uncertain ? '#d6bd91' : a.is_ai ? '#e18773' : '#a7b79a'}15` }}>
                    {a.kind === 'text' ? <FileText size={14} /> : a.kind === 'video' ? <Film size={14} /> : <Image size={14} />}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-white font-medium truncate">{a.label}</p>
                    <p className="text-[11px] text-slate-500">{a.kind === 'text' ? 'Text' : a.kind} · {a.confidence}% confidence</p>
                  </div>
                  <span className="text-[11px] font-semibold px-2 py-1 rounded-lg flex-shrink-0"
                    style={{ background: `${a.uncertain ? '#d6bd91' : a.is_ai ? '#e18773' : '#a7b79a'}15`, color: a.uncertain ? '#d6bd91' : a.is_ai ? '#e18773' : '#a7b79a' }}>{a.prediction}</span>
                </motion.div>
              ))}
              {!recentAnalyses.length && !error && <p className="text-sm text-slate-500">No analyses recorded yet.</p>}
            </div>
          </motion.div>
        </div>
      </div>
    </main>
  )
}
