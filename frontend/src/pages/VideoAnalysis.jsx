import { useEffect, useRef, useState } from 'react'
import { AlertTriangle, CheckCircle, Film, Loader2, Upload, X } from 'lucide-react'
import { analyzeVideo, getAnalysisErrorMessage } from '../utils/api'

const MAX_UPLOAD_MB = 200
const ACCEPTED_EXTENSIONS = /\.(mp4|mov|avi|mkv|webm)$/i

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60)
  const remainder = Math.floor(seconds % 60)
  return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
}

export default function VideoAnalysis() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [selectedFrame, setSelectedFrame] = useState(0)
  const [frameView, setFrameView] = useState('overlay')
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef(null)

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      return undefined
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const selectFile = (selected) => {
    if (!selected) return
    if (!ACCEPTED_EXTENSIONS.test(selected.name)) {
      setError('Unsupported video format. Use MP4, MOV, AVI, MKV, or WEBM.')
      return
    }
    if (selected.size > MAX_UPLOAD_MB * 1024 * 1024) {
      setError(`The video must be smaller than ${MAX_UPLOAD_MB} MB.`)
      return
    }
    setFile(selected)
    setResult(null)
    setSelectedFrame(0)
    setError(null)
  }

  const handleAnalyze = async () => {
    if (!file || loading) return
    setLoading(true)
    setError(null)
    try {
      setResult(await analyzeVideo(file))
      setSelectedFrame(0)
    } catch (requestError) {
      setError(getAnalysisErrorMessage(requestError, 'Video'))
    } finally {
      setLoading(false)
    }
  }

  const frames = result?.frames || []
  const frame = frames[selectedFrame] || frames[0]
  const isAI = result?.ai_confidence > 0.5
  const isLowConfidence = result?.low_confidence ?? result?.status === 'uncertain'
  const resultColor = isLowConfidence ? '#d6bd91' : isAI ? '#e18773' : '#a7b79a'
  const prediction = result?.status === 'uncertain'
    ? isAI ? 'AI-generated video' : 'Real video'
    : result?.prediction
  const duration = result?.duration_seconds || 0

  return (
    <main className="relative z-10 pt-28 pb-20 px-6">
      <div className="max-w-6xl mx-auto">
        <p className="section-label mb-2">Video Forensics</p>
        <h1 className="text-4xl font-bold text-white mb-2">Video <span className="accent-text">Analysis</span></h1>
        <p className="text-slate-400 max-w-2xl mb-10">
          Review the detector result alongside Grad-CAM explanations for the exact representative frames used in scoring.
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          <section className="card lg:col-span-2 h-fit">
            <div
              className={`rounded-xl border-2 border-dashed transition-colors ${dragOver ? 'border-sage-400 bg-sage-400/5' : 'border-white/15'}`}
              onDragOver={(event) => { event.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(event) => {
                event.preventDefault()
                setDragOver(false)
                selectFile(event.dataTransfer.files?.[0])
              }}>
              <label className="flex min-h-48 cursor-pointer flex-col items-center justify-center p-5 text-center">
                <input
                  ref={inputRef}
                  type="file"
                  accept="video/mp4,video/quicktime,video/x-msvideo,video/webm,video/x-matroska"
                  className="hidden"
                  onChange={(event) => {
                    selectFile(event.target.files?.[0])
                    event.target.value = ''
                  }}
                />
                <Upload size={24} className="text-sage-400 mb-3" />
                <span className="text-white font-semibold">Drop a video here or browse</span>
                <span className="text-slate-500 text-xs mt-2">MP4, MOV, AVI, MKV, or WEBM · up to 200 MB</span>
              </label>
            </div>

            {previewUrl && (
              <div className="mt-4">
                <video src={previewUrl} controls className="w-full rounded-xl bg-black/50" aria-label="Uploaded video preview" />
                <div className="mt-3 flex items-center gap-3 rounded-xl bg-white/[0.03] p-3">
                  <Film size={18} className="text-sage-400" />
                  <span className="text-sm text-slate-300 truncate flex-1">{file.name}</span>
                  <button type="button" onClick={() => { setFile(null); setResult(null); setError(null) }}
                    className="text-slate-500 hover:text-white" aria-label="Remove video">
                    <X size={17} />
                  </button>
                </div>
              </div>
            )}

            {error && <p role="alert" className="mt-4 text-sm text-coral-400">{error}</p>}
            <button type="button" onClick={handleAnalyze} disabled={!file || loading}
              className="btn-primary w-full mt-5 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed">
              {loading ? <><Loader2 size={16} className="animate-spin" /> Sampling and explaining frames…</> : 'Analyze video'}
            </button>
            <p className="text-xs leading-relaxed text-slate-500 mt-4">
              The model classifies sampled frames. Grad-CAM highlights model activation and does not prove that a frame is manipulated.
            </p>
          </section>

          <section className="lg:col-span-3 space-y-5">
            {!result && !loading && (
              <div className="card min-h-64 flex flex-col items-center justify-center text-center">
                <Film size={30} className="text-slate-600 mb-3" />
                <p className="text-slate-400">Your video result and frame explanations will appear here.</p>
              </div>
            )}
            {loading && (
              <div role="status" className="card min-h-64 flex flex-col items-center justify-center text-center">
                <Loader2 size={32} className="text-sage-400 animate-spin mb-4" />
                <p className="text-white font-semibold">Analyzing sampled frames</p>
                <p className="text-slate-500 text-sm mt-1">Running the video classifier and Grad-CAM.</p>
              </div>
            )}
            {result && (
              <>
                <div className="card" style={{ borderColor: `${resultColor}35` }}>
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div className="flex items-center gap-3">
                      {isLowConfidence
                        ? <AlertTriangle size={22} style={{ color: resultColor }} />
                        : isAI
                          ? <AlertTriangle size={22} style={{ color: resultColor }} />
                          : <CheckCircle size={22} style={{ color: resultColor }} />}
                      <div>
                        <p className="section-label mb-1">Overall classification</p>
                        <h2 className="text-2xl font-black text-white">{prediction}</h2>
                        <p className="text-xs text-slate-500 mt-1">{result.model_used} · {result.processing_time_ms} ms</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-4xl font-black" style={{ color: resultColor }}>{(result.confidence * 100).toFixed(1)}%</p>
                      <p className="text-xs text-slate-400">Top class score</p>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4 mt-5">
                    <div>
                      <div className="flex justify-between text-xs mb-1"><span className="text-slate-400">AI-generated</span><span className="text-coral-300">{(result.ai_confidence * 100).toFixed(1)}%</span></div>
                      <div className="confidence-bar"><div className="h-full rounded-full bg-coral-400" style={{ width: `${result.ai_confidence * 100}%` }} /></div>
                    </div>
                    <div>
                      <div className="flex justify-between text-xs mb-1"><span className="text-slate-400">Real</span><span className="text-sage-300">{(result.real_confidence * 100).toFixed(1)}%</span></div>
                      <div className="confidence-bar"><div className="h-full rounded-full bg-sage-400" style={{ width: `${result.real_confidence * 100}%` }} /></div>
                    </div>
                  </div>
                  {isLowConfidence && (
                    <p role="status" className="text-sm text-brass-300 mt-4">
                      The model leans toward {isAI ? 'AI-generated' : 'Real'} ({(result.confidence * 100).toFixed(1)}%), but this is a low-confidence prediction.
                    </p>
                  )}
                </div>

                <div className="card">
                  <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                    <div>
                      <h3 className="text-white font-semibold">Sampled-frame timeline</h3>
                      <p className="text-xs text-slate-500 mt-1">{result.frame_count} frames · select a marker to inspect its explanation</p>
                    </div>
                    <span className="text-xs text-slate-500">{formatTime(duration)}</span>
                  </div>
                  <div className="relative h-2 rounded-full bg-white/10 mb-5" aria-label="Suspicious frame timeline">
                    {frames.map((item, index) => {
                      const position = duration > 0
                        ? Math.min(100, Math.max(0, item.timestamp_seconds / duration * 100))
                        : frames.length > 1 ? index / (frames.length - 1) * 100 : 50
                      return (
                        <button key={item.frame_index} type="button"
                          title={`${formatTime(item.timestamp_seconds)} · AI score ${(item.ai_confidence * 100).toFixed(1)}%`}
                          aria-label={`Select frame at ${formatTime(item.timestamp_seconds)}${item.is_suspicious ? ', suspicious score' : ''}`}
                          aria-pressed={selectedFrame === index}
                          onClick={() => setSelectedFrame(index)}
                          className={`absolute top-1/2 -translate-x-1/2 -translate-y-1/2 w-4 h-4 rounded-full border-2 transition-transform hover:scale-125 ${item.is_suspicious ? 'bg-coral-400 border-coral-200' : 'bg-sage-400 border-sage-200'} ${selectedFrame === index ? 'ring-2 ring-white/70 ring-offset-2 ring-offset-dark-800' : ''}`}
                          style={{ left: `${position}%` }}
                        />
                      )
                    })}
                  </div>
                  <div className="flex gap-2 overflow-x-auto pb-2">
                    {frames.map((item, index) => (
                      <button key={item.frame_index} type="button" onClick={() => setSelectedFrame(index)}
                        aria-pressed={selectedFrame === index}
                        className={`min-w-24 rounded-lg border p-2 text-left transition-colors ${selectedFrame === index ? 'border-sage-400/50 bg-sage-400/10' : 'border-white/10 hover:border-white/25'}`}>
                        <span className="block text-xs text-white">{formatTime(item.timestamp_seconds)}</span>
                        <span className={`block text-[10px] mt-1 ${item.is_suspicious ? 'text-coral-300' : 'text-sage-300'}`}>
                          AI {(item.ai_confidence * 100).toFixed(0)}%
                        </span>
                      </button>
                    ))}
                  </div>
                </div>

                {frame && (
                  <div className="card">
                    <div className="flex flex-wrap items-start justify-between gap-3 mb-4">
                      <div>
                        <h3 className="text-white font-semibold">Frame explanation · {formatTime(frame.timestamp_seconds)}</h3>
                        <p className="text-xs text-slate-400 mt-1">{frame.region_summary}</p>
                      </div>
                      <p className="text-sm font-semibold text-coral-300">AI score {(frame.ai_confidence * 100).toFixed(1)}%</p>
                    </div>
                    <div className="flex flex-wrap gap-2 mb-4" role="group" aria-label="Frame visualization">
                      {[
                        ['original', 'Original'],
                        ['heatmap', 'Heatmap'],
                        ['overlay', 'Overlay'],
                      ].map(([value, label]) => (
                        <button key={value} type="button" onClick={() => setFrameView(value)}
                          aria-pressed={frameView === value}
                          className={`px-3 py-1.5 rounded-lg text-xs border transition-colors ${frameView === value ? 'text-sage-300 border-sage-400/40 bg-sage-400/10' : 'text-slate-400 border-white/10 hover:text-white'}`}>
                          {label}
                        </button>
                      ))}
                    </div>
                    <img src={frame[frameView]} alt={`${frameView} explanation for video frame ${frame.frame_index}`}
                      className="w-full max-h-[28rem] rounded-xl object-contain bg-black/30" />
                  </div>
                )}
              </>
            )}
          </section>
        </div>
      </div>
    </main>
  )
}
