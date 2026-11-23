import { useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, Image, Loader2, AlertTriangle, CheckCircle, Eye, ChevronDown, ChevronUp, X, Zap } from 'lucide-react'
import { analyzeImage, getAnalysisErrorMessage } from '../utils/api'

const MAX_UPLOAD_MB = 20
const ACCEPTED_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/gif'])

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  show: (i = 0) => ({ opacity: 1, y: 0, transition: { duration: 0.5, delay: i * 0.08 } }),
}

function ConfidenceBar({ value, color, label }) {
  return (
    <div className="mb-3">
      <div className="flex justify-between text-xs mb-1.5">
        <span className="text-slate-400">{label}</span>
        <span className="font-semibold" style={{ color }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="confidence-bar">
        <motion.div initial={{ width: 0 }} animate={{ width: `${value * 100}%` }}
          transition={{ duration: 1, ease: 'easeOut' }} className="h-full rounded-full"
          style={{ background: `linear-gradient(90deg, ${color}80, ${color})` }} />
      </div>
    </div>
  )
}

export default function ImageAnalysis() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [showGradcam, setShowGradcam] = useState(true)
  const [showModels, setShowModels] = useState(true)
  const [dragOver, setDragOver] = useState(false)
  const [imageView, setImageView] = useState('overlay')
  const inputRef = useRef(null)

  const handleFile = useCallback((f) => {
    if (!f) return
    if (!ACCEPTED_TYPES.has(f.type)) {
      setFile(null); setPreview(null); setResult(null)
      setError('Unsupported image type. Use JPG, PNG, WEBP, or GIF.')
      return
    }
    if (f.size > MAX_UPLOAD_MB * 1024 * 1024) {
      setFile(null); setPreview(null); setResult(null)
      setError(`The image must be smaller than ${MAX_UPLOAD_MB} MB.`)
      return
    }
    setFile(f); setResult(null); setError(null)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }, [])

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true); setResult(null); setError(null)
    try {
      const data = await analyzeImage(file)
      setResult(data)
    } catch (requestError) {
      setError(getAnalysisErrorMessage(requestError, 'Image'))
    } finally { setLoading(false) }
  }

  const isAI = result?.status !== 'uncertain' && result?.ai_confidence > 0.5
  const mainColor = result?.status === 'uncertain' ? '#d6bd91' : isAI ? '#e18773' : '#a7b79a'

  return (
    <main className="relative z-10 pt-28 pb-20 px-6">
      <div className="max-w-6xl mx-auto">
        <motion.div variants={fadeUp} initial="hidden" animate="show" custom={0} className="mb-10">
          <p className="section-label mb-2">Computer Vision</p>
          <h1 className="text-4xl font-bold text-white mb-2">Image <span className="accent-text">Analysis</span></h1>
          <p className="text-slate-400 max-w-2xl">Score an image with the configured AI-versus-real detector. Results are marked uncertain when the model score is close to its decision boundary.</p>
        </motion.div>

        <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
          <motion.div variants={fadeUp} initial="hidden" animate="show" custom={1} className="xl:col-span-2 flex flex-col gap-4">
            <div className="card">
              <div className="flex items-center gap-2 mb-4">
                <Image size={16} className="text-sage-400" />
                <span className="text-white font-semibold text-sm">Upload Image</span>
              </div>
              <div
                className={`relative rounded-xl border-2 border-dashed transition-all duration-300 cursor-pointer mb-4 ${dragOver ? 'border-sage-400 bg-sage-400/5' : 'border-white/10 hover:border-sage-400/40'}`}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]) }}
                onClick={() => inputRef.current?.click()}>
                <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp,image/gif" className="hidden" onChange={e => handleFile(e.target.files[0])} />
                {preview ? (
                  <div className="relative">
                    <img src={preview} alt="Preview" className="w-full rounded-xl object-cover max-h-64" />
                    <button onClick={(e) => { e.stopPropagation(); setFile(null); setPreview(null); setResult(null) }}
                      className="absolute top-2 right-2 w-7 h-7 rounded-full bg-black/70 flex items-center justify-center">
                      <X size={13} className="text-white" />
                    </button>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-12 text-center">
                    <div className="w-14 h-14 rounded-2xl flex items-center justify-center mb-3"
                      style={{ background: 'rgba(167,183,154,0.08)', border: '1px solid rgba(167,183,154,0.2)' }}>
                      <Upload size={22} className="text-sage-400" />
                    </div>
                    <p className="text-sm text-slate-300 font-medium">Drop image here</p>
                    <p className="text-xs text-slate-500 mt-1">or click to browse · JPG, PNG, WEBP · max {MAX_UPLOAD_MB} MB</p>
                  </div>
                )}
              </div>
              {error && <p role="alert" className="text-sm text-coral-400 mb-4">{error}</p>}
              {file && (
                <div className="flex items-center gap-2 text-xs text-slate-400 mb-4 px-3 py-2 rounded-lg bg-white/3">
                  <Image size={12} className="text-sage-400" />
                  <span className="truncate">{file.name}</span>
                  <span className="ml-auto">{(file.size / 1024).toFixed(0)} KB</span>
                </div>
              )}
              <button id="analyze-image-btn" onClick={handleAnalyze} disabled={!file || loading}
                className="btn-primary w-full flex items-center justify-center gap-2 py-3.5 disabled:opacity-50 disabled:cursor-not-allowed">
                {loading ? <><Loader2 size={16} className="animate-spin" /> Analyzing…</> : <><Zap size={16} /> Detect AI Image</>}
              </button>
            </div>

            <div className="card">
              <p className="text-white font-semibold text-sm mb-3">Image result</p>
              {result ? (
                <div className="flex items-center justify-between gap-3">
                  <p className="font-bold" style={{ color: mainColor }}>{result.prediction}</p>
                  <p className="text-lg font-black text-white">{(result.confidence * 100).toFixed(1)}%</p>
                </div>
              ) : (
                <p className="text-xs text-slate-400">The result will appear here after analysis.</p>
              )}
            </div>
          </motion.div>

          <div className="xl:col-span-3 flex flex-col gap-5">
            <AnimatePresence mode="wait">
              {!result && !loading && (
                <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="card flex flex-col items-center justify-center text-center py-20">
                  <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4"
                    style={{ background: 'rgba(167,183,154,0.08)', border: '1px solid rgba(167,183,154,0.15)' }}>
                    <Image size={28} className="text-slate-600" />
                  </div>
                  <p className="text-slate-500 font-medium">Upload an image to analyze</p>
                  <p className="text-slate-600 text-sm mt-1">Detector score and status will appear here</p>
                </motion.div>
              )}
              {loading && (
                <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="card flex flex-col items-center justify-center text-center py-20">
                  <div className="relative mb-6">
                    {preview && <img src={preview} alt="" className="w-20 h-20 rounded-xl object-cover opacity-50" />}
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="w-8 h-8 border-2 border-sage-400 border-t-transparent rounded-full animate-spin" />
                    </div>
                  </div>
                  <p className="text-white font-semibold">Running image detector</p>
                  <p className="text-slate-500 text-sm mt-1">Analyzing the uploaded image…</p>
                  <div className="w-48 h-1 bg-white/5 rounded-full mt-6 overflow-hidden">
                    <motion.div className="h-full bg-sage-400 rounded-full"
                      animate={{ x: ['-100%', '200%'] }} transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }} />
                  </div>
                </motion.div>
              )}
              {result && (
                <motion.div key="result" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col gap-5">
                  <div className="card relative overflow-hidden"
                    style={{ borderColor: `${mainColor}30`, background: '#171d19' }}>
                    <div className="absolute top-0 right-0 w-40 h-40 rounded-full pointer-events-none"
                      style={{ background: `radial-gradient(circle, ${mainColor}15, transparent)` }} />
                    <div className="flex items-start justify-between mb-5">
                      <div>
                        <p className="text-xs text-slate-400 mb-1">Image detector result</p>
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: `${mainColor}18` }}>
                            {result.status === 'uncertain' ? <AlertTriangle size={20} style={{ color: mainColor }} /> : isAI ? <AlertTriangle size={20} style={{ color: mainColor }} /> : <CheckCircle size={20} style={{ color: mainColor }} />}
                          </div>
                          <div>
                            <h2 className="text-2xl font-black text-white">{result.prediction}</h2>
                            <p className="text-xs text-slate-400">{result.processing_time_ms}ms · {result.model_used}</p>
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-5xl font-black" style={{ color: mainColor }}>{(result.confidence * 100).toFixed(1)}%</p>
                        <p className="text-xs text-slate-400">Confidence in {result.prediction}</p>
                      </div>
                    </div>
                    <ConfidenceBar value={result.ai_confidence} color="#e18773" label="AI Generated" />
                    <ConfidenceBar value={result.real_confidence} color="#a7b79a" label="Real" />
                    {result.status === 'uncertain' && <p role="status" className="mt-3 text-sm text-brass-300">The model is not confident enough to classify this image. Treat this result as inconclusive.</p>}
                  </div>

                  <div className="card">
                    <button onClick={() => setShowGradcam(!showGradcam)}
                      className="w-full flex items-center justify-between text-sm font-semibold text-white">
                      <span className="flex items-center gap-2"><Eye size={15} className="text-coral-400" /> Explanation</span>
                      {showGradcam ? <ChevronUp size={14} className="text-slate-400" /> : <ChevronDown size={14} className="text-slate-400" />}
                    </button>
                    <AnimatePresence>
                      {showGradcam && (
                        <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                          {result.explanation?.available && result.gradcam ? (
                            <>
                              <p className="text-xs text-slate-400 mt-4 mb-3" title="Grad-CAM uses gradients from the classifier's final spatial feature layer.">
                                {result.explanation.region_summary} Highlights show model activation, not a localized manipulation.
                              </p>
                              <div className="flex flex-wrap gap-2 mb-4" role="group" aria-label="Image explanation view">
                                {[
                                  ['model_input', 'Model input'],
                                  ['heatmap', 'Heatmap'],
                                  ['overlay', 'Overlay'],
                                ].map(([value, label]) => (
                                  <button key={value} type="button" onClick={() => setImageView(value)}
                                    aria-pressed={imageView === value}
                                    className={`px-3 py-1.5 rounded-lg text-xs border transition-colors ${imageView === value ? 'text-sage-300 border-sage-400/40 bg-sage-400/10' : 'text-slate-400 border-white/10 hover:text-white'}`}>
                                    {label}
                                  </button>
                                ))}
                              </div>
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <figure>
                                  <figcaption className="text-xs text-slate-400 mb-2">Original upload</figcaption>
                                  <img src={preview} alt="Original uploaded image"
                                    className="w-full max-h-[28rem] rounded-xl object-contain bg-black/30" />
                                </figure>
                                <figure>
                                  <figcaption className="text-xs text-slate-400 mb-2">
                                    {imageView === 'model_input' ? 'Model input' : `Grad-CAM ${imageView}`}
                                  </figcaption>
                                  <img src={result.gradcam[imageView]}
                                    alt={`${imageView} Grad-CAM view of the analyzed image`}
                                    className="w-full max-h-[28rem] rounded-xl object-contain bg-black/30" />
                                </figure>
                              </div>
                            </>
                          ) : (
                            <p className="text-xs text-slate-500 mt-4">No Grad-CAM visualization was returned for this analysis.</p>
                          )}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>

                  <div className="card">
                    <button onClick={() => setShowModels(!showModels)}
                      className="w-full flex items-center justify-between text-sm font-semibold text-white">
                      <span className="flex items-center gap-2"><Eye size={15} className="text-brass-400" /> Model details</span>
                      {showModels ? <ChevronUp size={14} className="text-slate-400" /> : <ChevronDown size={14} className="text-slate-400" />}
                    </button>
                    <AnimatePresence>
                      {showModels && (
                        <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                          <p className="text-sm text-slate-300 mt-4">{result.model_used}</p>
                          <p className="text-xs text-slate-500 mt-2">AI probability: {(result.ai_confidence * 100).toFixed(1)}% · Real probability: {(result.real_confidence * 100).toFixed(1)}%</p>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </main>
  )
}
