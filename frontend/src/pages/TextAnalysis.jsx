import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { FileText, Zap, Loader2, AlertTriangle, CheckCircle } from 'lucide-react'
import { analyzeAIAuthorship, analyzeFakeNews, getAnalysisErrorMessage } from '../utils/api'

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  show: (i = 0) => ({ opacity: 1, y: 0, transition: { duration: 0.5, delay: i * 0.08 } }),
}

function ConfidenceBar({ value, color, label }) {
  if (value == null) return null
  return (
    <div className="mb-3">
      <div className="flex justify-between text-xs mb-1.5">
        <span className="text-slate-400">{label}</span>
        <span className="font-semibold" style={{ color }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="confidence-bar">
        <motion.div initial={{ width: 0 }} animate={{ width: `${value * 100}%` }}
          transition={{ duration: 1, ease: 'easeOut' }}
          className="h-full rounded-full"
          style={{ background: `linear-gradient(90deg, ${color}80, ${color})` }} />
      </div>
    </div>
  )
}

function TextShapExplanation({ result }) {
  const highlightedText = result?.highlighted_text
  const shapValues = result?.shap_values

  if (!highlightedText?.length && !shapValues?.length) return null

  return (
    <div className="mt-5 pt-5 border-t border-white/8">
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-white">Text attribution · SHAP</h3>
        {result.explanation?.summary && (
          <p className="mt-1 text-xs leading-relaxed text-slate-400">{result.explanation.summary}</p>
        )}
      </div>

      {highlightedText?.length > 0 && (
        <div className="rounded-lg border border-white/8 bg-black/10 p-3 mb-4 leading-7 text-sm text-slate-300">
          {highlightedText.map((segment, index) => {
            const value = Number(segment.shap || 0)
            const color = value > 0
              ? 'rgba(225, 135, 115, 0.28)'
              : value < 0
                ? 'rgba(167, 183, 154, 0.28)'
                : 'transparent'
            return (
              <span
                key={`${index}-${segment.text}`}
                className="rounded px-0.5"
                style={{ backgroundColor: color }}
                title={`SHAP: ${value > 0 ? '+' : ''}${value.toFixed(4)}`}
              >
                {segment.text}
              </span>
            )
          })}
        </div>
      )}

      {shapValues?.length > 0 && (
        <div className="flex flex-col gap-2">
          {shapValues.slice(0, 12).map((item, index) => {
            const value = Number(item.shap || 0)
            const pushesAI = value > 0
            return (
              <div key={`${index}-${item.feature}`} className="grid grid-cols-[minmax(0,1fr)_4.5rem] items-center gap-3">
                <span className="truncate text-xs text-slate-400" title={item.feature}>{item.feature}</span>
                <span className={`text-right text-xs font-mono ${pushesAI ? 'text-coral-300' : 'text-sage-300'}`}>
                  {value > 0 ? '+' : ''}{value.toFixed(4)}
                </span>
              </div>
            )
          })}
        </div>
      )}

      <p className="mt-4 text-xs leading-relaxed text-slate-500">
        Coral segments push toward AI-generated; sage segments push toward human-written.
      </p>
    </div>
  )
}

export default function TextAnalysis() {
  const [text, setText] = useState('')
  const [analysisType, setAnalysisType] = useState('misinformation')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleAnalyze = async () => {
    if (!text.trim() || text.trim().length < 20) {
      setError('Please enter at least 20 characters.')
      return
    }
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const analyze = analysisType === 'misinformation' ? analyzeFakeNews : analyzeAIAuthorship
      setResult(await analyze(text))
    } catch (requestError) {
      setError(getAnalysisErrorMessage(requestError, analysisType === 'misinformation' ? 'Misinformation' : 'AI authorship'))
    } finally {
      setLoading(false)
    }
  }

  const isMisinformationMode = analysisType === 'misinformation'
  const notApplicable = isMisinformationMode && result?.status === 'not_applicable'
  const uncertain = result?.status === 'uncertain' || notApplicable
  const positivePrediction = isMisinformationMode
    ? result?.prediction === 'Fake News'
    : result?.ai_confidence > 0.5
  const mainColor = uncertain ? '#d6bd91' : positivePrediction ? '#e18773' : '#a7b79a'
  const verdict = !result
    ? ''
    : isMisinformationMode
      ? notApplicable
        ? 'Not a news claim'
        : result.status === 'uncertain'
          ? 'Inconclusive'
          : positivePrediction
            ? 'Potential misinformation'
            : 'Likely credible'
      : result.prediction
  const samples = isMisinformationMode
    ? [
      'The government announced a new policy after the election results were certified.',
      'Police reported that the hospital attack occurred on Saturday.',
      'According to officials, the new vaccine approval was announced after clinical trials.',
    ]
    : [
      'The implementation of blockchain technology in modern supply chain management systems facilitates unprecedented levels of traceability and transparency.',
      'I went to the market yesterday and honestly the tomatoes looked terrible. My neighbor said the same thing happened last week.',
    ]

  const chooseAnalysis = (type) => {
    setAnalysisType(type)
    setResult(null)
    setError(null)
  }

  return (
    <main className="relative z-10 pt-28 pb-20 px-6">
      <div className="max-w-6xl mx-auto">
        <motion.div variants={fadeUp} initial="hidden" animate="show" custom={0} className="mb-10">
          <p className="section-label mb-2">NLP Analysis</p>
          <h1 className="text-4xl font-bold text-white mb-2">Text <span className="accent-text">Analysis</span></h1>
          <p className="text-slate-400 max-w-2xl">Choose AI authorship or misinformation. Each check runs separately.</p>
        </motion.div>

        <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
          <motion.div variants={fadeUp} initial="hidden" animate="show" custom={1} className="xl:col-span-2 flex flex-col gap-4">
            <div className="card flex-1">
              <div className="grid grid-cols-2 gap-2 mb-5">
                {[
                  ['misinformation', 'Misinformation'],
                  ['authorship', 'AI vs Human'],
                ].map(([type, label]) => (
                  <button key={type} type="button" aria-pressed={analysisType === type}
                    disabled={loading} onClick={() => chooseAnalysis(type)}
                    className={`rounded-lg px-3 py-2 text-sm font-semibold transition-colors ${analysisType === type ? 'bg-sage-400/15 text-sage-300 border border-sage-400/40' : 'text-slate-400 border border-white/10 hover:text-white'}`}>
                    {label}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-2 mb-4">
                <FileText size={16} className="text-sage-400" />
                <span className="text-white font-semibold text-sm">{isMisinformationMode ? 'Claim or News Text' : 'Text to Check'}</span>
                <span className="ml-auto text-xs text-slate-500 font-mono">{text.length} chars</span>
              </div>
              <textarea id="text-input" className="input-field resize-none mb-4" rows={12}
                placeholder={isMisinformationMode
                  ? 'Paste a claim or news text to assess possible misinformation...'
                  : 'Paste text to check whether it may be AI-generated or human-written...'}
                value={text} onChange={e => { setText(e.target.value); setError(null) }} />
              {error && (
                <div className="flex items-center gap-2 text-coral-400 text-xs mb-3 px-3 py-2 rounded-lg bg-coral-400/10">
                  <AlertTriangle size={13} /> {error}
                </div>
              )}
              <button id="analyze-text-btn" onClick={handleAnalyze} disabled={loading || !text.trim()}
                className="btn-primary w-full flex items-center justify-center gap-2 py-3.5 disabled:opacity-50 disabled:cursor-not-allowed">
                {loading
                  ? <><Loader2 size={16} className="animate-spin" /> Checking…</>
                  : <><Zap size={16} /> {isMisinformationMode ? 'Check for Misinformation' : 'Check AI vs Human'}</>}
              </button>
              <div className="mt-4">
                <p className="text-xs text-slate-500 mb-2">Try a sample:</p>
                <div className="flex flex-col gap-2">
                  {samples.map((sample, index) => (
                    <button key={`${analysisType}-${index}`}
                      onClick={() => { setText(sample); setResult(null); setError(null) }}
                      className="text-left text-xs px-3 py-2 rounded-lg border border-white/8 text-slate-400 hover:text-white hover:border-sage-400/30 transition-all">
                      {sample.slice(0, 100)}{sample.length > 100 ? '…' : ''}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>

          <div className="xl:col-span-3 flex flex-col gap-5">
            <AnimatePresence mode="wait">
              {!result && !loading && (
                <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="card flex flex-col items-center justify-center text-center py-20">
                  <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4"
                    style={{ background: 'rgba(167,183,154,0.08)', border: '1px solid rgba(167,183,154,0.15)' }}>
                    <FileText size={28} className="text-slate-600" />
                  </div>
                  <p className="text-slate-500 font-medium">Results will appear here</p>
                  <p className="text-slate-600 text-sm mt-1">Choose a check, enter text, and analyze</p>
                </motion.div>
              )}
              {loading && (
                <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="card flex flex-col items-center justify-center text-center py-20">
                  <div className="relative mb-6">
                    <div className="w-16 h-16 rounded-2xl flex items-center justify-center"
                      style={{ background: 'rgba(167,183,154,0.08)', border: '1px solid rgba(167,183,154,0.2)' }}>
                      <FileText size={28} className="text-sage-400" />
                    </div>
                    <div className="absolute -inset-2 rounded-3xl border border-sage-400/20 animate-ping" />
                  </div>
                  <p className="text-white font-semibold">
                    {isMisinformationMode ? 'Assessing claim' : 'Checking authorship'}
                  </p>
                  <p className="text-slate-500 text-sm mt-1">
                    {isMisinformationMode ? 'Checking misinformation patterns…' : 'Running the AI-text detector…'}
                  </p>
                </motion.div>
              )}
              {result && (
                <motion.div key={`result-${analysisType}`} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                  <div className="card relative overflow-hidden"
                    style={{ borderColor: `${mainColor}30`, background: '#171d19' }}>
                    <div className="absolute top-0 right-0 w-48 h-48 rounded-full pointer-events-none"
                      style={{ background: `radial-gradient(circle, ${mainColor}15, transparent)` }} />
                    <div className="flex items-start justify-between mb-5">
                      <div>
                        <p className="text-xs text-slate-400 mb-1">
                          {isMisinformationMode ? 'Misinformation assessment' : 'AI authorship assessment'}
                        </p>
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: `${mainColor}18` }}>
                            {positivePrediction
                              ? <AlertTriangle size={20} style={{ color: mainColor }} />
                              : <CheckCircle size={20} style={{ color: mainColor }} />}
                          </div>
                          <h2 className="text-2xl font-black text-white">{verdict}</h2>
                        </div>
                      </div>
                      {result.confidence != null && (
                        <div className="text-right">
                          <p className="text-4xl font-black" style={{ color: mainColor }}>{(result.confidence * 100).toFixed(1)}%</p>
                          <p className="text-xs text-slate-400">Model confidence</p>
                        </div>
                      )}
                    </div>
                    {notApplicable && <p role="status" className="mb-4 text-sm text-brass-300">{result.reason}</p>}
                    {uncertain && !notApplicable && <p role="status" className="mb-4 text-sm text-brass-300">The model is unsure; treat this result as inconclusive.</p>}
                    {isMisinformationMode ? (
                      <>
                        <ConfidenceBar value={result.fake_score} color="#e18773" label="Misinformation likelihood" />
                        <ConfidenceBar value={result.real_score} color="#a7b79a" label="Credible-label likelihood" />
                        <p className="mt-3 text-xs leading-relaxed text-slate-500">
                          This is a screening estimate, not source-based fact-checking. Verify important claims using reliable sources.
                        </p>
                      </>
                    ) : (
                      <>
                        <ConfidenceBar value={result.ai_confidence} color="#e18773" label="AI Generated" />
                        <ConfidenceBar value={result.human_confidence} color="#a7b79a" label="Human Written" />
                        <p className="mt-3 text-xs leading-relaxed text-slate-500">
                          AI-authorship scores are estimates and are not proof of who wrote the text.
                        </p>
                        <TextShapExplanation result={result} />
                      </>
                    )}
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
