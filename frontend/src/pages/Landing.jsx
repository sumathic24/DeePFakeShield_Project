import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import {
  ArrowRight, BarChart3, Brain, FileText, Film,
  Globe, Image, Lock, Shield,
} from 'lucide-react'

const reveal = {
  hidden: { opacity: 0, y: 18 },
  show: (index = 0) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.38, delay: index * 0.07, ease: 'easeOut' },
  }),
}

const analysisPaths = [
  { icon: FileText, type: 'TEXT', title: 'Authorship signals', method: 'SHAP word attribution', tone: '#a7b79a', href: '/text-analysis' },
  { icon: Image, type: 'IMAGE', title: 'Artificial vs. real', method: 'Grad-CAM region map', tone: '#d0af7b', href: '/image-analysis' },
  { icon: Film, type: 'VIDEO', title: 'Frame-by-frame review', method: 'Scores + Grad-CAM timeline', tone: '#e18773', href: '/video-analysis' },
]

const capabilities = [
  { icon: FileText, title: 'Text authorship', description: 'Score text with the configured detector. SHAP shows which words move the AI probability up or down.', tone: '#a7b79a' },
  { icon: Image, title: 'Image classification', description: 'Compare artificial and real class scores with a Grad-CAM view of the model input.', tone: '#d0af7b' },
  { icon: Film, title: 'Video frame review', description: 'Inspect the same sampled frames used for the overall score, with individual timestamps and maps.', tone: '#e18773' },
  { icon: Brain, title: 'Uncertainty, made visible', description: 'Near-boundary results are labeled uncertain rather than forced into a confident verdict.', tone: '#a7b79a' },
  { icon: BarChart3, title: 'Recorded activity', description: 'Review analyses and modality breakdowns from the project’s local SQLite history.', tone: '#d0af7b' },
  { icon: Globe, title: 'Separate news workflow', description: 'A separate pretrained classifier estimates fake-news likelihood; it does not verify claims against sources.', tone: '#e18773' },
]

const pipeline = [
  ['01', 'Submit', 'Provide text, an image, or a video.'],
  ['02', 'Process', 'Existing tokenization, transforms, and frame sampling.'],
  ['03', 'Score', 'The configured detector evaluates the input.'],
  ['04', 'Inspect', 'Review the model score and its explanation.'],
]

export default function Landing() {
  return (
    <main className="relative z-10 pt-20">
      <section className="px-6 pt-14 pb-20 lg:pt-20">
        <div className="max-w-7xl mx-auto grid lg:grid-cols-[1.05fr_.95fr] gap-12 items-center">
          <div>
            <motion.div variants={reveal} initial="hidden" animate="show" custom={0}
              className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.025] px-3.5 py-2 text-[11px] font-medium tracking-wide text-slate-300 mb-7">
              <span className="w-1.5 h-1.5 rounded-full bg-brass-400" />
              MULTIMODAL CONTENT REVIEW
            </motion.div>
            <motion.h1 variants={reveal} initial="hidden" animate="show" custom={1}
              className="text-[2.8rem] sm:text-6xl xl:text-7xl font-semibold tracking-[-0.045em] leading-[1.04] text-balance mb-6">
              Evidence first.<br />
              <span className="accent-text">Certainty, never assumed.</span>
            </motion.h1>
            <motion.p variants={reveal} initial="hidden" animate="show" custom={2}
              className="text-base sm:text-lg leading-7 text-slate-400 max-w-xl mb-8">
              Review text, images, and video with model-backed scores and explanations. Each signal is inspectable; none is presented as proof of authenticity.
            </motion.p>
            <motion.div variants={reveal} initial="hidden" animate="show" custom={3}
              className="flex flex-col sm:flex-row gap-3">
              <Link to="/text-analysis" className="btn-primary inline-flex items-center justify-center gap-2">
                Start an analysis <ArrowRight size={15} />
              </Link>
              <Link to="/explainability" className="btn-ghost inline-flex items-center justify-center gap-2">
                How explanations work
              </Link>
            </motion.div>
            <motion.p variants={reveal} initial="hidden" animate="show" custom={4}
              className="mt-7 text-xs tracking-wide text-slate-500">
              TEXT <span className="mx-2 text-slate-700">/</span> IMAGE <span className="mx-2 text-slate-700">/</span> VIDEO
              <span className="mx-2 text-slate-700">·</span> SHAP <span className="mx-1 text-slate-700">+</span> GRAD-CAM
            </motion.p>
          </div>

          <motion.aside initial={{ opacity: 0, x: 14 }} animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.45, delay: 0.18 }} className="card p-5 sm:p-7">
            <div className="flex items-start justify-between gap-4 border-b border-white/[0.07] pb-5 mb-2">
              <div>
                <p className="section-label mb-2">Review workspace</p>
                <h2 className="text-xl font-semibold text-white tracking-tight">Choose an analysis path</h2>
              </div>
              <div className="w-10 h-10 rounded-xl border border-brass-400/20 bg-brass-400/[0.07] flex items-center justify-center">
                <Shield size={19} className="text-brass-300" />
              </div>
            </div>
            <div className="divide-y divide-white/[0.07]">
              {analysisPaths.map(({ icon: Icon, ...path }) => (
                <Link key={path.type} to={path.href}
                  className="group flex items-center gap-4 py-4 rounded-lg focus-visible:outline-offset-2">
                  <div className="w-10 h-10 shrink-0 rounded-xl flex items-center justify-center"
                    style={{ color: path.tone, background: `${path.tone}12`, border: `1px solid ${path.tone}26` }}>
                    <Icon size={18} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-[10px] tracking-[0.14em] font-semibold" style={{ color: path.tone }}>{path.type}</span>
                      <span className="text-xs text-slate-600">/</span>
                      <span className="text-sm font-medium text-slate-200">{path.title}</span>
                    </div>
                    <p className="text-xs text-slate-500">{path.method}</p>
                  </div>
                  <ArrowRight size={15} className="text-slate-600 transition-transform group-hover:translate-x-1 group-hover:text-slate-300" />
                </Link>
              ))}
            </div>
            <div className="mt-2 border-t border-white/[0.07] pt-4 flex items-start gap-2.5">
              <Lock size={14} className="text-brass-400 mt-0.5 shrink-0" />
              <p className="text-xs leading-relaxed text-slate-500">
                Outputs are screening signals, not a substitute for provenance checks or expert review.
              </p>
            </div>
          </motion.aside>
        </div>
      </section>

      <section className="px-6 pb-12">
        <div className="max-w-7xl mx-auto grid grid-cols-2 lg:grid-cols-4 border-y border-white/[0.08]">
          {[
            ['03', 'Content modalities'],
            ['SHAP', 'Text attribution'],
            ['Grad-CAM', 'Visual attribution'],
            ['SQLite', 'Local analysis history'],
          ].map(([value, label], index) => (
            <div key={label} className={`py-5 px-4 sm:px-6 ${index > 0 ? 'border-l border-white/[0.08]' : ''}`}>
              <p className="text-lg sm:text-xl font-semibold text-slate-100 tracking-tight">{value}</p>
              <p className="text-xs text-slate-500 mt-1">{label}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="px-6 py-20">
        <div className="max-w-7xl mx-auto">
          <motion.div variants={reveal} initial="hidden" whileInView="show" viewport={{ once: true }} className="mb-9">
            <p className="section-label mb-2">Capabilities</p>
            <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-3">
              <h2 className="text-3xl sm:text-4xl font-semibold tracking-tight text-white">A careful look across media</h2>
              <p className="text-sm text-slate-500 max-w-md leading-relaxed">Clear boundaries between what a model predicts, what influenced it, and what remains unknown.</p>
            </div>
          </motion.div>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {capabilities.map(({ icon: Icon, ...item }, index) => (
              <motion.article key={item.title} variants={reveal} initial="hidden" whileInView="show"
                viewport={{ once: true }} custom={index * 0.4} className="card min-h-48">
                <div className="flex items-center justify-between mb-5">
                  <span className="w-9 h-9 rounded-lg flex items-center justify-center"
                    style={{ color: item.tone, background: `${item.tone}12` }}>
                    <Icon size={17} />
                  </span>
                  <span className="w-8 h-px" style={{ background: `${item.tone}70` }} />
                </div>
                <h3 className="text-base font-semibold text-white mb-2">{item.title}</h3>
                <p className="text-sm leading-relaxed text-slate-400">{item.description}</p>
              </motion.article>
            ))}
          </div>
        </div>
      </section>

      <section className="px-6 py-16">
        <div className="max-w-7xl mx-auto grid lg:grid-cols-[.8fr_1.2fr] gap-10 items-start">
          <div>
            <p className="section-label mb-2">Method</p>
            <h2 className="text-3xl font-semibold tracking-tight text-white mb-3">From input to review</h2>
            <p className="text-sm leading-relaxed text-slate-500">
              The explanation follows the same configured model and preprocessing used to produce the score.
            </p>
          </div>
          <ol className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {pipeline.map(([step, title, description]) => (
              <li key={step} className="flex gap-4 p-4 rounded-xl border border-white/[0.07] bg-white/[0.015]">
                <span className="font-mono text-xs text-brass-400 pt-0.5">{step}</span>
                <div>
                  <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
                  <p className="text-xs text-slate-500 leading-relaxed mt-1">{description}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="px-6 pt-10 pb-24">
        <div className="max-w-7xl mx-auto rounded-2xl border border-white/[0.09] bg-dark-800 p-7 sm:p-10 flex flex-col md:flex-row md:items-center md:justify-between gap-6">
          <div>
            <p className="section-label mb-2">Forensic review</p>
            <h2 className="text-2xl sm:text-3xl font-semibold text-white tracking-tight">Start with the evidence you have.</h2>
            <p className="text-sm text-slate-400 mt-2">Use the dashboard to review recorded analyses and trends.</p>
          </div>
          <Link to="/dashboard" className="btn-ghost inline-flex items-center justify-center gap-2 shrink-0">
            Open dashboard <ArrowRight size={15} />
          </Link>
        </div>
      </section>
    </main>
  )
}
