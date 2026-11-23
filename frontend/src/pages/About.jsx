import { motion } from 'framer-motion'
import { Shield, Database, Cpu, BookOpen, Globe, Award } from 'lucide-react'

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  show: (i = 0) => ({ opacity: 1, y: 0, transition: { duration: 0.5, delay: i * 0.08 } }),
}

const datasets = [
  { name: 'HC3', type: 'Text · human vs ChatGPT', samples: 'Downloaded by dataset_prep.py', color: '#d0af7b' },
  { name: 'LIAR', type: 'Text · news veracity', samples: 'Downloaded by dataset_prep.py', color: '#a7b79a' },
  { name: 'CIFAKE', type: 'Image · AI vs real', samples: 'Up to 5,000 per class', color: '#e18773' },
]

const models = [
  { name: 'Configured text detector', role: 'AI-authorship score', type: 'Pretrained Hugging Face model', acc: 'Not evaluated here', color: '#a7b79a' },
  { name: 'Configured image detector', role: 'Artificial vs real image score', type: 'Binary image classifier', acc: 'Not evaluated here', color: '#d0af7b' },
  { name: 'Configured video detector', role: 'Frame-level deepfake score', type: 'ConvNeXt frame sampler', acc: 'Not evaluated here', color: '#e18773' },
  { name: 'RoBERTa / LinguaForge-X', role: 'Optional text training workflows', type: 'Experimental', acc: 'Train to evaluate', color: '#a7b79a' },
  { name: 'Fake-news BERT', role: 'Optional news classification', type: 'Available after training', acc: 'Train to evaluate', color: '#d0af7b' },
  { name: 'EfficientNet / ViT', role: 'Optional image training workflows', type: 'Binary classifier', acc: 'Train to evaluate', color: '#e18773' },
]

const xaiTools = [
  { name: 'Uncertainty', desc: 'Near-boundary scores are marked uncertain. This conservative threshold is not a calibrated probability guarantee.', color: '#e18773' },
  { name: 'Text attribution', desc: 'SHAP measures how masked text segments change the active detector’s AI probability. Signed token contributions are shown with the original text.', color: '#d0af7b' },
  { name: 'Image & video Grad-CAM', desc: 'Gradient-based heatmaps show spatial model activation for image predictions and the exact sampled video frames. They do not prove manipulation.', color: '#a7b79a' },
]

const techStack = [
  { layer: 'Frontend', items: ['React 18', 'Tailwind CSS', 'Framer Motion', 'Recharts', 'Axios'] },
  { layer: 'Backend', items: ['FastAPI', 'PyTorch', 'Transformers'] },
  { layer: 'Models', items: ['Configured text/image/video detectors', 'Optional training families'] },
  { layer: 'Evaluation', items: ['Accuracy', 'Precision', 'Recall', 'F1', 'Confusion matrix'] },
  { layer: 'Database', items: ['SQLite analysis history'] },
  { layer: 'DevOps', items: ['Docker', 'Docker Compose', 'Nginx'] },
]

export default function About() {
  return (
    <main className="relative z-10 pt-28 pb-20 px-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div variants={fadeUp} initial="hidden" animate="show" custom={0} className="mb-16 text-center">
          <p className="section-label mb-3">About the Project</p>
          <h1 className="text-5xl font-black text-white mb-4">
            <span className="accent-text">DeepFakeShield</span>
          </h1>
          <p className="text-slate-400 max-w-3xl mx-auto leading-relaxed text-lg">
            A multimodal screening application built as a Final Year Major Project for B.Tech CSE (Data Science).
            It offers separate NLP, image, and video detectors with explicit uncertainty handling and documents the limits of each classifier.
          </p>
          <div className="mt-6 inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium"
            style={{ background: 'rgba(167,183,154,0.08)', border: '1px solid rgba(167,183,154,0.2)', color: '#a7b79a' }}>
            <Globe size={14} /> Aligned with SDG 16: Peace, Justice & Strong Institutions
          </div>
        </motion.div>

        {/* Objective */}
        <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={0}
          className="card mb-8">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0"
              style={{ background: 'rgba(167,183,154,0.1)', border: '1px solid rgba(167,183,154,0.2)' }}>
              <Shield size={22} className="text-sage-400" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white mb-2">Project Objective</h2>
              <p className="text-slate-400 leading-relaxed">
                DeepFakeShield provides separate model-backed analysis paths for text, images, and video. Its active image model is a
                binary artificial-versus-real classifier, not an image-edit localization tool. Fake-news scoring uses a separately
                downloaded pretrained classifier and estimates dataset similarity rather than verifying claims. The project includes
                training and held-out evaluation workflows, but no benchmark results are claimed until those workflows have been run
                on a suitable source-disjoint dataset.
              </p>
            </div>
          </div>
        </motion.div>

        {/* Models */}
        <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={0} className="mb-10">
          <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
            <Cpu size={20} className="text-brass-400" /> AI Models
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {models.map((m, i) => (
              <motion.div key={m.name} variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={i * 0.1}
                className="p-4 rounded-xl transition-all border border-white/[0.07] bg-white/[0.015] hover:border-white/[0.14]"
                style={{ background: `${m.color}06`, border: `1px solid ${m.color}20` }}>
                <div className="flex items-center justify-between mb-2">
                  <p className="font-bold text-white text-sm">{m.name}</p>
                  <span className="text-xs font-black" style={{ color: m.color }}>{m.acc}</span>
                </div>
                <p className="text-xs text-slate-300 mb-1">{m.role}</p>
                <span className="text-[10px] px-2 py-0.5 rounded-full" style={{ background: `${m.color}15`, color: m.color }}>{m.type}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Datasets */}
        <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={0} className="mb-10">
          <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
            <Database size={20} className="text-sage-400" /> Datasets
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {datasets.map((d, i) => (
              <motion.div key={d.name} variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={i * 0.1}
                className="p-4 rounded-xl flex items-center gap-3"
                style={{ background: `${d.color}06`, border: `1px solid ${d.color}20` }}>
                <div className="w-2 h-10 rounded-full flex-shrink-0" style={{ background: d.color }} />
                <div>
                  <p className="font-semibold text-white text-sm">{d.name}</p>
                  <p className="text-xs text-slate-400">{d.type}</p>
                  <p className="text-xs mt-0.5" style={{ color: d.color }}>{d.samples}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* XAI */}
        <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={0} className="mb-10">
          <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
            <BookOpen size={20} className="text-coral-400" /> Explainable AI (XAI)
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {xaiTools.map((x, i) => (
              <motion.div key={x.name} variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={i * 0.1}
                className="card">
                <p className="text-2xl font-black mb-2" style={{ color: x.color }}>{x.name}</p>
                <p className="text-sm text-slate-400 leading-relaxed">{x.desc}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Tech Stack */}
        <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={0} className="mb-10">
          <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
            <Cpu size={20} className="text-brass-400" /> Technology Stack
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {techStack.map((t, i) => (
              <motion.div key={t.layer} variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={i * 0.1}
                className="card p-4">
                <p className="text-xs font-bold text-sage-400 mb-3 uppercase tracking-wider">{t.layer}</p>
                <div className="flex flex-col gap-1.5">
                  {t.items.map(item => (
                    <span key={item} className="text-xs text-slate-400">{item}</span>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* SDG */}
        <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={0}
          className="card text-center py-12">
          <Award size={40} className="text-sage-400 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-white mb-2">SDG 16 Alignment</h2>
          <p className="text-slate-400 max-w-xl mx-auto leading-relaxed">
            DeepFakeShield directly supports <strong className="text-white">Sustainable Development Goal 16</strong> —
            Peace, Justice and Strong Institutions — by combating AI-generated misinformation that undermines
            democratic processes, public trust, and social stability worldwide.
          </p>
        </motion.div>
      </div>
    </main>
  )
}
