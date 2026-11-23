import { motion } from 'framer-motion'
import { ArrowRight, Brain, FileText, Film, Image, Info } from 'lucide-react'
import { Link } from 'react-router-dom'

const methods = [
  {
    icon: FileText,
    title: 'Text · SHAP',
    color: '#a7b79a',
    description: 'SHAP masks text segments and measures how the active text classifier’s AI probability changes. Coral highlights push the score toward AI-generated; sage highlights push it toward human-written.',
    details: 'The ranked chart shows signed local feature contributions. A contribution explains this model’s output for this input; it is not proof of who wrote the text.',
    href: '/text-analysis',
    action: 'Analyze text',
  },
  {
    icon: Image,
    title: 'Image · Grad-CAM',
    color: '#d0af7b',
    description: 'Grad-CAM combines gradients and spatial activations from the configured image classifier to create a heatmap of regions that influenced its predicted class.',
    details: 'The active image checkpoint classifies artificial versus real. Its heatmap is not a manipulation mask and does not establish that an image was edited.',
    href: '/image-analysis',
    action: 'Analyze an image',
  },
  {
    icon: Film,
    title: 'Video · Frame-wise Grad-CAM',
    color: '#e18773',
    description: 'The video pipeline samples representative frames, scores those same frames with the ConvNeXt detector, and computes Grad-CAM for each sampled frame.',
    details: 'The timeline exposes frame timestamps and scores from the sampled frames. Highlights show classifier activation, not proof of deepfake artifacts.',
    href: '/video-analysis',
    action: 'Analyze a video',
  },
]

export default function Explainability() {
  return (
    <main className="relative z-10 pt-28 pb-20 px-6">
      <div className="max-w-6xl mx-auto">
        <motion.header initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} className="mb-10">
          <p className="section-label mb-2">Explainable AI</p>
          <h1 className="text-4xl md:text-5xl font-black text-white mb-3">
            How the models <span className="accent-text">reach a result</span>
          </h1>
          <p className="text-slate-400 max-w-3xl leading-relaxed">
            Each explanation is generated from the same configured detector and input used for its prediction. Explanations describe model behavior; they are not independent verification of authenticity.
          </p>
        </motion.header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          {methods.map(({ icon: Icon, ...method }, index) => (
            <motion.article key={method.title} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.08 }} className="card flex flex-col">
              <div className="w-11 h-11 rounded-xl flex items-center justify-center mb-5"
                style={{ background: `${method.color}18`, border: `1px solid ${method.color}35` }}>
                <Icon size={20} style={{ color: method.color }} />
              </div>
              <h2 className="text-xl font-bold text-white mb-3">{method.title}</h2>
              <p className="text-sm leading-relaxed text-slate-300">{method.description}</p>
              <p className="text-xs leading-relaxed text-slate-500 mt-4 flex-1">{method.details}</p>
              <Link to={method.href} className="mt-6 inline-flex items-center gap-2 text-sm font-semibold"
                style={{ color: method.color }}>
                {method.action} <ArrowRight size={15} />
              </Link>
            </motion.article>
          ))}
        </div>

        <section className="card mt-6">
          <h2 className="flex items-center gap-2 text-lg font-bold text-white mb-5">
            <Brain size={19} className="text-brass-400" /> Analysis pipeline
          </h2>
          <ol className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              ['01', 'Input', 'Text, image, or video is validated.'],
              ['02', 'Preprocess', 'Existing tokenization, image transforms, or frame sampling are applied.'],
              ['03', 'Predict', 'The configured pretrained detector produces class scores.'],
              ['04', 'Explain', 'SHAP or Grad-CAM attributes the same model output to text or spatial features.'],
            ].map(([step, title, description]) => (
              <li key={step} className="p-4 rounded-xl bg-white/[0.025] border border-white/[0.06]">
                <span className="text-xs font-mono text-sage-400">{step}</span>
                <h3 className="text-sm font-semibold text-white mt-2">{title}</h3>
                <p className="text-xs text-slate-500 leading-relaxed mt-1">{description}</p>
              </li>
            ))}
          </ol>
          <p className="flex items-start gap-2 text-xs text-slate-500 mt-5">
            <Info size={15} className="text-brass-400 shrink-0 mt-0.5" />
            Confidence scores are model outputs and are not calibrated accuracy estimates. Low-confidence results remain uncertain.
          </p>
        </section>
      </div>
    </main>
  )
}
