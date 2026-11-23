import { Link } from 'react-router-dom'
import { Shield } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="border-t border-white/5 mt-20">
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-10">
          <div className="col-span-1 md:col-span-2">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-8 rounded-xl flex items-center justify-center border border-sage-400/25 bg-sage-400/10">
                <Shield size={16} className="text-sage-300" />
              </div>
              <span className="font-semibold text-white">DeepFake<span className="accent-text">Shield</span></span>
            </div>
            <p className="text-sm text-slate-500 leading-relaxed max-w-xs">
              Multimodal content-scoring project. Model results are screening signals, not proof of authenticity. Aligned with SDG 16.
            </p>
          </div>
          <div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-4">Platform</p>
            <div className="flex flex-col gap-2">
              {[['/', 'Home'], ['/dashboard', 'Dashboard'], ['/text-analysis', 'Text Analysis'], ['/image-analysis', 'Image Analysis'], ['/video-analysis', 'Video Analysis'], ['/explainability', 'Explainability']].map(([path, label]) => (
                <Link key={path} to={path} className="text-sm text-slate-400 hover:text-sage-300 transition-colors">{label}</Link>
              ))}
            </div>
          </div>
          <div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-4">Models</p>
            <div className="flex flex-col gap-2">
              {['Text detection', 'Image detection', 'Video detection', 'Uncertain results', 'Source-aware evaluation'].map(m => (
                <span key={m} className="text-sm text-slate-400">{m}</span>
              ))}
            </div>
          </div>
        </div>
        <div className="pt-6 border-t border-white/5 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-xs text-slate-600">DeepFakeShield · Major Project — CSE (Data Science)</p>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-brass-400" />
            <span className="text-xs text-slate-500">Model scores are advisory</span>
          </div>
        </div>
      </div>
    </footer>
  )
}
