import { useState, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowRight, Shield, Menu, X } from 'lucide-react'

const navLinks = [
  { label: 'Home', path: '/' },
  { label: 'Dashboard', path: '/dashboard' },
  { label: 'Text Analysis', path: '/text-analysis' },
  { label: 'Image Analysis', path: '/image-analysis' },
  { label: 'Video Analysis', path: '/video-analysis' },
  { label: 'Explainability', path: '/explainability' },
  { label: 'About', path: '/about' },
]

export default function Navbar() {
  const location = useLocation()
  const [scrolled, setScrolled] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => setMenuOpen(false), [location])

  return (
    <>
      <motion.nav
        initial={{ y: -16, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.32, ease: 'easeOut' }}
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
          scrolled ? 'glass-strong shadow-2xl' : 'bg-transparent'
        }`}
      >
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-3 group">
            <div className="relative">
              <div className="w-9 h-9 rounded-xl flex items-center justify-center border border-sage-400/25 bg-sage-400/10 transition-colors group-hover:bg-sage-400/15">
                <Shield size={18} className="text-sage-300" />
              </div>
            </div>
            <div>
              <span className="text-white font-semibold text-[15px] tracking-tight">DeepFake</span>
              <span className="accent-text font-semibold text-[15px] tracking-tight">Shield</span>
            </div>
          </Link>

          <div className="hidden xl:flex items-center gap-1">
            {navLinks.map((link) => (
              <Link
                key={link.path}
                to={link.path}
                aria-current={location.pathname === link.path ? 'page' : undefined}
                className={`relative px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                  location.pathname === link.path
                    ? 'text-sage-300'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {location.pathname === link.path && (
                  <motion.span
                    layoutId="nav-active"
                    className="absolute inset-0 rounded-lg"
                    style={{ background: 'rgba(167,183,154,0.08)', border: '1px solid rgba(167,183,154,0.18)' }}
                    transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                  />
                )}
                <span className="relative z-10">{link.label}</span>
              </Link>
            ))}
          </div>

          <div className="hidden xl:flex items-center gap-3">
            <Link to="/text-analysis" className="btn-primary flex items-center gap-2">
              <ArrowRight size={14} />
              Analyze Now
            </Link>
          </div>

          <button
            type="button"
            aria-label={menuOpen ? 'Close navigation menu' : 'Open navigation menu'}
            aria-expanded={menuOpen}
            className="xl:hidden p-2 rounded-lg text-slate-400 hover:text-white transition-colors"
            onClick={() => setMenuOpen(!menuOpen)}
          >
            {menuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </motion.nav>

      <AnimatePresence>
        {menuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
            className="fixed top-16 left-4 right-4 z-40 glass-strong rounded-2xl p-4 xl:hidden"
          >
            {navLinks.map((link) => (
              <Link
                key={link.path}
                to={link.path}
                aria-current={location.pathname === link.path ? 'page' : undefined}
                className={`flex items-center px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 ${
                  location.pathname === link.path
                    ? 'text-sage-300 bg-sage-400/10'
                    : 'text-slate-400 hover:text-white hover:bg-white/5'
                }`}
              >
                {link.label}
              </Link>
            ))}
            <div className="mt-3 pt-3 border-t border-white/5">
              <Link to="/text-analysis" className="btn-primary w-full text-center block">
                Analyze Now
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
