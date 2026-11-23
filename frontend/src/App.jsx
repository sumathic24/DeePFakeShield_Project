import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import Navbar from './components/Navbar'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import TextAnalysis from './pages/TextAnalysis'
import ImageAnalysis from './pages/ImageAnalysis'
import VideoAnalysis from './pages/VideoAnalysis'
import About from './pages/About'
import Explainability from './pages/Explainability'
import Footer from './components/Footer'

export default function App() {
  return (
    <Router>
      <div className="min-h-screen bg-dark-950 relative overflow-x-hidden">
        <Navbar />

        <AnimatePresence mode="wait">
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/text-analysis" element={<TextAnalysis />} />
            <Route path="/image-analysis" element={<ImageAnalysis />} />
            <Route path="/video-analysis" element={<VideoAnalysis />} />
            <Route path="/about" element={<About />} />
            <Route path="/explainability" element={<Explainability />} />
          </Routes>
        </AnimatePresence>

        <Footer />
      </div>
    </Router>
  )
}
