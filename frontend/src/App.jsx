import { Routes, Route, NavLink } from 'react-router-dom'
import Projects from './screens/Projects'
import NewProject from './screens/NewProject'
import Workspace from './screens/Workspace'
import StoryboardReview from './screens/StoryboardReview'
import ImageSelection from './screens/ImageSelection'
import VideoGeneration from './screens/VideoGeneration'
import Assembly from './screens/Assembly'
import Metadata from './screens/Metadata'
import FinalReview from './screens/FinalReview'
import Settings from './screens/Settings'

export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <NavLink to="/" className="brand">
          AI Shorts Studio
        </NavLink>
        <nav className="topnav">
          <NavLink to="/" className={({ isActive }) => (isActive ? 'active' : '')}>
            Projects
          </NavLink>
          <NavLink
            to="/settings"
            className={({ isActive }) => (isActive ? 'active' : '')}
          >
            Settings
          </NavLink>
        </nav>
      </header>
      <main className="content">
        <Routes>
          <Route path="/" element={<Projects />} />
          <Route path="/new" element={<NewProject />} />
          <Route path="/projects/:id" element={<Workspace />} />
          <Route path="/projects/:id/story" element={<StoryboardReview />} />
          <Route path="/projects/:id/images" element={<ImageSelection />} />
          <Route path="/projects/:id/videos" element={<VideoGeneration />} />
          <Route path="/projects/:id/assembly" element={<Assembly />} />
          <Route path="/projects/:id/metadata" element={<Metadata />} />
          <Route path="/projects/:id/final" element={<FinalReview />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>
    </div>
  )
}