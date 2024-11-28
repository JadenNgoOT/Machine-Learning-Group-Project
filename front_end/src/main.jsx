import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import homePage from './homePage'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <homePage />
  </StrictMode>,
)
