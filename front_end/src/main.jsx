import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import homePage from './HomePage';
import 'bootstrap/dist/css/bootstrap.min.css';
import * as bootstrap from 'bootstrap'; // if you need JS components
import App from './App';

createRoot(document.getElementById('root')).render(
  <div>
    <App></App>
  </div>,
)
