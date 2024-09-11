import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import LoadingPage from './pages/LoadingPage';
import ResultPage from './pages/ResultPage';

/**
 * App component serves as the main entry point for the application.
 * It sets up the routing for the application, defining the paths and corresponding components.
 * 
 * @component
 * @returns {JSX.Element} The rendered App component with routing configuration.
 */
const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/loading" element={<LoadingPage />} />
        <Route path="/result" element={<ResultPage />} />
      </Routes>
    </Router>
  );
};

export default App;
