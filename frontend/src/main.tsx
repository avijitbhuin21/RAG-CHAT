import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';

import App from './App';
import { RequireAdmin, RequireUser, SessionProvider } from './lib/auth';
import Admin from './routes/Admin';
import AdminLogin from './routes/AdminLogin';
import Chat from './routes/Chat';
import Login from './routes/Login';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <SessionProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<App />}>
            <Route path="/" element={<Navigate to="/chat" replace />} />
            <Route path="/login" element={<Login />} />
            <Route path="/admin/login" element={<AdminLogin />} />
            <Route
              path="/chat"
              element={
                <RequireUser>
                  <Chat />
                </RequireUser>
              }
            />
            <Route
              path="/admin"
              element={
                <RequireAdmin>
                  <Admin />
                </RequireAdmin>
              }
            />
          </Route>
        </Routes>
      </BrowserRouter>
    </SessionProvider>
  </React.StrictMode>,
);
