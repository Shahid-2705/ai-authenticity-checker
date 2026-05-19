import React from 'react';
import { NavLink } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Image, Film, Mic, Layers, LayoutDashboard, Clock, Activity, LogOut } from 'lucide-react';
import useForensicStore from '../store/useForensicStore';
import useAuthStore from '../store/useAuthStore';
import { isAuthEnabled } from '../services/supabase';
import logo from '../assets/logo.jpeg';

const NAV_LINKS = [
  { to: '/',           icon: LayoutDashboard, label: 'Dashboard', exact: true },
  { to: '/image',      icon: Image,           label: 'Image' },
  { to: '/video',      icon: Film,            label: 'Video' },
  { to: '/audio',      icon: Mic,             label: 'Audio' },
  { to: '/multimodal', icon: Layers,          label: 'Multimodal' },
  { to: '/history',    icon: Clock,           label: 'History' },
];

function SidebarLink({ to, exact, icon: Icon, label, onClick }) {
  return (
    <NavLink
      to={to}
      end={exact}
      className={({ isActive }) => `nav-item w-full ${isActive ? 'active' : ''}`}
      onClick={onClick}
    >
      {({ isActive }) => (
        <>
          {isActive && (
            <motion.span
              layoutId="sidebar-active"
              className="absolute left-0 top-1.5 bottom-1.5 w-[2px] rounded-full bg-accent"
              transition={{ type: 'spring', stiffness: 350, damping: 30 }}
            />
          )}
          <Icon size={15} className="flex-shrink-0" />
          <span>{label}</span>
        </>
      )}
    </NavLink>
  );
}

export default function Sidebar({ open = true, onClose = () => {} }) {
  const { systemStatus } = useForensicStore();
  const { user, signOut } = useAuthStore();
  const loadedCount = systemStatus?.loaded_models?.length || 0;

  return (
    <>
      {/* Mobile backdrop */}
      {open && (
        <div
          className="md:hidden fixed inset-0 z-[45] bg-black/50"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={`
          w-[200px] flex flex-col h-screen fixed left-0 top-0 z-50
          transition-transform duration-200 ease-out
          md:translate-x-0
          bg-[rgba(12,15,22,0.95)] backdrop-blur-xl border-r border-border-dim
          ${open ? 'translate-x-0' : '-translate-x-full'}
        `}
      >
        {/* Brand */}
        <div className="px-4 pt-5 pb-4 flex items-center gap-2.5">
          <img
            src={logo}
            alt="ProofyX"
            className="w-6 h-6 rounded-lg flex-shrink-0 object-cover"
          />
          <span className="text-xs font-bold tracking-[0.12em] uppercase gradient-text font-display">
            PROOFYX
          </span>
        </div>

        <div className="mx-4 mb-3 divider" />

        {/* Main nav */}
        <nav className="flex-1 px-3 space-y-0.5 overflow-y-auto no-scrollbar" aria-label="Main navigation">
          {NAV_LINKS.map((link) => (
            <SidebarLink key={link.to} {...link} onClick={onClose} />
          ))}
        </nav>

        {/* Bottom section */}
        <div className="px-3 pb-4 mt-2 space-y-1">
          <div className="mx-1 mb-2 divider" />

          {/* System Status */}
          <SidebarLink to="/system" icon={Activity} label="System Status" onClick={onClose} />

          {/* Auth user */}
          {isAuthEnabled() && user && (
            <div className="flex items-center justify-between px-3 py-2 mt-1">
              <p className="text-xs font-medium truncate min-w-0 text-text-2">
                {user.email}
              </p>
              <button
                onClick={signOut}
                title="Sign out"
                aria-label="Sign out"
                className="p-1 rounded transition-colors ml-2 flex-shrink-0 hover:bg-white/5 text-text-3"
              >
                <LogOut size={13} />
              </button>
            </div>
          )}

          {/* System status indicator */}
          <div className="flex items-center gap-2 px-3 py-2">
            <span className="relative flex-shrink-0">
              <span
                className={`block w-1.5 h-1.5 rounded-full ${loadedCount > 0 ? 'bg-risk-clear' : 'bg-risk-critical'}`}
              />
            </span>
            <span className="text-xs text-text-3">
              {loadedCount} model{loadedCount !== 1 ? 's' : ''} loaded
            </span>
          </div>
        </div>
      </aside>
    </>
  );
}
