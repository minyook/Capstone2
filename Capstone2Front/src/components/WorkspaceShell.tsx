import { Link, NavLink, Outlet } from "react-router-dom";

import "./WorkspaceShell.css";

export function WorkspaceShell() {
  return (
    <div className="workspace">
      <aside className="workspace__sidebar" aria-label="문서·챗봇·휴지통">
        <nav className="workspace__nav" aria-label="섹션">
          <NavLink
            to="/projects"
            className={({ isActive }) => "workspace__nav-item" + (isActive ? " workspace__nav-item--active" : "")}
            end
          >
            <span className="workspace__nav-ico" aria-hidden>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V7z" />
              </svg>
            </span>
            문서
          </NavLink>

          <NavLink
            to="/chatbot"
            className={({ isActive }) => "workspace__nav-item" + (isActive ? " workspace__nav-item--active" : "")}
          >
            <span className="workspace__nav-ico" aria-hidden>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path
                  d="M21 15a4 4 0 01-4 4H8l-4 3V7a4 4 0 014-4h9a4 4 0 014 4v8z"
                  strokeLinejoin="round"
                />
              </svg>
            </span>
            챗봇
          </NavLink>
        </nav>

        <div className="workspace__side-foot">
          <Link to="/trash" className="workspace__foot-link">
            <span className="workspace__foot-ico" aria-hidden>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14zM10 11v6M14 11v6" />
              </svg>
            </span>
            휴지통
          </Link>
        </div>
      </aside>

      <div className="workspace__main">
        <Outlet />
      </div>
    </div>
  );
}
