import { useEffect, useRef } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { ServiceStatus } from "./ServiceStatus";
import { ThemeToggle } from "./ThemeToggle";

const NAV_ITEMS = [
  { to: "/", label: "Detect" },
  { to: "/performance", label: "Performance" },
  { to: "/method", label: "Method" },
];

export function Layout() {
  const { pathname } = useLocation();
  const main = useRef<HTMLElement>(null);
  const firstRender = useRef(true);

  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    main.current?.focus();
    window.scrollTo(0, 0);
  }, [pathname]);

  return (
    <div className="app">
      <a className="skip-link" href="#main">Skip to main content</a>
      <header className="site-header">
        <div className="site-header__inner">
          <NavLink to="/" className="brand" aria-label="AI Image Detection, home">
            <svg className="brand__mark" viewBox="0 0 20 20" aria-hidden="true">
              <rect x="1" y="1" width="18" height="18" rx="3" />
              <path d="M6 6h8v8H6z" />
            </svg>
            <span className="brand__name">AI Image Detection</span>
          </NavLink>
          <nav className="site-nav" aria-label="Main">
            <ul>
              {NAV_ITEMS.map((item) => (
                <li key={item.to}>
                  <NavLink to={item.to} end className="site-nav__link">{item.label}</NavLink>
                </li>
              ))}
            </ul>
          </nav>
          <ThemeToggle />
        </div>
      </header>
      <main id="main" ref={main} tabIndex={-1} className="site-main">
        <Outlet />
      </main>
      <footer className="site-footer">
        <div className="site-footer__inner">
          <p>Final-year project · Results are statistical estimates, not proof.</p>
          <ServiceStatus />
        </div>
      </footer>
    </div>
  );
}
