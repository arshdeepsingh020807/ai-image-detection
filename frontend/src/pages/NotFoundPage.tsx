import { Link, useLocation } from "react-router-dom";
import { PageHeader } from "../components/PageHeader";

export function NotFoundPage() {
  const { pathname } = useLocation();
  return (
    <div className="page not-found">
      <PageHeader title="Page not found" eyebrow="Error 404">
        <p>There is no page at <code>{pathname}</code>. It may have moved, or the address may be mistyped.</p>
      </PageHeader>
      <ul className="link-list">
        <li><Link to="/">Analyse an image</Link></li>
        <li><Link to="/performance">See model performance</Link></li>
        <li><Link to="/method">Read how the method works</Link></li>
      </ul>
    </div>
  );
}
