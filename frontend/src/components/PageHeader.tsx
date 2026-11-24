import { useEffect, type ReactNode } from "react";

interface PageHeaderProps {
  title: string;
  eyebrow?: string;
  children?: ReactNode;
}

export function PageHeader({ title, eyebrow, children }: PageHeaderProps) {
  useEffect(() => {
    document.title = `${title} · AI Image Detection`;
  }, [title]);
  return (
    <div className="page-header">
      {eyebrow && <p className="page-header__eyebrow">{eyebrow}</p>}
      <h1>{title}</h1>
      {children && <div className="page-header__lead">{children}</div>}
    </div>
  );
}
