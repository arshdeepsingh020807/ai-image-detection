import type { ReactNode, Ref } from "react";

interface NoticeProps {
  tone: "warning" | "danger" | "info";
  title: string;
  children?: ReactNode;
  actions?: ReactNode;
  role?: "alert" | "note";
  headingRef?: Ref<HTMLHeadingElement>;
  level?: 2 | 3;
}

export function Notice({ tone, title, children, actions, role = "note", headingRef, level = 3 }: NoticeProps) {
  const Heading = level === 2 ? "h2" : "h3";
  return (
    <div className={`notice notice--${tone}`} role={role}>
      <Heading className="notice__title" ref={headingRef} tabIndex={headingRef ? -1 : undefined}>{title}</Heading>
      {children && <div className="notice__body">{children}</div>}
      {actions && <div className="notice__actions">{actions}</div>}
    </div>
  );
}
