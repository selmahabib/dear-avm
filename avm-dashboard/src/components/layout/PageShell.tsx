import type { ReactNode } from "react";

export function PageShell({ children }: { children: ReactNode }) {
  return (
    <main className="mx-auto max-w-7xl px-6 py-8">
      {children}
    </main>
  );
}