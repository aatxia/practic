import { Hand } from "lucide-react";
import type { Metadata } from "next";
import "./globals.css";
import { ConnectionStatus } from "@/components/ConnectionStatus";

export const metadata: Metadata = {
  title: "UKSL Translator",
  description: "Двосторонній перекладач української жестової мови у реальному часі",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}): React.ReactElement {
  return (
    <html lang="uk">
      <body className="min-h-screen bg-slate-100">
        <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6">
          <div className="overflow-hidden rounded-[32px] bg-white shadow-sm ring-1 ring-slate-200">
            <header className="flex items-center justify-between px-6 py-5 sm:px-10">
              <span className="flex items-center gap-2.5">
                <span className="flex h-9 w-9 items-center justify-center rounded-full bg-brand-600">
                  <Hand className="h-5 w-5 text-white" aria-hidden />
                </span>
                <span className="text-lg font-semibold tracking-tight text-slate-900">
                  UKSL Translator
                </span>
              </span>
              <ConnectionStatus />
            </header>
            <main className="px-6 pb-10 sm:px-10">{children}</main>
          </div>
        </div>
      </body>
    </html>
  );
}
