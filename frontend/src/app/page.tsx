import { BackendStatus } from "@/components/backend-status";
import { ThemeToggle } from "@/components/theme-toggle";

export default function Home() {
  return (
    <div className="flex flex-1 flex-col">
      <header className="flex items-center justify-between border-b px-6 py-4">
        <span className="font-semibold tracking-tight">ResearchMind AI</span>
        <ThemeToggle />
      </header>

      <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col justify-center gap-6 px-6 py-16">
        <div className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight">ResearchMind AI</h1>
          <p className="text-muted-foreground">
            An AI-powered research platform for understanding, comparing, and reasoning over
            scientific papers. This dashboard is being built incrementally — see{" "}
            <code className="font-mono text-xs">docs/milestones.md</code> for the roadmap.
          </p>
        </div>

        <BackendStatus />
      </main>
    </div>
  );
}
