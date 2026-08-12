import Link from "next/link";

import { BackendStatus } from "@/components/backend-status";
import { ThemeToggle } from "@/components/theme-toggle";
import { buttonVariants } from "@/components/ui/button";

export default function Home() {
  return (
    <div className="flex flex-1 flex-col">
      <header className="flex items-center justify-between border-b px-6 py-4">
        <span className="font-semibold tracking-tight">ResearchMind AI</span>
        <div className="flex items-center gap-2">
          <Link href="/login" className={buttonVariants({ variant: "ghost", size: "sm" })}>
            Sign in
          </Link>
          <Link href="/register" className={buttonVariants({ size: "sm" })}>
            Get started
          </Link>
          <ThemeToggle />
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col justify-center gap-6 px-6 py-16">
        <div className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight">ResearchMind AI</h1>
          <p className="text-muted-foreground">
            An AI-powered research platform for understanding, comparing, and reasoning over
            scientific papers — hybrid retrieval, an agentic reasoning layer, and an automatically
            built knowledge graph, with verifiable citations.
          </p>
        </div>

        <BackendStatus />
      </main>
    </div>
  );
}
