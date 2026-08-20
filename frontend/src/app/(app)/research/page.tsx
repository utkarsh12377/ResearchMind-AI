import type { Metadata } from "next";

import { PageHeader } from "@/components/layout/page-header";
import { ResearchConsole } from "@/components/research/research-console";

export const metadata: Metadata = {
  title: "Research · ResearchMind AI",
};

export default function ResearchPage() {
  return (
    <div className="space-y-6 p-8">
      <PageHeader
        title="Research"
        description="Ask questions across your library. A team of agents plans, retrieves, verifies, and cites."
      />
      <ResearchConsole />
    </div>
  );
}
