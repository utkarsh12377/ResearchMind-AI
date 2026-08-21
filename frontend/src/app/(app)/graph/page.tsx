import type { Metadata } from "next";

import { GraphExplorer } from "@/components/graph/graph-explorer";
import { PageHeader } from "@/components/layout/page-header";

export const metadata: Metadata = {
  title: "Knowledge graph · ResearchMind AI",
};

export default function GraphPage() {
  return (
    <div className="space-y-6 p-8">
      <PageHeader
        title="Knowledge graph"
        description="Papers, authors, datasets, models, and methods, and the paths between them."
      />
      <GraphExplorer />
    </div>
  );
}
