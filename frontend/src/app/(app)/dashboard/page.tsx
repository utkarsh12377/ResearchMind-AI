import type { Metadata } from "next";

import { BackendStatus } from "@/components/backend-status";
import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export const metadata: Metadata = {
  title: "Overview · ResearchMind AI",
};

const CAPABILITIES = [
  {
    title: "Ingestion",
    body: "Upload papers at scale — layout-aware parsing, OCR for scans, and table, figure, equation, and reference extraction.",
  },
  {
    title: "Hybrid retrieval",
    body: "Dense embeddings fused with BM25, filtered by metadata, then reranked by a cross-encoder.",
  },
  {
    title: "Agentic reasoning",
    body: "A LangGraph agent team plans, retrieves, verifies, critiques, and cites before answering.",
  },
  {
    title: "Knowledge graph",
    body: "Papers, authors, datasets, models, and methods linked in Neo4j and queryable in natural language.",
  },
];

export default function DashboardPage() {
  return (
    <div className="space-y-6 p-8">
      <PageHeader
        title="Overview"
        description="Your research workspace at a glance."
      />

      <BackendStatus />

      <div className="grid gap-4 sm:grid-cols-2">
        {CAPABILITIES.map(({ title, body }) => (
          <Card key={title}>
            <CardHeader>
              <CardTitle className="text-base">{title}</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">{body}</CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
