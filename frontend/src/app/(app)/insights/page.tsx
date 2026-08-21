import type { Metadata } from "next";

import { InsightsWorkbench } from "@/components/insights/insights-workbench";
import { PageHeader } from "@/components/layout/page-header";

export const metadata: Metadata = {
  title: "Insights · ResearchMind AI",
};

export default function InsightsPage() {
  return (
    <div className="space-y-6 p-8">
      <PageHeader
        title="Insights"
        description="Compare results, track how the field moved, find conflicts, and draft a cited review."
      />
      <InsightsWorkbench />
    </div>
  );
}
