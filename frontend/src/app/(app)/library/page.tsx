import type { Metadata } from "next";

import { PageHeader } from "@/components/layout/page-header";
import { SearchPanel } from "@/components/library/search-panel";

export const metadata: Metadata = {
  title: "Library · ResearchMind AI",
};

export default function LibraryPage() {
  return (
    <div className="space-y-6 p-8">
      <PageHeader
        title="Library"
        description="Search every indexed passage. Each result shows which retriever found it and where the reranker moved it."
      />
      <SearchPanel />
    </div>
  );
}
