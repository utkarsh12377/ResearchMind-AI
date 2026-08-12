import type { Metadata } from "next";

import { PageHeader } from "@/components/layout/page-header";
import { ApiKeysPanel } from "@/components/settings/api-keys-panel";

export const metadata: Metadata = {
  title: "API keys · ResearchMind AI",
};

export default function ApiKeysPage() {
  return (
    <div className="space-y-6 p-8">
      <PageHeader
        title="API keys"
        description="Authenticate programmatic access with the X-API-Key header."
      />
      <ApiKeysPanel />
    </div>
  );
}
