"use client";

import { motion } from "framer-motion";

import { useBackendHealth } from "@/lib/api/hooks";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function BackendStatus() {
  const { data, isLoading, isError } = useBackendHealth();

  const variant = isLoading ? "secondary" : isError ? "destructive" : "default";
  const label = isLoading ? "Checking…" : isError ? "Unreachable" : `Online (${data?.status})`;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between text-base">
            Backend API
            <Badge variant={variant}>{label}</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          Live status of the FastAPI service at{" "}
          <code className="font-mono text-xs">
            {process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"}
          </code>
          . If this shows &ldquo;Unreachable&rdquo;, start the backend (see README) and this card
          will pick it up automatically.
        </CardContent>
      </Card>
    </motion.div>
  );
}
