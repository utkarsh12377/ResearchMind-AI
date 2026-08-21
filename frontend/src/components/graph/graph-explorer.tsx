"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { Loader2, Network, Search } from "lucide-react";
import { useState } from "react";

import { api } from "@/lib/api/client";
import type { GraphNode } from "@/lib/api/types";
import { ForceGraph } from "@/components/graph/force-graph";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";

const VIEWS = [
  { id: "entities", label: "Shared entities", hint: "Papers linked by datasets, models, methods" },
  { id: "citations", label: "Citations", hint: "Who cites whom inside your library" },
  { id: "authors", label: "Co-authorship", hint: "Authors who publish together" },
] as const;

type ViewId = (typeof VIEWS)[number]["id"];

function OverviewCards() {
  const { data, isLoading } = useQuery({
    queryKey: ["graph-overview"],
    queryFn: () => api.graphOverview(),
  });

  if (isLoading) {
    return (
      <div className="grid gap-4 sm:grid-cols-3">
        {[0, 1, 2].map((index) => (
          <Skeleton key={index} className="h-32" />
        ))}
      </div>
    );
  }

  if (!data) return null;

  const relationships = data.counts._relationships ?? 0;
  const nodeCount = Object.entries(data.counts)
    .filter(([key]) => !key.startsWith("_"))
    .reduce((total, [, count]) => total + count, 0);

  return (
    <div className="grid gap-4 sm:grid-cols-3">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">Graph size</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-semibold">{nodeCount}</p>
          <p className="text-xs text-muted-foreground">
            nodes · {relationships} relationship{relationships === 1 ? "" : "s"}
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Most common entities
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-1">
          {data.top_entities.length === 0 ? (
            <p className="text-xs text-muted-foreground">Nothing extracted yet.</p>
          ) : (
            data.top_entities.slice(0, 4).map((entity) => (
              <div key={`${entity.label}-${entity.name}`} className="flex justify-between text-xs">
                <span className="truncate">{entity.name}</span>
                <span className="text-muted-foreground">{entity.paper_count} papers</span>
              </div>
            ))
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Best connected papers
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-1">
          {data.most_connected.length === 0 ? (
            <p className="text-xs text-muted-foreground">Nothing extracted yet.</p>
          ) : (
            data.most_connected.slice(0, 4).map((paper) => (
              <div key={paper.paper_id} className="flex justify-between gap-2 text-xs">
                <span className="truncate">{paper.title}</span>
                <span className="shrink-0 text-muted-foreground">{paper.entity_count}</span>
              </div>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function NaturalLanguageQuery() {
  const [question, setQuestion] = useState("");
  const query = useMutation({
    mutationFn: () => api.graphQuery(question.trim()),
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Ask the graph</CardTitle>
        <p className="text-sm text-muted-foreground">
          Questions are translated to read-only Cypher and validated against a fixed schema
          before anything runs.
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        <form
          className="flex gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            if (question.trim()) query.mutate();
          }}
        >
          <Input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Which papers evaluate on the same dataset?"
            aria-label="Graph question"
          />
          <Button type="submit" disabled={!question.trim() || query.isPending}>
            {query.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Search className="h-4 w-4" />
            )}
          </Button>
        </form>

        {query.data && (
          <div className="space-y-2">
            {query.data.cypher && (
              <pre className="overflow-x-auto rounded-md bg-muted p-3 text-xs">
                {query.data.cypher}
              </pre>
            )}
            {query.data.error ? (
              <p className="text-sm text-muted-foreground">{query.data.error}</p>
            ) : (
              <pre className="max-h-64 overflow-auto rounded-md bg-muted p-3 text-xs">
                {JSON.stringify(query.data.rows, null, 2)}
              </pre>
            )}
          </div>
        )}

        {query.isError && (
          <p className="text-sm text-destructive">{(query.error as Error).message}</p>
        )}
      </CardContent>
    </Card>
  );
}

export function GraphExplorer() {
  const [view, setView] = useState<ViewId>("entities");
  const [selected, setSelected] = useState<GraphNode | null>(null);

  const network = useQuery({
    queryKey: ["graph-network", view],
    queryFn: () => api.graphNetwork(view),
  });

  const active = VIEWS.find((item) => item.id === view);

  return (
    <div className="space-y-6">
      <OverviewCards />

      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <CardTitle className="flex items-center gap-2 text-base">
                <Network className="h-4 w-4" />
                {active?.label}
              </CardTitle>
              <p className="text-sm text-muted-foreground">{active?.hint}</p>
            </div>
            <div className="flex gap-1.5">
              {VIEWS.map((item) => (
                <Button
                  key={item.id}
                  size="sm"
                  variant={item.id === view ? "default" : "outline"}
                  onClick={() => {
                    setView(item.id);
                    setSelected(null);
                  }}
                >
                  {item.label}
                </Button>
              ))}
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {network.isLoading ? (
            <Skeleton className="h-[420px] w-full" />
          ) : network.isError ? (
            <p className="text-sm text-destructive">{(network.error as Error).message}</p>
          ) : (
            <ForceGraph
              nodes={network.data?.nodes ?? []}
              edges={network.data?.edges ?? []}
              onSelect={setSelected}
            />
          )}

          {selected && (
            <div className="rounded-md border p-3">
              <div className="flex flex-wrap items-center gap-2">
                <Badge>{selected.kind}</Badge>
                <span className="text-sm font-medium">{selected.label}</span>
                <span className="text-xs text-muted-foreground">weight {selected.weight}</span>
              </div>
              {Object.keys(selected.metadata).length > 0 && (
                <pre className="mt-2 overflow-x-auto text-xs text-muted-foreground">
                  {JSON.stringify(selected.metadata, null, 2)}
                </pre>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      <NaturalLanguageQuery />
    </div>
  );
}
