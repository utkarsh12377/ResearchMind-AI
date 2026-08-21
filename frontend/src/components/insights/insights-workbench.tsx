"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { AlertTriangle, Check, Download, Loader2, Sparkles } from "lucide-react";
import { useState } from "react";

import { api } from "@/lib/api/client";
import type { ReviewResponse } from "@/lib/api/types";
import { ProgressionChart, TimelineChart } from "@/components/insights/trend-chart";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const DIRECTION_VARIANT: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
  rising: "default",
  emerging: "default",
  steady: "secondary",
  new: "outline",
  declining: "destructive",
};

function ComparisonTab() {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["insights-comparison"],
    queryFn: () => api.comparison(),
  });

  if (isLoading) return <Skeleton className="h-64 w-full" />;
  if (isError) return <p className="text-sm text-destructive">{(error as Error).message}</p>;
  if (!data || data.rows.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No comparable results yet. Numbers become comparable once two papers report the same
        metric on the same dataset — until then there is nothing to line up.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      {data.ungrouped_results > 0 && (
        <p className="text-xs text-muted-foreground">
          {data.ungrouped_results} extracted result
          {data.ungrouped_results === 1 ? " has" : "s have"} no dataset attached, so
          {data.ungrouped_results === 1 ? " it is" : " they are"} not shown here.
        </p>
      )}

      {data.rows.map((row) => (
        <Card key={`${row.dataset}-${row.metric}`}>
          <CardHeader className="pb-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <CardTitle className="text-sm font-medium">
                {row.dataset} · {row.metric}
              </CardTitle>
              {row.spread > 0 && (
                <span className="text-xs text-muted-foreground">
                  spread {row.spread.toFixed(2)}
                </span>
              )}
            </div>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model</TableHead>
                  <TableHead>Value</TableHead>
                  <TableHead>Paper</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {row.cells.map((cell) => (
                  <TableRow key={`${cell.paper_id}-${cell.model}-${cell.value}`}>
                    <TableCell className="font-medium">{cell.model}</TableCell>
                    <TableCell className={cell.is_best ? "font-semibold" : undefined}>
                      {cell.value}
                      {cell.unit ?? ""}
                      {cell.is_best && (
                        <Check className="ml-1 inline h-3.5 w-3.5 text-primary" aria-label="best" />
                      )}
                    </TableCell>
                    <TableCell className="text-muted-foreground">{cell.paper_title}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function TrendsTab() {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["insights-trends"],
    queryFn: () => api.trends("this library"),
  });

  if (isLoading) return <Skeleton className="h-64 w-full" />;
  if (isError) return <p className="text-sm text-destructive">{(error as Error).message}</p>;
  if (!data) return null;

  return (
    <div className="space-y-6">
      {data.narrative && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <Sparkles className="h-4 w-4" />
              What changed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm">{data.narrative}</p>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">Papers by year</CardTitle>
          {data.undated_papers > 0 && (
            <p className="text-xs text-muted-foreground">
              {data.undated_papers} paper{data.undated_papers === 1 ? "" : "s"} without a
              publication year {data.undated_papers === 1 ? "is" : "are"} excluded.
            </p>
          )}
        </CardHeader>
        <CardContent>
          <TimelineChart buckets={data.timeline} />
        </CardContent>
      </Card>

      {data.entity_trends.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Entity momentum</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {data.entity_trends.map((trend) => (
                <Badge
                  key={`${trend.label}-${trend.name}`}
                  variant={DIRECTION_VARIANT[trend.direction] ?? "secondary"}
                >
                  {trend.name} · {trend.direction}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {data.progressions.map((progression) => (
        <Card key={`${progression.dataset}-${progression.metric}`}>
          <CardHeader className="pb-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <CardTitle className="text-sm font-medium">
                {progression.metric} on {progression.dataset}
              </CardTitle>
              {progression.improvement !== null && (
                <span className="text-xs text-muted-foreground">
                  {progression.improvement >= 0 ? "+" : ""}
                  {progression.improvement.toFixed(2)} over the window
                </span>
              )}
            </div>
          </CardHeader>
          <CardContent>
            <ProgressionChart progression={progression} />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function ConsistencyTab() {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["insights-consistency"],
    queryFn: () => api.consistency([], true),
  });

  if (isLoading) return <Skeleton className="h-64 w-full" />;
  if (isError) return <p className="text-sm text-destructive">{(error as Error).message}</p>;
  if (!data) return null;

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">{data.summary}</p>

      {data.numeric_conflicts.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <AlertTriangle className="h-4 w-4 text-destructive" />
              Conflicting numbers
            </CardTitle>
            <p className="text-xs text-muted-foreground">
              Same model, same dataset, same metric — different results.
            </p>
          </CardHeader>
          <CardContent className="space-y-3">
            {data.numeric_conflicts.map((conflict, index) => (
              <div key={index} className="rounded-md border p-3 text-sm">
                <p className="font-medium">
                  {conflict.model} on {conflict.dataset} ({conflict.metric})
                </p>
                <p className="mt-1 text-muted-foreground">
                  {conflict.left_value} in “{conflict.left_paper}” vs {conflict.right_value} in “
                  {conflict.right_paper}” — {(conflict.relative_gap * 100).toFixed(0)}% apart
                </p>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {data.contradictions.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Contradicting claims</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {data.contradictions.map((pair, index) => (
              <div key={index} className="space-y-2 rounded-md border p-3 text-sm">
                <p className="text-xs text-muted-foreground">{pair.reason}</p>
                <p className="border-l-2 border-destructive/50 pl-3">
                  <span className="text-xs text-muted-foreground">{pair.left_paper}: </span>
                  {pair.left_claim}
                </p>
                <p className="border-l-2 border-destructive/50 pl-3">
                  <span className="text-xs text-muted-foreground">{pair.right_paper}: </span>
                  {pair.right_claim}
                </p>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {data.agreements.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Corroborating claims</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {data.agreements.slice(0, 5).map((pair, index) => (
              <p key={index} className="text-sm text-muted-foreground">
                {pair.left_paper} and {pair.right_paper} — {pair.reason}
              </p>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function GapsTab() {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["insights-gaps"],
    queryFn: () => api.gaps("this library"),
  });

  if (isLoading) return <Skeleton className="h-64 w-full" />;
  if (isError) return <p className="text-sm text-destructive">{(error as Error).message}</p>;
  if (!data) return null;

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">{data.summary}</p>

      {data.directions.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Suggested directions</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {data.directions.map((direction, index) => (
              <div key={index} className="rounded-md border p-3">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <p className="text-sm font-medium">{direction.gap}</p>
                  <Badge variant="outline" className="text-[10px]">
                    {(direction.confidence * 100).toFixed(0)}% confidence
                  </Badge>
                </div>
                <p className="mt-1 text-sm text-muted-foreground">{direction.rationale}</p>
                {direction.suggested_direction && (
                  <p className="mt-2 text-sm">→ {direction.suggested_direction}</p>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {data.structural.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Found in the data</CardTitle>
            <p className="text-xs text-muted-foreground">
              Derived from the extracted rows, not from a model&apos;s guess about the field.
            </p>
          </CardHeader>
          <CardContent className="space-y-2">
            {data.structural.map((gap, index) => (
              <div key={index} className="flex flex-wrap items-baseline gap-2 text-sm">
                <Badge variant="secondary" className="text-[10px]">
                  {gap.kind.replace(/_/g, " ")}
                </Badge>
                <span>{gap.description}</span>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {data.stated.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Stated by the authors</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {data.stated.map((gap, index) => (
              <div key={index} className="border-l-2 pl-3 text-sm">
                <p className="text-xs text-muted-foreground">{gap.paper_title}</p>
                <p>{gap.text}</p>
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function ReviewTab() {
  const [topic, setTopic] = useState("");
  const review = useMutation<ReviewResponse, Error, void>({
    mutationFn: () => api.review(topic.trim()),
  });

  const download = (contents: string, filename: string, type: string) => {
    const url = URL.createObjectURL(new Blob([contents], { type }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-4">
      <form
        className="flex gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          if (topic.trim()) review.mutate();
        }}
      >
        <Input
          value={topic}
          onChange={(event) => setTopic(event.target.value)}
          placeholder="Topic to review, e.g. dense retrieval for scientific search"
          aria-label="Review topic"
        />
        <Button type="submit" disabled={!topic.trim() || review.isPending}>
          {review.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          Generate
        </Button>
      </form>

      {review.isPending && (
        <p className="text-sm text-muted-foreground">
          Planning an outline, then drafting each section against its own retrieved sources.
        </p>
      )}

      {review.isError && <p className="text-sm text-destructive">{review.error.message}</p>}

      {review.data && (
        <Card>
          <CardHeader>
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <CardTitle className="text-base">{review.data.title}</CardTitle>
                <p className="text-xs text-muted-foreground">
                  {review.data.word_count} words · {review.data.sources.length} source
                  {review.data.sources.length === 1 ? "" : "s"}
                </p>
              </div>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() =>
                    download(review.data.markdown, "literature-review.md", "text/markdown")
                  }
                >
                  <Download className="mr-1.5 h-3.5 w-3.5" />
                  Markdown
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={!review.data.bibtex}
                  onClick={() => download(review.data.bibtex, "references.bib", "text/plain")}
                >
                  <Download className="mr-1.5 h-3.5 w-3.5" />
                  BibTeX
                </Button>
              </div>
            </div>
          </CardHeader>

          <CardContent className="space-y-5">
            {review.data.sections.map((section) => (
              <div key={section.heading} className="space-y-1.5">
                <h3 className="text-sm font-semibold">{section.heading}</h3>
                <p className="text-sm leading-relaxed">{section.text}</p>
                {section.source_indices.length > 0 && (
                  <p className="text-xs text-muted-foreground">
                    cites {section.source_indices.map((index) => `[${index}]`).join(" ")}
                  </p>
                )}
              </div>
            ))}

            {review.data.sources.length > 0 && (
              <>
                <Separator />
                <div className="space-y-1">
                  <h3 className="text-sm font-semibold">References</h3>
                  {review.data.sources.map((source) => (
                    <p key={source.index} className="text-xs text-muted-foreground">
                      [{source.index}] {source.citation}
                    </p>
                  ))}
                </div>
              </>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export function InsightsWorkbench() {
  return (
    <Tabs defaultValue="comparison" className="space-y-4">
      <TabsList>
        <TabsTrigger value="comparison">Comparison</TabsTrigger>
        <TabsTrigger value="trends">Trends</TabsTrigger>
        <TabsTrigger value="consistency">Consistency</TabsTrigger>
        <TabsTrigger value="gaps">Gaps</TabsTrigger>
        <TabsTrigger value="review">Review</TabsTrigger>
      </TabsList>

      <TabsContent value="comparison">
        <ComparisonTab />
      </TabsContent>
      <TabsContent value="trends">
        <TrendsTab />
      </TabsContent>
      <TabsContent value="consistency">
        <ConsistencyTab />
      </TabsContent>
      <TabsContent value="gaps">
        <GapsTab />
      </TabsContent>
      <TabsContent value="review">
        <ReviewTab />
      </TabsContent>
    </Tabs>
  );
}
