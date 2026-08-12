"use client";

import { useState } from "react";

import { ApiError } from "@/lib/api/client";
import { useApiKeys, useCreateApiKey, useRevokeApiKey } from "@/lib/api/hooks";
import type { ApiKeyCreated } from "@/lib/api/types";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

function formatDate(value: string | null): string {
  if (!value) return "—";
  return new Date(value).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function ApiKeysPanel() {
  const { data: keys, isPending, error } = useApiKeys();
  const createKey = useCreateApiKey();
  const revokeKey = useRevokeApiKey();

  const [name, setName] = useState("");
  const [justCreated, setJustCreated] = useState<ApiKeyCreated | null>(null);

  const handleCreate = (event: React.FormEvent) => {
    event.preventDefault();
    if (!name.trim()) return;
    createKey.mutate(name.trim(), {
      onSuccess: (created) => {
        setJustCreated(created);
        setName("");
      },
    });
  };

  return (
    <div className="space-y-6">
      {justCreated && (
        <Alert>
          <AlertTitle>Copy your API key now</AlertTitle>
          <AlertDescription className="space-y-2">
            <p>
              This is the only time the full key is shown — we store only a hash, so it can&rsquo;t
              be recovered later.
            </p>
            <code className="block overflow-x-auto rounded bg-muted px-3 py-2 font-mono text-xs">
              {justCreated.api_key}
            </code>
          </AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Create an API key</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleCreate} className="flex gap-2">
            <Input
              placeholder="e.g. ingestion pipeline"
              value={name}
              onChange={(event) => setName(event.target.value)}
              maxLength={255}
            />
            <Button type="submit" disabled={createKey.isPending || !name.trim()}>
              {createKey.isPending ? "Creating…" : "Create"}
            </Button>
          </form>
          {createKey.error instanceof ApiError && (
            <p className="mt-2 text-sm text-destructive">{createKey.error.message}</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Your API keys</CardTitle>
        </CardHeader>
        <CardContent>
          {isPending ? (
            <div className="space-y-2">
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
            </div>
          ) : error ? (
            <p className="text-sm text-destructive">
              {error instanceof ApiError ? error.message : "Could not load API keys"}
            </p>
          ) : keys && keys.length > 0 ? (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Prefix</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead>Last used</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead />
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {keys.map((key) => (
                    <TableRow key={key.id}>
                      <TableCell className="font-medium">{key.name}</TableCell>
                      <TableCell className="font-mono text-xs">{key.prefix}…</TableCell>
                      <TableCell>{formatDate(key.created_at)}</TableCell>
                      <TableCell>{formatDate(key.last_used_at)}</TableCell>
                      <TableCell>
                        <Badge variant={key.revoked_at ? "secondary" : "default"}>
                          {key.revoked_at ? "Revoked" : "Active"}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        {!key.revoked_at && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => revokeKey.mutate(key.id)}
                            disabled={revokeKey.isPending}
                          >
                            Revoke
                          </Button>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              No API keys yet. Create one to call the API programmatically.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
