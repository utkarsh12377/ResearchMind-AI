"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

import { useCurrentUser } from "@/lib/api/hooks";
import { Skeleton } from "@/components/ui/skeleton";

/**
 * Client-side route protection.
 *
 * The session token lives in localStorage (see `tokenStorage`), which the
 * server can't read, so the redirect has to happen after hydration rather than
 * in middleware. This is a UX guard, not a security boundary — every protected
 * resource is independently authorized by the backend.
 */
export function AuthGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const { data: user, isPending, isError } = useCurrentUser();

  const isSignedOut = !isPending && !user;

  useEffect(() => {
    if (isSignedOut) router.replace("/login");
  }, [isSignedOut, router]);

  if (isPending || isSignedOut || isError) {
    return (
      <div className="space-y-4 p-8">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  return <>{children}</>;
}
