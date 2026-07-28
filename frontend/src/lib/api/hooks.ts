import { useQuery } from "@tanstack/react-query";

import { fetchHealth } from "@/lib/api/client";

export function useBackendHealth() {
  return useQuery({
    queryKey: ["backend-health"],
    queryFn: ({ signal }) => fetchHealth(signal),
    retry: 1,
    refetchInterval: 15_000,
  });
}
