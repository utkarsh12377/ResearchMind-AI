"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";

import { ApiError, api, fetchHealth, tokenStorage } from "@/lib/api/client";
import type { LoginPayload, RegisterPayload, User } from "@/lib/api/types";

export const queryKeys = {
  health: ["backend-health"] as const,
  currentUser: ["current-user"] as const,
  apiKeys: ["api-keys"] as const,
};

export function useBackendHealth() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: ({ signal }) => fetchHealth(signal),
    retry: 1,
    refetchInterval: 15_000,
  });
}

/**
 * Resolves the signed-in user from the stored token.
 *
 * Returns `null` (rather than throwing) when there is no token or the token is
 * rejected, so callers can treat "signed out" as a normal state instead of an
 * error to be retried.
 */
export function useCurrentUser() {
  return useQuery<User | null>({
    queryKey: queryKeys.currentUser,
    queryFn: async () => {
      if (!tokenStorage.get()) return null;
      try {
        return await api.me();
      } catch (error) {
        if (error instanceof ApiError && error.isUnauthorized) {
          tokenStorage.clear();
          return null;
        }
        throw error;
      }
    },
    retry: false,
    staleTime: 5 * 60_000,
  });
}

export function useLogin() {
  const queryClient = useQueryClient();
  const router = useRouter();

  return useMutation({
    mutationFn: async (payload: LoginPayload) => {
      const token = await api.login(payload);
      tokenStorage.set(token.access_token);
      return api.me();
    },
    onSuccess: (user) => {
      queryClient.setQueryData(queryKeys.currentUser, user);
      router.push("/dashboard");
    },
  });
}

export function useRegister() {
  const login = useLogin();

  return useMutation({
    mutationFn: async (payload: RegisterPayload) => {
      await api.register(payload);
      // Registration doesn't return a token, so sign the new user straight in
      // rather than bouncing them to the login form.
      return login.mutateAsync({ email: payload.email, password: payload.password });
    },
  });
}

export function useLogout() {
  const queryClient = useQueryClient();
  const router = useRouter();

  return () => {
    tokenStorage.clear();
    queryClient.setQueryData(queryKeys.currentUser, null);
    queryClient.clear();
    router.push("/login");
  };
}

export function useApiKeys() {
  return useQuery({
    queryKey: queryKeys.apiKeys,
    queryFn: () => api.listApiKeys(),
  });
}

export function useCreateApiKey() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (name: string) => api.createApiKey(name),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.apiKeys }),
  });
}

export function useRevokeApiKey() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => api.revokeApiKey(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.apiKeys }),
  });
}
