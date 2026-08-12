"use client";

import { KeyRound, LayoutDashboard, LogOut } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";
import { useCurrentUser, useLogout } from "@/lib/api/hooks";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { ThemeToggle } from "@/components/theme-toggle";

// Entries are added here as the milestone that builds each route lands, so the
// nav never links to a page that doesn't exist yet.
const NAV_ITEMS = [
  { href: "/dashboard", label: "Overview", icon: LayoutDashboard },
  { href: "/settings/api-keys", label: "API keys", icon: KeyRound },
];

export function AppNav() {
  const pathname = usePathname();
  const logout = useLogout();
  const { data: user } = useCurrentUser();

  return (
    <aside className="flex w-60 shrink-0 flex-col border-r">
      <div className="px-4 py-4">
        <Link href="/dashboard" className="font-semibold tracking-tight">
          ResearchMind AI
        </Link>
      </div>

      <Separator />

      <nav className="flex-1 space-y-1 p-2">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const isActive = pathname === href || pathname.startsWith(`${href}/`);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors",
                isActive
                  ? "bg-accent text-accent-foreground font-medium"
                  : "text-muted-foreground hover:bg-accent/50 hover:text-foreground",
              )}
            >
              <Icon className="h-4 w-4" />
              {label}
            </Link>
          );
        })}
      </nav>

      <Separator />

      <div className="space-y-2 p-3">
        <div className="flex items-center justify-between gap-2">
          <span className="truncate text-xs text-muted-foreground" title={user?.email}>
            {user?.email}
          </span>
          <ThemeToggle />
        </div>
        <Button variant="ghost" size="sm" className="w-full justify-start" onClick={logout}>
          <LogOut className="mr-2 h-4 w-4" />
          Sign out
        </Button>
      </div>
    </aside>
  );
}
