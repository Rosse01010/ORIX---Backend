import clsx, { type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Conditional Tailwind className helper.
 * Merges clsx + tailwind-merge so duplicate/conflicting classes collapse cleanly.
 *
 *   cn("p-2", isActive && "bg-blue-500", "p-4") // -> "bg-blue-500 p-4"
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
