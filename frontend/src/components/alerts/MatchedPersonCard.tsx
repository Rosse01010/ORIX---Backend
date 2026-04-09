import { useEffect, useState } from "react";
import { getPerson } from "../../api/recognitionApi";
import type { PersonDetails } from "../../api/recognitionApi";

/**
 * Shows the consented social-profile links of a matched person.
 *
 * Given a `personId` from a `matched` detection alert, fetches the full
 * person record from the backend (name + social handles) and renders
 * clickable chips. Handles the common case of "not yet reachable /
 * backend down" gracefully — nothing here should crash the dashboard.
 */
interface Props {
  personId: string;
  onClose?: () => void;
}

function instagramUrl(handle: string): string {
  const clean = handle.replace(/^@/, "").trim();
  return `https://www.instagram.com/${clean}/`;
}

function twitterUrl(handle: string): string {
  const clean = handle.replace(/^@/, "").trim();
  return `https://x.com/${clean}`;
}

export default function MatchedPersonCard({ personId, onClose }: Props) {
  const [person, setPerson] = useState<PersonDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getPerson(personId)
      .then((data) => {
        if (!cancelled) setPerson(data);
      })
      .catch((err) => {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : "Unknown error";
        setError(msg);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [personId]);

  return (
    <div className="rounded-md border border-orix-accent/40 bg-orix-bg/80 p-3 text-xs space-y-2">
      <div className="flex items-start justify-between gap-2">
        <p className="text-[10px] uppercase tracking-wider text-orix-accent">
          Matched person
        </p>
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="text-zinc-500 hover:text-zinc-200 text-xs leading-none"
            aria-label="Close"
          >
            ✕
          </button>
        )}
      </div>

      {loading && <p className="text-zinc-500">Loading person details…</p>}

      {error && (
        <p className="text-red-400">
          Could not load person: {error}
        </p>
      )}

      {person && (
        <>
          <p className="text-white text-sm font-semibold break-words">
            {person.name}
          </p>

          <div className="flex flex-wrap gap-1.5">
            {person.linkedin_url && (
              <a
                href={person.linkedin_url}
                target="_blank"
                rel="noopener noreferrer"
                className="px-2 py-1 rounded bg-blue-900/40 text-blue-200 hover:bg-blue-900/70"
              >
                LinkedIn
              </a>
            )}
            {person.instagram_handle && (
              <a
                href={instagramUrl(person.instagram_handle)}
                target="_blank"
                rel="noopener noreferrer"
                className="px-2 py-1 rounded bg-pink-900/40 text-pink-200 hover:bg-pink-900/70"
              >
                Instagram {person.instagram_handle}
              </a>
            )}
            {person.twitter_handle && (
              <a
                href={twitterUrl(person.twitter_handle)}
                target="_blank"
                rel="noopener noreferrer"
                className="px-2 py-1 rounded bg-zinc-800 text-zinc-200 hover:bg-zinc-700"
              >
                X {person.twitter_handle}
              </a>
            )}
            {!person.linkedin_url &&
              !person.instagram_handle &&
              !person.twitter_handle && (
                <span className="text-zinc-500">
                  No social links on file.
                </span>
              )}
          </div>

          {person.notes && (
            <p className="text-zinc-400 italic break-words">
              {person.notes}
            </p>
          )}
        </>
      )}
    </div>
  );
}
