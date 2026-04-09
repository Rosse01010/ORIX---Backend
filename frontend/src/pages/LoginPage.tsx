import { useState, type FormEvent } from "react";
import { useAuthStore } from "../store/authStore";
import Button from "../components/common/Button";

export default function LoginPage() {
  const login = useAuthStore((s) => s.login);
  const loading = useAuthStore((s) => s.loading);
  const error = useAuthStore((s) => s.error);

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    try {
      await login(username, password);
    } catch {
      /* surfaced via store.error */
    }
  };

  return (
    <form
      onSubmit={onSubmit}
      className="card w-full max-w-sm p-6 sm:p-8 space-y-4 sm:space-y-5"
      aria-label="Login form"
    >
      <div className="space-y-1 text-center">
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">ORIX</h1>
        <p className="text-xs text-zinc-400">
          Intelligent Video Surveillance System
        </p>
      </div>

      <div className="space-y-3">
        <div>
          <label
            htmlFor="username"
            className="block text-xs uppercase tracking-wider text-zinc-400 mb-1"
          >
            Username
          </label>
          <input
            id="username"
            className="input"
            placeholder="admin"
            autoComplete="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </div>
        <div>
          <label
            htmlFor="password"
            className="block text-xs uppercase tracking-wider text-zinc-400 mb-1"
          >
            Password
          </label>
          <input
            id="password"
            type="password"
            className="input"
            placeholder="••••••••"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
      </div>

      {error && (
        <div
          role="alert"
          className="rounded-md bg-red-900/40 border border-red-700/60 px-3 py-2 text-sm text-red-200"
        >
          {error}
        </div>
      )}

      <Button type="submit" className="w-full" loading={loading}>
        {loading ? "Signing in…" : "Sign in"}
      </Button>

      <div className="text-[11px] text-zinc-500 leading-relaxed border-t border-orix-border pt-3">
        <p className="font-semibold text-zinc-400">Demo credentials:</p>
        <p>admin / admin123</p>
        <p>operator / operator123</p>
        <p>user / user123</p>
      </div>
    </form>
  );
}
