import type { User, UserRole } from "../types/User";
import { sleep } from "../utils/helpers";

/**
 * Local-only authentication.
 *
 * The ORIX backend (FastAPI / recognition API) does not expose any
 * `/auth` endpoints, so login is resolved entirely on the client using
 * the demo credential table below. This keeps the UI role-gating working
 * without pretending there's a server-side session.
 */
const MOCK_USERS: Record<string, { password: string; role: UserRole }> = {
  admin: { password: "admin123", role: "admin" },
  operator: { password: "operator123", role: "operator" },
  user: { password: "user123", role: "user" },
};

export interface LoginResponse {
  user: User;
}

/** Validates credentials against the local demo table. */
export async function loginRequest(
  username: string,
  password: string
): Promise<LoginResponse> {
  await sleep(300);
  const record = MOCK_USERS[username.toLowerCase()];
  if (!record || record.password !== password) {
    throw new Error("Invalid username or password");
  }
  const user: User = {
    id: `u_${username}`,
    username,
    role: record.role,
    token: `demo-token-${username}-${Date.now()}`,
  };
  return { user };
}

/** Client-side only; nothing to revoke on the backend. */
export async function logoutRequest(_token: string | null): Promise<void> {
  void _token;
}
