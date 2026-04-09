import type { User, UserRole } from "../types/User";
import { sleep } from "../utils/helpers";

/**
 * Local-only user management.
 *
 * The ORIX backend does not expose a `/users` endpoint yet — the
 * recognition service tracks *persons* (face embeddings), not app users.
 * All user-management state below lives in memory for the session.
 */

/** Public user info — never includes a token. */
export type UserDto = Omit<User, "token">;

const MOCK_USERS: UserDto[] = [
  { id: "u_admin", username: "admin", role: "admin" },
  { id: "u_operator", username: "operator", role: "operator" },
  { id: "u_user", username: "user", role: "user" },
];

export async function fetchUsers(_token: string | null): Promise<UserDto[]> {
  void _token;
  await sleep(150);
  return [...MOCK_USERS];
}

export async function createUser(
  payload: { username: string; password: string; role: UserRole },
  _token: string | null
): Promise<UserDto> {
  void _token;
  await sleep(150);
  const created: UserDto = {
    id: `u_${payload.username}_${Date.now()}`,
    username: payload.username,
    role: payload.role,
  };
  MOCK_USERS.push(created);
  return created;
}

export async function deleteUser(
  id: string,
  _token: string | null
): Promise<void> {
  void _token;
  await sleep(100);
  const idx = MOCK_USERS.findIndex((u) => u.id === id);
  if (idx !== -1) MOCK_USERS.splice(idx, 1);
}
