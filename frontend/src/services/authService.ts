import type { User } from "../types/User";
import { loginRequest, logoutRequest } from "../api/authApi";
import { STORAGE_KEYS, storage } from "../utils/storage";

/**
 * High-level authentication orchestration. Wraps the raw api layer and
 * handles client-side session persistence.
 */
export async function login(
  username: string,
  password: string
): Promise<User> {
  const { user } = await loginRequest(username, password);
  storage.set(STORAGE_KEYS.AUTH_USER, user);
  storage.set(STORAGE_KEYS.AUTH_TOKEN, user.token);
  return user;
}

export async function logout(): Promise<void> {
  const token = getStoredToken();
  await logoutRequest(token);
  storage.remove(STORAGE_KEYS.AUTH_USER);
  storage.remove(STORAGE_KEYS.AUTH_TOKEN);
}

export function getStoredUser(): User | null {
  return storage.get<User | null>(STORAGE_KEYS.AUTH_USER, null);
}

export function getStoredToken(): string | null {
  return storage.get<string | null>(STORAGE_KEYS.AUTH_TOKEN, null);
}
