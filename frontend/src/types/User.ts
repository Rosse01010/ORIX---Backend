/**
 * User roles available in the ORIX platform.
 * - admin:    full access, can manage users and cameras
 * - operator: can view all cameras and acknowledge alerts
 * - user:     read-only access to assigned cameras
 */
export type UserRole = "admin" | "operator" | "user";

export interface User {
  id: string;
  username: string;
  role: UserRole;
  token: string;
}
