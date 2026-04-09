import { useEffect, useState, type FormEvent } from "react";
import Button from "../components/common/Button";
import { fetchUsers, createUser, deleteUser, type UserDto } from "../api/userApi";
import { getStoredToken } from "../services/authService";
import type { UserRole } from "../types/User";
import { toast } from "react-toastify";

/**
 * Admin-only page to list, create and delete users. Ideally it would call
 * a dedicated service, but the CRUD here is thin enough that it reaches
 * into the api layer directly.
 */
export default function UserManagementPage() {
  const [users, setUsers] = useState<UserDto[]>([]);
  const [loading, setLoading] = useState(false);

  // New user form state
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<UserRole>("user");
  const [creating, setCreating] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const data = await fetchUsers(getStoredToken());
      setUsers(data);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to load users");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const onCreate = async (e: FormEvent) => {
    e.preventDefault();
    if (!username || !password) return;
    setCreating(true);
    try {
      const created = await createUser(
        { username, password, role },
        getStoredToken()
      );
      setUsers((prev) => [...prev, created]);
      setUsername("");
      setPassword("");
      setRole("user");
      toast.success(`User "${created.username}" created`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to create user");
    } finally {
      setCreating(false);
    }
  };

  const onDelete = async (id: string) => {
    try {
      await deleteUser(id, getStoredToken());
      setUsers((prev) => prev.filter((u) => u.id !== id));
      toast.info("User deleted");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete user");
    }
  };

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold">User Management</h1>
        <p className="text-xs text-zinc-400">
          Admin-only. Create, list and remove ORIX operators.
        </p>
      </div>

      <form
        onSubmit={onCreate}
        className="card p-4 grid grid-cols-1 sm:grid-cols-4 gap-3"
      >
        <input
          className="input"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
        />
        <input
          className="input"
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <select
          className="input"
          value={role}
          onChange={(e) => setRole(e.target.value as UserRole)}
        >
          <option value="user">user</option>
          <option value="operator">operator</option>
          <option value="admin">admin</option>
        </select>
        <Button type="submit" loading={creating}>
          Create
        </Button>
      </form>

      <div className="card overflow-hidden">
        <div className="px-4 py-2 border-b border-orix-border flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-400">
            Users ({users.length})
          </h2>
          <Button variant="outline" size="sm" onClick={load} loading={loading}>
            Refresh
          </Button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-orix-bg/50 text-left text-xs uppercase tracking-wider text-zinc-500">
              <tr>
                <th className="px-4 py-2">Username</th>
                <th className="px-4 py-2">Role</th>
                <th className="px-4 py-2 text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.length === 0 && !loading ? (
                <tr>
                  <td colSpan={3} className="px-4 py-6 text-center text-zinc-500">
                    No users.
                  </td>
                </tr>
              ) : (
                users.map((u) => (
                  <tr
                    key={u.id}
                    className="border-t border-orix-border hover:bg-orix-bg/30"
                  >
                    <td className="px-4 py-2 font-medium">{u.username}</td>
                    <td className="px-4 py-2 text-zinc-400 uppercase text-xs">
                      {u.role}
                    </td>
                    <td className="px-4 py-2 text-right">
                      <Button
                        variant="danger"
                        size="sm"
                        onClick={() => onDelete(u.id)}
                      >
                        Delete
                      </Button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
