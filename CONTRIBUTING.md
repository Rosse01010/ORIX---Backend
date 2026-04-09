# Contributing to orix-frontend

## Branch Naming

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feat/short-description` | `feat/user-auth` |
| Bug fix | `fix/short-description` | `fix/login-redirect` |
| Hotfix | `hotfix/short-description` | `hotfix/token-expiry` |
| Chore | `chore/short-description` | `chore/update-deps` |
| Release | `release/version` | `release/1.2.0` |

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(scope): short description

[optional body]

[optional footer]
```

**Types:** `feat`, `fix`, `chore`, `docs`, `style`, `refactor`, `test`, `perf`, `ci`

**Examples:**

```
feat(auth): add JWT refresh token logic
fix(dashboard): resolve chart rendering on mobile
chore(deps): bump axios to 1.14.0
```

## Pull Request Process

1. Branch off `main` (or `develop` if it exists).
2. Keep PRs small and focused — one concern per PR.
3. Fill in the PR template completely.
4. Request at least **1 reviewer**.
5. All CI checks must pass before merging.
6. Squash merge into `main`.

## Code Style

- Run `npm run lint` before pushing.
- No `console.log` in production code.
- Components go in `src/components/`, pages in `src/pages/`.
- Co-locate tests next to the file they test (`Component.test.tsx`).
