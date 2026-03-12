# Fetch Unresolved PR Comments

Fetches all unresolved review threads for a given PR.

**Important:** Inline values directly into the GraphQL query string. Do NOT use `-f`/`-F` flags with GraphQL `$variables` — bash mangles the `$` signs. See [common-issues](./common-issues.md) for details.

```bash
gh api graphql -f query='
query {
  repository(owner: "OWNER", name: "REPO") {
    pullRequest(number: NUMBER) {
      reviewThreads(first: 100) {
        nodes {
          id
          isResolved
          comments(first: 50) {
            nodes {
              id
              databaseId
              body
              path
              line
              originalLine
              diffHunk
              author { login }
              createdAt
            }
          }
        }
      }
    }
  }
}'
```

Replace `OWNER`, `REPO`, `NUMBER` with actual values (e.g., `"ChaoWao"`, `"simpler"`, `276`).

Use `--jq` to filter unresolved threads:

```bash
gh api graphql -f query='...' \
  --jq '[.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false)]'
```

**Limits:**
- `reviewThreads(first: 100)`: Fetches up to 100 threads.
- `comments(first: 50)`: Fetches up to 50 comments per thread.

Output: JSON array of unresolved threads. Each thread has:
- `id` — GraphQL node ID (for resolving via mutation)
- `comments.nodes[].databaseId` — REST API ID (for replying)
