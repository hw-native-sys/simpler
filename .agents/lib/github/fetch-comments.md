# Fetch Unresolved PR Comments

Fetch unresolved review threads with GraphQL. Inline values into the query to avoid shell quoting
issues with GraphQL variables.

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

Replace `OWNER`, `REPO`, and `NUMBER`.

For filtering:

```bash
gh api graphql -f query='...' \
  --jq '[.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false)]'
```

Use the thread `id` for GraphQL resolution and `databaseId` for REST replies.
