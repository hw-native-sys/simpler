# Fetch Unresolved PR Comments

Fetches all unresolved review threads for a given PR.

```bash
COMMENTS_JSON=$(gh api graphql -f owner="$PR_REPO_OWNER" -f repo="$PR_REPO_NAME" -F number=$PR_NUMBER \
  -f query='
query($owner: String!, $repo: String!, $number: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
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
              author { login }
            }
          }
        }
      }
    }
  }
}')

# Filter to unresolved threads only
UNRESOLVED=$(echo "$COMMENTS_JSON" | jq '.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false)')

# Check if we hit the pagination limit
THREAD_COUNT=$(echo "$COMMENTS_JSON" | jq '.data.repository.pullRequest.reviewThreads.nodes | length')
if [ "$THREAD_COUNT" -eq 100 ]; then
  echo "Warning: PR has 100+ review threads. Some threads may not be fetched."
  echo "Consider manually checking the PR for additional unresolved comments."
fi
```

**Limits:**
- `reviewThreads(first: 100)`: Fetches up to 100 threads. PRs with more threads will show a warning.
- `comments(first: 50)`: Fetches up to 50 comments per thread. Sufficient for most review discussions.

Output: JSON array of unresolved threads with their comments.
