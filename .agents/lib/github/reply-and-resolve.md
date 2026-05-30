# Reply And Resolve

For each addressed review comment, reply and resolve the thread.

## Reply

```bash
gh api "repos/${PR_REPO_OWNER}/${PR_REPO_NAME}/pulls/${PR_NUMBER}/comments/${COMMENT_DATABASE_ID}/replies" \
  -f body='Fixed in <commit-hash>: description'
```

Use single quotes for `-f body='...'`.

## Resolve

Use the review thread GraphQL node `id`, not the comment `databaseId`:

```bash
gh api graphql -f query='
mutation ResolveThread($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) {
    thread { isResolved }
  }
}' \
-f threadId="$THREAD_ID"
```

Both steps are mandatory unless the user explicitly asks not to resolve.
