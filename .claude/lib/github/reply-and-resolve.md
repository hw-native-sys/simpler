# Reply and Resolve

For each comment, reply then resolve the thread. Both steps are required.

## Step 1: Reply to the comment

```bash
gh api "repos/${PR_REPO_OWNER}/${PR_REPO_NAME}/pulls/${PR_NUMBER}/comments/${COMMENT_DATABASE_ID}/replies" \
  -f body='Fixed in <commit-hash> — description'
```

**Important:** Always use single quotes for `-f body='...'` to avoid bash history expansion issues with `!` and other special characters.

**Response templates:**

- Fixed: `'Fixed in <commit-hash> — description'`
- Skip: `'Current code follows .claude/rules/<file>'`
- Acknowledged: `'Acknowledged, thank you!'`

## Step 2: Resolve the thread

Use the thread `id` (the GraphQL node ID from [fetch-comments](./fetch-comments.md), NOT the `databaseId`):

```bash
gh api graphql -f query='
mutation ResolveThread($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) {
    thread { isResolved }
  }
}' \
-f threadId="$THREAD_ID"
```

`$THREAD_ID` is the `id` field from the `reviewThreads.nodes[]` returned by [fetch-comments](./fetch-comments.md).

**Both steps are mandatory.** Do not skip the resolve step.
