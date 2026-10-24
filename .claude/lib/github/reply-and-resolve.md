# Reply and Resolve

How to answer each of the three feedback surfaces from
[fetch-comments](./fetch-comments.md). Every actionable item gets a disposition; informational noise needs no reply;
only surface A can additionally be resolved. Use existing publication authority;
these API examples do not independently authorize posting or resolving.

## Surface A — inline review threads

For accepted and verified fixes, reply and resolve. Disputed or unverified
findings stay open pending reviewer or authorized maintainer agreement.

### Step 1: Reply to the comment

```bash
gh api "repos/${PR_REPO_OWNER}/${PR_REPO_NAME}/pulls/${PR_NUMBER}/comments/${COMMENT_DATABASE_ID}/replies" \
  -f body='Fixed — description'
```

**Important:** Always use single quotes for `-f body='...'` to avoid bash history expansion issues with `!` and other special characters.

**Response templates:**

- Fixed: `'Fixed — description'`
- Discuss: explain why the current code follows a rule; keep the disputed finding pending
- Acknowledged: `'Acknowledged, thank you!'`

### Step 2: Resolve the thread

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

Resolve settled findings; do not resolve a disagreement merely because an
explanation was posted. Thread resolution does not dismiss a blocking review.

## Surfaces B and C — review bodies and conversation comments

These have **no resolve mutation** — `resolveReviewThread` accepts only a
review-thread node ID and errors on a review or issue-comment ID. Answer them
with a PR conversation comment and record the ID in the ledger.

```bash
gh pr comment "$PR_NUMBER" --body "@${AUTHOR} Fixed — description"
echo "$NODE_ID" >> "/tmp/pr-${PR_NUMBER}-handled.txt"
```

- **Address the author by `@login`** — a conversation comment is not threaded, so without the mention it is unclear which feedback it answers. Quote the point being answered when several are outstanding.
- **Batch one reply per iteration.** Answering five conversation comments with five separate posts spams the PR; one comment with a short bullet per point is the norm.
- **Record processing, not acceptance.** The ID avoids duplicate replies, but a
  disputed item remains pending. Reconsider edited feedback and new replies; an ID
  alone cannot prove a concern resolved.
- Feedback that is purely informational (bot walkthrough, "nice work") needs no reply — just record its ID.

Template for a batched reply:

```text
@reviewer Addressed:
- Rebased onto main.
- Null check added on the predicate path.
- Kept the existing enum style — follows .claude/rules/codestyle.md §2.
```
