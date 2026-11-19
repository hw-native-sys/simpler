# Fetch PR Feedback (all three comment surfaces)

A PR carries feedback on **three separate surfaces**. Fetching only the inline
review threads silently drops the other two — which is where "please rebase",
"this whole approach is wrong", and most bot summaries actually land.

| # | Surface | GraphQL field | Resolvable? | Reply with |
| - | ------- | ------------- | ----------- | ---------- |
| A | Inline review threads (anchored to a file/line) | `reviewThreads.nodes` | Yes (`resolveReviewThread`) | thread reply on the comment's `databaseId` |
| B | Review summary bodies (text submitted with APPROVE / REQUEST_CHANGES / COMMENT) | `reviews.nodes[].body` | No | a PR conversation comment |
| C | PR conversation comments (issue comments, incl. `@`-mentions and most bot posts) | `comments.nodes` | No | a PR conversation comment |

**Important:** Inline values directly into the GraphQL query string. Do NOT use `-f`/`-F` flags with GraphQL `$variables` — bash mangles the `$` signs. See [common-issues](./common-issues.md) for details.

## One query for all three

```bash
gh api graphql -f query='
query {
  repository(owner: "OWNER", name: "REPO") {
    pullRequest(number: NUMBER) {
      reviewThreads(first: 100) {
        nodes {
          id
          isResolved
          isOutdated
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
      reviews(first: 100) {
        nodes {
          id
          databaseId
          state
          body
          author { login }
          submittedAt
        }
      }
      comments(first: 100) {
        nodes {
          id
          databaseId
          body
          author { login }
          createdAt
        }
      }
    }
  }
}' > /tmp/pr-feedback.json
```

Replace `OWNER`, `REPO`, `NUMBER` with actual values (e.g., `"hw-native-sys"`, `"simpler"`, `276`).

Split the surfaces out of the saved file (`jq` on a *file* avoids the `gh --jq`
`$`-quoting pitfalls):

```bash
P='.data.repository.pullRequest'

# A — unresolved inline threads
jq "[${P}.reviewThreads.nodes[] | select(.isResolved == false)]" /tmp/pr-feedback.json

# B — review bodies that actually say something (empty body = drive-by approve)
jq "[${P}.reviews.nodes[] | select(.body != \"\")]" /tmp/pr-feedback.json

# C — conversation comments
jq "${P}.comments.nodes" /tmp/pr-feedback.json
```

**Limits / pagination:** each connection is capped at `first: 100` (and
`comments(first: 50)` per thread). Add `pageInfo { hasNextPage endCursor }` and
re-query with `after: "<endCursor>"` on any connection reporting
`hasNextPage: true`.

## Surfaces B and C have no resolved bit — track them yourself

Only surface A carries `isResolved`. A review body or conversation comment stays
in the API response forever, so re-fetching it every iteration makes the same
feedback look perpetually unaddressed. Keep a handled-ID ledger per PR:

```bash
LEDGER="/tmp/pr-<NUMBER>-handled.txt"   # one comment/review node id per line
touch "$LEDGER"

# Filter out what was handled in an earlier iteration
jq -r "${P}.comments.nodes[].id" /tmp/pr-feedback.json | grep -Fxv -f "$LEDGER"

# After replying to (or consciously skipping) one, record it
echo "$NODE_ID" >> "$LEDGER"
```

This ledger only avoids duplicate replies; it is not proof of acceptance. Keep
unresolved disputes visible, and re-read edited feedback and new replies even when
an earlier ID was recorded. If the ledger is missing, reconstruct handling from
actual replies and current code. Timestamps or a later push do not prove old
feedback was read or addressed.

## Filtering bot noise (surface C)

Bots post status and walkthrough comments that carry no request. Treat a
conversation comment as **non-actionable** when its body matches any of:

- `<!-- This is an auto-generated comment: summarize by coderabbit.ai -->`
- `<!-- walkthrough_start -->` without an `Actionable comments posted: <n>` where `n > 0`
- `Review skipped` / `Pre-merge checks` status blocks
- A pure CI/deploy status echo

Everything else on surface C is real feedback and must be classified, including
short human one-liners (`please rebase onto main`, `can you split this?`) that
never appear as an inline thread.

A bot's *actionable* findings usually arrive as surface A threads, with B/C
holding only the summary — so **dedup B/C against A** before presenting: if a
review body just enumerates its own inline threads, address the threads and
treat the body as informational.

Output per surface:

- A — `id` (GraphQL node ID, for `resolveReviewThread`), `comments.nodes[].databaseId` (REST ID, for threaded replies), `path`/`line` for context
- B — `state` (`CHANGES_REQUESTED` outranks `COMMENTED`), `body`, `author`
- C — `databaseId`, `body`, `author`, `createdAt`
