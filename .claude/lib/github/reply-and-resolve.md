# Reply and Resolve

Reply using:

```bash
gh api repos/${PR_REPO_OWNER}/${REPO_NAME}$/pulls/<number>/comments/<comment_id>/replies -f body="..."
```

Then resolve thread with GraphQL `resolveReviewThread` mutation.

**Response templates:**

- Fixed: "Fixed in `<commit>` — description"
- Skip: "Current code follows `.claude/rules/<file>`"
- Acknowledged: "Acknowledged, thank you!"
