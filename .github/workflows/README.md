CI runs `pytest` on every pull request. A green check on each PR is a small
detail that makes the repo read as maintained rather than submitted.

If the model download makes CI slow, mark the model-dependent tests with
`@pytest.mark.slow` and run only the fast ones here.
