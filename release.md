# Versioning And Releases

Package versions are derived from Git tags using `setuptools-scm`; no version
number is maintained in the source. A clean checkout tagged `v0.1.0` builds
version `0.1.0`. Commits after a tag and dirty working trees produce development
versions instead of reusing the release version.

After committing and testing a release, push its version tag to publish it:

```bash
git tag -a v0.1.0 -m "Release 0.1.0"
git push origin v0.1.0
```

Pushing a `v*` tag runs `.github/workflows/publish.yml`. The workflow runs tests,
builds a wheel and source distribution, verifies that both versions match the
tag, smoke-tests a clean wheel installation, and publishes to production PyPI.
Prerelease tags such as `v0.2.0rc1` are supported; development and local versions
are rejected. Creating a GitHub release is not required.

Before the first automated release, create a GitHub Actions environment named
`pypi` under the repository's Settings > Environments. Add required reviewers if
you want to approve uploads. Configure the PyPI Trusted Publisher with your
GitHub owner and repository, workflow filename `publish.yml`, and environment
name `pypi`. No API token or repository secret is needed. Allow release tags in
the environment's deployment rules if you restrict deployments.

To build and check distributions locally without publishing:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

Choose a new version tag for each release and build from that tagged commit in a
clean working tree. Use an empty build output directory to avoid mixing older
artifacts with the new release. PyPI does not allow replacing uploaded files.

GitHub Actions builds must fetch the full history and tags by setting
`fetch-depth: 0` on `actions/checkout`. Install from a Git clone or a generated
source distribution, not a GitHub source ZIP: a source distribution retains the
resolved version and can build without Git metadata.

`xenium_splitter.__version__` reads the installed distribution's version. Reinstall
an editable installation after changing tags or commits to refresh that metadata.
