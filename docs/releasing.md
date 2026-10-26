# Releasing

Releases go to [PyPI](https://pypi.org/p/s1ght) from GitHub Actions (`.github/workflows/release.yml`). Pushing a version tag starts them; no token is stored anywhere.

## One-time setup

PyPI and TestPyPI publish through **trusted publishing**: the workflow proves its identity with a short-lived OIDC token.

1. On [pypi.org](https://pypi.org/manage/account/publishing/), add a **pending publisher** for the project `s1ght`:
   - owner `Doga0`
   - repository `S1GHT`
   - workflow `release.yml`
   - environment `pypi`
2. On [test.pypi.org](https://test.pypi.org/manage/account/publishing/), add the same with environment `testpypi`.
3. In the GitHub repository's *Settings → Environments*, create the `pypi` and `testpypi` environments. Optionally, require a reviewer on `pypi`, so every release waits for one approval.

## Every release

1. Set the version in `src/s1ght/__init__.py` (`__version__ = "0.2.0"`). It is the only place: `pyproject.toml` reads it.
2. In `CHANGELOG.md`, move the entries under `## [Unreleased]` into a new `## [0.2.0] - YYYY-MM-DD` section.
3. Merge both into `main` through a pull request. CI's `package` job builds the sdist and wheel and checks them.
4. Tag the merge commit and push the tag:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

The workflow then:

1. builds the sdist and wheel;
2. checks them with `scripts/check_dist.py`:
   - the tag matches `__version__`;
   - the package data is inside (`registry.yaml`, `cfg/default.yaml`, `py.typed`);
   - no design notes or notebooks are inside;
   - the PyPI README has no relative links;
3. runs `twine check`;
4. runs the offline tests against the installed wheel on Python 3.11 and 3.13, with the source tree moved away;
5. publishes to PyPI and creates a GitHub release whose notes come from the changelog section.

## Dry run

*Actions → Release → Run workflow* builds, tests and publishes to **TestPyPI** instead. Check the result with:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ s1ght
```

Locally, without publishing:

```bash
pip install build twine
python -m build
python scripts/check_dist.py dist/ --tag v0.2.0
twine check --strict dist/*
```

## Versions

Semantic versioning. Before 1.0, a minor version (0.2 → 0.3) may change the API, and a patch version (0.2.0 → 0.2.1) only fixes bugs. PyPI never accepts the same version twice: if a release is broken, publish a new patch version.
