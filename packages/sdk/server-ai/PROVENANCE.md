## Verifying SDK build provenance with the SLSA framework

LaunchDarkly uses the [SLSA framework](https://slsa.dev/spec/v1.0/about) (Supply-chain Levels for Software Artifacts) to help developers make their supply chain more secure by ensuring the authenticity and build integrity of our published SDK packages.

As part of [SLSA requirements for level 3 compliance](https://slsa.dev/spec/v1.0/requirements), LaunchDarkly publishes provenance about our SDK package builds using [GitHub's generic SLSA3 provenance generator](https://github.com/slsa-framework/slsa-github-generator/blob/main/internal/builders/generic/README.md#generation-of-slsa3-provenance-for-arbitrary-projects) for distribution alongside our packages. These attestations are available for download from the GitHub release page for the release version under Assets > `multiple.intoto.jsonl`.

To verify SLSA provenance attestations, we recommend using [slsa-verifier](https://github.com/slsa-framework/slsa-verifier). Example usage for verifying a package is included below:

<!-- x-release-please-start-version -->

```
# Set the version of the library to verify
VERSION=0.14.1
```

<!-- x-release-please-end -->

```
# Download package from PyPi
$ pip download --only-binary=:all: launchdarkly-server-sdk-ai==${VERSION}

# Download provenance from Github release into same directory
$ curl --location -O \
  https://github.com/launchdarkly/python-server-sdk-ai/releases/download/${VERSION}/multiple.intoto.jsonl

# Run slsa-verifier to verify provenance against package artifacts
$ slsa-verifier verify-artifact \
--provenance-path multiple.intoto.jsonl \
--source-uri github.com/launchdarkly/python-server-sdk-ai \
launchdarkly_server_sdk_ai-${VERSION}-py3-none-any.whl
```

Below is a sample of expected output.

```
Verified signature against tlog entry index 150910243 at URL: https://rekor.sigstore.dev/api/v1/log/entries/108e9186e8c5677ab3f14fc82cd3deb769e07ef812cadda623c08c77d4e51fc03124ee7542c470a1
Verified build using builder "https://github.com/slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@refs/tags/v2.0.0" at commit 8e2d4094b4833d075e70dfce43bbc7176008c4a1
Verifying artifact launchdarkly_server_sdk_ai-0.3.0-py3-none-any.whl: PASSED

PASSED: SLSA verification passed
```

Alternatively, to verify the provenance manually, the SLSA framework specifies [recommendations for verifying build artifacts](https://slsa.dev/spec/v1.0/verifying-artifacts) in their documentation.

**Note:** These instructions do not apply when building our libraries from source.
