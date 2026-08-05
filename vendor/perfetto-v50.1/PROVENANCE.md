# Perfetto C++ SDK provenance

- Project: `google/perfetto`
- Release: `v50.1`
- Commit: `2c4d2ffa7ff300e0b0feb8b8553e42afc7945870`
- License: Apache-2.0 (see `LICENSE`)
- Retrieved: 2026-08-05

The v50.1 GitHub release API no longer lists the `perfetto-cpp-sdk-src.zip` asset named by the integration contract. The two release-tagged amalgamation files remain available at immutable tag URLs and are vendored without modification:

| File | Source URL | Bytes | SHA-256 |
|---|---|---:|---|
| `sdk/perfetto.h` | `https://raw.githubusercontent.com/google/perfetto/v50.1/sdk/perfetto.h` | 7,380,597 | `c13259b528e932d96a71945392ec1a0c6e8be3f9a4b7ac1c9f746e7d7e046c12` |
| `sdk/perfetto.cc` | `https://raw.githubusercontent.com/google/perfetto/v50.1/sdk/perfetto.cc` | 2,669,294 | `012cf1575fe33f93386114d84c2e3c9c814366f329cd99ef2e403a174f9ade49` |
| `LICENSE` | `https://raw.githubusercontent.com/google/perfetto/v50.1/LICENSE` | 14,169 | `df3384bec476fac3525c723d0c10da9ac9820ef56cbb065ec035d32e9b55bd7a` |

The pinned Linux capture and analysis tools are distributed separately and are not vendored:

| Asset | Source URL | Bytes | SHA-256 |
|---|---|---:|---|
| `linux-amd64.zip` | `https://github.com/google/perfetto/releases/download/v50.1/linux-amd64.zip` | 9,792,693 | `7165d3ee71e13204f37c16ff0bff8bb6e73f2c484a33b85d1a185bd81930c3a5` |

Only the self-contained C++ SDK amalgamation and upstream license are compiled into the optional tracing build. No build-time download is performed.
