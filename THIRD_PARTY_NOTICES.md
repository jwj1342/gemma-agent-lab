# Third-party material

This private collaboration repository contains team-authored material alongside
reference snapshots. No repository-wide open-source
license has been selected for the team's original work. Upstream rights and
competition terms continue to apply to third-party material.

- `resource/competition/`, `resource/notebooks/`, `resource/discussions/`,
  `resource/models/`, and `resource/leaderboard/` preserve dated Kaggle sources,
  authors, identifiers, and links in their indexes and metadata. Access to a
  public page does not by itself grant permission to relicense its contents.
- `resource/harness-source/` is a static inspection copy of official evaluation
  package sources. Package metadata and source notices are retained alongside it.
- The presentation is maintained in a [separate Overleaf project](https://www.overleaf.com/project/6ac31a4a617496d875323ad8).
  Its theme and asset attribution are retained with that project; slide sources,
  images, fonts, and PDFs are excluded from the current GitHub tree.

Model weights and competition dataset archives are excluded from Git. Each team
member obtains authorized access through Kaggle and follows the applicable model,
dataset, and competition terms. Before publishing code or redistributing an
upstream artifact, check that specific artifact's license and preserve attribution.

## SWE training trajectories

The local SFT pilot uses the public
[SWE-bench/SWE-smith-trajectories dataset](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories),
revision `08e109b4a59eaeebf80e4675cd125d42e7ac99a4`, whose metadata declares MIT. Raw and
converted trajectories remain excluded from Git. Team conversion code is an original implementation;
original tool observations and teacher identifiers are retained in local provenance. Repository code
inside trajectories can carry its own upstream notices. Consult the pinned dataset and source
repository licenses before redistributing those materials.
