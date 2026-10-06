# Gemma 4 Developer Agent: Team Kickoff

Thirty-two English pages for a first team discussion, approximately 30 minutes: 26 content pages, a cover, and five chapter dividers.
The cover has one title, `Gemma 4 Developer Agent Competition`, and the subtitle
`Methodology TBD`. The five-line author block uses 9.3 pt type (reduced from 11 pt).
The deck covers the competition, proposed research directions, and repository
navigation. It contains no cluster instructions or role assignments.

## Authors

The original Overleaf author is retained, with four collaborators added:

- Wenjia Jiang — University of Alberta
- Di Liao — Hangzhou Dianzi University
- Sholpan Amankeldi — Westlake University
- Junzhe Jiao — Tongji University
- Xuanlin Guo — Fudan University

The title page lists five separate lines in the form `Name (University)`, with no
affiliation superscripts or separate institution list. Job titles, degree
descriptions, and the Singapore Management University affiliation have been removed. Official university marks
are displayed without redrawing their artwork; lettering within the original
marks is preserved. Asset provenance is recorded in `assets/logos/README.md`.

## Template

The presentation loads the original `Jacquenetta` theme supplied in Overleaf.
All five `sty/*.sty` files are copied unchanged from the original project.
The native title page, Helvetica typography, blue/orange palette, dark page border,
left-rule blocks, and page numbering are retained. Five native black section pages
organize agent basics, competition rules, research ideas, Kaggle training, and collaboration. University marks appear only
on the title page at 6 mm maximum height. Alberta uses the shield-only crop,
centered in a 6 mm-wide slot to match the other university marks.

## Files and build

- `main.tex`: build entry point; loads the preamble, metadata, and sections.
- `preamble.tex`: shared packages, typography, and diagram styles.
- `metadata.tex`: title, authors, affiliations, and title-page logos.
- `sections/00_opening.tex`: title page and meeting agenda.
- `sections/01_agent_basics.tex`: model/tool feedback loop, a real tool call, and a concrete repair example.
- `sections/02_competition.tex`: task, submission, data, evaluation, resources, a first Kaggle exercise, leaderboard, and verified results.
- `sections/03_research.tex`: hypotheses, experiments, and fine-tuning roadmap.
- `sections/04_kaggle_training.tex`: Kaggle setup, training-data preparation, and a small LoRA pilot.
- `sections/05_collaboration.tex`: repository tour and collaboration.
- `sections/06_next_steps.tex`: milestones and discussion.
- `sections/07_references.tex`: reference links and reading list.
- `figures/`: independently editable task, training, and collaboration diagrams.
- `main.pdf`: compiled presentation included in Git for teammates to read without a TeX installation; rebuild it when changing the slide sources.
- `latexmkrc`: XeLaTeX build configuration.
- `assets/logos/`: university marks and their source records.
- `assets/illustrations/` and `assets/emoji/`: two official illustrations and four Twemoji icons.
  See `assets/README.md`, `assets/sources.json`, and the bundled emoji graphics license.
- `sty/`: original Jacquenetta theme by Samuel Manchajm, CC BY-SA 4.0.
- `fonts/`: the license is retained; unused font binaries from the earlier Chinese version stay local and are excluded from Git.

Build using `latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex`.
On Overleaf, select `main.tex` as the main document and XeLaTeX as the compiler.
The English deck uses standard TeX Live fonts and needs no custom font install.
The original template also supports Overleaf's default pdfLaTeX compiler.

## Audience and presenter notes

The deck assumes basic Python/Git knowledge and no previous LLM-agent experience.
An independent reviewer, given only the visible deck and no conversation history,
reviewed it twice from this audience's perspective. Revisions clarify inference
versus training, agent ZIP versus per-task patch, one-task onboarding, SFT
context/targets, and held-out evaluation. The second review found no remaining
substantive comprehension blockers; its layout comments were also addressed.
A second reviewer independently challenged the final visible deck without project
or conversation context and found no must-fix comprehension blockers. Final checks
also balanced two chapter-title line breaks while preserving the original theme.
Three introductory pages explain the model/tool loop, an actual tool call, and the
verified Rich repair. SFT, LoRA, and RL are defined before research proposals. The later Kaggle
training section explains how to prepare offline inputs, build examples, save
adapters, and return to the official evaluator. This training path is proposed;
only inference/evaluation has been executed and verified.
Rendered slides use short statements; `\note{...}` blocks hold English presenter
explanations and evidence limits. Notes are hidden in the normal PDF. To produce a
speaker-note version, add `\setbeameroption{show notes}` to a separate build copy.

## Evidence and timestamps

Official rules and team proposals are distinguished on the slides. Per-slide
source footnotes are omitted from the rendered deck; their provenance is retained
as comments in the section sources and in the final reference slide. Visuals include a tool-call conversation, a before/after repair, an experiment map,
and a small training journey. Original black divider pages mark five chapters.
Sources for illustrations and emoji are credited on the final reference slide.
The leaderboard snapshot is October 5, 2026, 03:50 UTC. The completed-runs slide was updated
using the 18:30 UTC status check: submission 56844353 is COMPLETE with public
score 0.12. The L4 x 4 notebook's downloaded logs confirm valid empty/gold
controls and 84 passing tests for the agent patch on one public task. The agent
phase took 74.10 seconds, excluding model startup. Private score is unavailable;
one task passing is not a full local benchmark or training validation.

The source ZIP is generated at
`resource/downloads/presentations/gemma4-kickoff-overleaf.zip` relative to the
project root. Build verification and synchronization receipts are recorded in
`resource/manifests/slides-kickoff.json`.

Overleaf project: https://www.overleaf.com/project/6ac31a4a617496d875323ad8
The existing remote template and logo files are retained. Credentials stay
outside this project and are never included in Git or the source ZIP.
