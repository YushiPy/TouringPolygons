# Repository scripts

Command-line utilities. The benchmark logic lives in `benchmarks/_internal/`
and is exposed by `python3 benchmarks/tpp.py` (the only public benchmark CLI);
the `.sh` shortcuts below only prepare the Python for `tpp.py` and call it.

| Script | What it does |
|---|---|
| `install_dependencies.sh` | sets up the repository's dependencies **without sudo**: downloads whatever is missing and can be obtained without an administrator (uv, Python 3.12, headers, the original Fekete et al. solver and its packages); only lists what needs an administrator. Uses Homebrew on macOS when present. `--solvers-only` skips Node, the dashboard and Playwright |
| `sanity_check.sh` | checks dependencies, builds the solvers and runs the C++ and Python tests in about a minute (with the build ready). `--full` adds the convex stress suites, about 25 more minutes. Runs no benchmarks |
| `verify_unordered.sh` | builds and runs the focused check of the free-order solver |
| `benchmark.sh` | shortcut for `tpp.py tui`: assembles (and prints, copies or runs) a benchmark command |
| `run_comparison.sh` | shortcut for `tpp.py free-compare`: prepares and runs/resumes the free-order comparison on Fekete's 558 cases |
| `run_tspn_comparison.sh` | shortcut for `tpp.py tspn-compare`: prepares and runs/resumes the full TSPN campaign |
| `run_paula_campaign.sh` | shortcut for `tpp.py paula-campaign`: sets everything up without sudo (uv, headers, GMP, the solver and its tests) and runs/resumes the long ILS + B&B campaign on Paula's instances (`--detach`, `--report`) |
| `run_fekete_paper.sh [SECONDS]` | configuration **A** of the original Fekete et al. code on the paper's 558 instances (CHR root, one run at a time with one thread per logical CPU of the performance cores, 3 repetitions; `--threads N` picks another number); in the background and resumable |
| `run_fekete_single_thread.sh [SECONDS]` | configuration **B** (one thread, CHR and LEFP roots, one run per performance core, 3 repetitions); waits for A to finish if it is running |
| `run_ours_single_thread.sh [SECONDS]` | configuration **C**: our solver in B's protocol, with the options of the 10/16 comparison; the directory name carries our code's commit |
| `run_fekete_paper_lefp.sh [SECONDS]` | configuration **D**: like A, with the LEFP root |
| `run_ours_threads.sh [SECONDS]` | configuration **E**: our solver in A's protocol, with the same threads |
| `machine_specs.sh` | prints this machine's hardware and software in Markdown, without sudo, for `benchmarks/machines.md` |
| `review_meeting_transcript.py` | contextual review of meeting transcripts (below) |

The shortcuts accept the same options as the subcommands
(`scripts/run_comparison.sh --help`). `TPP_PYTHON` picks the Python (3.12 or
newer). The commands, options, results and the Fekete setup are documented in
[`../benchmarks/README.md`](../benchmarks/README.md) (in Portuguese).

## Reviewing a transcript

```bash
python3 scripts/review_meeting_transcript.py \
	"docs/meetings/AAAA-MM-DD/transcrição-bruta.txt"
```

By default the result is written to the same folder as
`transcrição-revisada.txt`.

Options:

- `transcript`: path to the raw transcript, required.
- `--output`: alternative path for the reviewed version.
- `--end-time`: ignores segments that start after the given time in seconds.

The script applies the recurring substitutions defined in `REPLACEMENTS`, plus
a few regular-expression normalizations. The automatic review does not
guarantee a perfectly literal transcript; uncertain passages must be checked
against the timestamps and the original recording. It lives here because
`docs/meetings/` is private and ignored by Git.
