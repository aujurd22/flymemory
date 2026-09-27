# LongMemEval-V2 Data Schema

## `questions.jsonl`

Each line is one JSON object with these fields:

- `id`: stable question id.
- `domain`: `web` or `enterprise`.
- `environment`: site/environment family label.
- `question_type`: memory ability category.
- `question`: question text.
- `image`: optional path under `question_screenshots/`; `null` for text-only questions.
- `answer`: reference answer used by the evaluator.
- `eval_function`: evaluator specification string.

## `trajectories.jsonl`

Each line is one JSON object with these fields:

- `id`: stable trajectory id.
- `domain`: `web` or `enterprise`.
- `environment`: trajectory environment family.
- `goal`: task goal for the trajectory.
- `outcome`: `success` or `failure`.
- `start_url`: URL of the first state.
- `states`: ordered state/action records.

Each state contains:

- `state_index`: zero-based index in the trajectory.
- `step`: original step number when available.
- `url`: browser URL at the state.
- `action`: action taken from this state, or `null` for the initial state.
- `thought`: agent thought text when available.
- `accessibility_tree`: text accessibility tree observation.
- `screenshot`: path under `screenshots/<trajectory_id>/<step>.png`.

## Haystacks

- `haystacks/lme_v2_small.json`: JSON object mapping each `question_id` to an ordered array of 100 `trajectory_id` strings. Within each domain, all questions share one 100-trajectory haystack.
- `haystacks/lme_v2_medium.json`: JSON object mapping each `question_id` to an ordered array of `trajectory_id` strings. Most questions have 500 trajectories; a small number of web-domain questions have fewer because the available matching pool is smaller.

Haystack ids resolve against `questions.jsonl` and `trajectories.jsonl`. Domain labels are stored on the question and trajectory records, not duplicated inside the haystack files.
