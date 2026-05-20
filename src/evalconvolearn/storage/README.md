# Storage

File-based persistence layer for sessions and student pools.

## `base.py` — abstract interfaces

- **`StudentPoolStorage`** — defines `save_pool`, `load_pool`, `pool_exists`.
- **`SessionStorage`** — defines `save_session_state`, `load_session_state`, `session_exists`.

Implement these to swap in a different storage backend (database, cloud, etc.).

## `file_storage.py` — concrete implementations

- **`FileStudentPoolStorage`** — persists a `StudentPool`'s practice history to a CSV file via `StudentPool.save_student_pool_practice_history_to_csv` / `load_student_pool_from_csv`. The pool ID is inferred from the directory name by stripping the timestamp suffix.
- **`FileSessionStorage`** — persists each conversation session as a JSON file named `<session_id>.json` under a given directory. Used by `SessionService` to checkpoint session state after every turn.
