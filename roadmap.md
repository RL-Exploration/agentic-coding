# MVP Roadmap: Automated RL Environments for Coding Agents

---

## Phase 1: The Data Generator (Today's Goal)

Write a Python script that calls the Anthropic API (Opus 4.6).

- Use a strict system prompt to generate **50 unique, non-standard coding puzzles**
  (e.g., *"Write a function to calculate the optimal path for a delivery drone with battery constraints"*).
- Save the outputs as local `.json` files containing:
  - `prompt`
  - `starter_code`
  - `reference_solution`
  - `unit_tests`

---

## Phase 2: The Execution Sandbox

Don't build this from scratch. Use an open-source sandbox tool like **E2B** (English2Bits) or a simple lightweight **Docker container** triggered via Python's `subprocess`.

- Write a script that takes a generated `reference_solution`, injects it into the `unit_tests` file, runs it in the sandbox, and parses the `stdout`/`stderr`.
- If it passes, the puzzle is **validated** and added to your active RL environment.

---

## Phase 3: The RL Environment Wrapper

Wrap your sandbox in a standard reinforcement learning interface (like OpenAI's **gymnasium**).

- **`reset()`** — Loads a random validated JSON puzzle and returns the prompt/starter code as the initial observation.
- **`step(action)`** — Takes the agent's generated code, runs it in the sandbox, parses the test results, and returns:
  - **reward** (e.g., `+1` for each passed test, `-1` for syntax errors)
  - **observation** (the compiler traceback)

---

## Phase 4: The Agent Loop

- Write a simple loop where a smaller, cheaper agent model (like **Claude 3.5 Haiku** or a local open-source model) tries to solve the environments, receives the traceback, and tries again.
- Save the trajectory to a database:
  - `prompt` → `bad code` → `traceback` → `good code`
