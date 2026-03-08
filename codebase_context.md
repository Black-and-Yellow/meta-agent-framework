# Meta-Agent Codebase Context

This document provides a comprehensive overview of the `LLM_Orch` codebase. It is designed to be passed to an LLM (like Claude) to give it a full architectural and operational understanding of the system, enabling it to make informed decisions for future development, debugging, or feature additions.

---

## 1. System Overview & Philosophy

**"The code writes the code that solves the problem."**
The Meta-Agent is an LLM orchestration framework that doesn't rely on hardcoded agent pipelines. Instead, for every user task, it dynamically **designs an architecture, instantiates the required agents, wires them together into a graph, executes the graph, evaluates the final output, and self-repairs if the quality is insufficient.**

### The Core Loop
1. **Plan**: `TaskPlanner` uses an LLM to decompose the user's prompt into sub-tasks and selects a topology.
2. **Design**: `BlueprintGenerator` strictly converts the plan into a deterministic, validated Pydantic model (`Blueprint`).
3. **Build**: `GraphBuilder` converts the blueprint into an executable `CompiledGraph` of `AgentNode`s.
4. **Execute**: `GraphExecutor` runs the graph (Sequential, Parallel, DAG).
5. **Evaluate**: `ResultEvaluator` uses an LLM-as-judge to score the final output.
6. **Repair**: If the score is below the threshold, the `MetaAgent` orchestrator triggers a refine parameter adjustment or a complete blueprint rebuild.

---

## 2. Pydantic Schemas (The Backbone)
The system relies on strict Pydantic v2 schemas (`src/meta_agent/schemas/`) to enforce structural integrity.

### `blueprint.py`
- **TopologyType**: `SEQUENTIAL`, `PARALLEL`, `HIERARCHICAL`, `DAG`
- **AgentType**: `RESEARCH`, `PLANNING`, `CODING`, `DATA_ANALYSIS`, `VERIFICATION`, `SUMMARIZATION`, `CRITIC`, `CUSTOM`
- **ToolType**: `WEB_SEARCH`, `CODE_EXECUTOR`, `DATABASE_QUERY`, `VECTOR_RETRIEVAL`, `API_CALLER`, `FILE_READER`
- **EdgeConfig**: Defines directional flow between agents. `condition_type` dictates when an edge is traversed (`ALWAYS`, `ON_SUCCESS`, `CONDITIONAL`, etc.).
- **AgentConfig**: Configures a dynamic agent instance (model, temperature, tools, `system_prompt_template`).
- **ExecutionGraph**: The full graph definition (nodes, edges, parallel groups).
- **EvaluationConfig**: Dimensions to score (e.g., correctness, completeness) and threshold levels.
- **Blueprint**: The master self-describing immutable snapshot of the entire generated architecture. 

### `state.py`
- **ExecutionContext**: The transient "short-term memory" shared across an execution run. It contains `intermediate_results` from each agent, `agent_records`, shared `messages`, and resource tracking (tokens/time).
- **AgentMessage**: An inter-node message.
- **EvaluationResult**: Output from the evaluator containing scores per dimension and a `RepairAction` (`ACCEPT`, `REFINE`, `REBUILD`).

---

## 3. Core Engine Mechanics (`src/meta_agent/core/`)

### `planner.py` (`TaskPlanner`)
- Uses `ChatOpenAI` (or compatible API) with a specific system prompt.
- Reads the raw user task and outputs a highly structured JSON plan.
- Identifies independent workstreams (creates `PARALLEL` topology) or linear tasks (`SEQUENTIAL` topology).
- Guesses the token bounds and required evaluation focus.

### `blueprint_generator.py` (`BlueprintGenerator`)
- A deterministic engine that translates `TaskPlanner`'s JSON output into a `Blueprint` model.
- Fetches default prompts and specific tool configurations natively for each archetype.
- Computes dependencies and depth grouping for parallel agent execution.
- Wires the `EdgeConfig`s automatically between agents based on the topology type.

### `meta_agent.py` (`MetaAgent`)
- The main entry point: `await meta.solve(task)`.
- Implements the **Outer Repair Loop**:
  - Starts up to `max_repair_iterations`.
  - Generates blueprint -> builds -> executes -> evaluates.
  - If `EvaluationResult.recommended_action == REBUILD`, it feeds the failure reasons and the previous score back to the `TaskPlanner` to design a *better* architecture.
  - If `REFINE`, it tweaks temperatures and token limits algorithmically via `blueprint.next_revision()`.

---

## 4. Orchestration (`src/meta_agent/orchestration/`)

### `graph_executor.py` (`GraphExecutor`)
- Drives the `CompiledGraph` to completion.
- Passes the `ExecutionContext` state through agents.
- **Topology Handlers**:
  - `_execute_sequential`: Simple linear traversal. **Also includes an Inner Repair Loop.**
  - `_execute_parallel`: Uses `asyncio.gather` for agents grouped in the same dependency depth tier, then fans back into remaining sequential nodes.
  - `_execute_dag`: A conditional edge traversal mechanism checking `edge.evaluate(state)`.
- **Inner Repair Loop (Sequential only)**:
  - If a Verification Agent runs directly after a Coding Agent and fails the verification, the executor detects it.
  - It pushes the verification text (feedback) into a special `_repair_feedback` state variable and specifically re-runs the previous Coding agent.
  - It limits repairs (default `max_inner_repairs = 1`) and tries to detect sandbox failures to inject rules dynamically (e.g., "SANDBOX RESTRICTION: No internet access allowed").
- **Dynamic Optimization**:
  - The executor can detect simplistic tasks using Regex (`write a function`, `convert`) and conditionally skip over the `PLANNING` agent to save time/tokens.
  - Skips verification agents if the coding agent's sandbox execution already yielded `success: True`.

---

## 5. Extensibility: Agents & Tools

### Agents (`src/meta_agent/agents/`)
- `AgentFactory` dynamically builds agent instances from `AgentConfig`. 
- Different specialized agents inherit from a `BaseAgent`. They accept kwargs for the LangChain/OpenAI LLM APIs.
- Instead of rigid class-based single-purpose actors, agents are fluid; changing the `system_prompt_template` morphs their behavior.

### Tools (`src/meta_agent/tools/`)
- Tools are sandboxed with timeouts and `permission_scope` logic.
- Notable integrations include an isolated `code_executor` sandbox for Python execution and `web_search` using DuckDuckGo (`ddgs` has recently been updated see recent commit references implicitly derived).

## 6. Project Setup & Deployment
- Frameworks: FastAPI + Uvicorn + Pydantic v2 + LangChain (+ LangGraph conceptual influences)
- Infrastructure configs live in `infra/` (Docker, Kubernetes, Prometheus tracking).
- Memory features ChromaDB for Long-Term memory (`MemoryEntry` in semantic space) and Redis for fast state lookups.

---

### How to Use This Context
When making code implementations or diagnosing logic bugs, remember:
1. **Never break the Blueprint validation:** Pydantic is intensely strict here.
2. **Avoid modifying `meta_agent.py` indiscriminately**, as the outer repair mechanic is fragile.
3. If altering execution mechanics or adding efficiency skips, `graph_executor.py` is the target.
4. If altering how the system responds to the user abstractly, update the LLM Prompts inside `planner.py` or default templates in `blueprint_generator.py`.
