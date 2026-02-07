# Clinical War Room

A multi-agent clinical decision support system that simulates a hospital clinical board meeting.

## Purpose

This system answers:

> "Given the available patient data, is it SAFE and JUSTIFIED to take an action,
> or should the system refuse, escalate to a human clinician, or request more data?"

**This is NOT a chatbot. This is NOT an autonomous doctor.**

## System Outputs

The system outputs:
- Risk assessment
- Confidence levels
- Reasoned recommendation
- Explanation of disagreement
- One of: `PROCEED_WITH_CAUTION` | `ESCALATE_TO_HUMAN` | `REFUSE_ACTION` | `REQUEST_MORE_DATA`

## Architecture

```
Patient Case
  → MCP Tools (objective computation, NO reasoning)
  → RAG (medical guidelines, NO decisions)
  → LLM Specialist Agents (reasoning, opinions)
  → Debate Engine (challenge & revision)
  → Rule-Based Safety Layer (hard overrides)
  → RL Coordinator (policy optimization)
  → Human-in-the-loop
  → Final Recommendation
```

## Quick Start

```bash
# Install dependencies
pip install -e .

# Copy environment template
cp .env.example .env

# Run the server
python main.py
```

API docs: http://localhost:8000/docs

## Implementation Phases

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 0** | ✅ Complete | Project skeleton & infrastructure |
| Phase 1 | ⏳ Pending | MCP Tools |
| Phase 2 | ⏳ Pending | RAG System |
| Phase 3 | ⏳ Pending | Specialist Agents |
| Phase 4 | ⏳ Pending | Debate Engine |
| Phase 5 | ⏳ Pending | Rules & Safety |
| Phase 6 | ⏳ Pending | RL Coordinator |
| Phase 7 | ⏳ Pending | UI |

## Key Design Principles

1. **LLMs are used ONLY for reasoning and debate** - never for math or thresholds
2. **MCP is the ONLY bridge** between reasoning and action
3. **Rules cannot be overridden** - safety is non-negotiable
4. **All decisions must be explainable**
5. **When uncertain, ESCALATE or REFUSE**

## Project Structure

```
clinical-warroom/
├── api/              # FastAPI routes
├── core/             # Infrastructure (config, logging, MCP)
├── tools/            # MCP-executable tools (NO LLM)
├── rag/              # Retrieval-Augmented Generation
├── agents/           # LLM specialist agents
├── debate/           # Board discussion logic
├── rules/            # Hard safety guardrails
├── rl/               # Reinforcement learning
├── coordinator/      # Decision synthesis
├── models/           # Shared schemas
├── ui/               # War Room interface
├── tests/            # Unit & integration tests
└── main.py           # Entry point
```

## License

MIT
