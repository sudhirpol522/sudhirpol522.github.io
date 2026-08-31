---
layout: page
title: Human-in-the-Loop Agent Systems
description: Agent workflows that combine structured planning, MCP tool access, traceable execution, and explicit human approval.
area: Agentic AI
stack: LangGraph, MCP, FastAPI, OpenTelemetry
importance: 1
category: featured
---

## Overview

I design agent systems as controlled workflows rather than unconstrained chains of model calls. The agent can interpret context, propose a plan, and coordinate tools, but consequential transitions remain observable and subject to explicit human approval.

Two implementations shaped this approach: an MCP-integrated evaluation workflow built during my Adobe internship and BrandForge, a production-style multi-agent platform with durable approval checkpoints.

> Agents propose and coordinate. Tools execute through defined interfaces. People retain authority at the decisions that matter.

## MCP-integrated evaluation at Adobe

I built a LangGraph agent that connected to Adobe PDF Spaces through Model Context Protocol. The workflow generated an evaluation plan, paused for human review before execution, and recorded agent activity with LangSmith tracing.

### Selected results

- Reduced an end-to-end evaluation feedback cycle from 48 hours to 30 minutes.
- Used MCP as a clear boundary between agent reasoning and document capabilities.
- Required human approval before a generated plan could execute.
- Traced the workflow so plans, tool activity, and outcomes could be inspected during review.
- Paired the workflow with a LiteLLM-based LLM-as-Judge harness whose rubric refinement with domain reviewers raised F1 on a labeled evaluation set from 0.79 to 0.92.

## BrandForge multi-agent platform

BrandForge applies the same control principles to a larger creative workflow. A deterministic state machine coordinates brand compilation, planning, eight-way generation, ranking, and four versioned approval gates.

### Selected results

- Built idempotent FastAPI endpoints with tenant isolation and optimistic concurrency so retries and simultaneous reviews could not silently corrupt workflow state.
- Implemented a two-stage multimodal ranker with policy, claims, and accessibility critics; Bradley-Terry preference scoring; MMR diversity; and vision-based alignment scores.
- Added OpenTelemetry traces with Tempo and Grafana for end-to-end observability.
- Built a tenant-isolated PostgreSQL and pgvector retrieval layer with row-level security, policy filters, idempotent indexing, and a bounded SQLite fallback.
- Verified workflow behavior with 90 automated tests.

## Design principles

- **Explicit state:** every transition is represented in a deterministic workflow rather than hidden inside a prompt loop.
- **Bounded tools:** MCP or typed service interfaces separate reasoning from side effects.
- **Human authority:** approval gates protect high-impact transitions without forcing people to perform repetitive orchestration.
- **Idempotency and concurrency control:** safe retries and version checks make long-running workflows reliable.
- **Traceability:** plans, tool calls, approvals, and outputs remain inspectable after execution.
