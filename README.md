# Problem

Clinical supply chains are complex, involving multiple data sources, regulatory requirements, and logistical constraints. Manual monitoring and decision-making for batch expiry, shortfall risks, and shelf-life extensions is error-prone, slow, and difficult to scale. Supply managers need reliable, explainable, and automated tools to ensure timely decisions and compliance across global operations.

# Solution

This project provides an agentic AI-powered control tower for clinical supply chains, integrating Python-based agents and n8n workflow automation. The system automates daily risk detection, expiry alerts, shortfall prediction, and scenario-based decision support. It leverages LLMs for advanced reasoning, explicit column context for robust SQL generation, and modular workflows for extensibility and notification. The result is a unified, transparent, and scalable platform for supply chain decision support, reducing manual effort and improving reliability.

# Clinical Supply Chain Control Tower – Master Overview

## Essence of the Architecture

This project delivers an agentic AI-powered solution for clinical supply chain management, integrating two complementary approaches:

### Python Agentic Architecture
- Scenario Strategist & Supply Watchdog: Python-based agents automate risk detection, expiry alerts, shortfall prediction, and scenario-based decision support.
- Conversational Decision Support: The Scenario Strategist agent answers ad-hoc queries (e.g., shelf-life extension requests) by checking technical, regulatory, and logistical constraints using direct SQL queries and explicit column context.
- Self-Healing & Reliable Reasoning: Agents use LLMs (Gemini API) for prompt engineering, error correction, and structured output, ensuring robust and explainable decisions.
- Data-Driven: All logic is based on querying PostgreSQL tables loaded from clinical CSVs, with clear requirements for a "YES" decision (all checks must pass).

### n8n Workflow Automation
- Visual Workflow Orchestration: n8n workflows automate supply chain monitoring, risk alerts, and scenario analysis using a visual, modular approach.
- Integration with LLMs & Databases: Workflows connect directly to PostgreSQL for data queries and use Gemini LLM for advanced reasoning and explanations.
- Extensible & Modular: New agents, logic, and notification channels can be added by updating workflow JSON files and connecting them to the database and LLM nodes.
- Automated Notifications: Scheduler and email nodes enable periodic risk checks and structured alerts to supply managers.

## Unified Value Proposition
- End-to-End Automation: From daily risk monitoring to complex scenario analysis, the system provides a seamless, data-driven, and explainable supply chain control tower.
- Human-in-the-Loop Decision Support: Managers can interact with agents or workflows to get actionable, transparent recommendations for shelf-life extensions and other supply chain decisions.
- Robustness & Reliability: Explicit column context, error handling, and modular design ensure the system is resilient to data ambiguity and SQL errors.
- Scalable & Extensible: Both Python agents and n8n workflows can be extended to cover new data sources, business rules, and notification channels.

## Technologies Used
- Python 3.13+
- PostgreSQL
- Gemini API (Google Generative AI)
- n8n Workflow Engine
- python-dotenv
- psycopg2

## Getting Started
- See individual README files in `src/` and `n8n/` for setup, configuration, and usage instructions.
- Load your clinical data into PostgreSQL and configure your environment variables and LLM credentials.

## License
This project is for demonstration and educational purposes.
