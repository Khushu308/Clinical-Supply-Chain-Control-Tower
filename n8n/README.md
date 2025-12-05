# n8n Workflows Overview

This folder contains n8n workflow JSON files for the Clinical Supply Chain Control Tower project. These workflows automate supply chain risk detection and scenario analysis using agentic AI principles.

## Components Used

- **n8n Workflow Engine:** Orchestrates automation and agent logic via visual workflows.
- **PostgreSQL Database:** All supply chain data (inventory, demand, regulatory, etc.) is loaded into a Postgres instance. Workflows use the PostgreSQL node to query tables directly (no API endpoints).
- **Gemini LLM (Google Generative AI):** Used for advanced reasoning, prompt-based decision support, and conversational responses within the workflows.
- **JSON Nodes:** Structure output for email alerts and agent communication.
- **Scheduler/Cron Nodes:** Automate periodic execution (e.g., daily risk checks).
- **Email Nodes:** Send structured risk alerts to supply managers.

## Workflow Files

- **Strategist-Agent.json:** Implements the "Scenario Strategist" workflow. Handles ad-hoc queries, shelf-life extension logic, and multi-constraint checks (technical, regulatory, logistical) using direct SQL queries and Gemini LLM reasoning.
- **Watchdog-Agent.json:** Implements the "Supply Watchdog" workflow. Monitors inventory health, expiry risks, and shortfall predictions. Generates JSON payloads for email alerts and triggers escalation logic.

## Key Features

- **Direct Table Access:** All logic is based on querying Postgres tables (e.g., `allocated_materials_to_orders`, `available_inventory_report`, `enrollment_rate_report`, `rim`, `re-evaluation`, `ip_shipping_timelines_report`).
- **LLM Integration:** Gemini LLM is used for prompt engineering, edge case handling, and generating human-readable explanations for decisions.
- **Modular Design:** Each workflow is modular, with clear separation of concerns between monitoring, reasoning, and notification components.
- **Extensible:** New agents or logic can be added by creating additional workflow JSON files and connecting them to the database and LLM nodes.

## Usage Instructions

1. Import the workflow JSON files into your n8n instance.
2. Configure the PostgreSQL node with your database credentials.
3. Set up Gemini LLM credentials for the LLM nodes.
4. Activate the workflows and set scheduling as needed.

---

For more details, see the individual workflow JSON files and the main project README.
