# Assignment: AI Implementation Engineer
## Topic: Clinical Supply Chain Control Tower – Agentic Architecture Design

### 1. Context & Scenario
**Global Pharma Inc.** manages complex clinical trials across 50+ countries. Currently, their supply chain data is fragmented across 40+ disparate tables (SAP dumps, SharePoint lists, Excel trackers). Supply Managers are overwhelmed by manual data reconciliation, leading to two major risks:
1.  **Stock-outs:** Patients not getting medication because enrollment spiked unexpectedly.
2.  **Waste:** High-value drugs expiring in warehouses because no one noticed they could be re-allocated or life-extended.

**Your Goal:** Design an **Agentic AI System** that automates risk detection and assists managers in making complex supply decisions.

### 2. The Data
You are provided with a raw dataset representing the internal database of the supply chain you can load these csv files in postgres sql database.
*   **Data Source:** "Attached zip file"
*   **Schema Overview:** The database contains ~40 tables covering:
    *   **Inventory:** Batch locations, expiry dates, quantities (e.g., *Affiliate Warehouse Inventory*).
    *   **Demand:** Patient enrollment forecasts and actuals (e.g., *Country Level Enrollment*).
    *   **Logistics:** Shipments, orders, and timelines (e.g., *Distribution Order Report*).
    *   **Regulatory/Quality:** Inspection lots, stability documents (e.g., *RIM*, *QDocs*).

### 3. Business Requirements (BRD)
You must design an AI architecture to handle **two distinct workflows**.

#### **Workflow A: The "Supply Watchdog" (Autonomous Monitoring)**
*   **Business Goal:** Eliminate manual checking of inventory health.
*   **Requirement:** The system must run autonomously (e.g., daily cron) to identify specific risks.
    1.  **Expiry Alert:** Identify reserved batches (`Allocated Materials`) that are expiring in **≤ 90 days**. Group them by criticality (Critical <30 days, High <60 days).
    2.  **Shortfall Prediction:** Compare *Projected Demand* (based on recent enrollment rates in `Enrollment Rate Report`) against *Current Inventory* (`Available Inventory Report`). Alert if stock runs out within 8 weeks.
*   **Output:** The Agent must generate a structured JSON payload for an email alert system, summarizing the risk.

#### **Workflow B: The "Scenario Strategist" (Conversational Assistant)**
*   **Business Goal:** Assist managers in "saving" expiring stock via Shelf-Life Extension.
*   **Requirement:** A conversational agent that answers ad-hoc user queries like: *"Can we extend the expiry of Batch #123 for the German trial?"* ( If you extend by answering more other queries related to whole database then that would be a plus)
*   **Logic:** The AI must checks three specific constraints:
    1.  **Technical:** Has this material been re-evaluated before? (Check `Re-Evaluation` table).
    2.  **Regulatory:** Is the extension submission approved in that country? (Check `RIM` or `Material Country Requirements`).
    3.  **Logistical:** Is there enough time to execute this? (Check `IP Shipping Timelines`).
*   **Constraint:** The AI must explicitly state *why* it says Yes/No citing the data found.

---

### 4. The Assignment Task
Please prepare a submission covering the following three sections.

#### **Part 1: Architectural Design (The Blueprint)**
*   Propose a **Multi-Agent Architecture** to solve the workflows above.
*   **Diagram/Flowchart:** Show how the User/Scheduler interacts with the Agents.
*   **Agent Definition:** Define the agents needed. (e.g., *Router Agent, Inventory Agent, Regulatory Agent*).
    *   Which agent is responsible for which tables?
    *   Which agents interact with each other?

#### **Part 2: Technical Implementation Strategy**
*   **Tool Design:** Define the tools the agents will use.
    *   *Example:* `run_sql_query(query: str)`
    *   *Task:* Write the **System Prompt** for the "Supply Watchdog" agent. How do you teach the LLM about the specific table schemas (e.g., specific column names like `wh_lpn_number` vs `LPN`) without overloading the context window?
*   **SQL Logic Generation:**
    *   Write the specific **SQL Query** (Postgres) (or the logic the agent would generate) to solve the **"Shortfall Prediction"** requirement in Workflow A. *Hint: You need to join Demand data with Supply data.*

#### **Part 3: Edge Case Handling**
*   Describe how your agents handle data ambiguity.
    *   *Scenario:* The user asks for "Trial ABC", but the database records it as "Trial_ABC_v2". How does the agent resolve this?
    *   *Scenario:* The agent generates an invalid SQL query. How do you design a self-healing loop?

---

### 5. Submission Guidelines
*   **Format:** A PDF document or a GitHub Repository with a Markdown `README.md`.
*   **Code:** Pseudo-code or Python snippets are highly encouraged for Part 2. Also you can try N8N for this implementation if you want it would be a big plus.
*   **Evaluation Criteria:**
    1.  **Schema Understanding:** Did you identify the correct tables for the logic?
    2.  **Agent Design:** Is the separation of concerns clear? (e.g., Don't make one agent do everything).
    3.  **Prompt Engineering:** Are the system prompts robust enough to handle the complex schema?

---
*Good luck. We look forward to seeing how you architect the future of our supply chain.*