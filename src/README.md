# Clinical Supply Chain Control Tower – Agentic Architecture

## Overview
This project implements an agentic AI system for automating risk detection and decision support in a clinical supply chain environment. It is designed for Global Pharma Inc. and addresses two major workflows:

- **Supply Watchdog**: Autonomous daily monitoring of inventory health, expiry risks, and shortfall prediction.
- **Scenario Strategist**: Conversational assistant for managers to evaluate shelf-life extension requests and other supply chain queries.

## Features
- **Automated Expiry Alerts**: Identifies batches at risk of expiry within 90 days and classifies them by criticality.
- **Shortfall Prediction**: Compares projected demand (from enrollment rates) against current inventory to alert if stock runs out within 8 weeks.
- **Conversational Query Handling**: Answers ad-hoc queries about batch expiry extension, checking technical, logistical, and regulatory constraints. For a "YES" response, all three checks (technical, regulatory, logistical) must pass based on the available data.
- **Self-Healing SQL Execution**: Agents automatically repair and retry invalid SQL queries using LLM feedback.
- **Explicit Column Context**: The agents use explicit column names for each table to prevent hallucination and ensure only valid columns are referenced in queries.
- **Environment Variable Support**: Uses `.env` for secure configuration.

## Folder Structure
```
Clinical-Supply-Chain-Control-Tower/
│   .gitignore
│   env/                # Python virtual environment
│   src/
│       .env            # Environment variables
│       requirements.txt
│       strategist_agent.py
│       watchdog_agent.py
```

## How It Works
### 1. Supply Watchdog (`watchdog_agent.py`)
- Runs daily (can be scheduled via cron or task scheduler).
- Connects to PostgreSQL, executes expiry and shortfall queries.
- Outputs a structured JSON payload for email/API alerts.

### 2. Scenario Strategist (`strategist_agent.py`)
- Accepts user queries (e.g., "Can we extend the expiry of Batch #123 for the German trial?").
- Uses an LLM (Gemini API) to reason through technical, logistical, and regulatory checks. The agent will only return "YES" if:
   - The batch has a "Complete" status for extension or retest in the technical table (`re-evaluation.csv`).
   - The trial/material has an "Accepted" or "Approved" regulatory outcome in the regulatory tables (`rim.csv`, `material_country_requirements.csv`).
   - The country for the trial/material has a valid shipping timeline in the logistics table (`ip_shipping_timelines_report.csv`).
- The agent uses explicit column context for each table to ensure reliable and accurate SQL generation.
- Returns a clear YES/NO answer with supporting data and reasoning.

## Setup Instructions
1. **Clone the repository**
2. **Create and activate a Python virtual environment**
   ```sh
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. **Install dependencies**
   ```sh
   pip install -r src/requirements.txt
   ```
4. **Configure environment variables**
   - Create a `.env` file in `src/` with your database and API keys:
     ```env
     GEMINI_API_KEY=your_gemini_api_key
     DB_HOST=localhost
     DB_NAME=supplychain
     DB_USER=n8nuser
     DB_PASSWORD=n8nuser
     DB_PORT=5432
     ```
5. **Load your clinical data into PostgreSQL**
   - Use the provided CSVs and schema overview to populate your database.

## Usage
- **Run Supply Watchdog**
  ```sh
  python src/watchdog_agent.py
  ```
- **Run Scenario Strategist**
  ```sh
  python src/strategist_agent.py
  ```

## Edge Case Handling
- **Data Ambiguity**: Agents use fuzzy matching and context to resolve naming mismatches (e.g., "Trial ABC" vs "Trial_ABC_v2").
- **SQL Errors**: Agents detect invalid SQL, provide error feedback to the LLM, and retry with corrected queries.
- **Missing Data**: If any required data for technical, regulatory, or logistical checks is missing, the agent will return "NO" and explain which check(s) failed.

## Technologies Used
- Python 3.13+
- PostgreSQL
- Gemini API (Google Generative AI)
- python-dotenv
- psycopg2

## License
This project is for demonstration and educational purposes.

# Clinical Supply Chain Control Tower – Agentic Architecture

## Overview
This project implements an agentic AI system for automating risk detection and decision support in a clinical supply chain environment. It is designed for Global Pharma Inc. and addresses two major workflows:

- **Supply Watchdog**: Autonomous daily monitoring of inventory health, expiry risks, and shortfall prediction.
- **Scenario Strategist**: Conversational assistant for managers to evaluate shelf-life extension requests and other supply chain queries.

## Features
- **Automated Expiry Alerts**: Identifies batches at risk of expiry within 90 days and classifies them by criticality.
- **Shortfall Prediction**: Compares projected demand (from enrollment rates) against current inventory to alert if stock runs out within 8 weeks.
- **Conversational Query Handling**: Answers ad-hoc queries about batch expiry extension, checking technical, logistical, and regulatory constraints.
- **Self-Healing SQL Execution**: Agents automatically repair and retry invalid SQL queries using LLM feedback.

## Folder Structure
```
Clinical-Supply-Chain-Control-Tower/
│   .gitignore
│   env/                # Python virtual environment
│   src/
│       .env            # Environment variables
│       requirements.txt
│       strategist_agent.py
│       watchdog_agent.py
```

## How It Works
### 1. Supply Watchdog (`watchdog_agent.py`)
- Runs daily (can be scheduled via cron or task scheduler).
- Connects to PostgreSQL, executes expiry and shortfall queries.
- Outputs a structured JSON payload for email/API alerts.

### 2. Scenario Strategist (`strategist_agent.py`)
- Accepts user queries (e.g., "Can we extend the expiry of Batch #123 for the German trial?").
- Uses an LLM (Gemini API) to reason through technical, logistical, and regulatory checks.
- Returns a clear YES/NO answer with supporting data and reasoning.

## Setup Instructions
1. **Clone the repository**
2. **Create and activate a Python virtual environment**
   ```sh
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. **Install dependencies**
   ```sh
   pip install -r src/requirements.txt
   ```
4. **Configure environment variables**
   - Create a `.env` file in `src/` with your database and API keys:
     ```env
     GEMINI_API_KEY=your_gemini_api_key
     DB_HOST=
     DB_NAME=
     DB_USER=
     DB_PASSWORD=
     DB_PORT=5432
     ```
5. **Load your clinical data into PostgreSQL**
   - Use the provided CSVs and schema overview to populate your database.

## Usage
- **Run Supply Watchdog**
  ```sh
  python src/watchdog_agent.py
  ```
- **Run Scenario Strategist**
   ```sh
   python src/strategist_agent.py
   ```

## Decision Logic Example

To receive a "YES" response for a shelf-life extension query, ensure:

- The batch has a "Complete" status for extension or retest in `re-evaluation.csv`.
- The trial/material has an "Accepted" or "Approved" regulatory outcome in `rim.csv` or `material_country_requirements.csv`.
- The country for the trial/material has a valid shipping timeline in `ip_shipping_timelines_report.csv`.

**Sample Query:**

```
Can we extend Batch LOT-66123852 for trial CT-6592-IDY?
```

If all checks pass, the agent will respond with YES and supporting reasoning for each check.

## Edge Case Handling
- **Data Ambiguity**: Agents use fuzzy matching and context to resolve naming mismatches (e.g., "Trial ABC" vs "Trial_ABC_v2").
- **SQL Errors**: Agents detect invalid SQL, provide error feedback to the LLM, and retry with corrected queries.

## Technologies Used
- Python 3.13+
- PostgreSQL
- Gemini API (Google Generative AI)
- python-dotenv
- psycopg2

## License
This project is for demonstration and educational purposes.
