# Agents Overview

## 1. Defined Agents

- **Strategist Agent (`strategist_agent.py`)**: Responsible for high-level planning, decision-making, and strategy formulation based on clinical supply chain data.
- **Watchdog Agent (`watchdog_agent.py`)**: Monitors data integrity, compliance, and operational status across the supply chain.

## 2. Agent Responsibilities by Table

| Agent             | Responsible Tables (CSV files)                                                                                   |
|-------------------|------------------------------------------------------------------------------------------------------------------|
| Strategist Agent  | - `enrollment_rate_report.csv`<br>- `country_level_enrollment_report.csv`<br>- `study_level_enrollment_report.csv`<br>- `metrics_over_time_report.csv`<br>- `material_requirements.csv`<br>- `materials_per_study.csv`<br>- `planned_orders.csv`<br>- `purchase_requirement.csv`<br>- `material_country_requirements.csv`<br>- `rim.csv`<br>- `re-evaluation.csv`<br>- `ip_shipping_timelines_report.csv` |
| Watchdog Agent    | - `allocated_materials_to_orders.csv`<br>- `available_inventory_report.csv`<br>- `affiliate_warehouse_inventory.csv`<br>- `complete_warehouse_inventory.csv`<br>- `inventory_detail_report.csv`<br>- `inspection_lot.csv`<br>- `lot_status_report.csv`<br>- `warehouse_and_site_shipment_tracking_report.csv`<br>- `shipment_status_report.csv`<br>- `shipment_summary_report.csv`<br>- `outstanding_site_shipment_status_report.csv`<br>- `batch_geneology.csv`<br>- `batch_master.csv`<br>- `bom_details.csv`<br>- `purchase_order_kpi.csv`<br>- `purchase_order_quantities.csv` |

## 3. Agent Interactions

- The **Watchdog Agent** autonomously monitors inventory, expiry, and shortfall risks using tables like `allocated_materials_to_orders`, `available_inventory_report`, and generates structured JSON alerts.
- The **Strategist Agent** uses validated data and alerts from the Watchdog Agent, and consults regulatory (`rim.csv`, `material_country_requirements.csv`), technical (`re-evaluation.csv`), and logistical (`ip_shipping_timelines_report.csv`) tables to answer scenario-based queries and recommend actions.
- Both agents share status updates and data summaries to ensure coordinated supply chain management and decision support.
