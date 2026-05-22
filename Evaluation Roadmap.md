# C-Arc Evaluation Roadmap

---

## **Phase 1: Individual Agent Audits (Modular Reliability)**

Each autonomous agent is verified in isolation to ensure that the "Digital Twin" construction is accurate before the system-wide integration.

* **Profiler Agent Audit**:
  * **Metric**: **Mean Squared Error (MSE)** & **Inference Jitter**.
  * **Method**: Feed the agent transcripts labeled with ground-truth Big Five scores. Compare the agent's live OCEAN vector against those labels.
  * **Stability Goal**: Ensure the personality vector reaches convergence and does not fluctuate more than **15%** between turns once stability is signaled.


* **IR Agent Audit**:
  * **Metric**: **Mapping Precision & Recall**.
  * **Method**: Present the agent with raw, non-standardized professional descriptions. It must correctly ground them to the **O*NET 30.2** database identifiers.
  * **Latency Target**: Local SQLite query execution must remain **<10ms**.


* **Evaluator Agent Audit**:
  * **Metric**: **Gate Logic Accuracy**.
  * **Method**: Provide "Incomplete" profiles (e.g., missing career history or $<5$ skills). The Evaluator must correctly trigger **Gate 2 (Density Audit)** to block the matching process.


* **Career Expert & Generator Audit**:
  * **Metric**: **Match Probability** & **DWA Alignment**.
  * **Method**: Audit the Resume Generator’s output to verify that every generated bullet point maps back to a valid **Detailed Work Activity (DWA)** for the suggested role.



---

## **Phase 2: System-Wide Simulation (The LLM Actor Test via API)**

This phase validates the **LangGraph** state machine and orchestration logic by using an LLM to simulate a human candidate in a live environment.

* **LLM Persona Modeling**: Deploy a high-reasoning LLM instance as an "Actor Agent." This agent is assigned a unique, complex persona including specific professional experience and hidden personality traits.


* **The Interview Loop**:
  * The Actor Agent engages in a **10–15 turn dialogue** with the system.
  * The system must autonomously extract skills and build a personality profile.


* **Verification Points**:
  * **State Control**: Verify that the system transitions between modes only after the **Triple-Gate Logic** (Variance, Density, and Stability) is satisfied.
  * **Convergence Speed**: Measure the number of turns required to achieve **Euclidean Stability ($d < \epsilon$)** across different persona types.



---

## **Phase 3: Adversarial Robustness & Data Augmentation**

This stage uses "Dirty Data" to ensure the system makes decisions based on professional logic rather than simple keyword patterns.

* **Manual Data Augmentation**: Generate 1,000 "Mismatched" synthetic profiles by intentionally decoupling skill sets from personality traits.

* **Test Scenarios**:
  * **The "Toxic Expert"**: A profile with 99th-percentile technical skills for a high-stakes role (e.g., Surgeon or Senior Manager) but with augmented personality data showing "Low Stress Tolerance" or "Low Agreeableness."
  * **The "Enthusiastic Amateur"**: A profile with perfect personality alignment for a role but zero grounded technical skills.


* **Validation of the Career Expert**:
  * The matching engine must downgrade the alignment score for these profiles.
  * A pass is achieved only if the model identifies that personality is a **Hard Constraint**, not just a secondary feature.

