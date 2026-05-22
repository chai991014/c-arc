# C-Arc Development Roadmap

---

## **Phase 1: Foundation & State Management**

Establish the standardized labor market vocabulary and the "nervous system" of the framework.

* **Step 1: Environment & Dependency Scaffolding** — Configure the Python environment, define secure `.env` pathing, and integrate the DeepSeek API for advanced reasoning tasks.
* **Step 2: O*NET Relational Ingestion** — Load the Unified **O*NET 30.2 Stack** into a local SQLite database (`onet.db`) with optimized indexing to ensure query latency stays below **10ms**.
* **Step 3: JSON Artifact Generation** — Extract technical skills and tools into high-speed `SKILL_POOL` and `OCEAN_POOL` JSON files for rapid look-aside retrieval.
* **Step 4: OCEAN Cross-walk** — Map the 16 O*NET Work Styles to the Big Five (OCEAN) personality dimensions to establish trait benchmarks for all 923 SOC codes.
* **Step 5: CArcState Schema Definition** — Engineer the `state.py` file to define the central `TypedDict` that manages the master profile, live OCEAN vector, and message history.

---

## **Phase 2: Directed Synthetic Pipeline (The Dataset Factory)**

Construct the grounded training stack using **100% Synthetic Data** to teach the system how to match humans to job roles without reliance on external resume noise.

* **Step 1: Synthetic Skeleton Generation** — Programmatically generate profiles for all 923 SOC codes by selecting random subsets of O*NET **Detailed Work Activities (DWAs)** and **Tasks**.
* **Step 2: Personality Anchoring** — Assign baseline OCEAN vectors to every synthetic profile based on the official O*NET Work Style benchmarks for that specific role.
* **Step 3: Gaussian Noise Injection** — Apply a normal distribution to personality scores to simulate realistic human variance and prevent overfitting:

$$Score_{synthetic} = \mu_{O*NET} + \mathcal{N}(0, 10)$$


* **Step 4: Binary Feature Mapping** — Transform the generated activities and skills into a sparse matrix of binary features for model ingestion.

---

## **Phase 3: Career Expert (The Matching Engine)**

Train the discriminative ML model that performs high-accuracy job-match ranking.

* **Step 1: XGBoost Architecture** — Architect the **XGBoost Classifier** and optimize it for local GPU inference.
* **Step 2: Discriminative Training** — Train the model from scratch on the 10,000 synthetic records from Phase 2 to learn the statistical alignment between skills and personality.
* **Step 3: Cold Start Probing** — Verify the model's ability to suggest "Interest-Aligned" career paths when no skills are present, utilizing only the personality vector.

---

## **Phase 4: Personality & Dialogue Layer (Dual-Mode Mentor)** // SFT or Prompt Eng. for Mentor

Fine-tune the interactive agents to conduct strategic interviews and provide empathetic career coaching.

LLM used: Gemma 4 E4B (**Mentor**), DeepSeek R1-Distill-Qwen 14B (**Profiler**)

[
* **Step 1a: Mentor (Interviewer) Fine-Tuning** — Perform **LoRA SFT** using **ConvCounsel** to master "Strategic Nudging" for fact extraction.
* **Step 1b: Mentor (Counselor) Fine-Tuning** — Perform **LoRA SFT** using **EmpatheticDialogues** for action-oriented, empathetic coaching.

//
* **Step 1: Mentor Agent** — Prompt engineering.

]


* **Step 2: Profiler Agent (Two-Stage SFT)** — Bifurcate the training pipeline for the DeepSeek-14B Profiler:
  * **Stage A (Knowledge Alignment):** Train on `butyuhao/OCEAN-Chat` to establish "High/Low" linguistic archetypes and trait definitions.
  * **Stage B (Nuance Refinement):** Fine-tune on `preke/PELD` to inject emotional sensitivity and calibrate the model on character-driven, multi-turn transitions.


* **Step 3: Logit-Based Rating Engine** — Implement a **Binary Logit Shift** rating system. Instead of predicting numerical strings, the model will output "High" or "Low" labels, with the final intensity score derived via the probability ratio of the first generated token:

$$Score_{trait} = \frac{P(\text{"High"})}{P(\text{"High"}) + P(\text{"Low"})}$$


* **Step 4: Recursive State Integration** — Engineer a **Stateful Accumulation** logic to prevent "memory reset" during periodic updates (every 10 turns). Integrate new evidence ($E_t$) into the previous persistent OCEAN vector ($P_{t-1}$) using an Exponential Moving Average (EMA):

$$P_{t(new)} = (1 - \alpha) P_{t-1} + \alpha E_t$$


* *Note: $\alpha$ (the learning rate) should decrease as turn count increases to simulate "profile inertia."*

---

## **Phase 5: IR Agent Development (The Translation Engine)**

Build the autonomous "Silent Observer" required to ground real-time conversation and external validation data.

* **Step 1: DeepSeek Extractor** — Engineer NLU prompts to identify ADD, UPDATE, and DELETE intents and capture raw professional entities from dialogue.
* **Step 2: Local IR Mapper** — Implement a `sentence-transformers` semantic search to ground raw text to official O*NET identifiers.
* **Step 3: Profile Reconciler** — Develop the logic to manage user-driven profile modifications and resolve data contradictions in the `CArcState`.

---

## **Phase 6: Evaluator Agent (The Logic Controller)**

Develop the "Brain" of the framework to manage the state machine and audit data maturity via **Triple-Gate Logic**.

* **Step 1: Gate 1 (Trait Variance)** — Implement logic to identify "Cold" traits by calculating cumulative variance between turns.
* **Step 2: Gate 2 (IR Density Audit)** — Implement a hard fact check requiring at least 5 O*NET-grounded skills and a complete role history.
* **Step 3: Euclidean Stability (Velocity Check)** — Update the stability logic to monitor the **velocity of change** in the persistent/accumulated vector rather than turn-by-turn snapshots. The gate opens when the magnitude of the update ($|P_t - P_{t-1}|$) consistently dampens:

$$d = \sqrt{\sum (P_{t} - P_{t-1})^2}$$

* **Step 4: State Control Logic** — Configure state signals like `FORCE_TOPIC` and `TRIGGER_GATEWAY` to orchestrate transitions between Mentor modes.

---

## **Phase 7: Asset Synthesis (Deployment)**

Translate the internal profile data into tangible professional assets and feedback.

* **Step 1: Resume Generation** — Use the DeepSeek API to map the verified Master Profile to the **Detailed Work Activities (DWAs)** of target SOC codes.
* **Step 2: Relational Bridge Integration** — Implement the logic to link SOC IDs to Task IDs and DWA IDs for high-precision bullet point generation.
* **Step 3: Counselor Feedback Loop** — Design the interface where the Mentor (Counselor) explains matching logic to the user.
* **Step 4: Automated Export** — Configure the workflow to export ATS-optimized documents in **LaTeX** or **Markdown**.