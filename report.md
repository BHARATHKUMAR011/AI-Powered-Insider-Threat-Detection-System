# PROJECT REPORT: AI-Powered Insider Threat Detection System

## 1. Abstract
The increasing frequency and sophistication of insider threats present significant challenges to modern cybersecurity. This project presents an AI-Powered Insider Threat Detection System that leverages behavioral analytics, Graph Neural Networks (GNNs), and Explainable AI (XAI) to proactively identify and mitigate malicious activities from within an organization. By ingesting and analyzing divergent data streams—including user logins, file access patterns, USB device usage, and NLP-processed email communications—the system constructs comprehensive profiles of typical user behavior. Employing unsupervised machine learning methodologies such as Isolation Forests, Autoencoders, and Graph representation learning, the platform successfully flags anomalous deviations indicative of potential threats. The integration of SHAP values provides transparent, interpretable evidence for each flagged anomaly, effectively bridging the gap between complex algorithmic outputs and actionable security insights through a robust, interactive dashboard.

## 2. Introduction
Cybersecurity frameworks traditionally focus on perimeter defenses to externalize threats. However, internal vulnerabilities—perpetrated by authorized users varying from disgruntled employees to compromised credentials—represent a disproportionate source of catastrophic data breaches. The complexity of identifying anomalous behavior among legitimate daily activities requires advanced, multi-faceted analytical strategies. This project develops a continuous monitoring and threat intelligence platform that models user behavior across disparate actions. Utilizing a combination of statistical feature engineering, natural language processing for text-based communications, and graph-based tracking of lateral movement, the system accurately separates benign operations from potential security incidents.

## 3. Problem Statement
Organizations routinely deploy preventative measures such as firewalls and access controls; however, traditional rules-based SIEM (Security Information and Event Management) systems often fail to identify insider threats due to excessive false positives and rigidity. Detecting an authorized user abusing their privileges requires understanding the contextual and historical baseline of their specific roles. The problem at hand is to build an intelligent, scalable system capable of detecting subtle, multi-stage insider attacks—such as low-profile data exfiltration, lateral privilege escalation, and evasion techniques—while simultaneously providing security analysts with clear, interpretable reasons for why an alert was generated.

## 4. Existing System
Historically, insider threat detection relied on static rule sets and threshold-based alerts (e.g., flagging whenever a user downloads more than 50 files). 
**Drawbacks of the Existing System:**
*   **High False Positive Rates:** Legitimate bulk operations heavily trigger inflexible rules.
*   **Lack of Contextual Awareness:** They do not consider the relationships between different data points (e.g., a login at 2 AM followed by a large email payload).
*   **Zero-day Evasion:** Rule-based systems cannot detect novel attack patterns that lack pre-configured signatures.
*   **Black-Box Outputs:** Even when traditional ML models are used, they rarely explain *why* an analyst should care, leading to alert fatigue.

## 5. Proposed System
The proposed system replaces static thresholds with dynamic, machine learning-driven behavioral analysis. It continually learns from simulated logs (logins, file accesses, USB usage, and emails) to establish behavioral baselines. 
**Key Advantages of the Proposed System:**
*   **Multivariate Feature Engineering:** Combines temporal frequency features, natural language sentiment metrics, and graph centrality measures.
*   **Unsupervised Anomaly Detection:** Utilizes algorithms (like Autoencoders and Isolation Forests) capable of detecting unknown threat vectors without requiring historically labeled attack data.
*   **Graph Neural Network Integration:** Models users, machines, and files as nodes to identify suspicious structural network patterns and lateral movement.
*   **Explainable AI (XAI):** Ensures that every generated anomaly score is accompanied by granular feature contributions via SHAP, explaining the exact behavioral shifts causing the alert.

## 6. System Architecture
The architecture is structured as a robust pipeline comprising data ingestion, processing, learning, and visualization stages:
1.  **Data Ingestion & Simulation Layer:** Captures multiple event vectors (USB logs, authentication events, file interactions) and simulates targeted red-team behaviors for training validation.
2.  **Feature Engineering Engine:** Merges discrete logs into tabular behavioral matrices. It executes NLP on email text and computes degree centralities via tools like NetworkX.
3.  **Modeling Core:** Generates risk scores. It utilizes Scikit-Learn models alongside PyTorch-based Graph Neural Networks (`gnn_model.pth`).
4.  **Explainability (XAI) Component:** Applies counterfactual and SHAP-based interpretations to the output tensors, determining the highest-contributing risk factors per user.
5.  **Interactive Visualization:** A Streamlit-based interface rendering interactive Risk Dashboards, dynamic network graphs (via PyVis), and individual user scrutiny panels.

## 7. Modules Description

*   **Log Simulation & Red Teaming Module:** Generates massive datasets to reflect normal corporate behavior while selectively injecting anomalous actions (e.g., mass USB transfers at strange hours, suspicious email content) using simulated red-team attacks.
*   **Feature Extraction & NLP Module:** Synthesizes raw logs into numerical and categorical configurations. It leverages Natural Language Processing to extract sentiment and entity-based risk features from corporate emails.
*   **Graph Analytics Module:** Constructs interaction mappings between network entities. Evaluates the sub-graphs of user activity to identify nodes exhibiting abnormal connectivity changes.
*   **Anomaly Detection Module:** Trains and runs inference using unsupervised algorithms (Isolation Forest/Autoencoders) and Graph Neural Networks to output a normalized anomaly score per user.
*   **Dashboard & XAI Module:** An interactive front-end. It features cross-filtering tables, an interactive "At-Risk Graph" using `vis-network.js`, and visual charts detailing the exact features driving up a user's risk score.

## 8. Technology Stack
*   **Programming Language:** Python 3.x
*   **Machine Learning / Deep Learning:** PyTorch, Scikit-Learn
*   **Graph Processing:** NetworkX, PyVis
*   **Natural Language Processing (NLP):** NLTK/Spacy
*   **Explainability:** SHAP
*   **Frontend / Dashboard:** Streamlit, HTML/CSS, JavaScript (vis-network.js)
*   **Data Manipulation:** Pandas, NumPy
*   **Development Tools:** Visual Studio Code, GitHub Copilot

## 9. Implementation Details
The system execution flows sequentially:
*   First, mock organizational logs are generated via `simulate_logs.py`.
*   Data is purposely compromised using `simulate_red_team.py` to test the system's robustness against adversarial activity.
*   The `merge_features.py` script normalizes and engineers a wide matrix encompassing frequency-based, NLP-based, and graph-based features into `merged_features.csv`.
*   The modeling layer, instantiated in `train.py` and `gnn_anomaly.py`, fits unsupervised algorithms against the benign baselines and evaluates anomaly scores on the merged dataset.
*   Counterfactuals and explainability vectors are calculated in the `explainability/` directory.
*   Finally, `combined_dashboard.py` serves the aggregated data, allowing an analyst to view the holistic network environment, investigate a single suspicious user, and interrogate the AI's logic behind the assigned anomaly score.

## 10. Results and Discussion
The implementation successfully discriminates between normal workforce routines and deviations injected by the simulated red team. The combined approach of utilizing standard numerical logs alongside NLP and Graph features proves significantly more effective than single-vector analysis. During testing, the system autonomously clustered simulated insider threats, accurately assigning them high anomaly scores. Furthermore, the integration of SHAP successfully correlated these high scores directly back to the anomalous actions—such as off-hours logins or unprecedented USB usage—demonstrating a highly interpretable and accountable ML pipeline.

## 11. Conclusion
This project successfully establishes a comprehensive, AI-driven framework for insider threat detection. By shifting the paradigm from static rule-based security to dynamic behavioral modeling, the system provides a resilient defense against complex internal vulnerabilities. Integrating unsupervised deep learning for anomaly detection with SHAP for interpretability creates a powerful, transparent, and scalable tool for security analysts, actively reducing false positives and accelerating incident response times.

## 12. Future Scope
*   **Real-Time Stream Processing:** Transitioning from batch CSV processing to real-time analysis using Apache Kafka or Spark Streaming.
*   **Advanced GNN Architectures:** Implementing dynamic temporal graph neural networks capable of recognizing shifting relationship graphs spanning large timeframes.
*   **Enhanced NLP Capabilities:** Expanding NLP feature extraction across wider corporate communication channels, such as Slack or Microsoft Teams chat semantics.
*   **Automated Mitigation:** Linking the output of the threat dashboard to automated incident response systems (e.g., automatically suspending compromised user accounts).