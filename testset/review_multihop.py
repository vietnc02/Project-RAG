"""Kết quả duyệt tay câu multi-hop (đối chiếu từng câu với 2 đoạn gốc trong multihop_raw.json).

    python testset/review_multihop.py      -> testset/multihop.json

Tiêu chí giữ: câu hỏi cần thông tin của CẢ HAI bài, tự đứng được (không "theo hai đoạn văn"),
đáp án chuẩn chỉ chứa thông tin có trong 2 đoạn gốc.
- DROP: câu bị loại (keep=false) kèm lý do
- EDITS: id -> (question mới, ground_truth mới); câu giữ lại đều được viết lại cho rõ và đúng với đoạn gốc
"""
import json
from pathlib import Path

DROP = {
    "m006": "ghép 2 ý không liên quan (độ chính xác ViT trong IDS IoT + thu nhỏ TinyViT)",
    "m008": "so sánh vô nghĩa (số honeypot vs cleanup rate 0,02%)",
    "m011": "ghép 2 ý không liên quan (công nghệ CMOS + IoT)",
    "m012": "một đoạn (Practical Challenges in Executing Shor's Algorithm) đã trả lời đủ",
    "m014": "ghép 2 ý không liên quan (hàm submodular + trò chơi Stackelberg)",
    "m016": "dùng lại đúng đoạn WAFFLED của m004; đáp án gán sai cơ chế cho WAFFLED",
    "m018": "so sánh gượng ép (mã 9-qubit vs trạng thái chồng chập)",
    "m019": "ghép 2 ý không liên quan (StreamingLLM + LFU cache)",
    "m021": "hai survey cùng tác giả, đoạn gần như trùng nhau -> một đoạn đã trả lời đủ",
    "m024": "ghép 2 ý không liên quan (machine_constraints của CVE + benchmark xOffense)",
    "m026": "ghép 2 ý không liên quan (Inception CNN + SNN của Masquelier & Thorpe)",
    "m030": "cả hai đoạn đều mô tả vanilla self-attention -> một đoạn đã trả lời đủ",
    "m033": "câu hỏi quá chung chung, đáp án chỉ nhắc lại phạm vi 2 bài survey",
    "m034": "so sánh vô nghĩa (số power trace của tấn công vs hiệu năng phần cứng của biện pháp chống)",
    "m035": "so sánh gượng ép (xLSTM vs baseline sliding window của StreamingLLM)",
    "m036": "so sánh gượng ép (bias BN trong Spiking ResNet vs chuẩn hoá trọng số)",
    # lượt sinh thêm (--append --seed 7)
    "m038": "so sánh gượng ép (hoạt động thưa của SCNNTraffic vs lỗi spike đầu tiên trên MNIST)",
    "m041": "đoạn QML for Cybersecurity một mình đã có đủ số liệu; đáp án gán sai số liệu cho IBM Quantum Platform",
    "m044": "ghép 2 ý không liên quan (cắt ngắn input của LongBench + AS-ES Learning)",
    "m045": "ghép 2 ý không liên quan (ràng buộc thưa trong Sparse Coding + VECTOR)",
    "m047": "đoạn NetSMT bị cắt trước phần cần hỏi; đáp án có chi tiết không có trong đoạn",
    "m049": "ghép 2 ý không liên quan (NetSMT + CLIP/SetFit)",
    "m051": "ghép 2 ý không liên quan (tấn công đối kháng lượng tử + ràng buộc QKD)",
    "m052": "so sánh vô nghĩa (công thức precision của 2 bài); đáp án sai",
    "m055": "ghép 2 ý không liên quan (tỉ lệ tham số embedding + cắt bớt encoder)",
    "m056": "so sánh gượng ép (NASim vs honeypot camera IP)",
    "m058": "dùng lại đúng đoạn survey của m042 (Degreaser)",
    "m059": "dùng lại đúng đoạn Neuronal Dynamics của m054",
    "m060": "cả hai số liệu nằm trong cùng một bảng của bài ICLSTM -> một đoạn đã trả lời đủ",
    "m062": "ghép 2 ý không liên quan (cắt bớt MobileBERT/TinyViT + KV cache của H2O)",
    "m063": "ghép 2 ý không liên quan (hướng phát triển xOffense + kết quả GenXSS)",
    "m065": "so sánh vô nghĩa (thời gian lập lịch 13 µs vs tham số T = 2)",
    "m066": "ghép 2 ý không liên quan (ngưỡng LIF + trọng số cân bằng lớp)",
    "m067": "ghép 2 ý không liên quan (kiểm chứng mẫu XSS + dropout)",
    "m069": "hai đoạn gần trùng nội dung về Kyber -> gần như một đoạn đã trả lời đủ",
    "m071": "đáp án tự sinh không đúng với đoạn gốc; hai ý ghép gượng ép",
}

EDITS = {
    "m001": ("How do concerns about harvest-now-decrypt-later (HNDL) attacks differ between security professionals "
             "and profit-motivated cybercriminals, and which kinds of data are considered prime targets for HNDL?",
             "An ISACA survey found that 56% of security professionals were specifically concerned about HNDL, i.e. that "
             "encrypted data stolen today could be broken later. Profit-motivated cybercriminals have less incentive to "
             "store data for a decade (e.g. stolen credit card numbers will be useless by then). Data with long "
             "confidentiality lifespans, such as medical records, government intelligence, banking archives and "
             "intellectual property, are prime targets, and nation-state actors are already conducting HNDL campaigns."),
    "m002": ("How are tasks organized in VulnBot's Penetration Task Graph (PTG) and in xOffense's Task Coordination "
             "Graph (TCG), and how does xOffense use its Knowledge Repository when planning tasks?",
             "In VulnBot's PTG each task depends on one or more preceding tasks, and the task list (with dependencies, "
             "instructions and actions) is given in JSON, ensuring a logical execution order. In xOffense the Task "
             "Orchestrator builds a TCG that decomposes the pentest objective into a sequence of tasks with defined "
             "dependencies, and it queries the Knowledge Repository, a vector database accessed through a RAG mechanism "
             "(from Langchain-Chatchat), to retrieve relevant penetration knowledge based on the task description, the "
             "current task's instruction or recent task results."),
    "m003": ("What role does Quantum Random Number Generation (QRNG) play in quantum key distribution (QKD), and how far "
             "has QKD technology progressed toward real-world deployment?",
             "QKD relies on physics rather than computational hardness to exchange keys provably securely and to detect "
             "eavesdroppers; QRNG supports it by producing true random numbers needed for key setting and growth. Over "
             "the past five years QKD has moved from laboratory experiments to early commercial deployments, for "
             "example financial institutions in Europe and Asia piloting QKD links; combined with one-time-pad or "
             "symmetric ciphers it can provide information-theoretic security."),
    "m004": ("How does the decision logic of the Adaptive Dual-Layer WAF (ADL-WAF) differ from the way conventional "
             "rule-based WAFs detect SQL injection?",
             "ADL-WAF uses a first ML layer to flag anomalies and a second layer that checks anomalies against threat "
             "models: a benign anomaly such as a mistyped '&' is allowed, while injected SQL code flagged as an anomaly "
             "and identified as an attack is blocked. Conventional rule-based WAFs extract request fields and match "
             "them against rules, often regular expressions scanning URL parameters or bodies for SQL keywords; on a "
             "match they block, log or challenge the request (e.g. CAPTCHA), otherwise they forward it."),
    "m005": ("What is the status of CRYSTALS-Dilithium support in cryptographic libraries such as OpenSSL and Botan, and "
             "which industry players have adopted Dilithium?",
             "Botan offers a rich PQC suite in its C++ API, and wolfSSL and Bouncy Castle already integrate Kyber and "
             "Dilithium into their TLS stacks, whereas OpenSSL did not officially support PQC as of early 2025 but plans "
             "to integrate Kyber, Dilithium and others. Dilithium is expected to be the default signature in most "
             "deployments, and Kyber and Dilithium have seen early adoption by Cloudflare, Amazon Web Services and IBM."),
    "m007": ("How do network tarpits distort ICMP-based Internet scans, and how does IP churn affect how long a tarpit "
             "can trap infected IoT devices in the US compared with China?",
             "Degreaser found that tarpitting subnets appear fully occupied with responding hosts in ICMP (ping) based "
             "Internet scans such as the ISI ANT census. In the tarpit feasibility study, a device that gets a new IP "
             "address must be re-learned by the tarpit, so in high-churn networks infected devices temporarily escape "
             "and keep operating; US devices churn much less than devices in China, so the tarpit traps US devices for "
             "longer periods without downtime."),
    "m009": ("How does lateral inhibition implement Winner-Takes-All competition among neurons, and what problem does "
             "Independent Component Analysis (ICA) address beyond PCA?",
             "In Winner-Takes-All the neuron most similar to the input is chosen as winner and, by lateral inhibition, "
             "is the only one allowed to update its weights, so different neurons specialize on different clusters; "
             "lateral connections with a nonlinearity create competition in which activations below a threshold are "
             "suppressed and those above inhibit the others. PCA finds maximally decorrelated directions of maximum "
             "variance, while ICA seeks maximally independent variables, removing higher-order correlations, and is "
             "related to Blind Source Separation."),
    "m010": ("According to RULER, how does Gemini-1.5-Pro's effective context length compare with the maximum length "
             "tested, and why does LongBench consider perplexity an inadequate metric for long-context models?",
             "RULER found that Gemini-1.5-Pro outperforms the other models by a large margin, with an effective length "
             "greater than the maximum length tested. LongBench notes that perplexity may not reflect a model's "
             "performance on sequence-level tasks in real applications, and that artificial tasks such as retrieval may "
             "also fall short of mirroring real-world scenarios."),
    "m013": ("How does the Expressive Leaky Memory (ELM) neuron differ from the leaky integrate-and-fire (LIF) neuron in "
             "its internal dynamics and output?",
             "The ELM neuron, modelled on a cortical pyramidal neuron, has input synapse dynamics, an integration "
             "mechanism (an MLP over synaptic state and memory), leaky memory units with their own timescales, and an "
             "output computed as a linear readout of the memory. The LIF neuron computes a weighted sum of inputs "
             "integrated over time with leakage, like an RC circuit, and emits a discrete spike when the integrated "
             "value crosses a threshold, encoding information in spike timing or frequency."),
    "m015": ("Why do modern attack landscapes call for adaptive honeypots, and what challenges do AI-powered honeypots "
             "face?",
             "Studies found modern attack landscapes are dominated by IoT-based malware and keylogging campaigns, which "
             "underscores the need for adaptive defenses, and cyber deception has moved from static honeypots to dynamic "
             "systems. AI-powered honeypots integrate machine learning, behavioral analytics and automated response, but "
             "face computational overhead, model training complexity and adversarial AI attacks in which attackers "
             "manipulate the ML models, requiring robust countermeasures."),
    "m017": ("Which generation metrics does the ORAN RAG benchmark use to compare Vector RAG, GraphRAG and Hybrid "
             "GraphRAG, and how do they differ from the comprehensiveness, diversity, empowerment and directness "
             "metrics used in earlier GraphRAG evaluation?",
             "The ORAN benchmark uses independent generation metrics, namely faithfulness, answer relevance, context "
             "relevance and factual correctness, and analyzes performance across question complexity levels. Earlier "
             "GraphRAG evaluation asked an LLM to judge pairs of answers on comprehensiveness, diversity, empowerment "
             "and directness, which compare final answers but do not evaluate the retrieval and generation parts "
             "separately."),
    "m020": ("What limitations of standard RAG on financial documents motivate HybridRAG, and why is applying RAG to "
             "small language models (SLMs) still considered an open challenge?",
             "RAG struggles with the specialized, intricate domain of financial documents, and context retrieved from a "
             "vast heterogeneous corpus can be inconsistent, leading to inaccurate and incomplete analyses, so more "
             "sophisticated methods are needed to integrate domain-specific information. For SLMs, techniques such as "
             "RAG and MoE could maintain or boost performance within constrained compute budgets, but integrating them "
             "into SLMs with inherently limited capabilities remains an unresolved challenge."),
    "m022": ("How does cyber deception affect attackers' behaviour, and what did evaluations of bidirectional deception "
             "and honeynets find?",
             "Deception increases attackers' cognitive burden, slowing attack progression, increasing operational "
             "mistakes and forcing more cautious, time-consuming and detectable approaches. Bidirectional deception, "
             "which camouflages authentic systems as honeypots alongside conventional honeypots, was found to increase "
             "attackers' uncertainty and improve defensive outcomes, and honeynets with many honeypots further confuse "
             "adversaries' ability to distinguish decoys from genuine systems."),
    "m023": ("How does the spiking neuromorphic transformer use STDP to compute Query-Key similarity in memory, and on "
             "what information does STDP base its synaptic weight updates?",
             "The Q-K similarity is not computed and stored separately; it is embedded in the synapse through STDP-driven "
             "plasticity, so both computation and storage happen locally in synaptic hardware, which is the principle of "
             "in-memory computing (each synapse acts as both compute and storage element). STDP adjusts the weight "
             "between a pre- and post-synaptic neuron according to their relative spike times within roughly tens of "
             "milliseconds, using information local to the synapse and local in time."),
    "m025": ("How does GenXSS differ from earlier XSS attack-generation work in handling attack generation and "
             "mitigation, and what limitation of WAF-A-MoLE's mutation-tree search do its authors acknowledge?",
             "GenXSS uses generative AI to simultaneously generate XSS attacks and the corresponding WAF rules, a unified "
             "and adaptive defense, whereas existing methods treat attack generation and mitigation separately and "
             "traditional practice relies on experts manually updating WAF rules. In WAF-A-MoLE each branch of the "
             "mutation tree evolves monotonically, so the search may stagnate on local minima, although experiments show "
             "it still finds injectable payloads."),
    "m027": ("When do experts expect cryptographically relevant quantum computers (CRQCs) able to break 2048-bit RSA, "
             "and what post-quantum algorithms has NIST standardized in response?",
             "Industry experts estimate that CRQCs capable of breaking 2048-bit RSA keys could emerge within the next "
             "decade, potentially sooner given exponential R&D investment, because integer factorization and elliptic "
             "curve discrete logarithms are efficiently solvable on a CRQC. After a multi-year process NIST standardized "
             "a first set of PQC algorithms, including lattice-based key encapsulation (CRYSTALS-Kyber) and digital "
             "signatures."),
    "m028": ("How does H2O's heavy-hitter eviction policy differ from VECTOR's three-way token routing for compressing "
             "the KV cache of large language models?",
             "H2O retains only the heavy-hitter (H2) tokens and the most recent tokens' KV embeddings, which largely "
             "reduces KV cache size without degrading OPT-30B performance, i.e. a binary keep-or-evict decision. VECTOR "
             "is a plug-in for eviction-based pipelines that routes tokens to retention, approximation or eviction, "
             "combining the base scorer's importance signal with a reconstructability signal from an offline-calibrated "
             "regression, so reconstructable tokens are approximated instead of being irreversibly evicted."),
    "m029": ("Why are rate-based training methods for spiking neural networks limited, and how do surrogate-gradient "
             "and spike-timing-gradient methods make use of temporal information?",
             "Rate-based methods only count spikes in a time window and ignore their timing, although spike trains with "
             "the same count can have distinct temporal patterns. Neftci et al.'s surrogate gradient algorithm provides "
             "spike-compatible backpropagation that preserves temporal patterns by approximating the true gradient. "
             "Temporal-coding methods instead treat outputs as firing times: SpikeProp linearized firing-time "
             "expressions to compute approximate gradients, and later work computed spike-timing gradients explicitly "
             "for non-leaky integrate-and-fire neurons with competitive performance."),
    "m031": ("What did the BABILong benchmark evaluate, and what does LongBench conclude about retrieval-based context "
             "compression for long-context models?",
             "BABILong extends existing tasks to very long contexts with distributed facts; it evaluated GPT-4 and RAG on "
             "needle-in-a-haystack question answering up to millions of tokens, and a recurrent memory transformer on "
             "inputs up to 11 million tokens, a record sequence size for a single model. LongBench concludes that "
             "context compression such as retrieval helps models with weak long-context ability, but their performance "
             "still lags behind models with strong long-context understanding."),
    "m032": ("How does WAFFLED use content parsing discrepancies to bypass web application firewalls, and why can "
             "semantically equivalent SQL injection payloads, as studied in WAF-A-MoLE, evade WAF detection?",
             "WAFFLED repurposes a grammar-based, structure-aware HTTP fuzzer to find differences in how WAFs and web "
             "frameworks parse complex content types (multipart/form-data, application/xml, application/json); most "
             "WAF-framework pairs could be bypassed with such content distortions. WAF-A-MoLE notes that WAFs must handle "
             "highly expressive languages such as SQL and HTML, so many semantically equivalent payloads exist (e.g. a "
             "plain tautology versus an obfuscated one) and a WAF relying on syntactic analysis may miss some of them."),
    "m039": ("How do Scm's sleep-based memory consolidation and the Recurrent Memory Transformer's self-retrieval each "
             "deal with the limits of keeping information over long interactions or sequences?",
             "Scm tags each stored concept with a four-dimensional importance vector, keeps recent experience in a "
             "strictly limited working memory, and in offline sleep cycles uses NREM consolidation to strengthen "
             "important associations, REM dreaming to create novel connections and an intentional forgetting module to "
             "prune low-value memories. In RMT a fixed-size recurrent state must hold information about the whole "
             "sequence, creating a bottleneck; self-retrieval keeps all previous recurrent states and retrieves the "
             "relevant ones for the current step, conceptually equivalent to attention for RNNs."),
    "m040": ("How did the systematic meta-survey of cyber deception and the 'Cyber Deception: State of the art, Trends "
             "and Open challenges' review select and assess the papers they included?",
             "The meta-survey ran a Scopus query (studies before 2025, 258 records) and two reviewers independently "
             "screened titles, abstracts and, when needed, full content, applying a lightweight quality-assessment "
             "protocol informed by PRISMA and the CASP checklists. The state-of-the-art review used exclusion criteria "
             "that removed articles based solely on keywords, articles from research areas unrelated to engineering or "
             "computer science (e.g. cognitive or psychological work) and articles not published in journals or "
             "conferences, followed by a subjective selection."),
    "m042": ("How could a community-run tarpit infrastructure help clean up IoT botnets, and what tool could attackers "
             "use to detect and avoid network tarpits?",
             "The tarpit feasibility study argues that, given enough participating devices (few enough for a community "
             "project), democratized botnet cleanup with tarpits could contain and eliminate the botnet threat, and it "
             "releases an open-source turn-key honeypot/tarpit that organizations and volunteers can run. Network "
             "tarpits masquerade as many fake hosts to deceive scanners, but Degreaser, an active probing detector based "
             "on packet fingerprinting, can detect tarpits and could be used by attackers for that purpose."),
    "m043": ("How has hybrid post-quantum key agreement been deployed in Chrome, and how do hybrid PQC deployments keep "
             "backward compatibility with parties that do not support it?",
             "From late 2022 into 2023 Google deployed a hybrid X25519+Kyber key agreement in Chrome (with BoringSSL), "
             "and by late 2024 it was enabled for all desktop use, likely the largest PQC deployment so far, done "
             "without user disruption. In hybrid deployments, modern clients and servers support both classical and "
             "PQC algorithms and negotiate mutually supported ones, preferring hybrid; with graceful degradation they "
             "fall back to classical algorithms if hybrid is unavailable, may offer pure PQC, and eventually phase out "
             "classical-only systems."),
    "m046": ("What applications of Quantum Neural Networks (QNNs) do quantum machine learning surveys highlight in "
             "healthcare and in cybersecurity?",
             "In healthcare, QML is promising for personalized medicine and drug discovery, with QNNs offering new "
             "approaches to analysing complex biological data. In cybersecurity, frameworks based on QSVMs, QNNs, VQCs "
             "and QGANs excel in both supervised and unsupervised settings (e.g. malware and botnet detection, "
             "telecommunication security), improving detection accuracy, exposing hidden anomalies and adapting to "
             "evolving threats, although current NISQ hardware imposes constraints."),
    "m048": ("What is the deception goal of a high-interaction client honeypot, and how did Bensalem et al. use deception "
             "against masquerade attacks?",
             "A high-interaction client honeypot aims to 'look' like a human user and be attacked: it mimics human "
             "behaviour by navigating the Internet and interacting with servers (high-interaction honeypots mimic users, "
             "browsers and active content, while low-interaction ones act as decoys). Bensalem et al. treated deception "
             "as a defense against masquerade attacks, where attackers pose as legitimate users (e.g. identity theft), "
             "using monitored honeyfiles acting as an IDS, and evaluated it with scenario-based user studies in which "
             "computer science students played the attackers."),
    "m050": ("How does the classical Hebbian learning rule update synaptic weights, and how does Spike-Time-Dependent "
             "Plasticity (STDP) take spike timing into account?",
             "The Hebbian rule ('cells that fire together wire together') updates a weight as the learning rate times "
             "the pre- and post-synaptic activities; for spike trains this is a dot product of binary activity vectors "
             "over time, which captures co-activation but not precise timing. STDP makes the update depend on the time "
             "difference between post- and pre-synaptic spikes with exponentially decaying terms: the weight is "
             "strengthened when the pre-synaptic spike likely caused the post-synaptic spike (correlated) and weakened "
             "when the pre-synaptic spike comes right after the post-synaptic one (anti-correlated)."),
    "m053": ("Which tools does VulnBot provide to its agents in the exploitation phase, and what limitation of public "
             "exploit sources such as Metasploit has been reported for reasoning about CVEs?",
             "In VulnBot's exploitation phase, agents use Metasploit to develop and execute exploit code and Hydra to "
             "brute-force credentials, exploiting vulnerabilities found earlier to gain access and escalate privileges. "
             "Metasploit, ExploitDB and GitHub are the major public exploit sources, but not all exploits are public or "
             "linked to a CVE, exploit source code follows no specification, and metadata is not always consistent, "
             "for example in Metasploit's documentation of the exploits for CVE-2021-38648 and CVE-2022-39952."),
    "m054": ("How are synaptic input currents usually modelled in spiking neural networks trained with surrogate "
             "gradients, and what does the amplitude of a postsynaptic current depend on according to Neuronal Dynamics?",
             "In SNNs the input current is generated by synaptic currents triggered by presynaptic spikes (spike trains "
             "written as sums of Dirac delta functions), and a common first-order approximation models their time course "
             "as an exponential decay. In Neuronal Dynamics, a spike generates a postsynaptic current whose amplitude is "
             "proportional to the difference between the membrane potential u0 and the synaptic reversal potential "
             "Esyn, so the postsynaptic response depends on the neuron's momentary state, especially for inhibitory "
             "synapses."),
    "m057": ("How does LLM-CAKG build an attack knowledge graph from threat reports, and how is AutoPen's knowledge graph "
             "structured?",
             "LLM-CAKG collects unstructured cyber threat reports and guides an LLM through staged, domain-specific "
             "prompts for entity extraction and relation identification; the triples are normalized and assembled into "
             "one graph, and an LLM-driven iterative clustering step validates, merges and labels similar entities. "
             "AutoPen organizes retrieved vulnerability-exploitation documents into a knowledge graph because standard "
             "RAG struggles with complex semantic relationships; it is a directed graph G=(V,E) whose nodes include "
             "system, software, vulnerability and resource nodes."),
    "m061": ("How do the Spiking STDP Transformer and the Spike Agreement Dependent Plasticity (SADP) experiments convert "
             "static images into spikes?",
             "In the Spiking STDP Transformer, the initial convolution block of Spiking Patch Splitting acts as a spike "
             "encoder that converts pixel intensities into temporal spike trains (Conv2d, batch normalization, "
             "max-pooling and spiking neurons), with two forms of spiking patch embedding: one keeps the original "
             "spatial resolution and one adds downsampling. In the SADP experiments, inputs are converted into "
             "10-timestep Poisson spike trains proportional to pixel intensity (rate coding), and hidden spikes over the "
             "10 timesteps form a 400-dimensional feature vector for a downstream classifier."),
    "m064": ("Why do large post-quantum keys and signatures cause problems for 5G network slices and handovers, and how "
             "large is the handshake overhead of hash-based signature schemes?",
             "In eMBB slices, the per-packet size increase from PQC can add substantial aggregate bandwidth and cause "
             "fragmentation; latency-critical alerting with sub-10 ms bounds can only use schemes with very low "
             "processing latency; and handovers require repeated authentication and key establishment, so larger PQC "
             "handshake messages can worsen signalling congestion. Hash-based signature schemes are especially costly: "
             "their large signatures cause handshake overheads of 245% to 890% relative to classical algorithms, mainly "
             "because signatures must be sent over multiple packets."),
    "m068": ("How does the Module-LWE problem underlying ML-KEM and ML-DSA relate to the basic Learning With Errors (LWE) "
             "problem in lattice-based cryptography?",
             "LWE asks to find a secret vector s such that A·s ≈ b (mod q) given A and b; its security comes from the "
             "difficulty of solving noisy linear systems, and together with SVP it is believed resistant to quantum "
             "algorithms, with ring variants such as Ring-LWE used for efficiency. Module-LWE adds ring/module "
             "structure: over the module R_q^k with R = Z[X]/(X^n+1), given (A, b = As + e) one must distinguish this "
             "distribution from uniform; this refinement underlies NIST's standardized ML-KEM and ML-DSA, balancing "
             "theoretical guarantees with practical efficiency."),
    "m070": ("How does InPars use a T5 model for reranking, and how does VulnBot's Memory Retriever retrieve and re-rank "
             "past tasks?",
             "InPars uses BM25 to select 1,000 candidates and re-ranks them with a fine-tuned T5-base (220M) model "
             "(monoT5) adapted as a binary classifier of document-query relevance. VulnBot's Memory Retriever uses the "
             "bce-embedding-base-v1 model for embeddings and the bce-reranker-base-v1 model for re-ranking: it retrieves "
             "the top 3 most similar vectors with a relevance score above 0.5, and the re-ranking algorithm then selects "
             "the most relevant tasks."),
    "m037": ("How do Quantum Gaussian Mixture Models (QGMMs) and Quantum Boltzmann Machines (QBMs) differ in how they "
             "represent probability distributions?",
             "QGMMs are quantum extensions of Gaussian mixture models for density estimation and clustering; they use "
             "quantum states to model the likelihood that data belong to different Gaussian distributions and are used "
             "for tasks such as image segmentation and pattern recognition. QBMs describe the target distribution as a "
             "quantum thermal (Gibbs) state of a parameterized Hamiltonian with classical energy terms plus transverse-"
             "field quantum fluctuations, allowing them to capture distributions intractable for classical models."),
}


def main():
    here = Path(__file__).resolve().parent
    items = json.loads((here / "multihop_raw.json").read_text(encoding="utf-8"))
    assert set(DROP) | set(EDITS) == {t["id"] for t in items} and not set(DROP) & set(EDITS)
    out = []
    for t in items:
        if t["id"] in DROP:
            continue
        t["original"] = {"question": t["question"], "ground_truth": t["ground_truth"]}
        t["question"], t["ground_truth"] = EDITS[t["id"]]
        t["reviewed"], t["keep"] = True, True
        out.append(t)
    (here / "multihop.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"giữ {len(out)} / {len(items)} câu -> multihop.json (loại {len(DROP)})")


if __name__ == "__main__":
    main()
