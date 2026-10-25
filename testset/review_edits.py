"""Kết quả duyệt tay testset (đối chiếu từng câu với chunk_text gốc).

    python testset/review_edits.py

- DROP: câu bị loại (keep=false) kèm lý do
- EDITS: id -> (question mới | None, ground_truth mới | None)
- Mọi câu còn lại: giữ nguyên, đặt reviewed=true
"""
import json
from pathlib import Path

DROP = {
    "q002": "hỏi công thức; công thức trích từ PDF bị vỡ ký tự",
    "q004": "chỉ hỏi tên tác giả (Tosun et al.)",
    "q018": "hỏi công thức (membrane potential), vỡ ký tự",
    "q033": "hỏi công thức SEW block",
    "q058": "công thức trích OCR lỗi, không đọc được",
    "q069": "đáp án tầm thường 'G(z)', không có nội dung",
    "q080": "hỏi độ phức tạp trong bảng, ký hiệu vỡ",
}

EDITS = {
    "q001": ("What is Shor's algorithm and how does it impact the security of current cryptographic systems?",
             "Shor's algorithm, proposed by Peter Shor in 1994, efficiently factors large integers and computes discrete logarithms in polynomial time, which directly undermines the security assumptions of current public-key cryptographic systems."),
    "q003": ("Which game-theoretic concept did Lu et al. base their argument on when discussing the scenarios under which KPSD is incentive-compatible?",
             "Nash equilibrium: Lu et al. use a game-theoretic argument based on Nash equilibrium to discuss when KPSD is incentive-compatible."),
    "q006": (None, "A photonic QGAN, implemented on a photonic processor (Quandela's Ascella) rather than qubits, which uses linear optical circuits and Fock-space encoding and is trained end-to-end to produce digit-like images."),
    "q007": (None, "QSVM uses a quantum circuit, the quantum feature map, to embed data into a quantum feature space (states |phi(x)> in a Hilbert space), exploiting the parallelism of quantum computers and avoiding the computationally intensive construction of a high-dimensional classical feature space."),
    "q008": ("According to the meta-survey of cyber deception, what is the 'need for intelligent adaptation' gap in honeypot research, and what practical deployment challenges are identified?",
             "Modern papers identify intelligent adaptation as a major gap: they advocate AI integration to build adaptive honeynets that learn from attacker interactions and improve deception strategies without manual reconfiguration. Practical challenges include high maintenance costs to keep deception environments current with legitimate system updates and patches, and the significant resource demands of high-interaction honeypots."),
    "q009": (None, "VECTOR integrates reconstructability-aware approximation into existing eviction methods, extending binary retain/evict decisions into retention, approximation and eviction under the same memory budget."),
    "q011": ("Which conventional public key cryptography methods are described as susceptible to quantum risks in the survey of pre- versus post-quantum security for 5G-enabled IoT?",
             "RSA, Diffie-Hellman, Secret Sharing, and Elliptic Curve Cryptography (ECC)."),
    "q013": ("In the experimental evaluation of honeypots and deception strategies, which system was used to simulate attacks and which tools performed the reconnaissance network scans?",
             "Kali Linux was used to simulate attacks; network scans targeting open ports and service banners were performed with tools such as Nmap and Nessus."),
    "q014": (None, "Post-training quantization from the AMD Quark library with INT8 power-of-two quantization, using mean squared error optimization and cross-layer equalization on a small calibration dataset."),
    "q015": ("In side-channel attacks on Kyber with an imperfect plaintext-checking oracle, how much of the Kyber-512 secret key was recovered by applying the CPA attack twice, and with how many power traces?",
             "About 60% of the Kyber-512 secret key was recovered using about 1619 power traces, leaving 40% erroneous coefficients."),
    "q016": ("How does the O(1) LFU cache algorithm organize its data structures when the first element is inserted, and what is the insertion complexity?",
             "The cache starts as an empty hash map and an empty frequency list. On the first insertion, a hash map entry pointing to the new element is created and a frequency node with value 1 is added to the frequency list; the element's node points back to its frequency node. Insertion runs in O(1)."),
    "q017": ("How are technology service providers using quantum resilience as a selling point?",
             "Certain VPN and secure communications products advertise that they are 'quantum-safe' by integrating PQC, and cloud companies like AWS, IBM and Microsoft have released tools for experimenting with PQC on their platforms."),
    "q020": (None, "AttacKG+ is an LLM-based approach that uses LLM-based rewriting preprocessing to remove noise, and primarily targets extraction of tactical-level threat information (rather than complete attack processes)."),
    "q022": ("What accuracy does the scalable quantum convolutional neural network (QCNN) achieve on MNIST, and with how many trainable parameters?",
             "The scalable QCNN achieves 98.7% accuracy (98.1% on validation) using only 45 independent rotational parameters, while the baseline QCNN is trapped at 52%."),
    "q023": ("In prior work on reasoning over CVEs with attack graphs, how did researchers visualize how CAPECs could be chained in an information system, and what inputs did they use?",
             "They developed a tool to visualize how CAPECs could be chained in an information system containing CVEs; they manually described the characteristics and settings of the information system and input data from the CVSS vectors related to the CVEs."),
    "q024": ("What approach was proposed to characterize encrypted and VPN traffic using time-related features, and what dataset was published?",
             "A flow-based classification method that uses only time-related features (with low computational complexity) to characterize encrypted and VPN traffic; the authors also published a labeled encrypted traffic dataset with 14 labels (7 regular encrypted, 7 VPN)."),
    "q025": ("Why can a SIMD engine for data-parallel floating-point applications minimize the energy cost of programmability?",
             "Because an FP operation takes around 1/10 the energy of a simple instruction, a SIMD engine performing the same operation on about 10 data lanes makes instruction energy dominated by the FP operations, minimizing the cost of programmability, as GPUs do."),
    "q026": ("In the ProbPol framework for probabilistic policy languages, what are crisp signals?",
             "Crisp signals, such as keyword match, group membership and token count, always return 0 or 1, unlike probabilistic signals (e.g., geometric embedding-based signals)."),
    "q027": ("In the study of tarpit feasibility against Internet worms, how many Mirai-infected IP addresses were trapped in the tarpit, and how many unique infections were observed per day on average?",
             "202,003 IP addresses (48%) were trapped in the tarpit; accounting for IP churn, there were on average 13,298 unique infections on a given day."),
    "q028": ("In moving target defense for cloud and SDN environments, how does the technique of rotating system components work?",
             "It continuously changes key system elements such as the operating system and web applications, at regular intervals or in response to specific events, making them harder for attackers to identify and exploit; this increases attack complexity and was shown to be implementable in SDN environments."),
    "q029": ("What is the purpose of a mobile honeypot system for industrial control systems?",
             "It can be placed in many network locations to provide an additional layer of defense that disrupts the attacker's reconnaissance activities, and it gives the defender early notification of incoming attacks."),
    "q030": ("What does decoupled weight decay (as in AdamW) improve for Adam?",
             "Decoupling weight decay from the gradient-based update decouples the optimal weight decay factor from the learning rate setting, and substantially improves Adam's generalization so that it can compete with SGD with momentum on image classification datasets, where Adam was previously outperformed."),
    "q031": ("How is the Multi-Layer Perceptron (MLP) described in the overview of deep learning for encrypted traffic classification, and what is its main drawback?",
             "An MLP has an input layer, an output layer and several hidden layers of densely connected neurons, each applying a non-linear activation to a weighted sum; a deep enough MLP can approximate any function, but its huge number of parameters makes it complex, inefficient and hard to train."),
    "q032": (None, "They used unsupervised STDP-based learning rules in a two-layer SNN and achieved a best performance of 95% on MNIST digit recognition."),
    "q034": ("In the streaming question-answering evaluation, how does StreamingLLM compare with dense attention and window attention?",
             "Dense attention runs out of memory; window attention is efficient but has low accuracy (random outputs) once the input exceeds the cache size; StreamingLLM handles the streaming format efficiently and matches the one-shot, sample-by-sample baseline accuracy."),
    "q035": ("For the encrypted traffic classification approach that converts each packet into a normalized byte sequence fed to a 1-D CNN and stacked auto-encoders, how well did it perform on VPN and on Tor traffic?",
             "It categorized unencrypted and VPN traffic well (around 93-94% average recall) without time-related features, but on the unencrypted vs. Tor dataset average recall dropped to 57% and average precision to 44%."),
    "q036": ("What kind of spiking neural network was used for encrypted internet traffic classification, how was it trained, and what accuracy did it reach on the ISCX datasets?",
             "A simple feedforward SNN with one fully connected hidden layer, using only packet size and arrival time, trained in a supervised manner with Surrogate Gradient Learning; it reached 95.9% accuracy on the ISCX datasets, outperforming previous approaches."),
    "q037": ("What type of neural network is used for end-to-end encrypted traffic classification that learns features directly from raw traffic?",
             "A one-dimensional convolutional neural network (1D-CNN), which learns features from raw traffic layer by layer and feeds high-level features to a softmax layer that directly outputs the predicted labels."),
    "q038": ("What does the stacked deep ensembles approach for encrypted HTTPS traffic classification contribute?",
             "A deployment-oriented, leakage-controlled recipe for flow-level encrypted HTTPS classification: a standardized pipeline, probability-level stacked ensembling, a like-for-like comparison of deep architectures under identical data, splits, preprocessing, imbalance handling and training budget, state-of-the-art results on the HTTPS/CESNET family, and a released codebase for reproducibility."),
    "q039": ("What does the network attack simulation methodology for evaluating deception and moving target defense allow defenders to determine?",
             "It quantitatively measures the effectiveness of honeypots and moving target defense in a simulation environment, enabling recommendations on how many honeypots to deploy and how quickly network addresses must be mutated to effectively disrupt attacks across network and attacker configurations."),
    "q040": ("How does ZoMBI improve Bayesian optimization on needle-in-a-haystack problems compared with standard Bayesian optimization?",
             "ZoMBI zooms the search bounds inward and prunes redundant memory points, so the needle-like optimum region is resolved accurately aligned with the true target, whereas standard Bayesian optimization does not progress significantly after 10 additional experiments."),
    "q042": ("What constraints does the H2O (Heavy-Hitter Oracle) eviction policy impose on the KV cache at each step?",
             "The KV cache size stays fixed at k, and at most one KV entry can be evicted per step (|S_i \\ S_{i-1}| <= 1, equivalently |S_i ∩ S_{i-1}| >= k-1)."),
    "q043": ("Why are FPGAs considered an ideal platform for SNN hardware accelerators, and what is the key challenge?",
             "FPGAs offer flexible hardware reconfigurability and parallel computing, with low latency, high efficiency and good power control; the key challenge is their limited resources, which makes balancing inference speed, power consumption and accuracy difficult."),
    "q044": ("In the network-centric Harvest-Now, Decrypt-Later (HNDL) model, what information besides application data is included in harvested ciphertext?",
             "Network-layer routing information, subscriber authentication vectors, and inter-domain trust relationships that persist in network registries and certificate transparency logs."),
    "q046": ("Which four evaluation metrics are used in the ICLSTM study of encrypted traffic service identification?",
             None),
    "q047": ("Why can attack data generated with tools like SQLMap, WPScan and Metasploit lead to mislabeled requests when building ML-based WAF datasets?",
             "Most of these tools first run a scanning process to gather information about the target website before launching attacks, so many normal requests get mixed into the attack requests, causing mislabeling."),
    "q048": ("In the analysis of the recurrent memory transformer on long-context needle tasks, what happens to the memory state when a fact is introduced, and what does this indicate?",
             "The memory state changes visibly when a fact is introduced, indicating that the model learned to distinguish important facts from background text and preserve them in memory until a question appears."),
    "q049": ("In CRYSTALS-Kyber, how is the shared key K derived during encapsulation?",
             None),
    "q050": ("How do Loihi's synaptic density and neuron density compare with TrueNorth?",
             "Loihi provides 2.1 million unique synaptic variables per mm2, over three times higher than TrueNorth, but its maximum neuron density of 2,184 per mm2 is marginally worse than TrueNorth's (a 2x reduction when process-normalized)."),
    "q051": ("Why does Loihi support a hierarchical network model, and how is a hierarchical template network formally represented?",
             "It significantly reduces the chip-wide connectivity and synaptic resources needed to map convolutional-style networks; the hierarchical template network is represented as a directed multigraph whose nodes are disjoint neuron population types and whose edges are sets of synapses connecting pairs of population types."),
    "q052": ("What does the LongBench benchmark contain, and what is its average length in English and Chinese?",
             "LongBench is a bilingual, multi-task benchmark with 21 datasets across 6 task categories in English and Chinese, with an average length of 6,711 words (English) and 13,386 characters (Chinese)."),
    "q053": (None, "Before exploration starts, NetSMT arranges branching variables (e.g., variables for matching conditions in route policies) to be explored first, and among them those corresponding to configurations in routers closer to the destination are explored first."),
    "q055": ("Which optimization method is used to approximate the functional contribution of the brain's optimization processes when training recurrent spiking neural networks such as LSNNs?",
             "Backpropagation through time (BPTT), combined with a rewiring algorithm that optimizes the network architecture."),
    "q056": ("How does a simple cell in the primary visual cortex respond to a light dot falling in its receptive field?",
             None),
    "q057": ("How is the state of a qubit defined in terms of probability amplitudes?",
             "A qubit is in a superposition state alpha|0> + beta|1>, where alpha and beta are complex probability amplitudes satisfying |alpha|^2 + |beta|^2 = 1."),
    "q059": ("In ML-empowered eBPF/XDP packet processing for intrusion detection, how much do integer-arithmetic-only NN and decision tree classifiers in kernel space reduce inference time compared with floating-point versions in user space?",
             "The integer-only NN reduces inference time to 15.3% and the DT to 1.6% of the floating-point user-space versions, with little degradation in classification performance."),
    "q060": ("What is the configuration of the MobileBERT model used as the study target for BFloat16 deployment of tiny transformers?",
             "24 encoder blocks, each with a 4-head multi-head self-attention kernel and a 2-stage feed-forward bottleneck layer, with a hidden projection size of 128."),
    "q062": (None, "Ring signatures, which let a signer sign on behalf of a ring of members so the verifier can validate the signature without knowing which member signed (as used in Monero), obscuring the sender's information."),
    "q063": ("What are the advantages and the main challenge of NV centers in diamond as a qubit technology?",
             "NV centers encode qubits in the spin of a lattice defect and offer long coherence times and room-temperature operation, but scaling them to large, controllable registers is difficult because of the difficulty of precise and reliable manufacturing."),
    "q064": ("Which physical effect was exploited in a 2019 jamming attack against QKD, and what was its impact?",
             "The Faraday effect; it disrupts QKD by increasing the QBER and halting key exchange."),
    "q065": (None, "The Dynamic-On-Demand Key Allocation scheme for QKD networks (DDKA-QKDN), which dynamically allocates key resources per application request in Quantum IoT scenarios; it is realized using SDN technology."),
    "q066": ("How is the trojan gate attack on quantum circuit models for 6G resource allocation constructed, and why is it stealthy?",
             "Adversarial perturbations are inserted as subcircuits into regions of the quantum circuit model with limited error detection; the natural noise and decoherence of quantum systems mask these perturbations, making errors hard to attribute to malicious interference."),
    "q067": ("In the hybrid QNN and QSVM pipeline for real-time network traffic monitoring, what accuracies and alerting latency were achieved?",
             "The QNN achieved 96% accuracy in anomaly detection, the QSVM 93%, with near-instant average alerting latency of about 10 ms."),
    "q068": ("Why is quantum key distribution (QKD) considered provably secure against eavesdropping?",
             "QKD relies on principles of physics rather than the computational complexity of mathematical problems, so anyone attempting to access the key exchange can be detected."),
    "q070": ("What is the Single NIAH task in the RULER benchmark, and how does Multi-keys NIAH differ from it?",
             "In Single NIAH, one key-value needle (e.g., a word key and a number value) is inserted into an essay haystack whose size scales with context length, and the model must retrieve the value. Multi-keys NIAH inserts multiple needles with different keys as distractors, and the model must retrieve the value of only the queried key."),
    "q071": (None, "SCNNTraffic uses a Spiking Convolutional Encoder to downsample and encode text-form inputs and a Spiking Cross-Gating Module for feature fusion; using spiking neurons reduces parameters and energy consumption, enabling deployment on edge devices."),
    "q072": ("Which layers make up the spiking convolutional encoder in SCNNTraffic?",
             "The encoder applies Conv1d, then BatchNorm, then a LIF spiking neuron, then MaxPool to the embedded packet payload vector: encoder(p) = MaxPool(LIF(BatchNorm(Conv1d(p))))."),
    "q073": ("What is the capacity limit of WorkingMemory in the Sleep-Consolidated Memory (SCM) system, and what principle does it follow?",
             "WorkingMemory is fixed at seven items, following Miller's Law on the limits of human working memory; when full, new episodes displace the oldest."),
    "q074": ("How are synaptic weights initialized in the Spike Agreement Dependent Plasticity (SADP) paradigm, and why?",
             "They are initialized from a Rademacher distribution (uniform over {-1, +1}), which promotes early activity symmetry and propagation."),
    "q075": ("On CIFAR100, how does Spikformer-4-384 compare with the ResNet-19 ANN in accuracy and parameter count?",
             "Spikformer-4-384 reaches 77.86% with 9.32M parameters versus 75.35% with 12.63M for ResNet-19 ANN, an improvement of 2.51%."),
    "q076": ("In differentiable plasticity, how is a synapse's weight composed, and which parts are meta-learned?",
             "The weight is w = w_BP + zeta * w_H, the sum of a backprop part and a Hebbian part scaled by a weighting factor zeta; the backprop part and the weighting constant are meta-learned."),
    "q077": ("In supervised spike agreement dependent plasticity, what role does the CNN-based feature extractor play?",
             "It acts as a frozen sensory front-end: convolutional preprocessing transforms raw sensory inputs into structured, semantically meaningful representations that are converted into spike trains suitable for agreement-driven plasticity."),
    "q078": ("How do Feedback Alignment and Direct Feedback Alignment relax the requirements of backpropagation?",
             "Feedback Alignment replaces the transposed forward weight matrix used in backpropagation with a fixed random matrix, while Direct Feedback Alignment directly propagates the errors from the top layer to the hidden layers through fixed random connections."),
    "q079": ("In Hebbian winner-take-all (WTA) learning, how is the weight update defined, and how does k-WTA differ?",
             "The update is delta w_ij = eta * r_j * (x_i - w_ij), where r_j is 1 for the winning neuron and 0 otherwise; in k-WTA the top-k neurons closest to the input are all winners (r_j = 1)."),
    "q081": ("What two complementary directions are pursued to explain how the brain performs computations, as discussed in the expressive leaky memory neuron work?",
             "One direction studies how computation arises from the collective activity of simple neurons (e.g., leaky integrators or ReLU units) connected in networks; the other proposes that the intrinsic computational power of individual neurons contributes significantly, supported by evidence that cortical neurons are remarkably sophisticated, comparable to multilayer ANNs."),
    "q082": ("Under the Harvest-Now, Decrypt-Later (HNDL) model, what is the status of encrypted data transmitted or stored today without quantum resistance?",
             "It is effectively 'pre-compromised', even if quantum computers capable of breaking RSA-2048 or ECC-256 are a decade away; once decrypted it could undermine national security, erode public trust and violate regulations such as GDPR and FISMA."),
    "q083": ("What are the three primary mechanisms employed by network tarpits?", None),
    "q085": ("How does VulnBot represent the dependencies among penetration testing tasks?",
             "As a Penetration Task Graph: a directed graph where an edge from T1 to T2 means T2 depends on T1, and these dependencies determine task execution order; each task has an id, dependencies, an instruction and an action (e.g., Shell)."),
    "q086": ("How does WAF-A-MoLE manage its payload pool during the mutation-based evasion algorithm?",
             "The payload pool is a priority queue Q, prioritized by the confidence value returned by the WAF's classifier; initially it contains only the initial payload p0, and payloads are mutated until one falls below the threshold t under which it is considered harmless."),
    "q087": ("In the WAFFLED study of 'forgot-password' requests on real websites, which content types did the websites use?",
             "More than two-thirds used application/x-www-form-urlencoded, about one-fourth used application/json or its variations, only two used multipart/form-data, and one used URL parameters."),
    "q088": ("What are the two categories of machine learning approaches for detecting web attacks, and how does supervised learning work?",
             "Unsupervised and supervised learning; supervised learning, the most common for classification, learns a mapping function Y = f(X) from labeled input data to output labels and then predicts labels for new inputs."),
    "q090": ("In the eIDPS evaluation, which cyberattacks were simulated and which tools were used to generate the malicious traffic?",
             "Six attacks: SYN Scan, OS Fingerprint, Brute Force SSH, Botnet ARES, and two DoS attacks (Slowloris and GoldenEye); malicious traffic was generated with tools such as hping3 and Metasploit."),
    "q091": ("In the ablation study of the hybrid HRSNN (RNN + SNN) intrusion detection model, what happens when SMOTE or RFE is removed?",
             "Without SMOTE the model becomes very sensitive to class imbalance, leading to more false negatives; without RFE redundant features are kept, increasing training time and slightly lowering accuracy."),
    "q092": ("When should organizations begin post-quantum cryptography migration, and why is waiting for Q-Day an error?",
             "Organizations should begin PQC migration now: with Q-Day estimated at 2030 ± 5 years and migration taking 2-5 years (plus 1-3 years of supply-chain coordination), migration must start in 2025-2028 to finish before 2030-2035; legacy constraints and long data confidentiality requirements also make waiting a strategic error."),
    "q093": ("In xOffense, how does the Task Orchestrator plan a penetration test and retrieve relevant penetration knowledge?",
             "It decomposes the pentest objective into a Task Coordination Graph (TCG) of tasks with dependencies, and queries the Knowledge Repository, a vector database, through a RAG mechanism (taken from Langchain-Chatchat) using the initial task description, the current task instruction or recent task results."),
}


def main():
    path = Path(__file__).with_name("testset.json")
    items = json.loads(path.read_text(encoding="utf-8"))
    ids = {t["id"] for t in items}
    assert set(DROP) <= ids and set(EDITS) <= ids, "id không tồn tại"
    for t in items:
        t["reviewed"] = True
        if t["id"] in DROP:
            t["keep"], t["review_note"] = False, DROP[t["id"]]
            continue
        t["keep"] = True
        if t["id"] in EDITS:
            q, gt = EDITS[t["id"]]
            t.setdefault("original", {"question": t["question"], "ground_truth": t["ground_truth"]})
            if q:
                t["question"] = q
            if gt:
                t["ground_truth"] = gt
            t["review_note"] = "edited"
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    kept = sum(t["keep"] for t in items)
    print(f"Duyệt xong: {kept} giữ ({len(EDITS)} đã sửa), {len(DROP)} loại / {len(items)} câu")


if __name__ == "__main__":
    main()
