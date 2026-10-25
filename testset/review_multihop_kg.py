"""Kết quả duyệt tay câu multi-hop trong 30 bài dựng KG (ý 2), đối chiếu từng câu với 2 đoạn gốc
trong multihop_kg_raw.json.

    python testset/review_multihop_kg.py      -> testset/multihop_kg.json

Tiêu chí giống testset/review_multihop.py: câu hỏi cần thông tin của CẢ HAI bài, tự đứng được
(không "theo hai đoạn văn / bài thứ hai"), đáp án chuẩn chỉ chứa thông tin có trong 2 đoạn gốc.
- DROP: câu bị loại kèm lý do
- EDITS: id -> (question mới, ground_truth mới); câu giữ lại đều được viết lại cho rõ và đúng với đoạn gốc
"""
import json
from pathlib import Path

DROP = {
    "g004": "so sánh gượng ép (thuật toán QNN-QRL cho QKD vs nhóm thuật toán quantum deep learning)",
    "g006": "ghép 2 ý không liên quan (dự án honeynet toàn cầu + xu hướng dùng ML trong honeypot); câu hỏi nhắc 'second paper'",
    "g008": "hỏi công thức cận độ dài khoá CV-QKD (ký hiệu vỡ) + so sánh vô nghĩa với AES",
    "g009": "so sánh gượng ép (chỉ số R(t) vs biểu đồ Adoption Rate / Attack Resilience)",
    "g010": "đoạn harvest-now-decrypt-later một mình đã liệt kê đủ giới hạn phần cứng NISQ; phần hash không liên quan",
    "g012": "một đoạn (bảng thư viện + CNSA 2.0) đã trả lời đủ; đáp án đọc sai bảng",
    "g013": "so sánh vô nghĩa (tỉ lệ khôi phục message của mask 6-share vs thời gian tấn công lattice)",
    "g017": "ghép 2 ý không liên quan (benchmark QML + telemetry triển khai PQC)",
    "g018": "đoạn A chỉ là danh mục tài liệu tham khảo",
    "g020": "câu hỏi quá chung chung (khuyến nghị chuyển đổi PQC), đáp án chỉ nhắc lại phần kết luận 2 bài",
    "g022": "đoạn A chỉ là danh mục tài liệu tham khảo",
    # lượt sinh thêm (--append --seed 7 / 11 / 13)
    "g024": "cả hai đoạn đều chỉ là danh mục tài liệu tham khảo",
    "g025": "ghép 2 ý không liên quan (trạng thái DFS trong giao thức MQPC + mô hình QGMM)",
    "g031": "ghép 2 ý không liên quan (khuyến nghị đào tạo của ETSI + blockchain hậu lượng tử)",
    "g032": "ghép 2 ý không liên quan (mô hình MHW cho CPA trên Kyber + Shor phá RSA)",
    "g033": "ghép 2 ý không liên quan (decoy-state trong QKD + QML phát hiện bất thường)",
    # lượt sinh trong 30 bài chủ đề quantum (--append --seed 17 / 19)
    "g038": "ghép 2 ý không liên quan (toán học của tấn công SIKE + ứng dụng của thuật toán Shor)",
    "g041": "ghép 2 ý không liên quan (DBO cho QHGAN dự đoán lỗi phần mềm + PSO)",
    "g047": "hỏi ký hiệu công thức (ma trận Ud, Ue của autoencoder) và ghép gượng ép với QGAN",
    "g049": "so sánh gượng ép (giảm nhiễu trong VQA vs mạch cộng của cài đặt Shor)",
    "g051": "hỏi công thức xác suất (12/p^α) ghép gượng ép với độ lệch 2^-f của Gidney",
}

EDITS = {
    "g001": ("Why does Shor's algorithm threaten RSA and elliptic-curve cryptography while Grover's algorithm can be "
             "countered by larger symmetric keys, and how many noisy qubits do theoretical estimates say are needed "
             "to break 2048-bit RSA?",
             "Shor's algorithm efficiently factors large integers and computes discrete logarithms, the hard problems "
             "underlying RSA and ECC, so a sufficiently powerful quantum computer could break them. Grover's algorithm "
             "only gives a quadratic speed-up for brute-force search, halving the security of symmetric ciphers "
             "(AES-128 would have 64-bit security), which can be countered by doubling key sizes. Theoretical resource "
             "estimates suggest 2048-bit RSA keys could be broken with fewer than a million noisy qubits."),
    "g002": ("How does Kyber compare with RSA in key exchange speed in practical deployment tests, and on which hard "
             "problem is Kyber's security based?",
             "Kyber (with Dilithium) has been integrated into OpenSSL prototypes and tested in TLS 1.3 handshakes, "
             "showing compatibility with current infrastructures; post-quantum algorithms like Kyber outperform RSA in "
             "key exchange speed. Kyber is a lattice-based scheme whose security depends on the Learning With Errors "
             "(LWE) problem, and it is used, for example, to protect data at rest in cloud storage."),
    "g003": ("Why do Smart City and IoVT applications need 5G unlike traditional IoT applications, and what security "
             "consequences does the predicted growth of IoT devices have?",
             "Traditional IoT applications such as Smart Homes, industries and farming need only minimal wireless "
             "capabilities, whereas Smart Cities and IoVT require significantly higher throughput, which 5G provides. "
             "Together with concepts such as CPS, M2M and 5G, this growth leads to predictions of over 30 billion IoT "
             "devices by 2030; their number and variety create numerous security risks, and IoT devices are among the "
             "most common targets of adversaries, who can compromise data security and user privacy or disrupt networks."),
    "g005": ("What problem with encoding classical data into quantum states does the data re-uploading strategy "
             "address, and what accuracy did a hybrid model with quantum-enhanced embeddings achieve for botnet "
             "detection?",
             "Amplitude encoding requires a prohibitively large number of qubits, while simpler angle encoding may "
             "give insufficient representational capacity; the data re-uploading strategy demonstrated by "
             "Pérez-Salinas et al. significantly improves data representation within quantum circuits. A hybrid model "
             "that mapped classical features into quantum states with quantum embeddings achieved up to 94.7% "
             "accuracy for botnet detection, significantly outperforming classical baselines."),
    "g007": ("Why are multivariate quadratic (MQ) cryptographic schemes considered quantum-resistant, what are they "
             "mainly used for, and what property makes a multivariate signature scheme suitable for authenticating "
             "edge nodes?",
             "MQ cryptography relies on solving systems of quadratic equations over finite fields, an NP-hard problem "
             "for which no efficient classical or quantum algorithm is known; MQ-based schemes are mainly used for "
             "digital signatures and public-key encryption. A multivariate signature scheme proposed by Akleylek et "
             "al. is immune to quantum attacks, has effective key and signature sizes, and its excellent parallelism "
             "support allows edge node authentication to be deployed effectively."),
    "g011": ("How many side-channel leakage points have been reported in Kyber's decryption procedure, and how did "
             "researchers use machine learning in a side-channel attack on a Kyber implementation?",
             "Three side-channel leakage points were reported in Kyber's decryption procedure, and side-channel "
             "attacks were completed on Kyber hardware implementations. Researchers used machine learning to exploit "
             "a power-consumption side channel against a Kyber implementation previously believed resistant to such "
             "attacks: energy fluctuations of processor or circuit operations reveal details about the data being "
             "processed."),
    "g014": ("How were honeypots traditionally classified, and how do modern honeypot works improve on traditional "
             "honeypots?",
             "Traditionally honeypots were classified by interaction level: low-interaction honeypots simulate basic "
             "network services, high-interaction honeypots provide fully functional environments (medium interaction "
             "is also used), and by goal (research vs. production). Traditional systems suffered from scalability, "
             "maintenance and detection problems, leading to hybrid and dynamic honeypots that mimic real systems "
             "better. Modern works (2020 onwards) emphasize believability through mimicking techniques that defeat "
             "fingerprinting and integration with AI-driven threat analysis and MITRE ATT&CK mapping."),
    "g015": ("How do Kyber's public key and Dilithium's signature sizes compare with classical ECC, and are these "
             "post-quantum schemes considered feasible and efficient enough for IoT and mainstream applications?",
             "Kyber's public key (~800 bytes) is larger than an ECC public key (~32 bytes) but much smaller than "
             "Classic McEliece's (>1 MB, which can exceed IoT RAM), and Dilithium signatures are a few kilobytes versus "
             "64 bytes for ECDSA. Efficient KEMs like Kyber are considered feasible at least at the first security "
             "level for a common IoT setup; Kyber operations are fast (comparable to RSA or faster) and Dilithium "
             "verifies thousands of signatures per second and suits fast signing when memory is sufficient, while "
             "FALCON is more memory-efficient."),
    "g016": ("How does quantum secure direct communication (QSDC) differ from quantum key distribution (QKD), and "
             "which intercity networks illustrate the development of practical QKD networks?",
             "QKD establishes shared secret keys between two parties (Alice and Bob) by exploiting quantum mechanics, "
             "whereas QSDC transmits the actual messages directly over a quantum channel, eliminating the need for "
             "cryptographic keys for encryption and decryption and giving confidential, near-instantaneous "
             "communication. Intercity networks such as the DARPA, Trieste and Tokyo networks illustrate the growth "
             "of practical QKD networks."),
    "g019": ("Why must the key storages of a QKD network be kept synchronized, and why does QKD still depend on "
             "classical cryptography in practice?",
             "A QKD network as a service generates, manages and supplies symmetric keys, so the contents of key "
             "storages must stay consistent or the service is not operational; key storage also decouples key "
             "generation from consumption to cope with QKD's limited key rates. QKD has no built-in authentication, so "
             "a man-in-the-middle could impersonate an endpoint; QKD links must be bootstrapped with an initial "
             "authentication method (a pre-shared key or classical public-key signature) and may need integration "
             "with classical key management systems."),
    "g021": ("How many power traces does the improved two-step side-channel attack need to recover Kyber's secret "
             "key, and what hardware efficiency cost does the hardware-friendly shuffling countermeasure for Kyber "
             "have?",
             "The two-step attack (CPA followed by a lattice attack) recovers the secret key s of Kyber-512, -768 and "
             "-1024 on an ARM Cortex-M4 board in about 9-10 minutes with at most 15 power traces. The shuffling "
             "countermeasure implemented on FPGA, verified with CPA and TVLA, shows only 8.7% degradation in "
             "hardware efficiency compared with the unprotected version, better than existing hardware hiding schemes."),
    "g023": ("What attacker-defender asymmetry does cyber deception address, and which sophisticated threats make "
             "traditional security measures inadequate and motivate the use of honeypots?",
             "Systems are still compromised despite security measures, e.g. through zero-day exploits or insiders. "
             "Cyber deception distracts adversaries from real targets by presenting simulated targets such as "
             "honeypots, addressing the asymmetry that the defender must succeed all of the time while the attacker "
             "needs to succeed only once. Sophisticated attacks such as ransomware, zero-day exploits and advanced "
             "persistent threats make traditional measures inadequate; honeypots serve as decoy systems that attract "
             "and monitor malicious actors."),
    "g026": ("How does higher-order masking, used to protect Kyber implementations against side-channel attacks, "
             "split a sensitive variable into shares, and how does threshold secret sharing split a secret among "
             "participants?",
             "In ω-order masking a sensitive variable x is split into ω+1 shares combined by arithmetic addition "
             "(arithmetic masking) or XOR (Boolean masking); random masks x1..xω are drawn and the last share is "
             "x − (x1 + ... + xω) or x ⊕ x1 ⊕ ... ⊕ xω, and operations are done on the shares so x is never used "
             "directly and does not leak. In threshold secret sharing each participant receives a value f(xi) of a "
             "random polynomial of degree t−1 whose constant term is the secret S; any t or more participants can "
             "reconstruct S by Lagrange interpolation, while fewer than t learn nothing."),
    "g027": ("How do the best reported quantum gate error rates compare with those of classical transistors, and what "
             "limitations did a quantum error correction demonstration with a Toffoli gate on superconducting qubits "
             "have?",
             "The best reported error rates (trapped ions) are about 10^-7 for single-qubit and 10^-5 for two-qubit "
             "gates, but effective rates in multi-qubit systems are closer to 10^-4 and 10^-2, whereas digital "
             "transistors reach 10^-23, so fault-tolerant computation depends on effective QEC. In the "
             "superconducting demonstration the Toffoli gate reached 85% (classical action) and 78% (quantum process) "
             "fidelity and QEC reduced the error rate quadratically, but it was limited by coherence times "
             "(T1 = 0.7-1.3 µs), gate fidelity insufficient for large-scale fault tolerance, and only two error models."),
    "g028": ("What is the status of Kyber and Dilithium support in the MbedTLS and Crypto++ libraries, and why are "
             "Kyber and Dilithium considered suitable for general-purpose deployment?",
             "MbedTLS does not yet implement Kyber or Dilithium (it supports LMS signatures) but intends to add Kyber "
             "soon and Dilithium later, which matters because it runs on millions of IoT devices. Crypto++ (version "
             "8.9) has no NIST PQC finalists in its mainline, but a third-party fork (cryptopp-pqc) adds "
             "CRYSTALS-Kyber, CRYSTALS-Dilithium, SABER and FrodoKEM. Kyber and Dilithium strike a balance between "
             "efficiency and simplicity and are considered the most mature post-quantum candidates."),
    "g029": ("What attack details do high-interaction honeypots such as Cowrie capture during credential-based "
             "attacks, and how was the NASim network attack simulator extended to evaluate honeypots?",
             "In brute-force login attacks on SSH and HTTP (with tools such as Hydra), high-interaction honeypots "
             "like Cowrie captured attack input including keystrokes and authentication attempts, and they also "
             "logged scanning methods and attacker intent. NASim was extended with honeypots implemented as sensitive "
             "hosts: if an agent exploits a honeypot with user or root privileges it loses; experiments varied the "
             "number of honeypots (0-10), movement time, number of hosts and agent behaviour (careful, standard, "
             "aggressive)."),
    "g030": ("What is the primary threat-actor behavior before Q-Day, and why does the 'Cryptographic Time Paradox' "
             "make this behavior dangerous?",
             "Before Q-Day the primary behavior, especially of nation-states, is 'Store/Harvest Now, Decrypt Later' "
             "(HNDL): collecting encrypted data now to decrypt once quantum computers break RSA/ECC. The Cryptographic "
             "Time Paradox is that RSA-encrypted data is safe only until a working CRQC is built; at that moment "
             "every piece of encrypted data ever collected becomes readable, and much data (diplomatic cables, "
             "genetic data, trade secrets) stays valuable for decades, so PQC readiness is needed by 2035 at the latest."),
    "g034": ("How does the security basis of quantum cryptography differ from traditional encryption, what are the two "
             "main categories of QKD protocols, and what practical requirement distinguishes QKD from traditional "
             "key agreement?",
             "Traditional encryption relies on the computational complexity of mathematical problems, whereas quantum "
             "cryptography relies on principles of quantum physics. QKD protocols are divided into Continuous Variable "
             "(CV) and Discrete Variable (DV) protocols, the latter producing discrete outcomes, e.g. from the "
             "polarization of single photons. QKD's keys are information-theoretically secure, but QKD requires "
             "specialized hardware; it has been tested for over two decades and only recently used commercially."),
    "g035": ("Which quantum mechanisms allow Shor's algorithm to factor large integers efficiently, and why does this "
             "threaten RSA?",
             "Shor's algorithm uses entanglement to exploit quantum parallelism, and at its core the Quantum Fourier "
             "Transform implements key subroutines exponentially faster than classical counterparts, so it factors "
             "large integers in polynomial time. RSA's security rests on the difficulty of factoring large numbers "
             "into their prime factors, so Shor's algorithm threatens RSA (and ECC)."),
    "g036": ("How are honeypots differentiated by interaction level and platform, why did client honeypots emerge, and "
             "what determines the value and effectiveness of honeypots?",
             "Low-interaction honeypots emulate vulnerabilities with limited interaction, high-interaction honeypots "
             "are vulnerable systems allowing interaction at all levels; physical honeypots run on physical machines "
             "and virtual honeypots on virtual machines. Because attackers exploit client programs such as browsers, "
             "client honeypots simulate human behavior to analyze such exploitation. Honeypots' main value is "
             "intelligence gathering (logging every command, file and lateral movement to build datasets of attacker "
             "TTPs), and their effectiveness depends on realism and proper isolation."),
    "g037": ("What does the security of SPHINCS+ (SLH-DSA) rely on compared with the other algorithms NIST selected, "
             "how does it achieve stateless operation, and which FIPS standards correspond to Kyber, Dilithium and "
             "SPHINCS+?",
             "Three of the four selected algorithms use structured lattices, while SPHINCS+ derives its security "
             "solely from the collision resistance of hash functions. It combines a hierarchy of hash trees with "
             "few-time signatures: WOTS+ signs individual nodes, FORS gives few-time signatures at the leaves, and the "
             "Hypertree links them to achieve stateless operation. FIPS 203 is ML-KEM (Kyber), FIPS 204 is ML-DSA "
             "(Dilithium) and FIPS 205 is SLH-DSA (SPHINCS+)."),
    "g039": ("On what is the security of quantum key distribution (QKD) based, why are practical QKD experiments with "
             "attenuated lasers potentially insecure, and how can quantum machine learning help detect eavesdroppers?",
             "Unlike conventional cryptography, which relies on unproven computational assumptions, QKD security is "
             "based on the fundamental laws of physics. In practice, highly attenuated lasers sometimes emit signals "
             "with more than one photon, so experiments can be insecure due to such real-life imperfections. Quantum "
             "machine learning algorithms such as quantum support vector machines (QSVM) and quantum convolutional "
             "neural networks (QCNN) can be leveraged for eavesdropping detection."),
    "g040": ("What speed-up does an FPGA accelerator with an optimized Number Theoretic Transform (NTT) achieve for "
             "Kyber, and why is polynomial multiplication in Kyber computed in the NTT domain?",
             "Implemented on an Artix-7 FPGA, the accelerator achieves a speed-up of up to 34.16x for Kyber and 9.6x "
             "for Dilithium, with a 44%-96% improvement in area-time product. Polynomial multiplication in the NTT "
             "domain reduces the complexity from O(N^2) to O(N log N), speeding up Kyber encryption and decryption; "
             "because Kyber has only a 256th primitive root of unity, the 256 coefficients are split into odd and "
             "even groups for the NTT."),
    "g042": ("How do quantum generative adversarial networks (QGANs) work, and how does EQGAN improve their training?",
             "A QGAN pairs a parameterized quantum generator with a quantum or classical discriminator in a two-player "
             "minimax game, using superposition and entanglement to encode data distributions; it is usually built "
             "as a variational quantum algorithm and trained in a hybrid loop where quantum measurements estimate the "
             "loss. EQGAN uses quantum circuits as generators and classical neural networks as discriminators, treats "
             "training targets as mutation operations and uses a fitness evaluation to select optimal generator "
             "configurations across generations, giving faster, more stable training at lower cost."),
    "g043": ("What reasons suggest that quantum computers can outperform classical computers, and what role does the "
             "precision of quantum state manipulation play?",
             "Describing the correlations of a few hundred qubits may need more bits than there are atoms in the "
             "visible universe, but this complexity alone does not make quantum computers more powerful; a key reason "
             "is that quantum algorithms solve problems believed hard for classical computers, the best known being "
             "factoring large integers. If precision grows polynomially with the input size, quantum computers appear "
             "to be a more powerful type of computer; the achievable precision is dictated by materials and "
             "architecture rather than fundamental physical laws."),
    "g044": ("Why are variational quantum algorithms (VQAs) considered the leading strategy for NISQ devices, which "
             "constraints must they account for, and why does improving gate accuracy matter for near-term platforms?",
             "Fault-tolerant quantum computers are still years or decades away, so a strategy must account for limited "
             "numbers of qubits, limited qubit connectivity and coherent and incoherent errors that limit circuit "
             "depth; VQAs handle these with an optimization- or learning-based approach. More accurate gates allow "
             "larger circuits to be executed, extending the power of NISQ technology, and lower the overhead of "
             "quantum error correction later."),
    "g045": ("What error rate does an eavesdropper who measures and resends all photons in the rectilinear basis "
             "induce in BB84, and which two sources contribute to detection events in decoy-state QKD?",
             "Such an eavesdropper learns the correct polarization of half the photons and induces disagreements in "
             "1/4 of the photons later re-measured in the original basis (no measurement can yield more than 1/2 "
             "expected bit about a key bit). In decoy-state QKD, the yield combines the detection of signal photons "
             "and background events such as dark counts (pdark, typically about 1e-5)."),
    "g046": ("How has research on quantum machine learning for quantum key distribution evolved, and what risk do "
             "QML-based applications themselves face in 6G networks?",
             "Research on eavesdropping detection and security analysis of QKD protocols with QML has increased "
             "significantly since 2010, aiming at more robust and intelligent cryptographic applications. Despite their "
             "learning capabilities, QML-based applications face several emerging security threats, including "
             "adversarial threats, which motivates studying QML for 6G network intelligence and its attacks."),
    "g048": ("How are variational quantum classifiers (VQCs) used for cyberattack detection, and why are variational "
             "quantum circuits the most common architecture for QGANs?",
             "VQCs combine parameterised quantum circuits with classical optimisers in hybrid feedback loops, so they "
             "can adapt to evolving attack strategies; they have been used, for example, for cyberattack detection on "
             "the NSL-KDD dataset and for DGA botnet classification. A variational quantum circuit consists of initial "
             "state preparation followed by layers of parameterized gates optimized with classical techniques, and "
             "its flexibility and trainability make it the most common QGAN architecture."),
    "g050": ("How are the generator and discriminator of quantum GANs trained, and how are the gradients obtained?",
             "Training follows an adversarial minimax framework: the discriminator parameters are updated to minimize "
             "its loss L_D (distinguishing real from generated data) and the generator parameters to minimize L_G "
             "(making fake data classified as real), until a Nash equilibrium. Gradients are typically estimated with "
             "the parameter-shift rule or stochastic approximation and optimization is done with classical algorithms; "
             "QuGANs may outperform classical GANs in representation."),
}


def main():
    here = Path(__file__).resolve().parent
    items = json.loads((here / "multihop_kg_raw.json").read_text(encoding="utf-8"))
    assert set(DROP) | set(EDITS) == {t["id"] for t in items} and not set(DROP) & set(EDITS)
    out = []
    for t in items:
        if t["id"] in DROP:
            continue
        t["original"] = {"question": t["question"], "ground_truth": t["ground_truth"]}
        t["question"], t["ground_truth"] = EDITS[t["id"]]
        t["reviewed"], t["keep"] = True, True
        out.append(t)
    (here / "multihop_kg.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"giữ {len(out)} / {len(items)} câu -> multihop_kg.json (loại {len(DROP)})")


if __name__ == "__main__":
    main()
