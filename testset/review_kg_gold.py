"""Kết quả duyệt tay bộ chuẩn đánh giá KG của ý 3 (đối chiếu từng câu với câu hỏi + đáp án chuẩn trong
kg_gold_raw.json do judge trích).

    python testset/review_kg_gold.py      -> testset/kg_gold.json

Tiêu chí:
- Thực thể: chỉ giữ thực thể kỹ thuật cụ thể cần có trong KG để trả lời câu hỏi (thuật toán, giao thức, tấn công,
  phòng thủ, hệ thống / thư viện, khái niệm, chuẩn, tổ chức, phần cứng). Bỏ từ chung chung ("data", "participants",
  "physics", "Europe"), con số / giá trị ("60%", "1619"), tên người ("Alice", "Akleylek et al.") và mục chỉ là
  diễn giải ("large integers" -> "integer factorization"). Dùng tên thông dụng; viết "Tên đầy đủ (VIẾT TẮT)" khi có.
- Quan hệ: chỉ giữ cặp thực thể mà đáp án chuẩn (hoặc câu hỏi) nói rõ là có quan hệ, kèm câu mô tả quan hệ;
  hai đầu mút phải nằm trong danh sách thực thể của câu. Bỏ quan hệ tới con số / từ chung chung, quan hệ suy diễn.
GOLD: id -> (thực thể, [(thực thể A, thực thể B, mô tả quan hệ)]) — viết lại toàn bộ cho từng câu.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

GOLD = {
    # ---------------- single-hop ----------------
    "q001": (["Shor's algorithm", "integer factorization", "discrete logarithm", "public-key cryptography"], [
        ("Shor's algorithm", "integer factorization", "Shor's algorithm efficiently factors large integers"),
        ("Shor's algorithm", "discrete logarithm", "Shor's algorithm computes discrete logarithms in polynomial time"),
        ("Shor's algorithm", "public-key cryptography", "Shor's algorithm undermines the security of current public-key cryptosystems")]),
    "q005": (["Libsodium", "NIST PQC finalists"], [
        ("Libsodium", "NIST PQC finalists", "Libsodium 1.0.x does not support any NIST PQC finalist algorithm")]),
    "q006": (["photonic QGAN", "linear optical circuit", "Fock-space encoding", "Ascella"], [
        ("photonic QGAN", "linear optical circuit", "the photonic QGAN uses linear optical circuits"),
        ("photonic QGAN", "Fock-space encoding", "the photonic QGAN uses Fock-space encoding"),
        ("photonic QGAN", "Ascella", "the photonic QGAN is implemented on Quandela's Ascella photonic processor")]),
    "q007": (["quantum support vector machine (QSVM)", "quantum feature map", "quantum feature space"], [
        ("quantum support vector machine (QSVM)", "quantum feature map", "QSVM uses a quantum circuit, the quantum feature map"),
        ("quantum feature map", "quantum feature space", "the quantum feature map embeds data into a quantum feature space (Hilbert space)")]),
    "q011": (["RSA", "Diffie-Hellman", "secret sharing", "Elliptic Curve Cryptography (ECC)"], []),
    "q015": (["Kyber-512", "side-channel attack", "plaintext-checking oracle", "correlation power analysis (CPA)",
              "power trace"], [
        ("correlation power analysis (CPA)", "Kyber-512", "applying the CPA attack twice recovered about 60% of the Kyber-512 secret key"),
        ("correlation power analysis (CPA)", "power trace", "the CPA attack used about 1619 power traces"),
        ("side-channel attack", "plaintext-checking oracle", "the side-channel attack on Kyber uses an imperfect plaintext-checking oracle")]),
    "q017": (["post-quantum cryptography (PQC)", "VPN", "Amazon Web Services (AWS)", "IBM", "Microsoft"], [
        ("VPN", "post-quantum cryptography (PQC)", "VPN products advertise being quantum-safe by integrating PQC"),
        ("Amazon Web Services (AWS)", "post-quantum cryptography (PQC)", "AWS released tools for experimenting with PQC"),
        ("IBM", "post-quantum cryptography (PQC)", "IBM released tools for experimenting with PQC"),
        ("Microsoft", "post-quantum cryptography (PQC)", "Microsoft released tools for experimenting with PQC")]),
    "q022": (["quantum convolutional neural network (QCNN)", "MNIST"], [
        ("quantum convolutional neural network (QCNN)", "MNIST", "the scalable QCNN achieves 98.7% accuracy on MNIST with 45 parameters")]),
    "q044": (["Harvest-Now, Decrypt-Later (HNDL)", "network-layer routing information",
              "subscriber authentication vector", "inter-domain trust relationship", "certificate transparency log"], [
        ("Harvest-Now, Decrypt-Later (HNDL)", "network-layer routing information", "HNDL harvested ciphertext includes network-layer routing information"),
        ("Harvest-Now, Decrypt-Later (HNDL)", "subscriber authentication vector", "HNDL harvested ciphertext includes subscriber authentication vectors"),
        ("Harvest-Now, Decrypt-Later (HNDL)", "inter-domain trust relationship", "HNDL harvested ciphertext includes inter-domain trust relationships"),
        ("inter-domain trust relationship", "certificate transparency log", "inter-domain trust relationships persist in certificate transparency logs")]),
    "q049": (["CRYSTALS-Kyber", "CPAPKE.Enc", "key derivation function (KDF)", "shared key K"], [
        ("CRYSTALS-Kyber", "CPAPKE.Enc", "Kyber encapsulation uses the CPAPKE.Enc function"),
        ("key derivation function (KDF)", "shared key K", "the shared key K is derived with a key derivation function"),
        ("CPAPKE.Enc", "shared key K", "the hash of the CPAPKE.Enc output is an input for deriving K")]),
    "q057": (["qubit", "superposition", "probability amplitude"], [
        ("qubit", "superposition", "a qubit is in a superposition state alpha|0> + beta|1>"),
        ("superposition", "probability amplitude", "alpha and beta are complex probability amplitudes with |alpha|^2 + |beta|^2 = 1")]),
    "q061": (["SIKE", "SIDH", "auxiliary point information"], [
        ("SIKE", "auxiliary point information", "the July 2022 attack on SIKE exploited auxiliary point information in the specification"),
        ("SIDH", "auxiliary point information", "the July 2022 attack on SIDH exploited auxiliary point information, enabling classical key recovery")]),
    "q062": (["ring signature", "blockchain", "Monero"], [
        ("ring signature", "blockchain", "ring signatures obscure the sender's information in blockchain-IoT transactions"),
        ("ring signature", "Monero", "ring signatures are used in Monero")]),
    "q063": (["nitrogen-vacancy (NV) center", "qubit", "coherence time"], [
        ("nitrogen-vacancy (NV) center", "qubit", "NV centers encode qubits in the spin of a lattice defect"),
        ("nitrogen-vacancy (NV) center", "coherence time", "NV centers offer long coherence times")]),
    "q064": (["Faraday effect", "jamming attack", "quantum key distribution (QKD)", "quantum bit error rate (QBER)"], [
        ("Faraday effect", "jamming attack", "the 2019 jamming attack exploited the Faraday effect"),
        ("jamming attack", "quantum key distribution (QKD)", "the jamming attack disrupts QKD and halts key exchange"),
        ("jamming attack", "quantum bit error rate (QBER)", "the jamming attack increases the QBER")]),
    "q065": (["DDKA-QKDN", "QKD network", "software-defined networking (SDN)", "Quantum IoT"], [
        ("DDKA-QKDN", "QKD network", "DDKA-QKDN dynamically allocates key resources in QKD networks"),
        ("DDKA-QKDN", "software-defined networking (SDN)", "DDKA-QKDN is realized using SDN technology"),
        ("DDKA-QKDN", "Quantum IoT", "DDKA-QKDN allocates keys per application request in Quantum IoT scenarios")]),
    "q066": (["trojan gate attack", "quantum circuit model", "6G resource allocation", "adversarial perturbation",
              "decoherence"], [
        ("trojan gate attack", "quantum circuit model", "the trojan gate attack inserts subcircuits into the quantum circuit model"),
        ("quantum circuit model", "6G resource allocation", "the quantum circuit model is used for 6G resource allocation"),
        ("trojan gate attack", "adversarial perturbation", "the trojan gate attack inserts adversarial perturbations as subcircuits"),
        ("decoherence", "adversarial perturbation", "natural noise and decoherence mask the adversarial perturbations")]),
    "q067": (["quantum neural network (QNN)", "quantum support vector machine (QSVM)", "anomaly detection"], [
        ("quantum neural network (QNN)", "anomaly detection", "the QNN achieved 96% accuracy in anomaly detection"),
        ("quantum support vector machine (QSVM)", "anomaly detection", "the QSVM achieved 93% accuracy in anomaly detection")]),
    "q068": (["quantum key distribution (QKD)", "eavesdropping", "computational complexity"], [
        ("quantum key distribution (QKD)", "eavesdropping", "QKD detects anyone attempting to access the key exchange"),
        ("quantum key distribution (QKD)", "computational complexity", "QKD relies on physics, not on the computational complexity of mathematical problems")]),
    "q082": (["Harvest-Now, Decrypt-Later (HNDL)", "quantum computer", "RSA-2048", "ECC-256", "GDPR", "FISMA"], [
        ("quantum computer", "RSA-2048", "quantum computers capable of breaking RSA-2048 may be a decade away"),
        ("quantum computer", "ECC-256", "quantum computers capable of breaking ECC-256 may be a decade away"),
        ("Harvest-Now, Decrypt-Later (HNDL)", "GDPR", "decrypted harvested data could violate regulations such as GDPR"),
        ("Harvest-Now, Decrypt-Later (HNDL)", "FISMA", "decrypted harvested data could violate regulations such as FISMA")]),
    "q092": (["post-quantum cryptography (PQC) migration", "Q-Day", "legacy system"], [
        ("post-quantum cryptography (PQC) migration", "Q-Day", "PQC migration takes 2-5 years and must finish before Q-Day (2030 +- 5 years)"),
        ("legacy system", "post-quantum cryptography (PQC) migration", "legacy constraints make waiting to migrate a strategic error")]),
    "s001": (["variational quantum circuit", "quantum simulation", "hardware efficient ansatz",
              "parametrized random circuit"], [
        ("hardware efficient ansatz", "variational quantum circuit", "the hardware efficient ansatz is a parameterization of variational quantum circuits"),
        ("hardware efficient ansatz", "quantum simulation", "in quantum simulation the approach is called hardware efficient ansatz"),
        ("hardware efficient ansatz", "parametrized random circuit", "the hardware efficient ansatz is a parametrized random circuit of varying depth")]),
    "s002": (["decoy-state method", "vacuum state", "weak decoy state", "signal state", "yield",
              "quantum bit error rate (QBER)"], [
        ("decoy-state method", "vacuum state", "the Vacuum+Weak decoy method uses the vacuum state"),
        ("decoy-state method", "weak decoy state", "the Vacuum+Weak decoy method uses a weak decoy state with mean photon number << 1"),
        ("decoy-state method", "signal state", "the Vacuum+Weak decoy method uses a signal state with mean photon number O(1)"),
        ("weak decoy state", "yield", "measuring the yields of decoy states bounds the single-photon parameters"),
        ("weak decoy state", "quantum bit error rate (QBER)", "measuring the QBER of decoy states bounds the single-photon error rate")]),
    "s004": (["Shor's algorithm", "discrete logarithm", "good (c, d) pair"], [
        ("Shor's algorithm", "discrete logarithm", "Shor's quantum algorithm computes discrete logarithms"),
        ("Shor's algorithm", "good (c, d) pair", "each good (c, d) pair is generated with probability at least 1/(20q^2)")]),
    "s005": (["quantum computing", "information technology", "startup"], [
        ("quantum computing", "information technology", "quantum computing is so different from today's information technology that its applications are hard to foresee"),
        ("startup", "quantum computing", "a surge of investment in quantum computing by large companies and startups came sooner than expected")]),
    "s006": (["BB84", "quantum public key distribution", "polarized photon", "eavesdropping", "one-time pad"], [
        ("BB84", "quantum public key distribution", "BB84 is a protocol for quantum public key distribution"),
        ("BB84", "polarized photon", "BB84 encodes bits in polarized photons"),
        ("BB84", "eavesdropping", "Alice and Bob publicly compare a random subset of bits to detect eavesdropping"),
        ("BB84", "one-time pad", "the remaining bits are used as a one-time pad")]),
    "s007": (["variational quantum algorithm (VQA)", "local parameter-update optimizer", "Anderson acceleration",
              "gradient-free optimization"], [
        ("local parameter-update optimizer", "variational quantum algorithm (VQA)", "local parameter-update optimizers optimize VQA parameters sequentially"),
        ("local parameter-update optimizer", "gradient-free optimization", "local parameter updates give a gradient-free method without hyper-parameters"),
        ("Anderson acceleration", "local parameter-update optimizer", "a variant with Anderson acceleration speeds up convergence")]),
    # ---------------- multi-hop ----------------
    "m001": (["Harvest-Now, Decrypt-Later (HNDL)", "ISACA", "cybercriminal", "nation-state actor", "medical record",
              "government intelligence", "banking archive", "intellectual property"], [
        ("ISACA", "Harvest-Now, Decrypt-Later (HNDL)", "an ISACA survey found 56% of security professionals concerned about HNDL"),
        ("cybercriminal", "Harvest-Now, Decrypt-Later (HNDL)", "profit-motivated cybercriminals have less incentive for HNDL"),
        ("nation-state actor", "Harvest-Now, Decrypt-Later (HNDL)", "nation-state actors are already conducting HNDL campaigns"),
        ("medical record", "Harvest-Now, Decrypt-Later (HNDL)", "medical records are prime HNDL targets"),
        ("government intelligence", "Harvest-Now, Decrypt-Later (HNDL)", "government intelligence is a prime HNDL target"),
        ("banking archive", "Harvest-Now, Decrypt-Later (HNDL)", "banking archives are prime HNDL targets"),
        ("intellectual property", "Harvest-Now, Decrypt-Later (HNDL)", "intellectual property is a prime HNDL target")]),
    "m003": (["Quantum Random Number Generation (QRNG)", "quantum key distribution (QKD)", "eavesdropping",
              "one-time pad", "information-theoretic security"], [
        ("Quantum Random Number Generation (QRNG)", "quantum key distribution (QKD)", "QRNG produces the true random numbers QKD needs for key setting and growth"),
        ("quantum key distribution (QKD)", "eavesdropping", "QKD detects eavesdroppers"),
        ("quantum key distribution (QKD)", "one-time pad", "QKD combined with a one-time pad provides information-theoretic security"),
        ("quantum key distribution (QKD)", "information-theoretic security", "QKD with one-time pad or symmetric ciphers provides information-theoretic security")]),
    "m005": (["CRYSTALS-Dilithium", "Kyber", "OpenSSL", "Botan", "wolfSSL", "Bouncy Castle", "Cloudflare",
              "Amazon Web Services (AWS)", "IBM"], [
        ("wolfSSL", "CRYSTALS-Dilithium", "wolfSSL integrates Dilithium into its TLS stack"),
        ("Bouncy Castle", "CRYSTALS-Dilithium", "Bouncy Castle integrates Dilithium into its TLS stack"),
        ("OpenSSL", "CRYSTALS-Dilithium", "OpenSSL plans to integrate Dilithium"),
        ("OpenSSL", "Kyber", "OpenSSL plans to integrate Kyber"),
        ("Cloudflare", "CRYSTALS-Dilithium", "Cloudflare adopted Dilithium early"),
        ("Amazon Web Services (AWS)", "CRYSTALS-Dilithium", "AWS adopted Dilithium early"),
        ("IBM", "CRYSTALS-Dilithium", "IBM adopted Dilithium early")]),
    "m027": (["cryptographically relevant quantum computer (CRQC)", "RSA-2048", "integer factorization",
              "elliptic curve discrete logarithm", "NIST", "CRYSTALS-Kyber", "post-quantum cryptography (PQC)"], [
        ("cryptographically relevant quantum computer (CRQC)", "RSA-2048", "CRQCs able to break RSA-2048 could emerge within the next decade"),
        ("cryptographically relevant quantum computer (CRQC)", "integer factorization", "integer factorization is efficiently solvable on a CRQC"),
        ("cryptographically relevant quantum computer (CRQC)", "elliptic curve discrete logarithm", "elliptic curve discrete logarithms are efficiently solvable on a CRQC"),
        ("NIST", "post-quantum cryptography (PQC)", "NIST standardized a first set of PQC algorithms"),
        ("NIST", "CRYSTALS-Kyber", "NIST standardized the lattice-based KEM CRYSTALS-Kyber")]),
    "m037": (["Quantum Gaussian Mixture Model (QGMM)", "Quantum Boltzmann Machine (QBM)", "Gaussian mixture model",
              "density estimation", "clustering", "Gibbs state", "Hamiltonian", "transverse field"], [
        ("Quantum Gaussian Mixture Model (QGMM)", "Gaussian mixture model", "QGMMs are quantum extensions of Gaussian mixture models"),
        ("Quantum Gaussian Mixture Model (QGMM)", "density estimation", "QGMMs are used for density estimation"),
        ("Quantum Gaussian Mixture Model (QGMM)", "clustering", "QGMMs are used for clustering"),
        ("Quantum Boltzmann Machine (QBM)", "Gibbs state", "QBMs describe the target distribution as a quantum thermal (Gibbs) state"),
        ("Quantum Boltzmann Machine (QBM)", "Hamiltonian", "the QBM Gibbs state is of a parameterized Hamiltonian"),
        ("Quantum Boltzmann Machine (QBM)", "transverse field", "QBMs add transverse-field quantum fluctuations")]),
    "m043": (["hybrid key exchange", "Chrome", "BoringSSL", "X25519", "Kyber", "graceful degradation"], [
        ("hybrid key exchange", "Chrome", "Google deployed hybrid X25519+Kyber key agreement in Chrome"),
        ("hybrid key exchange", "X25519", "the hybrid key agreement combines X25519"),
        ("hybrid key exchange", "Kyber", "the hybrid key agreement combines Kyber"),
        ("Chrome", "BoringSSL", "Chrome's hybrid key agreement is implemented with BoringSSL"),
        ("hybrid key exchange", "graceful degradation", "hybrid deployments fall back to classical algorithms via graceful degradation")]),
    "m046": (["quantum machine learning (QML)", "quantum neural network (QNN)", "drug discovery",
              "personalized medicine", "malware detection", "botnet detection", "NISQ"], [
        ("quantum machine learning (QML)", "drug discovery", "QML is promising for drug discovery"),
        ("quantum machine learning (QML)", "personalized medicine", "QML is promising for personalized medicine"),
        ("quantum neural network (QNN)", "malware detection", "QNN-based frameworks are used for malware detection"),
        ("quantum neural network (QNN)", "botnet detection", "QNN-based frameworks are used for botnet detection"),
        ("NISQ", "quantum neural network (QNN)", "current NISQ hardware constrains QNN-based frameworks")]),
    "m064": (["post-quantum cryptography (PQC)", "5G network slicing", "enhanced mobile broadband (eMBB)", "handover",
              "fragmentation", "signalling congestion", "hash-based signature", "handshake overhead"], [
        ("post-quantum cryptography (PQC)", "5G network slicing", "large PQC keys and signatures cause problems for 5G network slices"),
        ("post-quantum cryptography (PQC)", "enhanced mobile broadband (eMBB)", "PQC per-packet size increase adds aggregate bandwidth in eMBB slices"),
        ("post-quantum cryptography (PQC)", "fragmentation", "larger PQC messages cause fragmentation"),
        ("handover", "signalling congestion", "repeated PQC handshakes during handovers worsen signalling congestion"),
        ("hash-based signature", "handshake overhead", "hash-based signatures cause 245%-890% handshake overhead")]),
    "m068": (["Module-LWE", "ML-KEM", "ML-DSA", "Learning With Errors (LWE)", "Ring-LWE", "lattice-based cryptography",
              "shortest vector problem (SVP)", "NIST"], [
        ("Module-LWE", "ML-KEM", "Module-LWE underlies ML-KEM"),
        ("Module-LWE", "ML-DSA", "Module-LWE underlies ML-DSA"),
        ("Learning With Errors (LWE)", "Module-LWE", "Module-LWE refines LWE with ring/module structure"),
        ("Learning With Errors (LWE)", "Ring-LWE", "Ring-LWE is a ring variant of LWE used for efficiency"),
        ("Learning With Errors (LWE)", "lattice-based cryptography", "LWE is a basic hard problem of lattice-based cryptography"),
        ("Learning With Errors (LWE)", "shortest vector problem (SVP)", "LWE together with SVP is believed resistant to quantum algorithms"),
        ("NIST", "ML-KEM", "NIST standardized ML-KEM"),
        ("NIST", "ML-DSA", "NIST standardized ML-DSA")]),
    "g001": (["Shor's algorithm", "Grover's algorithm", "RSA", "Elliptic Curve Cryptography (ECC)",
              "integer factorization", "discrete logarithm", "symmetric cipher", "AES", "RSA-2048", "noisy qubit"], [
        ("Shor's algorithm", "RSA", "Shor's algorithm threatens RSA"),
        ("Shor's algorithm", "Elliptic Curve Cryptography (ECC)", "Shor's algorithm threatens ECC"),
        ("Shor's algorithm", "integer factorization", "Shor's algorithm efficiently factors large integers"),
        ("Shor's algorithm", "discrete logarithm", "Shor's algorithm efficiently computes discrete logarithms"),
        ("Grover's algorithm", "symmetric cipher", "Grover's quadratic speed-up halves symmetric-cipher security, countered by doubling key sizes"),
        ("Grover's algorithm", "AES", "under Grover's algorithm AES-128 has 64-bit security"),
        ("RSA-2048", "noisy qubit", "RSA-2048 could be broken with fewer than a million noisy qubits")]),
    "g002": (["Kyber", "RSA", "Dilithium", "OpenSSL", "TLS 1.3", "Learning With Errors (LWE)",
              "lattice-based cryptography"], [
        ("Kyber", "RSA", "Kyber outperforms RSA in key exchange speed"),
        ("Kyber", "Learning With Errors (LWE)", "Kyber's security depends on the LWE problem"),
        ("Kyber", "lattice-based cryptography", "Kyber is a lattice-based scheme"),
        ("Kyber", "OpenSSL", "Kyber was integrated into OpenSSL prototypes"),
        ("Kyber", "TLS 1.3", "Kyber was tested in TLS 1.3 handshakes")]),
    "g003": (["Smart City", "Internet of Vehicles and Things (IoVT)", "5G", "Internet of Things (IoT)", "Smart Home",
              "cyber-physical system (CPS)", "machine-to-machine (M2M)"], [
        ("Smart City", "5G", "Smart Cities require the higher throughput that 5G provides"),
        ("Internet of Vehicles and Things (IoVT)", "5G", "IoVT requires the higher throughput that 5G provides"),
        ("Smart Home", "Internet of Things (IoT)", "Smart Homes are traditional IoT applications needing minimal wireless capabilities")]),
    "g005": (["amplitude encoding", "angle encoding", "qubit", "data re-uploading", "quantum circuit",
              "quantum embedding", "botnet detection"], [
        ("amplitude encoding", "qubit", "amplitude encoding requires a prohibitively large number of qubits"),
        ("data re-uploading", "quantum circuit", "data re-uploading improves data representation within quantum circuits"),
        ("quantum embedding", "botnet detection", "a hybrid model with quantum embeddings achieved up to 94.7% accuracy for botnet detection")]),
    "g007": (["multivariate quadratic (MQ) cryptography", "system of quadratic equations", "digital signature",
              "public-key encryption", "multivariate signature scheme", "edge node authentication"], [
        ("multivariate quadratic (MQ) cryptography", "system of quadratic equations", "MQ cryptography relies on solving systems of quadratic equations over finite fields (NP-hard)"),
        ("multivariate quadratic (MQ) cryptography", "digital signature", "MQ schemes are mainly used for digital signatures"),
        ("multivariate quadratic (MQ) cryptography", "public-key encryption", "MQ schemes are used for public-key encryption"),
        ("multivariate signature scheme", "edge node authentication", "the scheme's parallelism allows effective edge node authentication")]),
    "g011": (["Kyber", "side-channel attack", "side-channel leakage", "machine learning", "power analysis"], [
        ("Kyber", "side-channel leakage", "three side-channel leakage points were reported in Kyber's decryption"),
        ("Kyber", "side-channel attack", "side-channel attacks were completed on Kyber hardware implementations"),
        ("machine learning", "power analysis", "researchers used machine learning to exploit a power-consumption side channel"),
        ("power analysis", "Kyber", "the power side channel was exploited against a Kyber implementation")]),
    "g015": (["Kyber", "Dilithium", "Elliptic Curve Cryptography (ECC)", "Classic McEliece", "ECDSA", "FALCON",
              "Internet of Things (IoT)", "RSA"], [
        ("Kyber", "Elliptic Curve Cryptography (ECC)", "Kyber's public key (~800 bytes) is larger than an ECC public key"),
        ("Kyber", "Classic McEliece", "Kyber's public key is much smaller than Classic McEliece's (>1 MB)"),
        ("Dilithium", "ECDSA", "Dilithium signatures are a few kilobytes versus 64 bytes for ECDSA"),
        ("Kyber", "Internet of Things (IoT)", "Kyber is considered feasible for a common IoT setup"),
        ("Kyber", "RSA", "Kyber operations are comparable to or faster than RSA"),
        ("FALCON", "Dilithium", "FALCON is more memory-efficient than Dilithium")]),
    "g016": (["quantum secure direct communication (QSDC)", "quantum key distribution (QKD)", "quantum channel",
              "DARPA quantum network", "Tokyo QKD network", "Trieste QKD network"], [
        ("quantum secure direct communication (QSDC)", "quantum key distribution (QKD)", "QSDC transmits messages directly whereas QKD establishes shared keys"),
        ("quantum secure direct communication (QSDC)", "quantum channel", "QSDC transmits messages directly over a quantum channel"),
        ("DARPA quantum network", "quantum key distribution (QKD)", "the DARPA network is a practical intercity QKD network"),
        ("Tokyo QKD network", "quantum key distribution (QKD)", "the Tokyo network is a practical intercity QKD network"),
        ("Trieste QKD network", "quantum key distribution (QKD)", "the Trieste network is a practical intercity QKD network")]),
    "g019": (["QKD network", "quantum key distribution (QKD)", "key storage", "key rate", "authentication",
              "man-in-the-middle attack", "pre-shared key", "key management system"], [
        ("QKD network", "key storage", "the key storages of a QKD network must stay consistent for the service to work"),
        ("key storage", "key rate", "key storage decouples generation from consumption to cope with limited QKD key rates"),
        ("quantum key distribution (QKD)", "authentication", "QKD has no built-in authentication"),
        ("quantum key distribution (QKD)", "man-in-the-middle attack", "without authentication a man-in-the-middle could impersonate a QKD endpoint"),
        ("quantum key distribution (QKD)", "pre-shared key", "QKD links are bootstrapped with a pre-shared key or classical signature"),
        ("quantum key distribution (QKD)", "key management system", "QKD may need integration with classical key management systems")]),
    "g021": (["Kyber", "two-step attack", "correlation power analysis (CPA)", "lattice attack", "power trace",
              "ARM Cortex-M4", "shuffling countermeasure", "FPGA", "TVLA"], [
        ("two-step attack", "Kyber", "the two-step attack recovers Kyber's secret key"),
        ("two-step attack", "correlation power analysis (CPA)", "the two-step attack starts with CPA"),
        ("two-step attack", "lattice attack", "the two-step attack follows CPA with a lattice attack"),
        ("two-step attack", "power trace", "the two-step attack needs at most 15 power traces"),
        ("two-step attack", "ARM Cortex-M4", "the two-step attack is run on an ARM Cortex-M4 board"),
        ("shuffling countermeasure", "Kyber", "the shuffling countermeasure protects Kyber"),
        ("shuffling countermeasure", "FPGA", "the shuffling countermeasure is implemented on FPGA with 8.7% efficiency loss"),
        ("shuffling countermeasure", "TVLA", "the shuffling countermeasure is verified with TVLA")]),
    "g026": (["higher-order masking", "Kyber", "side-channel attack", "arithmetic masking", "Boolean masking",
              "threshold secret sharing", "Lagrange interpolation"], [
        ("higher-order masking", "Kyber", "higher-order masking protects Kyber implementations"),
        ("higher-order masking", "side-channel attack", "masking is a countermeasure against side-channel attacks"),
        ("higher-order masking", "arithmetic masking", "shares can be combined by arithmetic addition (arithmetic masking)"),
        ("higher-order masking", "Boolean masking", "shares can be combined by XOR (Boolean masking)"),
        ("threshold secret sharing", "Lagrange interpolation", "t or more participants reconstruct the secret by Lagrange interpolation")]),
    "g027": (["gate error rate", "trapped ion", "transistor", "quantum error correction (QEC)",
              "fault-tolerant quantum computation", "Toffoli gate", "superconducting qubit", "coherence time"], [
        ("trapped ion", "gate error rate", "trapped ions have the best reported gate error rates (1e-7 / 1e-5)"),
        ("transistor", "gate error rate", "digital transistors reach error rates of 1e-23"),
        ("fault-tolerant quantum computation", "quantum error correction (QEC)", "fault-tolerant computation depends on effective QEC"),
        ("quantum error correction (QEC)", "Toffoli gate", "QEC was demonstrated with a Toffoli gate"),
        ("Toffoli gate", "superconducting qubit", "the Toffoli gate was implemented on superconducting qubits"),
        ("quantum error correction (QEC)", "coherence time", "the QEC demonstration was limited by coherence times")]),
    "g028": (["Kyber", "Dilithium", "MbedTLS", "Crypto++", "LMS", "cryptopp-pqc", "SABER", "FrodoKEM",
              "Internet of Things (IoT)"], [
        ("MbedTLS", "Kyber", "MbedTLS intends to add Kyber soon"),
        ("MbedTLS", "Dilithium", "MbedTLS intends to add Dilithium later"),
        ("MbedTLS", "LMS", "MbedTLS supports LMS signatures"),
        ("MbedTLS", "Internet of Things (IoT)", "MbedTLS runs on millions of IoT devices"),
        ("Crypto++", "cryptopp-pqc", "cryptopp-pqc is a third-party fork of Crypto++"),
        ("cryptopp-pqc", "Kyber", "cryptopp-pqc adds CRYSTALS-Kyber"),
        ("cryptopp-pqc", "Dilithium", "cryptopp-pqc adds CRYSTALS-Dilithium"),
        ("cryptopp-pqc", "SABER", "cryptopp-pqc adds SABER"),
        ("cryptopp-pqc", "FrodoKEM", "cryptopp-pqc adds FrodoKEM")]),
    "g030": (["Harvest-Now, Decrypt-Later (HNDL)", "Q-Day", "nation-state actor", "RSA",
              "Elliptic Curve Cryptography (ECC)", "Cryptographic Time Paradox",
              "cryptographically relevant quantum computer (CRQC)", "post-quantum cryptography (PQC)"], [
        ("nation-state actor", "Harvest-Now, Decrypt-Later (HNDL)", "HNDL is the primary behavior of nation-states before Q-Day"),
        ("Harvest-Now, Decrypt-Later (HNDL)", "Q-Day", "HNDL actors collect data now to decrypt after Q-Day"),
        ("Harvest-Now, Decrypt-Later (HNDL)", "RSA", "harvested data is decrypted once quantum computers break RSA"),
        ("Harvest-Now, Decrypt-Later (HNDL)", "Elliptic Curve Cryptography (ECC)", "harvested data is decrypted once quantum computers break ECC"),
        ("Cryptographic Time Paradox", "cryptographically relevant quantum computer (CRQC)", "encrypted data becomes readable the moment a CRQC is built"),
        ("Cryptographic Time Paradox", "RSA", "RSA-encrypted data is safe only until a working CRQC exists"),
        ("Cryptographic Time Paradox", "post-quantum cryptography (PQC)", "the paradox means PQC readiness is needed by 2035")]),
    "g034": (["quantum cryptography", "quantum physics", "quantum key distribution (QKD)",
              "continuous-variable QKD (CV-QKD)", "discrete-variable QKD (DV-QKD)", "single photon",
              "information-theoretic security"], [
        ("quantum cryptography", "quantum physics", "quantum cryptography relies on principles of quantum physics"),
        ("quantum key distribution (QKD)", "continuous-variable QKD (CV-QKD)", "CV protocols are one category of QKD protocols"),
        ("quantum key distribution (QKD)", "discrete-variable QKD (DV-QKD)", "DV protocols are one category of QKD protocols"),
        ("discrete-variable QKD (DV-QKD)", "single photon", "DV protocols use discrete outcomes such as single-photon polarization"),
        ("quantum key distribution (QKD)", "information-theoretic security", "QKD keys are information-theoretically secure")]),
    "g035": (["Shor's algorithm", "entanglement", "quantum parallelism", "Quantum Fourier Transform (QFT)",
              "integer factorization", "RSA", "Elliptic Curve Cryptography (ECC)"], [
        ("Shor's algorithm", "entanglement", "Shor's algorithm uses entanglement"),
        ("Shor's algorithm", "quantum parallelism", "Shor's algorithm exploits quantum parallelism"),
        ("Shor's algorithm", "Quantum Fourier Transform (QFT)", "the QFT is at the core of Shor's algorithm"),
        ("Shor's algorithm", "integer factorization", "Shor's algorithm factors large integers in polynomial time"),
        ("RSA", "integer factorization", "RSA's security rests on the difficulty of factoring large numbers"),
        ("Shor's algorithm", "RSA", "Shor's algorithm threatens RSA"),
        ("Shor's algorithm", "Elliptic Curve Cryptography (ECC)", "Shor's algorithm threatens ECC")]),
    "g037": (["SPHINCS+ (SLH-DSA)", "hash function", "WOTS+", "FORS", "hypertree", "NIST", "Kyber", "Dilithium",
              "FIPS 203", "FIPS 204", "FIPS 205"], [
        ("SPHINCS+ (SLH-DSA)", "hash function", "SPHINCS+ security relies solely on the collision resistance of hash functions"),
        ("SPHINCS+ (SLH-DSA)", "WOTS+", "SPHINCS+ uses WOTS+ to sign individual nodes"),
        ("SPHINCS+ (SLH-DSA)", "FORS", "SPHINCS+ uses FORS few-time signatures at the leaves"),
        ("SPHINCS+ (SLH-DSA)", "hypertree", "SPHINCS+ links its trees in a hypertree for stateless operation"),
        ("NIST", "SPHINCS+ (SLH-DSA)", "NIST selected SPHINCS+"),
        ("FIPS 203", "Kyber", "FIPS 203 is ML-KEM (Kyber)"),
        ("FIPS 204", "Dilithium", "FIPS 204 is ML-DSA (Dilithium)"),
        ("FIPS 205", "SPHINCS+ (SLH-DSA)", "FIPS 205 is SLH-DSA (SPHINCS+)")]),
    "g039": (["quantum key distribution (QKD)", "attenuated laser", "quantum machine learning (QML)",
              "quantum support vector machine (QSVM)", "quantum convolutional neural network (QCNN)",
              "eavesdropping detection"], [
        ("quantum key distribution (QKD)", "attenuated laser", "attenuated lasers sometimes emit multi-photon signals, making practical QKD insecure"),
        ("quantum machine learning (QML)", "eavesdropping detection", "QML algorithms can be leveraged for eavesdropping detection"),
        ("quantum support vector machine (QSVM)", "eavesdropping detection", "QSVM can be used for eavesdropping detection"),
        ("quantum convolutional neural network (QCNN)", "eavesdropping detection", "QCNN can be used for eavesdropping detection")]),
    "g040": (["Number Theoretic Transform (NTT)", "Kyber", "Dilithium", "Artix-7 FPGA", "polynomial multiplication"], [
        ("Number Theoretic Transform (NTT)", "Kyber", "the NTT-optimized accelerator speeds up Kyber by up to 34.16x"),
        ("Number Theoretic Transform (NTT)", "Dilithium", "the NTT-optimized accelerator speeds up Dilithium by 9.6x"),
        ("Number Theoretic Transform (NTT)", "Artix-7 FPGA", "the NTT accelerator is implemented on an Artix-7 FPGA"),
        ("Number Theoretic Transform (NTT)", "polynomial multiplication", "the NTT reduces polynomial multiplication from O(N^2) to O(N log N)"),
        ("polynomial multiplication", "Kyber", "polynomial multiplication dominates Kyber encryption and decryption")]),
    "g042": (["quantum generative adversarial network (QGAN)", "quantum generator", "discriminator",
              "minimax game", "variational quantum algorithm (VQA)", "EQGAN", "fitness evaluation"], [
        ("quantum generative adversarial network (QGAN)", "quantum generator", "a QGAN has a parameterized quantum generator"),
        ("quantum generative adversarial network (QGAN)", "discriminator", "a QGAN has a quantum or classical discriminator"),
        ("quantum generative adversarial network (QGAN)", "minimax game", "QGAN training is a two-player minimax game"),
        ("quantum generative adversarial network (QGAN)", "variational quantum algorithm (VQA)", "a QGAN is usually built as a variational quantum algorithm"),
        ("EQGAN", "quantum generative adversarial network (QGAN)", "EQGAN improves QGAN training"),
        ("EQGAN", "fitness evaluation", "EQGAN uses fitness evaluation to select generator configurations")]),
    "g043": (["quantum computer", "classical computer", "quantum algorithm", "integer factorization",
              "precision of quantum state manipulation"], [
        ("quantum algorithm", "classical computer", "quantum algorithms solve problems believed hard for classical computers"),
        ("quantum algorithm", "integer factorization", "factoring large integers is the best-known such problem"),
        ("precision of quantum state manipulation", "quantum computer", "if precision grows polynomially with input size, quantum computers are more powerful")]),
    "g044": (["variational quantum algorithm (VQA)", "NISQ", "qubit connectivity", "circuit depth", "gate fidelity",
              "quantum error correction (QEC)"], [
        ("variational quantum algorithm (VQA)", "NISQ", "VQAs are the leading strategy for NISQ devices"),
        ("variational quantum algorithm (VQA)", "qubit connectivity", "VQAs account for limited qubit connectivity"),
        ("variational quantum algorithm (VQA)", "circuit depth", "VQAs account for errors that limit circuit depth"),
        ("gate fidelity", "NISQ", "more accurate gates extend the power of NISQ technology"),
        ("gate fidelity", "quantum error correction (QEC)", "more accurate gates lower the overhead of later QEC")]),
    "g045": (["BB84", "intercept-resend attack", "rectilinear basis", "decoy-state QKD", "yield", "dark count"], [
        ("intercept-resend attack", "BB84", "measuring and resending all photons induces errors in 1/4 of BB84 bits"),
        ("intercept-resend attack", "rectilinear basis", "the eavesdropper measures all photons in the rectilinear basis"),
        ("decoy-state QKD", "yield", "decoy-state QKD analyses the yield"),
        ("yield", "dark count", "the yield combines signal-photon detection and background events such as dark counts")]),
    "g046": (["quantum machine learning (QML)", "quantum key distribution (QKD)", "eavesdropping detection",
              "6G network", "adversarial attack"], [
        ("quantum machine learning (QML)", "quantum key distribution (QKD)", "research on QML for QKD has increased since 2010"),
        ("quantum machine learning (QML)", "eavesdropping detection", "QML is used for eavesdropping detection in QKD"),
        ("quantum machine learning (QML)", "6G network", "QML is studied for 6G network intelligence"),
        ("quantum machine learning (QML)", "adversarial attack", "QML-based applications face adversarial threats")]),
    "g048": (["variational quantum classifier (VQC)", "parameterized quantum circuit", "classical optimizer",
              "cyberattack detection", "NSL-KDD", "DGA botnet", "variational quantum circuit",
              "quantum generative adversarial network (QGAN)"], [
        ("variational quantum classifier (VQC)", "parameterized quantum circuit", "VQCs combine parameterised quantum circuits with classical optimisers"),
        ("variational quantum classifier (VQC)", "classical optimizer", "VQCs use classical optimisers in hybrid feedback loops"),
        ("variational quantum classifier (VQC)", "cyberattack detection", "VQCs are used for cyberattack detection"),
        ("variational quantum classifier (VQC)", "NSL-KDD", "VQCs were used for cyberattack detection on NSL-KDD"),
        ("variational quantum classifier (VQC)", "DGA botnet", "VQCs were used for DGA botnet classification"),
        ("variational quantum circuit", "quantum generative adversarial network (QGAN)", "variational quantum circuits are the most common QGAN architecture")]),
    "g050": (["quantum generative adversarial network (QGAN)", "generator", "discriminator", "Nash equilibrium",
              "parameter-shift rule", "stochastic approximation"], [
        ("quantum generative adversarial network (QGAN)", "generator", "the QGAN generator is trained to make fake data classified as real"),
        ("quantum generative adversarial network (QGAN)", "discriminator", "the QGAN discriminator is trained to distinguish real from generated data"),
        ("quantum generative adversarial network (QGAN)", "Nash equilibrium", "QGAN training proceeds until a Nash equilibrium"),
        ("quantum generative adversarial network (QGAN)", "parameter-shift rule", "QGAN gradients are estimated with the parameter-shift rule"),
        ("quantum generative adversarial network (QGAN)", "stochastic approximation", "QGAN gradients may be estimated by stochastic approximation")]),
}


def main():
    raw = json.loads((HERE / "kg_gold_raw.json").read_text(encoding="utf-8"))
    assert {r["id"] for r in raw} == set(GOLD), set(GOLD) ^ {r["id"] for r in raw}
    out = []
    for r in raw:
        ents, rels = GOLD[r["id"]]
        assert len(ents) == len(set(ents)), r["id"]
        for a, b, _ in rels:
            assert a in ents and b in ents and a != b, (r["id"], a, b)
        out.append({"id": r["id"], "type": r["type"], "question": r["question"],
                    "entities": ents, "relations": [list(x) for x in rels], "reviewed": True})
    (HERE / "kg_gold.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(out)} câu, {sum(len(x['entities']) for x in out)} thực thể, "
          f"{sum(len(x['relations']) for x in out)} quan hệ -> {HERE / 'kg_gold.json'}")


if __name__ == "__main__":
    main()
