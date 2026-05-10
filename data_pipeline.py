"""
RAG Preprocessing Pipeline for Security Academic Papers.
Data source: Hugging Face `clouditera/security-paper-datasets`.
"""

import warnings
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14",
)

import re
import uuid
import logging
from typing import Any
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# =============================================================================
# Domain Taxonomy — keyword -> tag mapping (EN + ZH bilingual)
# =============================================================================
# Each category has English AND Chinese keywords.
# Chinese keywords are matched literally (no \b — \b only works for ASCII \w).
# English keywords use \b word boundaries to avoid false partial matches.

SECURITY_TAXONOMY: dict[str, list[str]] = {
    "Vulnerability Analysis": [
        # English
        "vulnerability", "CVE", "exploit", "zero-day", "zero day",
        "fuzzing", "fuzz", "bug bounty", "patch", "proof of concept",
        "attack surface", "privilege escalation", "sandbox escape",
        "code execution", "command injection", "race condition",
        # Chinese
        "漏洞分析", "漏洞挖掘", "漏洞利用", "漏洞扫描", "漏洞复现",
        "反序列化", "命令注入", "代码执行", "提权攻击", "沙箱逃逸",
        "零日漏洞", "模糊测试", "补丁分析", "POC", "EXP",
        "攻击面", "权限提升", "任意代码执行",
    ],
    "Cryptography": [
        # English
        "cryptograph", "encrypt", "decrypt", "cipher", "hash function",
        "digital signature", "PKI", "SSL", "TLS", "RSA", "AES", "ECC",
        "elliptic curve", "homomorphic encryption", "quantum-safe",
        "post-quantum", "key exchange", "zero-knowledge", "commitment scheme",
        "symmetric encryption", "public key",
        # Chinese
        "密码学", "加密算法", "解密", "国密", "哈希算法", "哈希函数",
        "数字签名", "密钥交换", "密钥协商", "同态加密", "量子密码",
        "零知识证明", "安全多方计算", "椭圆曲线", "分组密码", "流密码",
        "公钥密码", "对称加密", "SM2", "SM3", "SM4", "SM9",
        "密码协议", "认证加密",
    ],
    "Malware Analysis": [
        # English
        "malware", "ransomware", "trojan", "virus", "worm", "spyware",
        "rootkit", "botnet", "backdoor", "dropper", "packer", "obfuscation",
        "polymorphic", "metamorphic", "command and control", "C2", "C&C",
        "keylogger", "adware", "fileless",
        # Chinese
        "恶意软件", "勒索软件", "木马", "病毒", "蠕虫", "间谍软件",
        "僵尸网络", "免杀", "加壳", "代码混淆", "多态病毒", "变形病毒",
        "命令与控制", "后门程序", "下载器", "键盘记录", "无文件攻击",
        "恶意代码", "APK", "样本分析", "沙箱检测",
    ],
    "Network Security": [
        # English
        "network security", "firewall", "IDS", "IPS", "intrusion",
        "packet inspection", "DNS secur", "DDoS", "denial of service",
        "VPN", "proxy", "man-in-the-middle", "MITM", "ARP spoof",
        "BGP hijack", "SDN security", "traffic analysis",
        "port scan", "packet filter",
        # Chinese
        "网络安全", "防火墙", "入侵防御", "入侵检测系统",
        "流量分析", "拒绝服务", "中间人攻击", "ARP欺骗",
        "协议安全", "抓包分析", "端口扫描", "DNS安全",
        "SDN安全", "BGP劫持", "深度包检测", "VPN安全",
        "网络隔离", "网络边界",
    ],
    "Binary Exploitation": [
        # English
        "binary", "assembly", "disassembly", "buffer overflow",
        "return-oriented programming", "ROP", "shellcode", "ASLR",
        "DEP", "stack canary", "control flow integrity", "CFI",
        "memory corruption", "use-after-free", "type confusion",
        "reverse engineering", "ghidra", "ida pro", "binary ninja",
        "heap spray", "format string",
        # Chinese
        "二进制安全", "汇编", "反汇编", "缓冲区溢出", "栈溢出",
        "堆溢出", "返回导向编程", "内存破坏", "UAF",
        "类型混淆", "逆向工程", "格式化字符串", "整数溢出",
        "ROP链", "Gadget", "shellcode", "地址随机化",
        "控制流完整性", "堆风水", "堆利用",
    ],
    "Web Security": [
        # English
        "XSS", "CSRF", "SQL injection", "cross-site", "web application",
        "browser security", "DOM", "same-origin", "CSP", "content security",
        "cookie", "session hijack", "OAuth", "open redirect", "SSRF",
        "file upload", "path traversal", "XXE",
        # Chinese
        "Web安全", "跨站脚本", "跨站请求伪造", "SQL注入", "文件上传漏洞",
        "反序列化漏洞", "命令执行", "WAF绕过", "渗透测试",
        "服务端请求伪造", "路径穿越", "XML外部实体", "同源策略",
        "内容安全策略", "会话劫持", "越权", "文件包含",
    ],
    "AI/ML Security": [
        # English
        "adversarial example", "adversarial attack", "deep learning",
        "neural network", "machine learning security", "model inversion",
        "membership inference", "backdoor attack", "poisoning attack",
        "federated learning", "GAN", "NLP security", "transformer",
        "large language model", "LLM security", "prompt injection",
        # Chinese
        "对抗样本", "对抗攻击", "深度学习", "神经网络", "模型安全",
        "模型窃取", "后门攻击", "投毒攻击", "联邦学习",
        "差分隐私", "AI安全", "大模型安全", "提示注入",
        "成员推断", "模型反演", "生成对抗网络",
    ],
    "Intrusion Detection": [
        # English
        "intrusion detection", "anomaly detection", "SIEM", "log analysis",
        "alert correlation", "false positive", "threat hunting",
        "endpoint detection", "EDR", "network monitor", "honeypot",
        "UEBA", "SOAR",
        # Chinese
        "入侵检测", "异常检测", "威胁狩猎", "端点检测与响应",
        "蜜罐", "态势感知", "安全运营中心", "SOC",
        "日志分析", "关联分析", "误报", "漏报",
        "用户实体行为分析", "安全编排自动化与响应",
        "威胁情报", "攻击链",
    ],
    "Blockchain Security": [
        # English
        "blockchain", "smart contract", "Ethereum", "Bitcoin", "DeFi",
        "consensus algorithm", "solidity", "decentralized", "NFT",
        "DAO", "oracle", "flash loan", "reentrancy",
        "MEV", "DeFi", "cross-chain",
        # Chinese
        "区块链安全", "智能合约", "以太坊", "比特币", "去中心化",
        "重入攻击", "闪电贷", "预言机", "共识算法", "跨链",
        "Solidity", "NFT安全", "DAO攻击", "抢先交易",
        "默克尔树", "钱包安全",
    ],
    "Cloud Security": [
        # English
        "cloud security", "AWS", "Azure", "GCP", "container security",
        "Kubernetes", "Docker", "virtualization", "serverless",
        "multi-tenan", "hypervisor", "IaaS", "PaaS", "SaaS",
        "microservice",
        # Chinese
        "云安全", "容器安全", "虚拟化安全", "多云安全",
        "超visor", "多租户", "微服务安全", "无服务器安全",
        "镜像安全", "编排安全",
    ],
    "IoT Security": [
        # English
        "IoT", "Internet of Things", "embedded security", "firmware",
        "PLC", "SCADA", "industrial control", "ICS", "CAN bus",
        "smart home", "wearable", "sensor network", "V2X",
        # Chinese
        "物联网安全", "嵌入式安全", "固件安全", "工控安全",
        "工业控制系统", "车联网安全", "智能家居安全",
        "传感器安全", "可编程逻辑控制器",
    ],
    "Privacy": [
        # English
        "privacy", "anonymous", "GDPR", "PII", "personally identifiable",
        "differential privacy", "data leak", "surveillance", "tracking",
        "fingerprint", "k-anonymity", "data protection",
        "personal data", "consent",
        # Chinese
        "隐私保护", "匿名化", "数据脱敏", "数据泄露", "个人信息保护",
        "个人信息", "隐私计算", "差分隐私", "数据安全法",
        "联邦学习", "安全多方计算", "去标识化", "匿名通信",
    ],
    "Social Engineering": [
        # English
        "phishing", "social engineering", "spear phishing", "whaling",
        "pretexting", "baiting", "credential theft", "smishing",
        "vishing", "business email compromise",
        # Chinese
        "社工", "社会工程学", "网络钓鱼", "鱼叉钓鱼",
        "鲸钓攻击", "电信诈骗", "仿冒", "凭证窃取",
        "伪基站", "钓鱼邮件", "短信钓鱼", "语音钓鱼",
    ],
    "Authentication & Access Control": [
        # English
        "authentication", "authorization", "OAuth", "SAML", "MFA",
        "biometric", "password", "JWT", "RBAC", "ABAC", "single sign-on",
        "identity management", "kerberos", "LDAP",
        "zero trust", "CAPTCHA", "FIDO",
        # Chinese
        "身份认证", "访问控制", "单点登录", "多因素认证",
        "生物识别", "权限管理", "零信任", "统一身份认证",
        "基于角色的访问控制", "基于属性的访问控制",
        "口令安全", "认证协议", "授权管理",
    ],
    "Digital Forensics": [
        # English
        "forensic", "digital evidence", "chain of custody",
        "disk forensics", "memory forensics", "file carving",
        "timeline analysis", "anti-forensic", "data recovery",
        "live forensics", "network forensics",
        # Chinese
        "数字取证", "电子数据取证", "电子证据", "证据链",
        "磁盘取证", "内存取证", "文件恢复", "文件雕刻",
        "时间线分析", "反取证", "数据恢复", "痕迹分析",
        "电子数据鉴定",
    ],
    # ---- NEW: Security Management & Compliance ----
    "Security Management & Compliance": [
        # English
        "security management", "compliance", "risk assessment",
        "security policy", "security audit", "incident response",
        "business continuity", "disaster recovery", "ISO 27001",
        "NIST", "security governance", "security baseline",
        "security awareness", "supply chain security",
        "third-party risk", "security assessment", "security framework",
        # Chinese
        "安全管理", "安全合规", "等保", "等级保护", "风险评估",
        "安全建设", "安全体系", "安全策略", "安全审计",
        "合规性", "监管合规", "信息安全管理", "业务连续性",
        "灾备", "应急响应", "安全治理", "安全基线",
        "安全评估", "供应链安全", "安全意识", "安全培训",
        "ISO27001", "网络安全法", "数据安全法", "个人信息保护法",
        "关基", "关键信息基础设施", "网络安全等级保护",
        "安全管理制度", "安全运维", "安全运营",
    ],
}

# =============================================================================
# Regex compilation: CJK-aware boundary handling
# =============================================================================
# Problem: \b in Python regex only recognizes ASCII \w characters.
# Chinese characters (e.g. 一-鿿) are NOT matched by \w,
# so \b漏洞分析\b will never match. We split each category's keywords
# into ASCII (wrap with \b) and CJK (literal match, no boundary).

_CJK_RANGE = re.compile(r"[一-鿿㐀-䶿豈-﫿]")


def _build_tag_pattern(keywords: list[str]) -> re.Pattern:
    """Compile a case-insensitive regex pattern from a mixed-language keyword list.

    ASCII keywords  → wrapped with \\b for word-boundary safety.
    CJK keywords   → matched literally (\\b does not work for Unicode in Python).
    """
    parts: list[str] = []
    for kw in keywords:
        escaped = re.escape(kw)
        if _CJK_RANGE.search(kw):
            # CJK keyword: literal substring match
            parts.append(escaped)
        else:
            # ASCII keyword: enforce word boundaries
            parts.append(r"\b" + escaped + r"\b")
    if not parts:
        return re.compile(r"(?!)")  # never matches
    return re.compile("|".join(parts), re.IGNORECASE)


_TAXONOMY_PATTERNS: dict[str, re.Pattern] = {
    tag: _build_tag_pattern(keywords)
    for tag, keywords in SECURITY_TAXONOMY.items()
}


# =============================================================================
# A. Academic Paper Deep Cleaner
# =============================================================================


class AcademicPaperCleaner:
    """Deep-clean academic papers extracted from PDFs.

    Handles: hyphenation repair, reference-stripping, header/footer removal,
    figure/table noise, whitespace normalization, and orphaned symbol cleanup.
    """

    # Lines that look like section headers for references
    _REF_HEADER = re.compile(
        r"^\s*(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY|"
        r"参考文献|參考文獻|Works\s+Cited|"
        r"REFERENCES\s+AND\s+NOTES)\s*$",
        re.IGNORECASE,
    )

    # Detects the start of a numbered reference entry — e.g. "[1] Smith, ..."
    _REF_ENTRY = re.compile(r"^\s*\[\d+\][\s,].+|^\s*\d+\.\s+[A-Z].+")

    # Hyphenated line-break: "vulnera-\nbility" → "vulnerability"
    _HYPHEN_BREAK = re.compile(r"(\w+)-\n(\w+)")

    # Page number standing alone on a line
    _PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")

    # Lines that are mostly special characters (from PDF artifacts)
    _NOISE_LINE = re.compile(r"^[\s\.\-_=*#~^]{10,}$")

    # DOI / URL on its own line (keep track, but remove standalone noise)
    _DOI_LINE = re.compile(r"^\s*(?:DOI|doi):\s*\S+")

    # Running-header patterns: "Conference Name 2023", "Author Name et al.", etc.
    _HEADER_PATTERNS = [
        re.compile(r"^\s*(?:Proceedings|Proc\.)\s+of\s+(?:the\s+)?\S+", re.IGNORECASE),
        re.compile(r"^\s*(?:IEEE|ACM|USENIX|NDSS|S\&P|CCS|RAID|ACSAC)\s*\d{4}", re.IGNORECASE),
    ]

    def __init__(self, strip_references: bool = True, min_line_len: int = 3):
        self.strip_references = strip_references
        self.min_line_len = min_line_len

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, raw_text: str) -> str:
        """Run the full cleaning pipeline on a single document."""
        if not raw_text or not raw_text.strip():
            return ""

        text = raw_text

        # 1. Repair hyphenation splits
        text = self._repair_hyphenation(text)

        # 2. Remove reference section
        if self.strip_references:
            text = self._strip_references(text)

        # 3. Remove figure/table captions (heuristic)
        text = self._strip_figure_table_captions(text)

        # 4. Remove header/footer noise
        text = self._strip_headers_footers(text)

        # 5. Collapse orphaned symbols and whitespace
        text = self._normalize_whitespace(text)

        # 6. Strip noisy short lines (page numbers, stray symbols)
        text = self._strip_noise_lines(text)

        return text.strip()

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    @staticmethod
    def _repair_hyphenation(text: str) -> str:
        """Fix PDF-induced word breaks like 'exploi-\ntation' → 'exploitation'."""
        return AcademicPaperCleaner._HYPHEN_BREAK.sub(r"\1\2", text)

    @staticmethod
    def _strip_references(text: str) -> str:
        """Truncate at the first reference-section header, if found."""
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if AcademicPaperCleaner._REF_HEADER.match(line):
                # Heuristic: references header followed by numbered entries
                if i + 1 < len(lines) and AcademicPaperCleaner._REF_ENTRY.match(lines[i + 1]):
                    return "\n".join(lines[:i])
                # If next line isn't clearly a reference entry, check a few more
                if i + 3 < len(lines):
                    ref_count = sum(
                        1
                        for j in range(i + 1, min(i + 5, len(lines)))
                        if AcademicPaperCleaner._REF_ENTRY.match(lines[j])
                    )
                    if ref_count >= 2:
                        return "\n".join(lines[:i])
        return text

    @staticmethod
    def _strip_figure_table_captions(text: str) -> str:
        """Remove lines that are pure figure/table labels with numbers."""
        # Strip lines like "Fig. 3.", "Figure 4:", "Table 2.", "Fig. 3: Overview"
        pattern = re.compile(
            r"^\s*(?:Fig(?:ure)?\.?\s*\d+|Table\.?\s*\d+)\s*[.:].*$",
            re.IGNORECASE,
        )
        lines = text.split("\n")
        lines = [ln for ln in lines if not pattern.match(ln.strip())]
        return "\n".join(lines)

    def _strip_headers_footers(self, text: str) -> str:
        """Remove suspected running headers/footers and page numbers."""
        lines = text.split("\n")
        cleaned: list[str] = []
        for line in lines:
            stripped = line.strip()
            # Skip standalone page numbers
            if self._PAGE_NUMBER.match(stripped):
                continue
            # Skip DOI-only lines (usually a footer artifact)
            if self._DOI_LINE.match(stripped):
                continue
            # Skip lines matching common header patterns
            if any(pat.match(stripped) for pat in self._HEADER_PATTERNS):
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse redundant whitespace while preserving paragraph boundaries."""
        # Collapse tabs and multiple spaces → single space
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse 3+ newlines → 2 newlines (preserve paragraph separation)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove trailing spaces on each line
        text = re.sub(r" +\n", "\n", text)
        return text

    @staticmethod
    def _strip_noise_lines(text: str) -> str:
        """Remove lines that are mostly noise: long symbol chains, etc."""
        lines = text.split("\n")
        lines = [
            ln for ln in lines if not AcademicPaperCleaner._NOISE_LINE.match(ln.strip())
        ]
        return "\n".join(lines)


# =============================================================================
# B. Context-Aware Chunker
# =============================================================================


class ContextAwareChunker:
    """Splits long academic texts with paragraph/sentence-aware boundaries."""

    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
    ):
        # Separators ordered from coarsest to finest — respects paragraph
        # and sentence integrity before falling back to whitespace
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "。 ", "? ", "! ", "; ", ": ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    def split(self, text: str) -> list[str]:
        """Chunk cleaned text. Returns list of chunk strings."""
        if not text or not text.strip():
            return []
        return self.splitter.split_text(text)


# =============================================================================
# C. Metadata Injector & Tag Extractor
# =============================================================================


class TagExtractor:
    """Extract security-domain tags from academic text via keyword matching."""

    def __init__(self, min_keyword_hits: int = 2):
        self.min_hits = min_keyword_hits

    def extract(self, text: str) -> list[str]:
        """Return a sorted list of matching domain tags."""
        tags: list[str] = []
        text_lower = text.lower()
        for tag, pattern in _TAXONOMY_PATTERNS.items():
            matches = pattern.findall(text_lower)
            if len(matches) >= self.min_hits:
                tags.append(tag)
        # Sort so output is deterministic
        return sorted(tags)


class MetadataInjector:
    """Attaches the standard RAG metadata schema to chunked documents."""

    def __init__(self, tag_extractor: TagExtractor | None = None):
        self.tag_extractor = tag_extractor or TagExtractor()

    def inject(
        self,
        chunks: list[str],
        *,
        source: str = "hf:clouditera/security-paper-datasets",
        paper_title: str = "",
    ) -> list[dict[str, Any]]:
        """Produce a list of {id, content, metadata} dicts."""
        records: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            tags = self.tag_extractor.extract(chunk)
            records.append({
                "id": str(uuid.uuid4()),
                "content": chunk.strip(),
                "metadata": {
                    "source": source,
                    "paper_title": paper_title or "Untitled",
                    "chunk_index": idx,
                    "word_count": len(chunk.split()),
                    "tags": tags,
                },
            })
        return records


# =============================================================================
# D. Full Pipeline Orchestrator
# =============================================================================


class SecurityPaperPipeline:
    """End-to-end pipeline: clean → chunk → inject metadata."""

    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        strip_references: bool = True,
    ):
        self.cleaner = AcademicPaperCleaner(strip_references=strip_references)
        self.chunker = ContextAwareChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.injector = MetadataInjector()

    def run(
        self,
        raw_text: str,
        paper_title: str = "",
        source: str = "hf:clouditera/security-paper-datasets",
    ) -> list[dict[str, Any]]:
        """Run the full pipeline on a single raw paper text."""
        # Step 1 — Deep clean
        cleaned = self.cleaner.clean(raw_text)
        if not cleaned:
            logger.warning("Text became empty after cleaning — skipping.")
            return []

        # Step 2 — Context-aware chunking
        chunks = self.chunker.split(cleaned)

        # Step 3 — Metadata injection
        records = self.injector.inject(chunks, source=source, paper_title=paper_title)

        return records

    def run_batch(
        self,
        texts: list[tuple[str, str]],  # (raw_text, paper_title)
        source: str = "hf:clouditera/security-paper-datasets",
    ) -> list[dict[str, Any]]:
        """Process multiple papers. Each item is (raw_text, paper_title)."""
        all_records: list[dict[str, Any]] = []
        for raw_text, title in texts:
            records = self.run(raw_text, paper_title=title, source=source)
            all_records.extend(records)
        return all_records


# =============================================================================
# E. HuggingFace Dataset Loader
# =============================================================================


def load_security_papers(
    limit: int | None = None,
) -> list[tuple[str, str]]:
    """Load papers from HuggingFace. Returns list of (text, title).

    The dataset has columns: text, category.
    We derive a title from the first non-empty line of text + the category.
    """
    from datasets import load_dataset

    ds = load_dataset("clouditera/security-paper-datasets", split="train")
    papers: list[tuple[str, str]] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        text = row.get("text") or ""
        category = row.get("category") or ""

        # Derive a title: first meaningful line of text
        title = ""
        for line in text.split("\n"):
            stripped = line.strip()
            if len(stripped) > 10 and not stripped.startswith(("http", "DOI", "©", "ISBN")):
                title = stripped[:120]
                break
        if not title:
            title = f"{category}_{i}" if category else f"paper_{i}"

        papers.append((text, title))
    return papers


# =============================================================================
# F. Diagnostic helpers
# =============================================================================


def compute_pipeline_stats(
    raw_text: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return before/after stats for dashboard display."""
    raw_words = len(raw_text.split()) if raw_text else 0
    cleaned_words = sum(r["metadata"]["word_count"] for r in records)
    all_tags = Counter()
    for r in records:
        for t in r["metadata"]["tags"]:
            all_tags[t] += 1
    return {
        "raw_chars": len(raw_text) if raw_text else 0,
        "raw_words": raw_words,
        "num_chunks": len(records),
        "total_cleaned_words": cleaned_words,
        "avg_chunk_words": round(cleaned_words / len(records), 1) if records else 0,
        "tag_distribution": all_tags.most_common(),
    }
