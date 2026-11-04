"""
RAG Quality, Information Retrieval & Groundedness Benchmark.

Evaluates:
1. Top-1 Page Accuracy & Top-K Recall
2. Mean Reciprocal Rank (MRR)
3. Distractor Terminology Discrimination
4. Out-of-Domain & Missing Attribute Refusal
5. Grounded Page Citation Verification (Live LLM if key configured, or deterministic IR metrics)
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional

from backend import (
    PDFProcessor,
    TextChunker,
    EmbeddingManager,
    FAISSVectorStore,
    LLMClient,
    verify_citations
)


def create_4page_benchmark_doc() -> bytes:
    """Creates a 4-page technical specification document."""
    doc = fitz.open()

    # Page 1: Overview and Performance SLA
    p1 = doc.new_page()
    p1.insert_text(
        (50, 70),
        "Project Nebula Technical Architecture Specification\n"
        "Document Version: 2.4 | Classification: Internal Engineering\n\n"
        "1. Executive Overview & System Performance\n"
        "Project Nebula is a globally distributed, partition-tolerant stream processing engine.\n"
        "The system achieves a sustained write throughput of 1,200,000 queries per second (QPS) "
        "with an end-to-end 99th-percentile (p99) latency of 4.8 milliseconds.\n"
        "Resource allocation requires minimum 64GB DDR5 memory and 16 NVMe SSD nodes per cluster."
    )

    # Page 2: Storage Engine
    p2 = doc.new_page()
    p2.insert_text(
        (50, 70),
        "2. Storage Subsystem & Data Compaction\n"
        "Nebula replaces standard B-Trees with a custom Log-Structured Merge-tree (LSM) engine.\n"
        "B-Trees are explicitly not supported due to write amplification on NVMe drives.\n"
        "Write amplification is reduced using a tiered Leveled Compaction strategy with Bloom filters.\n"
        "Data blocks are compressed using Zstandard (zstd) at compression level 3, achieving a 3.4x compression ratio."
    )

    # Page 3: Security & Consensus
    p3 = doc.new_page()
    p3.insert_text(
        (50, 70),
        "3. Distributed Consensus & Network Security\n"
        "Leader election and distributed state machine replication rely on an optimized Multi-Raft protocol.\n"
        "Heartbeat intervals are calibrated to 150 milliseconds to ensure rapid failover under 500 milliseconds.\n"
        "All node-to-node transport is encrypted using TLS 1.3 with mutual certificate authentication (mTLS)."
    )

    # Page 4: Disaster Recovery & Operations
    p4 = doc.new_page()
    p4.insert_text(
        (50, 70),
        "4. Operational Runbook & Disaster Recovery\n"
        "Automated snapshot backups are scheduled nightly at 02:00 UTC.\n"
        "The system commits to a Recovery Time Objective (RTO) of under 15 minutes.\n"
        "The Recovery Point Objective (RPO) is strictly bounded to under 30 seconds of data loss.\n"
        "Failover across availability zones triggers automatically upon 3 consecutive missed Raft heartbeats."
    )

    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def run_benchmark():
    print("=" * 80)
    print("  RAG BOT DETERMINISTIC IR & GROUNDEDNESS BENCHMARK")
    print("=" * 80)

    # 1. Ingestion
    print("\n[1/3] Building in-memory FAISS vector index from 4-page test specification...")
    pdf_bytes = create_4page_benchmark_doc()
    pages = PDFProcessor.extract_pages(pdf_bytes)
    chunks = TextChunker.chunk_pages(pages, document_name="nebula_architecture.pdf", document_id="nebula_4p")
    texts = [c.text for c in chunks]
    embeddings = EmbeddingManager.encode_texts(texts)

    store = FAISSVectorStore(dimension=embeddings.shape[1])
    store.add_chunks(chunks, embeddings)
    print(f"  -> Extracted {len(pages)} pages | Created {len(chunks)} chunks | Indexed {store.index.ntotal} vectors\n")

    # 2. Benchmark Test Suite
    test_cases: List[Dict[str, Any]] = [
        {
            "id": "TC-1",
            "category": "Direct Factual Retrieval (Page 1)",
            "query": "What is the sustained write throughput and p99 latency SLA?",
            "expected_page": 1,
            "expected_keywords": ["1,200,000", "4.8", "throughput", "latency"],
            "type": "factual"
        },
        {
            "id": "TC-2",
            "category": "Storage & Compression (Page 2)",
            "query": "What storage engine and compression algorithm are utilized?",
            "expected_page": 2,
            "expected_keywords": ["Log-Structured", "LSM", "Zstandard", "zstd"],
            "type": "factual"
        },
        {
            "id": "TC-3",
            "category": "Consensus & Transport Security (Page 3)",
            "query": "Which consensus protocol and transport encryption are used?",
            "expected_page": 3,
            "expected_keywords": ["Raft", "TLS 1.3", "mTLS"],
            "type": "factual"
        },
        {
            "id": "TC-4",
            "category": "Multi-Page / SLA Recovery Objectives (Page 4)",
            "query": "What are the RTO and RPO recovery objectives in disaster recovery?",
            "expected_page": 4,
            "expected_keywords": ["RTO", "15 minutes", "RPO", "30 seconds"],
            "type": "factual"
        },
        {
            "id": "TC-5",
            "category": "Distractor Terminology Discrimination (Page 2)",
            "query": "Does the system support B-Tree indexing for relational tables?",
            "expected_page": 2,
            "expected_keywords": ["B-Trees", "replaces", "not supported"],
            "type": "factual"
        },
        {
            "id": "TC-6",
            "category": "Out-of-Domain Query (Completely Absent Subject)",
            "query": "What is the market capitalization and stock price of Tesla or Apple?",
            "expected_page": None,
            "expected_keywords": [],
            "type": "out_of_domain"
        },
        {
            "id": "TC-7",
            "category": "Ambiguous / Missing Document Attribute (Entity Match)",
            "query": "Who is the Chief Executive Officer (CEO) and founder of Project Nebula?",
            "expected_page": None,
            "expected_keywords": [],
            "type": "missing_fact"
        }
    ]

    print("[2/3] Executing Benchmark Queries & Measuring IR Metrics (Top-K=4, Threshold=0.20)...")
    score_threshold = 0.20
    top_k = 4

    reciprocal_ranks = []
    top1_hits = 0
    topk_hits = 0
    factual_count = 0
    refusal_count = 0
    expected_refusals = 0

    llm_client = LLMClient()
    is_live_llm = llm_client.is_configured()
    mode_str = f"[LIVE LLM: {llm_client.model}]" if is_live_llm else "[OFFLINE: IR Precision Mode]"
    print(f"  Mode: {mode_str}\n")

    for test in test_cases:
        query = test["query"]
        q_emb = EmbeddingManager.encode_query(query)
        retrieved = store.similarity_search(q_emb, query_text=query, top_k=top_k, score_threshold=score_threshold)

        print(f"[{test['id']}] {test['category']}")
        print(f"  Query: \"{query}\"")

        if test["type"] == "factual":
            factual_count += 1
            if not retrieved:
                print(f"  -> Result: FAIL (No passages retrieved above threshold {score_threshold})\n")
                reciprocal_ranks.append(0.0)
                continue

            # Check ranking of expected page
            retrieved_pages = [r.chunk.page_number for r in retrieved]
            top_page = retrieved[0].chunk.page_number
            top_score = retrieved[0].score

            # Top-1 accuracy
            if top_page == test["expected_page"]:
                top1_hits += 1

            # Top-K recall & Reciprocal Rank
            if test["expected_page"] in retrieved_pages:
                topk_hits += 1
                rank = retrieved_pages.index(test["expected_page"]) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

            # Keyword presence in retrieved chunks
            all_retrieved_text = " ".join([r.chunk.text for r in retrieved]).lower()
            kw_matches = [kw for kw in test["expected_keywords"] if kw.lower() in all_retrieved_text]
            kw_coverage = len(kw_matches) / len(test["expected_keywords"]) if test["expected_keywords"] else 1.0

            print(f"  -> Top-1 Page: {top_page} (Expected: {test['expected_page']}) | Top Score: {top_score:.3f}")
            print(f"  -> Top-K Pages Retrieved: {retrieved_pages}")
            print(f"  -> Keyword Coverage: {len(kw_matches)}/{len(test['expected_keywords'])} ({kw_coverage*100:.0f}%)")

            # If Live LLM available, test generation and citations
            if is_live_llm:
                resp = llm_client.generate_grounded_answer(query, retrieved)
                print(f"  -> LLM Answer: \"{resp.answer[:120]}...\"")
                citations = verify_citations(resp.answer, retrieved)
                print(f"  -> Citation Validation: Valid={citations['is_valid']}, Cited={citations['cited_pages']}")

            status = "PASS" if (test["expected_page"] in retrieved_pages and kw_coverage >= 0.5) else "FAIL"
            print(f"  -> Status: {status}\n")

        elif test["type"] in ("out_of_domain", "missing_fact"):
            expected_refusals += 1
            top_score = retrieved[0].score if retrieved else 0.0
            print(f"  -> Chunks above threshold: {len(retrieved)} (Max Score: {top_score:.3f})")

            if test["type"] == "out_of_domain":
                # For pure out-of-domain query, similarity score should be 0 or below threshold
                if len(retrieved) == 0 or top_score < score_threshold:
                    refusal_count += 1
                    print("  -> Out-of-Domain Refusal: PASS (Retrieval threshold correctly rejected unrelated query)\n")
                else:
                    print(f"  -> Out-of-Domain Refusal: FAIL (Unwarranted high similarity score: {top_score:.3f})\n")

            elif test["type"] == "missing_fact":
                # Entity 'Project Nebula' exists, so chunks will be retrieved, but LLM must reject
                if is_live_llm:
                    resp = llm_client.generate_grounded_answer(query, retrieved)
                    if "does not contain sufficient information" in resp.answer.lower():
                        refusal_count += 1
                        print("  -> Missing-Fact Refusal: PASS (Live LLM correctly refused to hallucinate missing CEO)\n")
                    else:
                        print(f"  -> Missing-Fact Refusal: FAIL (LLM generated unsupported response: {resp.answer})\n")
                else:
                    # In offline mode, verify that passages do NOT contain 'CEO' or 'founder'
                    all_text = " ".join([r.chunk.text for r in retrieved]).lower()
                    if "ceo" not in all_text and "founder" not in all_text:
                        refusal_count += 1
                        print("  -> Context Analysis: PASS (Verified retrieved context contains zero mention of CEO/founder)\n")
                    else:
                        print("  -> Context Analysis: FAIL\n")

    # 3. Quantitative Summary
    mrr = statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    top1_acc = (top1_hits / factual_count) * 100 if factual_count else 0.0
    topk_recall = (topk_hits / factual_count) * 100 if factual_count else 0.0
    refusal_acc = (refusal_count / expected_refusals) * 100 if expected_refusals else 0.0

    print("=" * 80)
    print("  QUANTITATIVE INFORMATION RETRIEVAL (IR) BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"  Total Benchmark Test Cases:       {len(test_cases)}")
    print(f"  Factual Retrieval Queries:        {factual_count}")
    print(f"  Top-1 Page Accuracy (P@1):        {top1_acc:.1f}% ({top1_hits}/{factual_count})")
    print(f"  Top-K Page Recall (Recall@{top_k}):    {topk_recall:.1f}% ({topk_hits}/{factual_count})")
    print(f"  Mean Reciprocal Rank (MRR):       {mrr:.3f}")
    print(f"  Unrelated / Out-of-Domain Refusal: {refusal_acc:.1f}% ({refusal_count}/{expected_refusals})")
    print("=" * 80)

    overall_pass = (topk_recall >= 80.0 and mrr >= 0.80 and refusal_acc >= 80.0)
    print(f"  Benchmark Overall Verdict: {'PASSED (High-Fidelity Retrieval)' if overall_pass else 'FAILED'}\n")
    return overall_pass


if __name__ == "__main__":
    import statistics
    success = run_benchmark()
    sys.exit(0 if success else 1)
