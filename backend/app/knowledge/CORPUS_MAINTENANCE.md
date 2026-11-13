# 📚 Geopolitical Knowledge Corpus Maintenance & Dispute Resolution

## 1. Overview
The GeoPol Bot Knowledge Pipeline is built on a **verifiable primary source doctrine**. All evidence retrieved by the bot must originate from permitted, official, and reputable geopolitical repositories (multilateral treaties, UN resolutions, government legislative acts, official joint declarations).

This document outlines protocols for updating the corpus, handling conflicting or superseded evidence, and preventing hallucinated citations.

---

## 2. Permitted Source Repositories & Ingestion Standards

All documents ingested into GeoPol Bot must satisfy the following provenance criteria:
1. **Verifiable Authority**: Originating from recognized international organizations (UN, NATO, ICJ), sovereign treaty depositaries, or official legislative repositories.
2. **Immutable Provenance Fields**:
   - `id`: Unique kebab-case identifier (e.g., `nato-founding-treaty-1949`).
   - `title`: Official treaty or resolution name.
   - `source_name`: Issuing body or depositary.
   - `source_url`: Permanent direct URL to the official text.
   - `publication_date`: ISO 8601 date (`YYYY-MM-DD`).
   - `topic`: Standardized geopolitical category.
3. **No Secondary Commentary Ingestion**: Primary texts, official treaties, and resolutions are prioritized over speculative journalistic op-eds to preserve factual integrity.

---

## 3. How to Update the Knowledge Corpus

### Adding or Updating Documents
1. **Append to Corpus Data**: Add new JSON entries to `backend/app/knowledge/corpus/corpus.json`.
2. **Run Ingestion Pipeline**:
   ```bash
   python -m backend.app.knowledge.ingest
   ```
   The ingestion script:
   - Chunks the document preserving title, source URL, date, and chunk index.
   - Computes normalized vector embeddings (`text-embedding-3-small`).
   - Upserts into `data/geopol_vectors.db` using `ON CONFLICT(chunk_id) DO UPDATE` to ensure idempotent re-indexing.

---

## 4. Handling Outdated, Amending, and Superseded Evidence

Geopolitical agreements evolve over time (e.g., treaties are amended, sanctions are lifted or snapped back, alliances admit new members).

### Strategies Implemented:
1. **Temporal Metadata Preservation**: Every chunk retains its exact `publication_date`. Downstream generation engines receive dates for each chunk to distinguish historic baselines from subsequent amendments.
2. **Supersession & Sunset Protocols**:
   - When a treaty or resolution is formally superseded (e.g., a newer UNSC resolution revoking sanctions), the new document is ingested with an explicit reference to the superseded resolution.
   - Chunks retain a status indicator if marked historical vs. active.
3. **Snapback & Reversion Handling**:
   - For agreements with contingency clauses (such as UNSC Res 2231's snapback mechanism), evidence chunks from both the original resolution and subsequent notifications are preserved, allowing the generation model to present conditional legal statuses accurately.

---

## 5. Handling Conflicting Geopolitical Claims

International relations frequently feature irreconcilable sovereign claims (e.g., maritime boundary disputes, territorial sovereignty in the Arctic or South China Sea).

### Handling Rules:
1. **Multi-Perspective Attribution**: The pipeline ingests the competing official declarations (e.g., UNCLOS provisions alongside specific state reservations or declarations).
2. **No Editorial Adjudication**: GeoPol Bot does not invent a compromise; it retrieves evidence from both recognized frameworks and explicitly presents the legal arguments of the contending parties, labeled with their respective official sources.
3. **Citation Inviolability**: The generation model is forbidden from inventing citations. Citations are verified post-generation against the retrieved chunk URLs.
