import Foundation
import Tokenizers

/// Block Recall archive — the app holds one of these and uses it identically whether it scores with
/// BGE (Phase A) or the learned ensemble indexer (Phase B). Data structure / injection / wiring are
/// the same; only the scoring differs (BLOCK_RECALL_SPEC §8).
public protocol RecallArchive: AnyObject {
    func reset()
    func sync(absorbed: [Int]) async
    func retrieve(query: String) async -> [Int]
}

/// Swift port of `hypernet_sp/block_recall.BlockArchive` (mode="bge") — Block Recall, Phase A.
///
/// SP (32-vector compression) keeps "what we were talking about" but drops exact values. Block
/// Recall archives the tokens that leave the exposure window into fixed 128-token blocks (each with
/// a BGE document vector), and at turn start re-injects the top-2 query-relevant blocks VERBATIM
/// between the SP and the raw window — so precise values survive SP compression (measured 0/10 → 7/10
/// on a 4.2k-token needle test). The memory layer (L1/L2/routing) is untouched; this is a
/// generation-context layer. OFF (or empty archive) ⇒ bit-identical to the prior behaviour.
public final class BlockArchive: RecallArchive {
    private struct Block { let ids: [Int]; let emb: [Float] }
    private var blocks: [Block] = []
    private var buf: [Int] = []        // evicted tokens not yet a full block
    private var archivedLen = 0        // how many of the absorbed-token stream we've consumed
    private let blockSize = 128

    private let tokenizer: Tokenizer
    private let bge: Embedder

    public init(tokenizer: Tokenizer, bge: Embedder) { self.tokenizer = tokenizer; self.bge = bge }

    public func reset() { blocks = []; buf = []; archivedLen = 0 }

    /// Seal newly-absorbed tokens into 128-blocks. `absorbed` = the full ordered prefix of tokens
    /// that have left the window (append-only; independent of SP mass-eviction).
    public func sync(absorbed: [Int]) async {
        guard absorbed.count > archivedLen else { return }
        buf.append(contentsOf: absorbed[archivedLen...])
        archivedLen = absorbed.count
        while buf.count >= blockSize {
            let ids = Array(buf.prefix(blockSize)); buf.removeFirst(blockSize)
            let text = tokenizer.decode(tokens: ids)
            let emb = await bge.encode(text, isQuery: false)
            blocks.append(Block(ids: ids, emb: emb))
        }
    }

    public func retrieve(query: String) async -> [Int] { await retrieve(query: query, k: 2) }

    /// Top-k blocks for `query`, chronological, concatenated ids (≤256 tok). Includes the pending
    /// buffer as a pseudo-block (≥16 tok) so the just-evicted boundary gap is reachable.
    public func retrieve(query: String, k: Int) async -> [Int] {
        var cands: [(ids: [Int], emb: [Float], order: Int)] = []
        for (i, b) in blocks.enumerated() { cands.append((b.ids, b.emb, i)) }
        if buf.count >= 16 {
            let emb = await bge.encode(tokenizer.decode(tokens: buf), isQuery: false)
            cands.append((buf, emb, blocks.count))
        }
        guard !cands.isEmpty else { return [] }
        let q = await bge.encode(query, isQuery: true)   // BGE query prefix applied inside encode
        let scored = cands.map { (c: $0, s: cosine($0.emb, q)) }
        let top = scored.sorted { $0.s > $1.s }.prefix(k).map { $0.c }.sorted { $0.order < $1.order }
        return top.flatMap { $0.ids }
    }

    public var blockCount: Int { blocks.count }
}
