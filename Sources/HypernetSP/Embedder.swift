import Foundation
import MLX
import MLXEmbedders
import Tokenizers

/// BGE-small-en-v1.5 sentence embedder (384-d), matching `runtime/rag.BGERetriever`:
/// **raw CLS pooling** (last_hidden_state[:, 0], NOT the BERT dense pooler) + L2 normalize, with the
/// BGE retrieval prefix for queries. Used for intent routing, specificity, and memory retrieval.
public final class Embedder {
    private let container: ModelContainer
    public let dim: Int

    /// `dim` defaults to 384 (BGE-small). Pass 768 for BGE-base (the bridge-head retriever's key tower).
    public static func load(directory: URL, dim: Int = 384) async throws -> Embedder {
        let cfg = ModelConfiguration(directory: directory)
        let container = try await MLXEmbedders.loadModelContainer(configuration: cfg)
        return Embedder(container: container, dim: dim)
    }
    init(container: ModelContainer, dim: Int = 384) { self.container = container; self.dim = dim }

    private static let queryPrefix = "Represent this sentence for searching relevant passages: "

    /// Encode one text → 384-d unit vector. `isQuery` adds the BGE retrieval prefix (for the query side).
    public func encode(_ text: String, isQuery: Bool = false) async -> [Float] {
        let input = isQuery ? Self.queryPrefix + text : text
        return await container.perform { model, tokenizer, _ in
            // BGE uses RAW CLS (last_hidden_state[:,0]) + L2, NOT the BERT dense pooler that the
            // auto-loaded `.cls` pooling would use — so build a `.first` pooler explicitly.
            let pooler = Pooling(strategy: .first)
            var ids = tokenizer.encode(text: input, addSpecialTokens: true)
            if ids.count > 512 { ids = Array(ids.prefix(512)) }   // BGE max length
            let tokens = MLXArray(ids.map { Int32($0) }, [1, ids.count])
            let mask = MLXArray.ones([1, ids.count], dtype: .int32)
            let tt = MLXArray.zeros([1, ids.count], dtype: .int32)
            let out = model(tokens, positionIds: nil, tokenTypeIds: tt, attentionMask: mask)
            let pooled = pooler(out, mask: mask, normalize: true)  // (1, dim) raw CLS, L2-normalized
            eval(pooled)
            return pooled[0].asArray(Float.self)
        }
    }

    /// Batch encode (documents).
    public func encode(_ texts: [String], isQuery: Bool = false) async -> [[Float]] {
        var out = [[Float]]()
        for t in texts { out.append(await encode(t, isQuery: isQuery)) }
        return out
    }
}

/// Cosine similarity of two unit vectors (already L2-normalized) — just the dot product.
public func cosine(_ a: [Float], _ b: [Float]) -> Float {
    var s: Float = 0
    let n = min(a.count, b.count)
    var i = 0
    while i < n { s += a[i] * b[i]; i += 1 }
    return s
}
