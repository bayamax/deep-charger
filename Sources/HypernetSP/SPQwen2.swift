import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// Vendored from mlx-swift-examples MLXLLM/Models/Qwen2.swift (tag 2.29.1), modified so the
// model can run a forward pass from injected input embeddings (the soft prompts + raw window
// produced by the SP-evict loop) and expose the token embedding lookup. Behaviour is otherwise
// identical to the stock Qwen2 (same RoPE / causal-mask / KV-cache semantics), so it matches the
// Python reference (`tiered_rag_mlx` / `sp_mlx`) which feeds `input_embeddings` to mlx-lm's Qwen2.
//
// Uses a local public config (MLXLLM's `SPQwen2Configuration` fields are `internal`).

/// Qwen2 configuration (public fields), decoded from the model's `config.json`.
public struct SPQwen2Configuration: Codable, Sendable {
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var rmsNormEps: Float
    public var vocabularySize: Int
    public var kvHeads: Int
    public var ropeTheta: Float
    public var ropeTraditional: Bool
    public var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

private class SPAttention: Module {
    let args: SPQwen2Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    init(_ args: SPQwen2Configuration) {
        self.args = args
        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads
        let headDim = args.hiddenSize / heads
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        self.rope = RoPE(
            dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta, scale: 1)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return wo(output)
    }

    /// Eager attention (explicit softmax(QK^T)) over a single full sequence — no KV cache. Returns
    /// the block output AND the per-head attention probabilities (B, heads, L, L), needed by the
    /// retrieval trigger (which reads attention mass from turn tokens onto the SP positions).
    func eager(_ x: MLXArray) -> (out: MLXArray, attn: MLXArray) {
        let (B, L) = (x.dim(0), x.dim(1))
        let hd = args.hiddenSize / args.attentionHeads
        var q = wq(x).reshaped(B, L, args.attentionHeads, hd).transposed(0, 2, 1, 3)
        var k = wk(x).reshaped(B, L, args.kvHeads, hd).transposed(0, 2, 1, 3)
        var v = wv(x).reshaped(B, L, args.kvHeads, hd).transposed(0, 2, 1, 3)
        q = rope(q); k = rope(k)
        // expand grouped KV heads to all query heads (repeat_interleave)
        let reps = args.attentionHeads / args.kvHeads
        if reps > 1 {
            k = broadcast(k.reshaped(B, args.kvHeads, 1, L, hd), to: [B, args.kvHeads, reps, L, hd])
                .reshaped(B, args.attentionHeads, L, hd)
            v = broadcast(v.reshaped(B, args.kvHeads, 1, L, hd), to: [B, args.kvHeads, reps, L, hd])
                .reshaped(B, args.attentionHeads, L, hd)
        }
        // attention math in float32 — fp16 would overflow the -inf mask to NaN (4-bit model runs fp16)
        let inDType = x.dtype
        var scores = matmul(q.asType(.float32), k.asType(.float32).transposed(0, 1, 3, 2)) * scale
        let idx = MLXArray((0 ..< L).map { Int32($0) })
        let causal = idx.reshaped([L, 1]) .>= idx.reshaped([1, L])   // true where key ≤ query
        scores = MLX.where(causal, scores, MLXArray(Float(-1e9)))
        let attn = softmax(scores, axis: -1)                          // (B, heads, L, L) float32
        var out = matmul(attn, v.asType(.float32)).transposed(0, 2, 1, 3)
            .reshaped(B, L, args.attentionHeads * hd).asType(inDType)
        out = wo(out)
        return (out, attn)
    }

    /// Pre-RoPE q_proj / k_proj output for `x` (already input-layernormed). (B, L, Hq*D) or (B, L, Hkv*D).
    func proj(_ x: MLXArray, wantQ: Bool) -> MLXArray { wantQ ? wq(x) : wk(x) }
}

private class SPMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

private class SPTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: SPAttention
    let mlp: SPMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: SPQwen2Configuration) {
        _attention.wrappedValue = SPAttention(args)
        self.mlp = SPMLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }

    /// Eager forward returning the block output and this layer's attention probabilities.
    func eager(_ x: MLXArray) -> (out: MLXArray, attn: MLXArray) {
        let (att, weights) = attention.eager(inputLayerNorm(x))
        let h = x + att
        let r = mlp(postAttentionLayerNorm(h))
        return (h + r, weights)
    }

    /// Pre-RoPE q_proj/k_proj output for this layer's input `x` (input_layernorm applied inside).
    func projInput(_ x: MLXArray, wantQ: Bool) -> MLXArray { attention.proj(inputLayerNorm(x), wantQ: wantQ) }
}

private class SPQwen2ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [SPTransformerBlock]
    let norm: RMSNorm

    init(_ args: SPQwen2Configuration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { _ in SPTransformerBlock(args) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    /// Forward from token ids OR from precomputed input embeddings.
    func callAsFunction(
        _ inputs: MLXArray?, inputEmbeddings: MLXArray? = nil, cache: [KVCache]? = nil
    ) -> MLXArray {
        var h = inputEmbeddings ?? embedTokens(inputs!)
        let mask = createAttentionMask(h: h, cache: cache)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }

    /// Run the forward (advancing `cache`), capturing each requested layer's pre-RoPE q_proj/k_proj
    /// output for its INPUT. Used by Phase-B Block Recall (contextual key/query features).
    func captureProj(_ inputEmbeddings: MLXArray, cache: [KVCache], captureLayers: Set<Int>, wantQ: Bool)
        -> [Int: MLXArray]
    {
        var h = inputEmbeddings
        let mask = createAttentionMask(h: h, cache: cache)
        var out = [Int: MLXArray]()
        for (i, layer) in layers.enumerated() {
            if captureLayers.contains(i) { let p = layer.projInput(h, wantQ: wantQ); eval(p); out[i] = p }
            h = layer(h, mask: mask, cache: cache[i])
        }
        return out
    }

    /// One forward returning BOTH the normed hidden state (for logits) AND the captured pre-RoPE q at
    /// `captureLayers` — for the in-generation recall gate (RECALL_V2 §3): one prefill per chunk yields
    /// both the next-token logits and the gate q-feature, no second pass.
    func callCapturing(_ inputEmbeddings: MLXArray, cache: [KVCache], captureLayers: Set<Int>)
        -> (hidden: MLXArray, q: [Int: MLXArray])
    {
        var h = inputEmbeddings
        let mask = createAttentionMask(h: h, cache: cache)
        var out = [Int: MLXArray]()
        for (i, layer) in layers.enumerated() {
            if captureLayers.contains(i) { let p = layer.projInput(h, wantQ: true); out[i] = p }
            h = layer(h, mask: mask, cache: cache[i])
        }
        return (norm(h), out)
    }
}

/// Qwen2 with input-embedding injection for the SP-evict generation loop.
public class SPQwen2Model: Module, LanguageModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let model: SPQwen2ModelInner
    let configuration: SPQwen2Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: SPQwen2Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = SPQwen2ModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    private func project(_ hidden: MLXArray) -> MLXArray {
        if let lmHead { return lmHead(hidden) }
        return model.embedTokens.asLinear(hidden)
    }

    /// Standard token-id forward (LanguageModel conformance).
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        project(model(inputs, cache: cache))
    }

    /// Forward from injected input embeddings (B, L, H) → logits (B, L, vocab).
    public func callAsFunction(inputEmbeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        project(model(nil, inputEmbeddings: inputEmbeddings, cache: cache))
    }

    /// Token embedding lookup (B, L) → (B, L, H), matching `model.model.embed_tokens`.
    public func embed(_ ids: MLXArray) -> MLXArray {
        model.embedTokens(ids)
    }

    /// Phase-B feature capture: forward `inputEmbeddings` with `cache` behind it, returning the
    /// pre-RoPE q_proj/k_proj output at each `layers` index (input_layernorm applied inside).
    public func captureProj(inputEmbeddings: MLXArray, cache: [KVCache], layers: [Int], wantQ: Bool)
        -> [Int: MLXArray]
    {
        model.captureProj(inputEmbeddings, cache: cache, captureLayers: Set(layers), wantQ: wantQ)
    }

    /// In-generation recall: one prefill → (next-token logits, pre-RoPE q at `layers`).
    public func callCapturing(inputEmbeddings: MLXArray, cache: [KVCache], layers: [Int])
        -> (logits: MLXArray, q: [Int: MLXArray])
    {
        let (hidden, q) = model.callCapturing(inputEmbeddings, cache: cache, captureLayers: Set(layers))
        return (project(hidden), q)
    }

    /// Retrieval-trigger features: for each `(layer, head)` pair, the mean attention mass that the
    /// turn tokens (positions ≥ `1 + nSP`) place on the SP positions (`1 ..< 1+nSP`). Runs eager
    /// attention only through `max(layer)` of 28 — one short forward, no generation, no lm_head.
    public func triggerMasses(inputEmbeddings: MLXArray, heads: [(layer: Int, head: Int)], nSP: Int) -> [Float] {
        let maxLayer = heads.map { $0.layer }.max() ?? 0
        let needed = Set(heads.map { $0.layer })
        var h = inputEmbeddings
        var attnByLayer = [Int: MLXArray]()
        for i in 0 ... maxLayer {
            let (out, attn) = model.layers[i].eager(h)
            h = out
            if needed.contains(i) { eval(attn); attnByLayer[i] = attn }
        }
        let off = 1 + nSP
        var masses = [Float]()
        for (l, hd) in heads {
            let A = attnByLayer[l]![0, hd]                  // (L, L)
            let block = A[off..., 1 ..< (1 + nSP)]          // turn rows × SP cols
            masses.append(MLX.mean(MLX.sum(block, axis: -1)).item(Float.self))
        }
        return masses
    }

    /// Minimal `LanguageModel` conformance — unused (the SP-evict loop drives prefill itself).
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        .tokens(input.text)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }
        return weights.filter { !$0.key.contains("self_attn.rotary_emb.inv_freq") }
    }
}
