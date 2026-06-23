import Foundation
import MLX
import MLXNN

/// MLX-Swift port of `pooler_mlx.PoolerMLX` (TSE.AttnPoolSP).
///
/// Numerically matches the PyTorch / MLX-Python pooler: a small transformer that
/// compresses a variable-length sequence of token embeddings (`pastEmb`) into
/// `numSoftTokens` soft-prompt vectors. Each block runs cross-attention (soft
/// queries attend to the past), self-attention over the soft queries, and a GELU
/// FFN. `forwardWithMass` additionally returns the per-past-token attention mass
/// (head-averaged, summed over soft queries and layers) used for eviction.
///
/// All math is done in float32 to match the reference, regardless of the model dtype.
public final class Pooler {

    public struct Config: Decodable {
        public let hidden_dim: Int
        public let num_soft_tokens: Int
        public let heads: Int
        public let layers: Int
        public let ffn: Int
        public let out_scale: Float
    }

    public let config: Config
    public var H: Int { config.hidden_dim }
    public var numSoftTokens: Int { config.num_soft_tokens }

    private let A: [String: MLXArray]      // weights (float32)
    private let query: MLXArray            // (n_sp, H)
    private let outScaleAbs: Float
    private let heads: Int
    private let layers: Int

    private var peCache: MLXArray?         // (L, H) sinusoidal positional encodings

    public init(weightsURL: URL, configURL: URL) throws {
        let cfgData = try Data(contentsOf: configURL)
        self.config = try JSONDecoder().decode(Config.self, from: cfgData)

        var arrays = try loadArrays(url: weightsURL)
        for (k, v) in arrays { arrays[k] = v.asType(.float32) }
        self.A = arrays

        guard let q = arrays["query"] else { throw PoolerError.missingKey("query") }
        self.query = q
        self.heads = config.heads
        self.layers = config.layers
        self.outScaleAbs = abs(config.out_scale)
    }

    enum PoolerError: Error { case missingKey(String) }

    // MARK: - primitives

    /// LayerNorm matching the reference `_ln` (eps 1e-5, affine).
    private func ln(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray, eps: Float = 1e-5) -> MLXArray {
        let mu = mean(x, axis: -1, keepDims: true)
        let d = x - mu
        let varr = mean(d * d, axis: -1, keepDims: true)
        return d / sqrt(varr + eps) * w + b
    }

    /// Manual multi-head attention matching torch nn.MultiheadAttention
    /// (packed in_proj QKV, head-averaged attention weights when `wantW`).
    /// Returns (output (B,Lq,H), attnHeadAvg (B,Lq,Lk) or nil).
    private func mha(
        q: MLXArray, k: MLXArray, v: MLXArray, prefix: String, wantW: Bool
    ) -> (MLXArray, MLXArray?) {
        let B = q.dim(0), Lq = q.dim(1)
        let Lk = k.dim(1)
        let hd = H / heads

        let Wi = A[prefix + "in_proj_weight"]!   // (3H, H)
        let bi = A[prefix + "in_proj_bias"]!     // (3H,)
        let Wo = A[prefix + "out_proj.weight"]!  // (H, H)
        let bo = A[prefix + "out_proj.bias"]!    // (H,)

        let qp = matmul(q, Wi[0 ..< H].transposed()) + bi[0 ..< H]
        let kp = matmul(k, Wi[H ..< 2 * H].transposed()) + bi[H ..< 2 * H]
        let vp = matmul(v, Wi[(2 * H) ..< (3 * H)].transposed()) + bi[(2 * H) ..< (3 * H)]

        func split(_ t: MLXArray, _ L: Int) -> MLXArray {
            t.reshaped(B, L, heads, hd).transposed(0, 2, 1, 3) // (B,heads,L,hd)
        }
        let qh = split(qp, Lq), kh = split(kp, Lk), vh = split(vp, Lk)

        let scores = matmul(qh, kh.transposed(0, 1, 3, 2)) / sqrt(Float(hd)) // (B,heads,Lq,Lk)
        let attn = softmax(scores, axis: -1)
        var out = matmul(attn, vh).transposed(0, 2, 1, 3).reshaped(B, Lq, H)
        out = matmul(out, Wo.transposed()) + bo
        let w: MLXArray? = wantW ? mean(attn, axis: 1) : nil  // (B,Lq,Lk)
        return (out, w)
    }

    private func gelu(_ x: MLXArray) -> MLXArray {
        // 0.5 * x * (1 + erf(x / sqrt(2)))  — matches reference _gelu
        0.5 * x * (1.0 + erf(x / Float(2.0).squareRoot()))
    }

    private func sinusoidal(_ L: Int) -> MLXArray {
        let need = max(L, 1024)
        if let pe = peCache, pe.dim(0) >= L {
            return pe[0 ..< L]
        }
        var pe = [Float](repeating: 0, count: need * H)
        let logBase = Foundation.log(10000.0)
        for pos in 0 ..< need {
            for i2 in stride(from: 0, to: H, by: 2) {
                let div = Foundation.exp(-logBase * Double(i2) / Double(H))
                let ang = Double(pos) * div
                pe[pos * H + i2] = Float(Foundation.sin(ang))
                if i2 + 1 < H { pe[pos * H + i2 + 1] = Float(Foundation.cos(ang)) }
            }
        }
        let arr = MLXArray(pe, [need, H])
        peCache = arr
        return arr[0 ..< L]
    }

    private func blockPrefix(_ i: Int, _ which: String) -> String { "blocks.\(i).\(which)." }

    private func lnWB(_ i: Int, _ name: String) -> (MLXArray, MLXArray) {
        (A["blocks.\(i).\(name).weight"]!, A["blocks.\(i).\(name).bias"]!)
    }

    // MARK: - forward

    /// Core run. `pastEmb` is (B, L, H) float32. Returns soft prompts (B, n_sp, H)
    /// and, if `wantMass`, the per-token eviction mass (B, L).
    private func run(_ pastEmb: MLXArray, wantMass: Bool) -> (MLXArray, MLXArray?) {
        let B = pastEmb.dim(0), L = pastEmb.dim(1)
        let past = L > 0 ? pastEmb + sinusoidal(L).expandedDimensions(axis: 0) : pastEmb
        var q = broadcast(query.expandedDimensions(axis: 0), to: [B, numSoftTokens, H])
        var mass: MLXArray? = wantMass ? MLXArray.zeros([B, L]) : nil

        for i in 0 ..< layers {
            let (w1, b1) = lnWB(i, "lnq1")
            let (wk, bk) = lnWB(i, "lnk")
            if L > 0 {
                let kn = ln(past, wk, bk)
                let (a, w) = mha(q: ln(q, w1, b1), k: kn, v: kn,
                                 prefix: blockPrefix(i, "cross"), wantW: wantMass)
                q = q + a
                if wantMass, let w {
                    mass = mass! + sum(w, axis: 1)   // (B,Lq,Lk) -> sum over Lq -> (B,Lk)
                }
            }
            let (w2, b2) = lnWB(i, "lnq2")
            let qn = ln(q, w2, b2)
            let (s, _) = mha(q: qn, k: qn, v: qn, prefix: blockPrefix(i, "selfa"), wantW: false)
            q = q + s
            let (w3, b3) = lnWB(i, "lnq3")
            let h = ln(q, w3, b3)
            var ff = matmul(h, A["blocks.\(i).ffn.0.weight"]!.transposed()) + A["blocks.\(i).ffn.0.bias"]!
            ff = gelu(ff)
            ff = matmul(ff, A["blocks.\(i).ffn.2.weight"]!.transposed()) + A["blocks.\(i).ffn.2.bias"]!
            q = q + ff
        }

        var sp = ln(q, A["ln_out.weight"]!, A["ln_out.bias"]!)
        let norm = sqrt(sum(sp * sp, axis: -1, keepDims: true))
        sp = sp / maximum(norm, MLXArray(Float(1e-6))) * outScaleAbs
        return (sp, mass)
    }

    /// Soft prompts only (B, n_sp, H), in float32.
    public func forward(_ pastEmb: MLXArray) -> MLXArray {
        run(pastEmb, wantMass: false).0
    }

    /// Soft prompts (B, n_sp, H) and per-token eviction mass (B, L), in float32.
    public func forwardWithMass(_ pastEmb: MLXArray) -> (MLXArray, MLXArray) {
        let (sp, mass) = run(pastEmb, wantMass: true)
        return (sp, mass!)
    }
}
