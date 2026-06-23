import Foundation
import MLX
import XCTest

import HypernetSP

/// Headless validation harness — runs the faithful SP-evict pipeline on hardcoded prompts and
/// checks it against the Python reference fixtures. Runs on the iOS Simulator (device-equivalent)
/// via `xcodebuild test`. The model + fixtures are read from the repo by absolute path; override
/// with the SP_REPO environment variable.
final class SPModelTests: XCTestCase {

    static var repo: URL {
        let p = ProcessInfo.processInfo.environment["SP_REPO"]
            ?? "/Users/oobayashikoushin/hypernet-sp-distill"
        return URL(fileURLWithPath: p)
    }
    static var assets: URL { repo.appendingPathComponent("app_assets") }
    static var fixtures: URL { assets.appendingPathComponent("fixtures") }

    static var paths: SPModel.Paths {
        SPModel.Paths(
            modelDir: repo.appendingPathComponent("fft_mlx4"),
            poolerWeights: assets.appendingPathComponent("pooler.safetensors"),
            poolerConfig: assets.appendingPathComponent("pooler_config.json"))
    }

    // load once, share across tests
    nonisolated(unsafe) static var shared: SPModel!

    override class func setUp() {
        super.setUp()
        // The iOS Simulator cannot init MLX's Metal GPU — run validation on CPU (same numerics).
        MLX.Device.setDefault(device: .cpu)
        let exp = XCTestExpectation(description: "load")
        Task {
            do { shared = try await SPModel.load(paths) } catch { XCTFail("load failed: \(error)") }
            exp.fulfill()
        }
        _ = XCTWaiter().wait(for: [exp], timeout: 600)
    }

    func jsonFixture(_ name: String) throws -> [String: Any] {
        let data = try Data(contentsOf: Self.fixtures.appendingPathComponent(name))
        return try JSONSerialization.jsonObject(with: data) as! [String: Any]
    }

    // MARK: 1. tokenizer parity (BOS + special tokens must match the Python q_ids)

    func test1TokenizerParity() throws {
        let ref = try jsonFixture("greedy_ref.json")
        let refQ = (ref["q_ids"] as! [Int])
        let prompt = ref["prompt"] as! String
        let text = "<｜User｜>\(prompt)<｜Assistant｜>"
        let q = Self.shared.tokenizer.encode(text: text, addSpecialTokens: true)
        print("q_ids swift: \(q)")
        print("q_ids ref  : \(refQ)")
        XCTAssertEqual(q, refQ, "tokenizer encoding diverged from Python reference")
    }

    // MARK: 2. pooler numerics (SP norm == out_scale; eviction-mass argmax matches)

    func test2PoolerNumerics() throws {
        let idsData = try Data(contentsOf: Self.fixtures.appendingPathComponent("ids.json"))
        let ids = try JSONSerialization.jsonObject(with: idsData) as! [Int]

        let emb = Self.shared.model.embed(MLXArray(ids.map { Int32($0) }, [1, ids.count]))
            .asType(.float32)
        let sp = Self.shared.pooler.forward(emb)
        eval(sp)
        let sp0 = sp[0, 0, 0...].asArray(Float.self)
        let norm0 = sqrt(sp0.reduce(0) { $0 + $1 * $1 })
        let outScale = Self.shared.pooler.config.out_scale
        print("SP norm0 swift=\(norm0)  out_scale=\(outScale)")
        XCTAssertEqual(norm0, abs(outScale), accuracy: 1e-2, "SP vector not normalized to out_scale")

        let (_, mass) = Self.shared.pooler.forwardWithMass(emb)
        let m = mass[0].asArray(Float.self)
        let argmax = m.firstIndex(of: m.max()!)!
        let massRef = try jsonFixture("mass_ref.json")
        let refArgmax = massRef["argmax"] as! Int
        print("mass argmax swift=\(argmax) ref=\(refArgmax)")
        XCTAssertEqual(argmax, refArgmax, "eviction-mass argmax diverged")
    }

    // MARK: 3. end-to-end greedy generation must reproduce the Python token stream

    func test3GreedyGenerationMatchesReference() throws {
        let ref = try jsonFixture("greedy_ref.json")
        let prompt = ref["prompt"] as! String
        let refGen = ref["gen_ids"] as! [Int]

        var opts = SPModel.Options()
        opts.chat = true
        opts.greedy = true
        opts.rw = ref["rw"] as! Int
        opts.chunk = ref["C"] as! Int
        opts.maxD = ref["maxD"] as! Int
        opts.genLen = ref["gen_len"] as! Int

        let r = Self.shared.generate(prompt: prompt, options: opts)
        print("gen swift (\(r.tokens.count)): \(r.tokens.prefix(20))")
        print("gen ref   (\(refGen.count)): \(refGen.prefix(20))")
        print("text: \(r.text.prefix(160))")

        // exact match expected with greedy decoding + identical numerics
        let n = min(r.tokens.count, refGen.count)
        var firstDiff = -1
        for i in 0 ..< n where r.tokens[i] != refGen[i] { firstDiff = i; break }
        XCTAssertEqual(firstDiff, -1, "token stream diverged at index \(firstDiff)")
        XCTAssertEqual(r.tokens.count, refGen.count, "length differs")
    }

    // MARK: 4. chat smoke test (sampling) — logs an answer, checks it terminates

    func test4ChatBatterySmoke() throws {
        var opts = SPModel.Options()
        opts.chat = true
        opts.temp = 0.6
        opts.rw = 1024
        opts.genLen = 80
        opts.seed = 1234
        let r = Self.shared.generate(prompt: "What is 17 multiplied by 24?", options: opts)
        print("[smoke] len=\(r.tokens.count) eos=\(r.eosHit) tok/s=\(String(format: "%.1f", r.tokensPerSecond)) evicts=\(r.evicts)")
        print("[smoke] tail: \(r.text.suffix(200))")
        XCTAssertGreaterThan(r.tokens.count, 0)
    }
}
