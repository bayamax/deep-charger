import Foundation
import MLX
import XCTest
import HypernetSP

/// Validates the ported BGE embedder + intent/specificity heads against the Python reference
/// (`app_assets/fixtures/route_ref.json`). Runs on macOS via xcodebuild.
final class RouteTests: XCTestCase {

    static var repo: URL {
        URL(fileURLWithPath: ProcessInfo.processInfo.environment["SP_REPO"]
            ?? "/Users/oobayashikoushin/hypernet-sp-distill")
    }
    static var assets: URL { repo.appendingPathComponent("app_assets") }

    func testRoutingAndRetrieval() async throws {
        MLX.Device.setDefault(device: .cpu)
        let a = Self.assets
        let bge = try await Embedder.load(directory: a.appendingPathComponent("bge_small"))
        let intent = try LinearHead.load(a.appendingPathComponent("intent_head.json"))
        let spec = try LinearHead.load(a.appendingPathComponent("specificity_head.json"))

        let ref = try JSONSerialization.jsonObject(
            with: Data(contentsOf: a.appendingPathComponent("fixtures/route_ref.json")))
            as! [String: Any]

        var labelHits = 0
        let sentences = ref["sentences"] as! [[String: Any]]
        for s in sentences {
            let text = s["text"] as! String
            let e = await bge.encode(text, isQuery: false)
            let (lab, prob, _) = intent.classify(e)
            let pspec = spec.scorePositive(e)
            let refIntent = s["intent"] as! String
            let refPspec = Float((s["pspec"] as! NSNumber).doubleValue)
            print("[route] \"\(text)\" → \(lab) (p=\(String(format: "%.2f", prob))) | pspec=\(String(format: "%.2f", pspec)) (ref \(refIntent)/\(refPspec))")
            if lab == refIntent { labelHits += 1 }
            XCTAssertEqual(pspec, refPspec, accuracy: 0.10, "specificity prob diverged for \(text)")
        }
        XCTAssertEqual(labelHits, sentences.count, "intent routing mismatch (\(labelHits)/\(sentences.count))")

        // retrieval: the matching doc must rank first and roughly match reference cosines
        let r = ref["retrieval"] as! [String: Any]
        let query = r["query"] as! String
        let docs = r["docs"] as! [String]
        let refSims = (r["sims"] as! [NSNumber]).map { Float($0.doubleValue) }
        let qv = await bge.encode(query, isQuery: true)
        let dv = await bge.encode(docs, isQuery: false)
        var sims = [Float]()
        for d in dv { sims.append(cosine(d, qv)) }
        print("[retrieval] q=\"\(query)\" sims=\(sims.map { String(format: "%.3f", $0) }) ref=\(refSims)")
        XCTAssertEqual(sims.firstIndex(of: sims.max()!), 0, "top retrieval doc should be the matching one")
        for i in sims.indices {
            XCTAssertEqual(sims[i], refSims[i], accuracy: 0.04, "cosine diverged for doc \(i)")
        }
    }
}
