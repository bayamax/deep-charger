import Foundation
import MLX
import XCTest
import HypernetSP

/// Validates the canonical-module ports (intent_route 3-band, calculator, decode_policy) against
/// reference values produced by the canonical Python (`app_assets/fixtures/canon_ref.json`).
final class CanonTests: XCTestCase {
    static var assets: URL {
        URL(fileURLWithPath: ProcessInfo.processInfo.environment["SP_REPO"]
            ?? "/Users/oobayashikoushin/hypernet-sp-distill").appendingPathComponent("app_assets")
    }

    func ref() throws -> [String: Any] {
        try JSONSerialization.jsonObject(
            with: Data(contentsOf: Self.assets.appendingPathComponent("fixtures/canon_ref.json")))
            as! [String: Any]
    }

    func testIntentRoute3Band() async throws {
        MLX.Device.setDefault(device: .cpu)
        let bge = try await Embedder.load(directory: Self.assets.appendingPathComponent("bge_small"))
        let intent = try LinearHead.load(Self.assets.appendingPathComponent("intent_head.json"))
        let routes = try ref()["routes"] as! [String: String]
        var hits = 0
        for (text, want) in routes {
            let doc = await bge.encode(text, isQuery: false)
            let got = routeIntent(text, intentHead: intent, docEmbedding: doc)
            print("[route] \"\(text)\" → \(got) (want \(want))\(got == want ? "" : "  ❌")")
            if got == want { hits += 1 }
        }
        XCTAssertEqual(hits, routes.count, "intent routing mismatch (\(hits)/\(routes.count))")
    }

    func testCalculator() throws {
        let cases = try ref()["calc"] as! [[String: Any]]
        for c in cases {
            let ans = c["ans"] as! String, body = c["body"] as! String
            let wantFixed = c["fixed"] as! String, wantN = c["n_corr"] as! Int
            let (fixed, corr) = repairAnswer(ans, fullBody: body)
            print("[calc] \"\(ans)\" → \"\(fixed)\" (corr=\(corr.count), want \"\(wantFixed)\"/\(wantN))")
            XCTAssertEqual(corr.count, wantN, "correction count for \(ans)")
            XCTAssertEqual(fixed, wantFixed, "repaired answer for \(ans)")
        }
    }

    func testDecodePolicyAsserts() throws {
        let dp = try ref()["decode"] as! [String: Any]
        let wantVals = dp["assert_values"] as! [String]
        let text = "We compute 2000*1.05 = 2100. So the total is 2315. Hmm, the total is 2315. Actually the total is 2315 again."
        XCTAssertEqual(assertValues(text), wantVals)
        XCTAssertEqual(canonNum("1,157.00"), dp["canon_1157"] as! String)
    }
}
