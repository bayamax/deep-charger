import Foundation
import MLX
import XCTest
import HypernetSP

/// Live L3 web-retrieval smoke test (network) — DuckDuckGo Instant Answer + Wikipedia JSON.
final class WebSearchTests: XCTestCase {

    static var assets: URL {
        URL(fileURLWithPath: ProcessInfo.processInfo.environment["SP_REPO"]
            ?? "/Users/oobayashikoushin/hypernet-sp-distill").appendingPathComponent("app_assets")
    }

    /// The BGE rerank should surface the CURRENT emperor (Naruhito) above the historical one.
    func testWebRerankCurrentEmperor() async throws {
        MLX.Device.setDefault(device: .cpu)
        let bge = try await Embedder.load(directory: Self.assets.appendingPathComponent("bge_small"))
        let web = WebSearch(n: 4, timeout: 15)
        let q = "Who is the current emperor of Japan?"
        let raw = await web.search(q)
        let ranked = await rerankByCosine(raw, query: q, bge: bge)
        print("[rerank] top: \(ranked.first?.prefix(120) ?? "—")")
        let top2 = ranked.prefix(2).joined(separator: " ").lowercased()
        XCTAssertTrue(top2.contains("naruhito"), "current emperor (Naruhito) should rank into the top 2: \(top2.prefix(140))")
    }
    func testLiveSearch() async throws {
        let web = WebSearch(n: 3, timeout: 15)

        let emperor = await web.search("who is the current emperor of Japan")
        print("[web] emperor → \(emperor.count) snippets")
        for s in emperor.prefix(3) { print("   • \(s.prefix(160))") }
        XCTAssertFalse(emperor.isEmpty, "no results for emperor query")
        let joined = emperor.joined(separator: " ").lowercased()
        XCTAssertTrue(joined.contains("japan") || joined.contains("emperor") || joined.contains("naruhito"),
                      "results not relevant: \(joined.prefix(120))")

        let fuji = await web.search("height of Mount Fuji")
        print("[web] fuji → \(fuji.count) snippets")
        for s in fuji.prefix(2) { print("   • \(s.prefix(160))") }
        XCTAssertFalse(fuji.isEmpty, "no results for Mount Fuji query")
    }
}
