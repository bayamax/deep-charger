import Foundation
import MLX
import XCTest
import HypernetSP

/// Validates the local-file retrieval tier: index a temp folder of notes, then retrieve a fact by
/// semantic query.
final class LocalFileTests: XCTestCase {
    static var assets: URL {
        URL(fileURLWithPath: ProcessInfo.processInfo.environment["SP_REPO"]
            ?? "/Users/oobayashikoushin/hypernet-sp-distill").appendingPathComponent("app_assets")
    }

    func testIndexAndSearch() async throws {
        MLX.Device.setDefault(device: .cpu)
        let bge = try await Embedder.load(directory: Self.assets.appendingPathComponent("bge_small"))

        // temp folder with two notes
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("sp_localtest_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try "Project Atlas notes.\n\nThe production deployment key is DPLOY-9931. Keep it secret.\n"
            .write(to: dir.appendingPathComponent("atlas.md"), atomically: true, encoding: .utf8)
        try "Grocery list: milk, eggs, bread. The cake should be chocolate.\n"
            .write(to: dir.appendingPathComponent("home.txt"), atomically: true, encoding: .utf8)

        let index = LocalFileIndex(bge: bge)
        let n = await index.build(from: dir)
        print("[localfiles] indexed \(index.indexedFiles) files, \(n) chunks")
        XCTAssertGreaterThan(n, 0)

        let hits = await index.search("what is the deployment key for production?", k: 2, minSim: 0.4)
        print("[localfiles] hits: \(hits.map { "\($0.source)(\(String(format: "%.2f", $0.sim)))" })")
        XCTAssertFalse(hits.isEmpty, "expected a local-file hit")
        XCTAssertTrue(hits.contains { $0.text.contains("DPLOY-9931") }, "deployment key not retrieved")
        XCTAssertEqual(hits.first?.source, "atlas.md")
    }
}
