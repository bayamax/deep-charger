import Foundation
import MLX
import XCTest
import HypernetSP

/// Unit tests for the fidelity helpers: groundedness guard, specificity span enumeration/scoring,
/// answer extraction.
final class RefineTests: XCTestCase {

    static var assets: URL {
        URL(fileURLWithPath: ProcessInfo.processInfo.environment["SP_REPO"]
            ?? "/Users/oobayashikoushin/hypernet-sp-distill").appendingPathComponent("app_assets")
    }

    func testIsGrounded() {
        // a verbatim code present in context → grounded
        XCTAssertTrue(isGrounded(answer: "Your policy number is POL-55821.",
                                 chunks: ["I saved your policy as POL-55821 last week."],
                                 question: "what is my policy number"))
        // answer only echoes the question entity → NOT grounded (no novel salient token)
        XCTAssertFalse(isGrounded(answer: "The CEO of OpenAI is OpenAI itself.",
                                  chunks: ["Sam Altman is the chief executive."],
                                  question: "who is the CEO of OpenAI"))
        // vague / empty → NOT grounded
        XCTAssertFalse(isGrounded(answer: "it is over there.", chunks: ["bay 12"], question: "where"))
    }

    func testExtractAnswer() {
        XCTAssertEqual(extractAnswer("<think>let me see…</think>\n4821"), "4821")
        XCTAssertEqual(extractAnswer("just text"), "just text")
    }

    func testSpecificCandidates() {
        let spans = specificCandidates("Pay $120 a day, locker code QX7-2291, meet Naruhito at 3pm.")
        print("[cands] \(spans)")
        XCTAssertTrue(spans.contains("$120"))
        XCTAssertTrue(spans.contains("QX7-2291"))
        XCTAssertTrue(spans.contains("Naruhito"))
        XCTAssertTrue(spans.contains(where: { $0.contains("3pm") }))
    }

    func testSpecificSpansScored() async throws {
        MLX.Device.setDefault(device: .cpu)
        let bge = try await Embedder.load(directory: Self.assets.appendingPathComponent("bge_small"))
        let spec = try LinearHead.load(Self.assets.appendingPathComponent("specificity_head.json"))
        // build a tiny Assistant-less scorer inline
        func spans(_ t: String) async -> [String] {
            var out = [(String, Float)]()
            for c in specificCandidates(t) {
                let e = await bge.encode(c, isQuery: false)
                out.append((c, spec.scorePositive(e)))
            }
            return out.filter { $0.1 >= 0.6 }.sorted { $0.1 > $1.1 }.map { $0.0 }
        }
        let s1 = await spans("the locker code is QX7-2291")
        print("[spans] \(s1)")
        XCTAssertTrue(s1.contains("QX7-2291"), "specific code should score specific")
    }
}
