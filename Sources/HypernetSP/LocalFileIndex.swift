import Foundation
#if canImport(PDFKit)
import PDFKit
#endif

/// A local-file retrieval tier: scans a folder for text/markdown/PDF, splits into chunks, embeds
/// each with BGE, and answers cosine queries. Used as an optional knowledge source for `lookup`
/// turns (Mac primarily, but works on iOS too). Nil-by-default in `Assistant` → zero behaviour change.
public final class LocalFileIndex {
    public struct Chunk { public let text: String; let emb: [Float]; public let source: String }

    private var chunks: [Chunk] = []
    private let bge: Embedder
    public private(set) var indexedFiles: Int = 0

    public init(bge: Embedder) { self.bge = bge }
    public var count: Int { chunks.count }

    private static let exts: Set<String> = ["txt", "md", "markdown", "text", "pdf"]

    /// Recursively index a directory. Replaces any prior index.
    @discardableResult
    public func build(from dir: URL, maxFiles: Int = 300, maxCharsPerFile: Int = 200_000) async -> Int {
        chunks.removeAll(); indexedFiles = 0
        let fm = FileManager.default
        guard let en = fm.enumerator(at: dir, includingPropertiesForKeys: nil) else { return 0 }
        let all = en.allObjects.compactMap { $0 as? URL }
        var files = [URL]()
        for url in all where Self.exts.contains(url.pathExtension.lowercased()) {
            files.append(url); if files.count >= maxFiles { break }
        }
        for url in files {
            guard let raw = extractText(url) else { continue }
            let body = String(raw.prefix(maxCharsPerFile))
            let name = url.lastPathComponent
            for piece in chunkText(body) {
                chunks.append(Chunk(text: piece, emb: await bge.encode(piece, isQuery: false), source: name))
            }
            indexedFiles += 1
        }
        return chunks.count
    }

    /// Top-`k` chunks by cosine ≥ `minSim`, as "filename: text".
    public func search(_ query: String, k: Int = 3, minSim: Float = 0.5) async
        -> [(text: String, source: String, sim: Float)] {
        guard !chunks.isEmpty else { return [] }
        let qv = await bge.encode(query, isQuery: true)
        var scored = chunks.map { (c: $0, s: cosine($0.emb, qv)) }.filter { $0.s >= minSim }
        scored.sort { $0.s > $1.s }
        return scored.prefix(k).map { ($0.c.text, $0.c.source, $0.s) }
    }

    // MARK: - extraction / chunking

    private func extractText(_ url: URL) -> String? {
        if url.pathExtension.lowercased() == "pdf" {
            #if canImport(PDFKit)
            return PDFDocument(url: url)?.string
            #else
            return nil
            #endif
        }
        return try? String(contentsOf: url, encoding: .utf8)
    }

    /// Pack paragraphs into ~`target`-char chunks; hard-split paragraphs longer than that.
    private func chunkText(_ text: String, target: Int = 500) -> [String] {
        let paras = text.components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { $0.count >= 20 }
        var out = [String](); var cur = ""
        func flush() { if !cur.isEmpty { out.append(cur); cur = "" } }
        for p in paras {
            if p.count > target {
                flush()
                var s = Substring(p)
                while s.count > target {
                    let idx = s.index(s.startIndex, offsetBy: target)
                    out.append(String(s[..<idx])); s = s[idx...]
                }
                cur = String(s)
            } else {
                if cur.count + p.count + 1 > target { flush() }
                cur = cur.isEmpty ? p : cur + "\n" + p
            }
        }
        flush()
        return out
    }
}
