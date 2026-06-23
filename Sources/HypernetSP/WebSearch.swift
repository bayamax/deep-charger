import Foundation

/// L3 web retrieval (keyless, JSON-only — no HTML scraping, robust on iOS). Swift port of the
/// Wikipedia backend in `runtime/web_search.py`, plus the DuckDuckGo Instant-Answer API for a quick
/// abstract. Returns short text snippets to inject into a `lookup` (world-question) turn.
public struct WebSearch: Sendable {
    public let n: Int
    private let session: URLSession
    private static let ua = "sp-distill-ios/1.0 (https://huggingface.co/baya1116/hypernet-sp-distill)"

    public init(n: Int = 3, timeout: TimeInterval = 10) {
        self.n = n
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = timeout
        // hard ceiling on the WHOLE operation (the Wikipedia path issues several sequential
        // requests; without this a stalled chain could outlive any single per-request timeout)
        cfg.timeoutIntervalForResource = timeout * 2
        cfg.waitsForConnectivity = false
        cfg.httpAdditionalHeaders = ["User-Agent": Self.ua, "Accept-Language": "en-US,en;q=0.9"]
        self.session = URLSession(configuration: cfg)
    }

    /// Search the web: DuckDuckGo Instant Answer first; **Wikipedia only as a fallback** when the
    /// instant-answer layer found nothing. Wikipedia is an encyclopedia — for time-sensitive queries
    /// ("today's oil price") it returns tangential articles ("1973 oil crisis") that pollute the
    /// result, so it must not be a co-equal source.
    public func search(_ query: String) async -> [String] {
        var out = [String]()
        if let ddg = await duckDuckGoInstant(query), ddg.count > 30 { out.append(ddg) }
        if out.isEmpty {
            out.append(contentsOf: await wikipedia(query, limit: n))
        }
        // de-dup near-identical snippets
        var seen = Set<String>(); var uniq = [String]()
        for s in out {
            let key = String(s.prefix(60)).lowercased()
            if seen.insert(key).inserted { uniq.append(s) }
        }
        return uniq
    }

    private func getJSON(_ url: URL) async -> Any? {
        guard let (data, _) = try? await session.data(from: url) else { return nil }
        return try? JSONSerialization.jsonObject(with: data)
    }

    // MARK: DuckDuckGo Instant Answer (JSON)

    private func duckDuckGoInstant(_ q: String) async -> String? {
        var c = URLComponents(string: "https://api.duckduckgo.com/")!
        c.queryItems = [
            .init(name: "q", value: q), .init(name: "format", value: "json"),
            .init(name: "no_html", value: "1"), .init(name: "skip_disambig", value: "1"),
        ]
        guard let url = c.url, let obj = await getJSON(url) as? [String: Any] else { return nil }
        let heading = (obj["Heading"] as? String) ?? ""
        if let abstract = obj["AbstractText"] as? String, !abstract.isEmpty {
            return heading.isEmpty ? abstract : "\(heading): \(abstract)"
        }
        if let answer = obj["Answer"] as? String, !answer.isEmpty {
            return answer
        }
        if let topics = obj["RelatedTopics"] as? [[String: Any]] {
            for t in topics { if let txt = t["Text"] as? String, txt.count > 30 { return txt } }
        }
        return nil
    }

    // MARK: rerank

    // MARK: Wikipedia (search → summary)

    private func wikipedia(_ q: String, limit: Int) async -> [String] {
        var sc = URLComponents(string: "https://en.wikipedia.org/w/api.php")!
        sc.queryItems = [
            .init(name: "action", value: "query"), .init(name: "list", value: "search"),
            .init(name: "srsearch", value: q), .init(name: "srlimit", value: String(limit)),
            .init(name: "format", value: "json"),
        ]
        guard let surl = sc.url, let obj = await getJSON(surl) as? [String: Any],
            let query = obj["query"] as? [String: Any],
            let results = query["search"] as? [[String: Any]]
        else { return [] }

        let titles = results.compactMap { $0["title"] as? String }
        var chunks = [String]()
        for t in titles {
            let safe = t.replacingOccurrences(of: " ", with: "_")
                .addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? t
            guard let url = URL(string: "https://en.wikipedia.org/api/rest_v1/page/summary/\(safe)"),
                let d = await getJSON(url) as? [String: Any],
                let extract = d["extract"] as? String, extract.count > 50
            else { continue }
            chunks.append("\(t): \(extract)")
        }
        return chunks
    }
}

/// Re-rank text chunks by BGE cosine similarity to the query (descending). Used so the most
/// query-relevant web snippet (e.g. "current emperor → Naruhito") ranks above a merely on-topic but
/// off-target one ("Hirohito"). Purely reorders — adds no new chunks.
public func rerankByCosine(_ chunks: [String], query: String, bge: Embedder) async -> [String] {
    guard chunks.count > 1 else { return chunks }
    let qv = await bge.encode(query, isQuery: true)
    var scored = [(c: String, s: Float)]()
    for c in chunks { scored.append((c, cosine(await bge.encode(c, isQuery: false), qv))) }
    return scored.sorted { $0.s > $1.s }.map { $0.c }
}
