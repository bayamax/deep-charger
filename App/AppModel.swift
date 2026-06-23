import Foundation
import MLX
import SwiftUI
import HypernetSP
#if canImport(UIKit)
import UIKit
import os
#endif

/// Megabytes available before the OS kills us. iOS: jetsam headroom; macOS has no such limit, so
/// report total physical RAM as a (large) proxy.
func availableMemoryMB() -> Int {
    #if os(iOS)
    return Int(os_proc_available_memory() / (1024 * 1024))
    #else
    return Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024))
    #endif
}

/// Thread-safe flag read from the generation background thread.
final class AtomicBool: @unchecked Sendable {
    private var v: Bool
    private let lock = NSLock()
    init(_ x: Bool) { v = x }
    var value: Bool { lock.lock(); defer { lock.unlock() }; return v }
    func set(_ x: Bool) { lock.lock(); v = x; lock.unlock() }
}

/// One line in the chat transcript.
struct ChatMessage: Identifiable, Equatable {
    enum Role { case user, assistant, system }
    let id = UUID()
    let role: Role
    var text: String
    var source: String? = nil      // tier label ("L3·web", "WM·pins", …)
    var intent: String? = nil      // routed intent
    var meta: String? = nil        // "12 tok · 8.3 tok/s"
    var pending: Bool = false       // assistant is still generating
}

/// Drives the composite on-device assistant: BGE routing + tiered memory + bounded SP-evict chat.
/// On a physical iPhone MLX uses the Metal GPU automatically.
@MainActor
final class AppModel: ObservableObject {
    enum Phase: Equatable { case idle, loading, ready, error(String) }

    @Published var phase: Phase = .idle
    @Published var deviceName: String = "—"
    @Published var log: String = ""
    @Published var busy = false
    @Published var input: String = ""
    @Published var messages: [ChatMessage] = []
    /// The user's chosen spoken language (BCP-47, e.g. "ja"/"en"/"zh-Hans"). Empty = not yet chosen →
    /// the onboarding language picker is shown. The model always runs in ENGLISH; a non-English choice
    /// routes the turn through the userLang↔EN translation shim.
    @Published var userLang: String = UserDefaults.standard.string(forKey: "userLang") ?? "" {
        didSet { UserDefaults.standard.set(userLang, forKey: "userLang") }
    }
    /// Show the "What language do you speak?" picker until a language is chosen.
    var needsLanguagePick: Bool { userLang.isEmpty }
    var translationSupported: Bool { TranslationService.isAvailable }

    let translation = TranslationService()

    /// Pick the user's language: store it, then pull the ENGLISH (core) + chosen-language packs up
    /// front (one system "Download" tap). English-only choice needs no translation/pack.
    func chooseLanguage(_ code: String) {
        userLang = code
        if code != "en", TranslationService.isAvailable {
            Task { await translation.prewarm(userLang: code) }
        }
    }

    private var assistant: Assistant?
    let foreground = AtomicBool(true)
    /// Set by the Stop button; the generation loop polls it via `shouldContinue` and bails out.
    let cancelled = AtomicBool(false)

    private func applyWebSetting() {
        assistant?.web = nil          // Web removed from this build (fully offline).
    }

    // Persistent conversation log in Documents/ — survives console drops, screen lock, and crashes.
    // Pull with: xcrun devicectl device copy from --domain-type appDataContainer
    //            --domain-identifier com.bayamax.hypernetsp --source Documents/convo_log.txt ...
    private lazy var convoLogURL: URL? = {
        try? FileManager.default.url(for: .documentDirectory, in: .userDomainMask,
                                     appropriateFor: nil, create: true)
            .appendingPathComponent("convo_log.txt")
    }()
    func logConvo(_ s: String) { append(convoLogURL, s) }

    private func append(_ url: URL?, _ s: String) {
        guard let u = url, let data = (s + "\n").data(using: .utf8) else { return }
        if let h = try? FileHandle(forWritingTo: u) {
            defer { try? h.close() }
            _ = try? h.seekToEnd(); try? h.write(contentsOf: data)
        } else {
            try? data.write(to: u)
        }
    }

    // Gap-case instrumentation (owner decision 2026-06-12): when a non-recall/lookup turn is
    // immediately followed by the user re-asking, that's a case a retrieval trigger would fill.
    // We log every (prev intent ∉ {recall,lookup}) → next-turn transition with a `reask` flag, so
    // the rate can be measured later to decide if turning the trigger on is worth it.
    private lazy var gapLogURL: URL? = {
        try? FileManager.default.url(for: .documentDirectory, in: .userDomainMask,
                                     appropriateFor: nil, create: true)
            .appendingPathComponent("gap_events.jsonl")
    }()
    private var prevIntent: String?
    private var prevUser: String?

    private func looksLikeReask(_ cur: String) -> Bool {
        let c = cur.lowercased()
        let markers = ["本当", "ほんと", "違", "ちが", "じゃなくて", "じゃない", "もう一", "もう少し",
                       "やっぱ", "正し", "検証", "ちゃんと", "つまり", "え？", "えっ", "って？",
                       "really", "are you sure", "sure?", "no,", "wrong", "actually", "again",
                       "verify", "i meant", "you mean", "what about", "recheck", "double check"]
        if markers.contains(where: { c.contains($0) }) { return true }
        let t = cur.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.count <= 12 && (t.contains("?") || t.contains("？"))
    }

    private func jsonStr(_ s: String) -> String {
        let e = s.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"").replacingOccurrences(of: "\n", with: " ")
        return "\"\(e)\""
    }

    private func recordGap(curUser: String, curIntent: String) {
        if let p = prevIntent, p != "recall", p != "lookup" {
            let pu = jsonStr(prevUser ?? "")
            append(gapLogURL, "{\"prev_intent\":\(jsonStr(p)),\"prev_user\":\(pu),"
                + "\"cur_user\":\(jsonStr(curUser)),\"cur_intent\":\(jsonStr(curIntent)),"
                + "\"reask\":\(looksLikeReask(curUser))}")
        }
        prevIntent = curIntent
        prevUser = curUser
    }

    init() {
        // iOS rejects GPU work in the background, so track foreground state. macOS runs the GPU in
        // the background fine — leave `foreground` permanently true there.
        #if canImport(UIKit)
        let nc = NotificationCenter.default
        nc.addObserver(forName: UIApplication.didBecomeActiveNotification, object: nil, queue: .main) {
            [foreground] _ in foreground.set(true)
        }
        nc.addObserver(forName: UIApplication.willResignActiveNotification, object: nil, queue: .main) {
            [foreground] _ in foreground.set(false)
        }
        #endif
    }

    private func bundlePaths() throws -> Assistant.Paths {
        let b = Bundle.main
        guard let modelDir = b.url(forResource: "fft_mlx4", withExtension: nil),
            let pw = b.url(forResource: "pooler", withExtension: "safetensors"),
            let pc = b.url(forResource: "pooler_config", withExtension: "json"),
            let bge = b.url(forResource: "bge_small", withExtension: nil),
            let ih = b.url(forResource: "intent_head", withExtension: "json"),
            let sh = b.url(forResource: "specificity_head", withExtension: "json")
        else { throw AppError("bundle assets missing") }
        let trig = b.url(forResource: "trigger_head", withExtension: "json")  // optional
        let phaseb = b.url(forResource: "phaseb_indexer", withExtension: "safetensors")  // optional
        let gate = b.url(forResource: "trigger_gate_v3", withExtension: "safetensors")  // optional
        let bgeBase = b.url(forResource: "bge_base", withExtension: nil)                 // optional (V2)
        let gateV2 = b.url(forResource: "recall_gate_v2", withExtension: "safetensors")  // optional (V2)
        let bgeHead = b.url(forResource: "recall_bge_head", withExtension: "safetensors")// optional (V2)
        // keep persistent memory in Application Support (not the user's Documents)
        let fm = FileManager.default
        let support = (try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                   appropriateFor: nil, create: true))?
            .appendingPathComponent("HypernetSP", isDirectory: true)
        if let support { try? fm.createDirectory(at: support, withIntermediateDirectories: true) }
        return Assistant.Paths(
            sp: .init(modelDir: modelDir, poolerWeights: pw, poolerConfig: pc),
            bgeDir: bge, intentHead: ih, specHead: sh,
            // L2 disk (cross-session persistence) intentionally DISABLED for this build — removes
            // session-crossing AND prevents stale persisted facts from a prior run bleeding in.
            l2URL: nil, triggerHead: trig,
            phasebIndexer: phaseb, gateHead: gate,
            bgeBaseDir: bgeBase, recallGateV2: gateV2, recallBgeHead: bgeHead)
    }

    func load() {
        guard phase == .idle else { return }
        phase = .loading
        deviceName = Self.deviceLabel()
        Task.detached(priority: .userInitiated) {
            do {
                let paths = try await MainActor.run { try self.bundlePaths() }
                let a = try await Assistant.load(paths)
                a.web = nil          // fully-offline build: no Web tier
                await MainActor.run {
                    self.assistant = a
                    self.deviceName = Self.deviceLabel()
                    self.phase = .ready
                    self.log = "Assistant ready on \(self.deviceName).\nType a message, or tap ‘Recall demo’.\n"
                    // If a non-English language was already chosen, pull its pack (+ English core) up
                    // front. First-launch language pick (and its download) is driven by the onboarding UI.
                    if !self.userLang.isEmpty, self.userLang != "en", TranslationService.isAvailable {
                        Task { await self.translation.prewarm(userLang: self.userLang) }
                    }
                    print("[SP] ready · device=\(self.deviceName) · avail=\(availableMemoryMB())MB")
                    // Optional headless validation hooks.
                    if ProcessInfo.processInfo.environment["SP_CONVO_TEST"] != nil {
                        self.runConvoTest()
                    } else if ProcessInfo.processInfo.environment["SP_RECALL_ONLY"] != nil {
                        self.runRecallOnlyTest()
                    } else if ProcessInfo.processInfo.environment["SP_RETR_PROBE"] != nil {
                        self.runRetrievalProbe()
                    } else if ProcessInfo.processInfo.environment["SP_JP_TEST"] != nil {
                        self.runJPDirectTest()
                    } else if ProcessInfo.processInfo.environment["SP_RF_BATTERY"] != nil {
                        self.runRFBattery()
                    } else if ProcessInfo.processInfo.environment["SP_REPLY_RECALL"] != nil {
                        self.runReplyRecallTest()
                    } else if ProcessInfo.processInfo.environment["SP_V2_GENNEEDLE"] != nil {
                        self.runV2GenNeedleTest()
                    } else if ProcessInfo.processInfo.environment["SP_V2_NEEDLE"] != nil {
                        self.runV2NeedleTest()
                    } else if ProcessInfo.processInfo.environment["SP_GATE_TEST"] != nil {
                        self.runGateTest()
                    } else if ProcessInfo.processInfo.environment["SP_ROUTE_TEST"] != nil {
                        self.runRouteTest()
                    } else if ProcessInfo.processInfo.environment["SP_NEEDLE_TEST"] != nil {
                        self.runNeedleTest()
                    } else if ProcessInfo.processInfo.environment["SP_STUCK_TEST"] != nil {
                        self.runStuckTest()
                    } else if ProcessInfo.processInfo.environment["SP_ANAPHORA_TEST"] != nil {
                        self.runAnaphoraTest()
                    } else if ProcessInfo.processInfo.environment["SP_TRIGGER_TEST"] != nil {
                        self.runTriggerTest()
                    } else if ProcessInfo.processInfo.environment["SP_TRANSLATE_TEST"] != nil {
                        self.runTranslateTest()
                    } else if ProcessInfo.processInfo.environment["SP_SHIMTEST"] != nil {
                        self.runShimTest()
                    } else if let ref = ProcessInfo.processInfo.environment["SP_PARITY"] {
                        self.runParity(URL(fileURLWithPath: ref))
                    } else if let ask = ProcessInfo.processInfo.environment["SP_ASK"] {
                        if let ask2 = ProcessInfo.processInfo.environment["SP_ASK2"] {
                            self.sendTurn(ask) { self.sendTurn(ask2) }   // chained 2-turn demo (screenshots)
                        } else {
                            self.sendTurn(ask)
                        }
                    } else if ProcessInfo.processInfo.environment["SP_BATTERY"] != nil {
                        self.runBattery()
                    } else if let dir = ProcessInfo.processInfo.environment["SP_INDEX_DIR"],
                       let q = ProcessInfo.processInfo.environment["SP_INDEX_QUERY"] {
                        self.indexFolder(URL(fileURLWithPath: dir)) { [weak self] in
                            self?.sendTurn(q)
                        }
                    } else {
                        self.messages.append(ChatMessage(
                            role: .system,
                            text: "Ready on \(self.deviceName). Ask me a question or give me some math to work through."))
                    }
                }
            } catch {
                await MainActor.run { self.phase = .error("\(error)") }
            }
        }
    }

    // MARK: - foreground keep-alive (GPU needs the app foreground)

    #if canImport(UIKit)
    private var bgTask: UIBackgroundTaskIdentifier = .invalid
    #endif
    private func beginCompute() {
        #if canImport(UIKit)
        UIApplication.shared.isIdleTimerDisabled = true
        bgTask = UIApplication.shared.beginBackgroundTask(withName: "sp-gen") { [weak self] in
            self?.endCompute()
        }
        #endif
    }
    private func endCompute() {
        #if canImport(UIKit)
        UIApplication.shared.isIdleTimerDisabled = false
        if bgTask != .invalid { UIApplication.shared.endBackgroundTask(bgTask); bgTask = .invalid }
        #endif
    }

    // MARK: - turns

    /// Run one assistant turn (canonical AppSession.turn pipeline). `then` chains follow-up turns.
    func sendTurn(_ text: String, then: (@MainActor @Sendable () -> Void)? = nil) {
        guard let assistant, !busy else { return }
        busy = true
        cancelled.set(false)
        beginCompute()
        let fg = foreground
        let cancel = cancelled
        let a = assistant
        let lang = userLang
        messages.append(ChatMessage(role: .user, text: text))
        let pendingID = ChatMessage(role: .assistant, text: "", pending: true)
        messages.append(pendingID)
        let idx = messages.count - 1
        // userLang in → English core pipeline → userLang out (CoT stays English; verbatim spans protected).
        let doTranslate = !lang.isEmpty && lang != "en" && TranslationService.isAvailable
        let svc = translation
        print("[SP] turn: \(text.prefix(50)) · lang=\(lang) translate=\(doTranslate) · avail=\(availableMemoryMB())MB")
        Task.detached(priority: .userInitiated) {
            var opts = SPModel.Options()
            // CoT open with a 4000-token think cap (in genOnce); keep genLen large so the ANSWER after
            // the closed </think> isn't truncated.
            opts.temp = 0.6; opts.rw = 512; opts.maxD = 4096; opts.genLen = 5600; opts.seed = 0
            opts.shouldContinue = { fg.value && !cancel.value }
            let tr: TranslationShim.Translate = { t, f, to in await svc.translate(t, from: f, to: to) }
            let coreText = doTranslate ? await TranslationShim.inbound(text, userLang: lang, translate: tr) : text
            // Translation failed (passthrough returns the input unchanged) → do NOT feed raw non-English
            // to the English model (it produces off-language garbage). Tell the user instead.
            if doTranslate, coreText == text {
                await MainActor.run {
                    guard idx < self.messages.count else { self.busy = false; self.endCompute(); return }
                    self.messages[idx] = ChatMessage(role: .system,
                        text: "Translation needs the language pack. Tap ‘Download’ on the iOS prompt at first send, or add your language + English in Settings → Translate, then send again.")
                    self.busy = false
                    self.endCompute()
                    then?()
                }
                return
            }
            // PURE SP: no memory / recall / web — just SP-evict generation (CoT open, 4000 think cap).
            let r = await a.turnSP(coreText, options: opts)
            var answer = cancel.value && r.answer.isEmpty ? "(stopped)" : r.answer
            if doTranslate, !answer.isEmpty {
                answer = await TranslationShim.outbound(answer, userLang: lang, translate: tr)
            }
            await MainActor.run {
                guard idx < self.messages.count else { self.busy = false; self.endCompute(); return }
                let c = a.mem.counts
                if r.acked {
                    self.messages[idx] = ChatMessage(role: .assistant, text: answer,
                        intent: r.intent, meta: "memory L1=\(c.l1) · pins=\(c.pins) · L2=\(c.l2)")
                    print("[SP] DONE [\(r.intent)] ack: \(r.answer.prefix(80))")
                } else {
                    self.messages[idx] = ChatMessage(role: .assistant, text: answer,
                        source: r.source, intent: r.intent,
                        meta: String(format: "%d tok · %.1f tok/s", r.tokens, r.tokensPerSecond))
                    print("[SP] ANSWER[\(r.intent)/\(r.source ?? "-")]: \(r.answer.replacingOccurrences(of: "\n", with: " ").prefix(160))")
                }
                // persistent transcript (includes the translated Japanese the user actually sees)
                self.logConvo("USER: \(text)")
                self.logConvo(String(format: "  route=%@/%@ translate=%@ trigger=%.2f%@",
                    r.intent, r.source ?? "-", doTranslate ? "Y" : "N",
                    a.lastTriggerScore, a.lastTriggerFired ? " FIRED↩︎" : ""))
                if doTranslate { self.logConvo("  EN-core: \(r.answer.replacingOccurrences(of: "\n", with: " ").prefix(280))") }
                self.logConvo("  SHOWN: \(answer.replacingOccurrences(of: "\n", with: " ").prefix(280))")
                self.logConvo("")
                self.recordGap(curUser: text, curIntent: r.intent)
                self.busy = false
                self.endCompute()
                then?()
            }
        }
    }

    /// Stop the in-flight generation; it returns whatever it has so far.
    func cancel() { cancelled.set(true) }

    func generate() {
        let t = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        input = ""
        sendTurn(t)
    }

    /// Env-gated composite battery (SP_BATTERY): exercises each turn() path end-to-end for validation.
    func runBattery() {
        let prompts = [
            "Remember my employee ID is EMP-4471.",            // persist/fact → ack
            "What is my employee ID?",                          // recall → EMP-4471
            "What is 47 multiplied by 9?",                      // math → 423 (calculator)
            "If I have 650 dollars and I spend 200, how much do I have left?",  // math → 450
            "Who wrote Romeo and Juliet?",                      // lookup → web → Shakespeare
            "Compute these: (a) 5 plus 3 (b) 10 minus 4.",      // multi-part → 8 and 6
        ]
        func step(_ i: Int) {
            guard i < prompts.count else { print("[SP] BATTERY DONE"); return }
            sendTurn(prompts[i]) { step(i + 1) }
        }
        step(0)
    }

    /// SP-evict parity check (SP_PARITY=<path to parity_reference.safetensors>). Reproduces the
    /// model-dev side's 4-step diagnostic: ① empty-past pooler has no NaN, ② all 32 SP vectors have
    /// L2 norm == |out_scale| (1.468531), and the max-abs-diff vs the reference `sp0`.
    func runParity(_ refURL: URL) {
        guard let a = assistant else { return }
        let sp = a.sp
        Task.detached(priority: .userInitiated) {
            do {
                let ref = try MLX.loadArrays(url: refURL)
                guard let sp0 = ref["sp0"] else { throw AppError("sp0 missing in ref") }
                // ① empty-past pooler (kept == []) → forward over zeros[1,0,H]
                let empty = MLXArray.zeros([1, 0, sp.H], dtype: .float32)
                let out = sp.pooler.forward(empty)          // (1, 32, H) float32
                eval(out)
                let total = MLX.sum(out).item(Float.self)   // NaN/Inf propagates to the sum
                let nanCount = (total.isNaN || total.isInfinite) ? 1 : 0
                // ② per-vector L2 norms
                let norms = MLX.sqrt(MLX.sum(out * out, axis: -1))[0]   // (32,)
                eval(norms)
                let normArr = norms.asArray(Float.self)
                let nmin = normArr.min() ?? 0, nmax = normArr.max() ?? 0
                // diff vs reference sp0 (fp32)
                let diff = MLX.abs(out[0] - sp0.asType(.float32))
                let maxDiff = MLX.max(diff).item(Float.self)
                let refNorms = ref["sp0_norms"]?.asArray(Float.self)
                let poolerVerdict = nanCount > 0 ? "FAIL — NaN in empty-past SP (candidate #1 CONFIRMED)"
                    : (maxDiff < 1e-3 ? "PASS — pooler matches reference; bug is downstream (cache/RoPE/forward)"
                                      : "FAIL — pooler diverges numerically (maxDiff \(maxDiff))")
                print("[SP] PARITY-1 nan=\(nanCount) normRange=[\(nmin),\(nmax)] maxDiff=\(maxDiff) :: \(poolerVerdict)")

                // ③/④ stage-2: replicate sp_evict_parity.py greedy SP-evict EXACTLY —
                // prompt q_ids PREFILLED (MQ=14), SP block injected after, greedy, no guards.
                // This isolates the block-injection / cache-crop / RoPE machinery.
                let qIds = (ref["q_ids"]?.asArray(Int32.self) ?? []).map { Int($0) }
                let refGreedy = (ref["greedy_tokens"]?.asArray(Int32.self) ?? []).map { Int($0) }
                let rw = 512, C = 64, STEPS = min(40, refGreedy.count)
                func embM(_ ids: [Int]) -> MLXArray {
                    ids.isEmpty ? MLXArray.zeros([1, 0, sp.H], dtype: sp.embedDtype)
                                : sp.model.embed(MLXArray(ids.map { Int32($0) }, [1, ids.count]))
                }
                let cache = sp.model.newCache(parameters: nil)
                _ = sp.model.callAsFunction(MLXArray(qIds.map { Int32($0) }, [1, qIds.count]), cache: cache)
                let MQ2 = cache[0].offset
                var gen = [Int](), kept = [Int](), absorbed = 0, toks = [Int](), done = false
                while gen.count < STEPS && !done {
                    let c0 = gen.count, R = min(c0, rw), ndEnd = c0 - R
                    if ndEnd > absorbed { kept.append(contentsOf: gen[absorbed ..< ndEnd]); absorbed = ndEnd }
                    let sp_ = sp.pooler.forward(embM(kept).asType(.float32)).asType(sp.embedDtype)
                    var block = sp_
                    if R > 0 { block = MLX.concatenated([sp_, embM(Array(gen[(c0 - R) ..< c0]))], axis: 1) }
                    for c in cache { _ = c.trim(c.offset - MQ2) }
                    var last = sp.model.callAsFunction(inputEmbeddings: block, cache: cache)[0..., -1, 0...]
                    eval(last)
                    for _ in 0 ..< min(C, STEPS - gen.count) {
                        let t = last[0].argMax().item(Int.self)
                        if t == sp.eos { done = true; break }
                        gen.append(t); toks.append(t)
                        last = sp.model.callAsFunction(inputEmbeddings: embM([t]), cache: cache)[0..., -1, 0...]
                        eval(last)
                    }
                }
                var firstDiff = -1
                for i in 0 ..< min(toks.count, refGreedy.count) where toks[i] != refGreedy[i] { firstDiff = i; break }
                let swiftHead = Array(toks.prefix(12))
                let refHead = Array(refGreedy.prefix(12))

                // stage-3: replicate canonical _gen_once's ACTUAL structure — BOS-only prefix
                // (MQ=1), prompt fed as FORCED tokens AFTER the empty-SP block, greedy. The
                // parity harness PREFILLS the prompt instead, so this is the untested path.
                let bos = 151646
                let cache3 = sp.model.newCache(parameters: nil)
                _ = sp.model.callAsFunction(MLXArray([Int32(bos)], [1, 1]), cache: cache3)
                let MQ3 = cache3[0].offset
                let feed = qIds                       // forced prompt tokens (== q_ids)
                var gen3 = [Int](), kept3 = [Int](), absorbed3 = 0, fi3 = 0
                var toks3 = [Int](), done3 = false
                let genCap = STEPS + feed.count
                while !done3 && gen3.count < genCap {
                    let c0 = gen3.count, R = min(c0, rw), ndEnd = c0 - R
                    if ndEnd > absorbed3 { kept3.append(contentsOf: gen3[absorbed3 ..< ndEnd]); absorbed3 = ndEnd }
                    let sp_ = sp.pooler.forward(embM(kept3).asType(.float32)).asType(sp.embedDtype)
                    var block = sp_
                    if R > 0 { block = MLX.concatenated([sp_, embM(Array(gen3[(c0 - R) ..< c0]))], axis: 1) }
                    for c in cache3 { _ = c.trim(c.offset - MQ3) }
                    var last = sp.model.callAsFunction(inputEmbeddings: block, cache: cache3)[0..., -1, 0...]
                    eval(last)
                    for _ in 0 ..< C {
                        var t: Int
                        if fi3 < feed.count { t = feed[fi3]; fi3 += 1 }
                        else {
                            t = last[0].argMax().item(Int.self)
                            if t == sp.eos { done3 = true; break }
                            toks3.append(t)
                        }
                        gen3.append(t)
                        last = sp.model.callAsFunction(inputEmbeddings: embM([t]), cache: cache3)[0..., -1, 0...]
                        eval(last)
                    }
                }
                let swift3Head = Array(toks3.prefix(12))
                var diff3 = -1
                for i in 0 ..< min(toks3.count, refGreedy.count) where toks3[i] != refGreedy[i] { diff3 = i; break }

                // stage-4: forced-feed structure with TEMP=0.6 multinomial sampling (the app's
                // actual path, minus DecodePolicy). Isolates whether temperature sampling itself
                // produces the ,1!#!!!! degeneration.
                MLXRandom.seed(0)
                let cache4 = sp.model.newCache(parameters: nil)
                _ = sp.model.callAsFunction(MLXArray([Int32(bos)], [1, 1]), cache: cache4)
                let MQ4 = cache4[0].offset
                var gen4 = [Int](), kept4 = [Int](), absorbed4 = 0, fi4 = 0
                var toks4 = [Int](), done4 = false
                while !done4 && gen4.count < genCap {
                    let c0 = gen4.count, R = min(c0, rw), ndEnd = c0 - R
                    if ndEnd > absorbed4 { kept4.append(contentsOf: gen4[absorbed4 ..< ndEnd]); absorbed4 = ndEnd }
                    let sp_ = sp.pooler.forward(embM(kept4).asType(.float32)).asType(sp.embedDtype)
                    var block = sp_
                    if R > 0 { block = MLX.concatenated([sp_, embM(Array(gen4[(c0 - R) ..< c0]))], axis: 1) }
                    for c in cache4 { _ = c.trim(c.offset - MQ4) }
                    var last = sp.model.callAsFunction(inputEmbeddings: block, cache: cache4)[0..., -1, 0...]
                    eval(last)
                    for _ in 0 ..< C {
                        var t: Int
                        if fi4 < feed.count { t = feed[fi4]; fi4 += 1 }
                        else {
                            t = MLXRandom.categorical(last * (1.0 / 0.6)).item(Int.self)
                            if t == sp.eos { done4 = true; break }
                            toks4.append(t)
                        }
                        gen4.append(t)
                        last = sp.model.callAsFunction(inputEmbeddings: embM([t]), cache: cache4)[0..., -1, 0...]
                        eval(last)
                    }
                }
                let toks4Text = sp.tokenizer.decode(tokens: Array(toks4.prefix(24)))
                let swift4Head = Array(toks4.prefix(12))

                // stage-5: forced-feed + temp=0.6, but rw=16 so the NON-EMPTY-kept pooling path
                // (pooler cross-attention over real generated tokens) activates after 16 tokens.
                // This is the path the empty-past reference never exercised. Generate 80 tokens.
                MLXRandom.seed(0)
                let rw5 = 16, STEPS5 = 80
                let cache5 = sp.model.newCache(parameters: nil)
                _ = sp.model.callAsFunction(MLXArray([Int32(bos)], [1, 1]), cache: cache5)
                let MQ5 = cache5[0].offset
                var gen5 = [Int](), kept5 = [Int](), absorbed5 = 0, fi5 = 0
                var toks5 = [Int](), done5 = false, evicted5 = false
                let genCap5 = STEPS5 + feed.count
                while !done5 && gen5.count < genCap5 {
                    let c0 = gen5.count, R = min(c0, rw5), ndEnd = c0 - R
                    if ndEnd > absorbed5 { kept5.append(contentsOf: gen5[absorbed5 ..< ndEnd]); absorbed5 = ndEnd }
                    if !kept5.isEmpty { evicted5 = true }
                    let sp_ = sp.pooler.forward(embM(kept5).asType(.float32)).asType(sp.embedDtype)
                    var block = sp_
                    if R > 0 { block = MLX.concatenated([sp_, embM(Array(gen5[(c0 - R) ..< c0]))], axis: 1) }
                    for c in cache5 { _ = c.trim(c.offset - MQ5) }
                    var last = sp.model.callAsFunction(inputEmbeddings: block, cache: cache5)[0..., -1, 0...]
                    eval(last)
                    for _ in 0 ..< C {
                        var t: Int
                        if fi5 < feed.count { t = feed[fi5]; fi5 += 1 }
                        else {
                            t = MLXRandom.categorical(last * (1.0 / 0.6)).item(Int.self)
                            if t == sp.eos { done5 = true; break }
                            toks5.append(t)
                        }
                        gen5.append(t)
                        last = sp.model.callAsFunction(inputEmbeddings: embM([t]), cache: cache5)[0..., -1, 0...]
                        eval(last)
                    }
                }
                let toks5Text = sp.tokenizer.decode(tokens: Array(toks5.prefix(40)))

                // stage-6: an arbitrary prompt (SP_PARITY_PROMPT) through forced-feed, generating
                // 220 tokens. `greedyAfter`=true emulates DecodePolicy (temp inside <think>, greedy
                // after </think>) to test whether greedy-after-think is what locks onto `!`.
                var stage6 = ""
                if let p6 = ProcessInfo.processInfo.environment["SP_PARITY_PROMPT"] {
                    let greedyAfter = ProcessInfo.processInfo.environment["SP_PARITY_GREEDYAFTER"] != nil
                    MLXRandom.seed(0)
                    let feed6 = sp.tokenizer.encode(text: "<｜User｜>\(p6)<｜Assistant｜>", addSpecialTokens: false)
                        + sp.tokenizer.encode(text: "<think>\n", addSpecialTokens: false)
                    let cache6 = sp.model.newCache(parameters: nil)
                    _ = sp.model.callAsFunction(MLXArray([Int32(bos)], [1, 1]), cache: cache6)
                    let MQ6 = cache6[0].offset
                    var gen6 = [Int](), kept6 = [Int](), absorbed6 = 0, fi6 = 0
                    var toks6 = [Int](), done6 = false, inThink6 = true
                    let genCap6 = 220 + feed6.count
                    while !done6 && gen6.count < genCap6 {
                        let c0 = gen6.count, R = min(c0, rw), ndEnd = c0 - R
                        if ndEnd > absorbed6 { kept6.append(contentsOf: gen6[absorbed6 ..< ndEnd]); absorbed6 = ndEnd }
                        let sp_ = sp.pooler.forward(embM(kept6).asType(.float32)).asType(sp.embedDtype)
                        var block = sp_
                        if R > 0 { block = MLX.concatenated([sp_, embM(Array(gen6[(c0 - R) ..< c0]))], axis: 1) }
                        for c in cache6 { _ = c.trim(c.offset - MQ6) }
                        var last = sp.model.callAsFunction(inputEmbeddings: block, cache: cache6)[0..., -1, 0...]
                        eval(last)
                        for _ in 0 ..< C {
                            var t: Int
                            if fi6 < feed6.count { t = feed6[fi6]; fi6 += 1 }
                            else {
                                let useGreedy = greedyAfter && !inThink6
                                t = useGreedy ? last[0].argMax().item(Int.self)
                                              : MLXRandom.categorical(last * (1.0 / 0.6)).item(Int.self)
                                if t == sp.eos { done6 = true; break }
                                toks6.append(t)
                            }
                            gen6.append(t)
                            if inThink6, sp.tokenizer.decode(tokens: Array(gen6.suffix(8))).contains("</think>") { inThink6 = false }
                            last = sp.model.callAsFunction(inputEmbeddings: embM([t]), cache: cache6)[0..., -1, 0...]
                            eval(last)
                        }
                    }
                    stage6 = sp.tokenizer.decode(tokens: toks6)
                    print("[SP] PARITY-6 prompt=\"\(p6.prefix(40))\" greedyAfter=\(greedyAfter) closed=\(!inThink6)")
                    print("[SP] PARITY-6 decoded=\(stage6.replacingOccurrences(of: "\n", with: "\\n").prefix(400))")

                    // stage-7: the REAL genOnce (policy noteText + salvage + 800 cap) — dump the
                    // FULL body so we see WHERE it degenerates (think vs answer vs salvage).
                    var opts7 = SPModel.Options()
                    opts7.temp = 0.6; opts7.rw = 512; opts7.maxD = 4096; opts7.genLen = 800; opts7.seed = 0
                    var st7 = SPModel.GenState()
                    let (ans7, body7) = sp.genOnce(p6, state: &st7, options: opts7, firstTurn: true)
                    print("[SP] PARITY-7 REAL genOnce answer=\(ans7.replacingOccurrences(of: "\n", with: "\\n").prefix(80))")
                    print("[SP] PARITY-7 REAL body[0:500]=\(body7.replacingOccurrences(of: "\n", with: "\\n").prefix(500))")
                    print("[SP] PARITY-7 REAL body[tail300]=\(String(body7.suffix(300)).replacingOccurrences(of: "\n", with: "\\n"))")
                }

                await MainActor.run {
                    self.log += "\n=== SP PARITY ===\n"
                    self.log += "① empty-past pooler NaN count: \(nanCount)\n"
                    self.log += String(format: "② SP L2 norms: min=%.6f max=%.6f (target 1.468531)\n", nmin, nmax)
                    self.log += String(format: "   ref sp0_norms[0]=%.6f\n", refNorms?.first ?? -1)
                    self.log += String(format: "③ max|sp_swift - sp0_ref| = %.6e\n", maxDiff)
                    self.log += "  pooler verdict: \(poolerVerdict)\n"
                    self.log += "④ greedy SP-evict (MQ=\(MQ2), expect 14):\n"
                    self.log += "   swift[:12] = \(swiftHead)\n"
                    self.log += "   ref[:12]   = \(refHead)\n"
                    self.log += "   first divergence @ token \(firstDiff)\n"
                    self.log += "⑤ _gen_once forced-feed (MQ=\(MQ3), BOS-only), greedy:\n"
                    self.log += "   swift3[:12] = \(swift3Head)\n"
                    self.log += "   first divergence vs ref @ token \(diff3)\n"
                    print("[SP] PARITY-2 MQ=\(MQ2) firstDiff=\(firstDiff)")
                    print("[SP] PARITY-2 swift=\(swiftHead)")
                    print("[SP] PARITY-2 ref  =\(refHead)")
                    self.log += "⑥ _gen_once forced-feed + temp=0.6 sampling:\n"
                    self.log += "   swift4[:12] = \(swift4Head)\n   decoded: \(toks4Text)\n"
                    print("[SP] PARITY-3 MQ=\(MQ3) firstDiff=\(diff3) forcedFeed_swift=\(swift3Head)")
                    print("[SP] PARITY-4 temp0.6 swift=\(swift4Head)")
                    print("[SP] PARITY-4 temp0.6 decoded=\(toks4Text.replacingOccurrences(of: "\n", with: " "))")
                    print("[SP] PARITY-5 rw=16 nonEmptyKept=\(evicted5) decoded=\(toks5Text.replacingOccurrences(of: "\n", with: " "))")
                }
            } catch {
                await MainActor.run { print("[SP] PARITY error: \(error)") }
            }
        }
    }

    /// SP_SHIMTEST: verify the translation shim's verbatim-span protection + restore round-trip
    /// using a mock translator that simulates an NMT (mangles plain words, copies [[i]] through).
    func runShimTest() {
        // mock: uppercases plain text (simulating "translation") but leaves placeholders intact
        let mock: TranslationShim.Translate = { t, _, _ in t.uppercased() }
        let cases = [
            "合言葉は「さくら」です。社員IDは EMP-4471。",
            "I have $650 and the code is QX7-2291.",
            "ロッカーの番号は4821、残りは450円。",
            // the device failure: ordinary words + bare numbers must NOT be masked (they over-masked
            // into [[N]] soup that Apple's NMT mangled). Expect span count to stay LOW here.
            "Step-by-step: 1. Multiply, 2. Divide. The Answer is 24.4. Start with 122 and 5.",
            // device failure repro: a math word problem — the QUANTITIES must NOT be masked or the
            // model can't compute with them.
            "リンゴ1個120円。5個買って1000円札で払ったらおつりは？",
        ]
        Task {
            for c in cases {
                let (masked, map) = TranslationShim.protect(c)
                let round = await TranslationShim.outbound(c, translate: mock)
                let ok = map.values.allSatisfy { round.contains($0) }
                print("[SP] SHIM in=\(c)")
                print("[SP] SHIM masked=\(masked) spans=\(map.count)")
                print("[SP] SHIM out=\(round)  verbatim-preserved=\(ok)")
            }
            print("[SP] SHIMTEST DONE")
        }
    }

    /// SP_TRANSLATE_TEST: probe the REAL Apple Translation engine — pack availability + a live
    /// JP→EN→JP round trip through the full shim. Tells us how far automated verification gets.
    func runTranslateTest() {
        let svc = translation
        Task {
            print("[SP] TR supported=\(TranslationService.isAvailable)")
            let jaEn = await svc.availability(from: "ja", to: "en")
            let enJa = await svc.availability(from: "en", to: "ja")
            print("[SP] TR pack ja→en: \(jaEn)")
            print("[SP] TR pack en→ja: \(enJa)")
            let tr: TranslationShim.Translate = { t, f, to in await svc.translate(t, from: f, to: to) }
            let sample = "合言葉は「さくら」です。47かける9はいくつ？"
            print("[SP] TR sample=\(sample)")
            let en = await TranslationShim.inbound(sample, translate: tr)
            print("[SP] TR inbound(EN)=\(en)")
            let back = await TranslationShim.outbound(en, translate: tr)
            print("[SP] TR outbound(JA)=\(back)")
            print("[SP] TR changed=\(en != sample)  (false ⇒ pack not active / fell back to passthrough)")
            print("[SP] TRANSLATE_TEST DONE")
        }
    }

    /// SP_TRIGGER_TEST: run a multi-turn sequence and print the retrieval-trigger score per turn.
    /// Expect: referential follow-ups ("are you sure?", "is that right?") score HIGH (fire ↩︎),
    /// fresh/control turns score LOW. Then the follow-up should be answered from context.
    func runTriggerTest() {
        guard let a = assistant else { return }
        let seq = [
            "The capital of France is Paris.",   // establishes context (fact/chitchat)
            "Are you sure about that?",          // referential → expect FIRE
            "What is 12 times 3?",               // fresh compute → expect low
            "Is that answer correct?",           // referential → expect FIRE
        ]
        Task { @MainActor in
            print("[SP] TRIG trigger loaded=\(a.trigger != nil) threshold=\(a.trigger?.fireThreshold ?? -1)")
            var opts = SPModel.Options()
            opts.temp = 0.6; opts.rw = 512; opts.maxD = 4096; opts.genLen = 400; opts.seed = 0
            for q in seq {
                let r = await a.turn(q, options: opts)
                print(String(format: "[SP] TRIG q=\"%@\"  score=%.3f %@  route=%@/%@",
                    q, a.lastTriggerScore, a.lastTriggerFired ? "FIRE" : "----",
                    r.intent, r.source ?? "-"))
                print("[SP] TRIG   ans=\(r.answer.replacingOccurrences(of: "\n", with: " ").prefix(140))")
            }
            print("[SP] TRIGGER_TEST DONE")
        }
    }

    /// SP_CONVO_TEST: run a real multi-turn conversation (full pipeline) and dump route / answer /
    /// body(CoT) per turn, to judge (a) conversational continuity ("the vibe carries") and (b) CoT.
    func runConvoTest() {
        guard let a = assistant else { return }
        let convo = [
            "I'm organizing a 3-day team offsite to Hakone next month.",
            "There are 12 people on the team.",
            "The total budget is 4200 dollars.",
            "We want a place with an onsen and a meeting room.",
            "What team-building activities would you suggest?",
            "Any tips for the food, since some of the team are vegetarian?",
            "How far is Hakone from Tokyo by train, roughly?",
            "What should everyone pack for March weather there?",
            "What's a good way to split the budget across lodging, food, and activities?",
            "Can you summarize the plan so far?",                 // continuity (genOnce, long context)
            "Remind me, how many people are coming again?",        // recall-needing
            "And what was the total budget I mentioned earlier?",  // recall-needing
            "So how much is that per person?",                     // CoT using recalled facts
            "Thanks — what was our destination again?",            // recall-needing
        ]
        Task { @MainActor in
            var opts = SPModel.Options()
            opts.temp = 0.6; opts.rw = 512; opts.maxD = 4096; opts.genLen = 300; opts.seed = 0
            for q in convo {
                let r = await a.turn(q, options: opts)
                let fired = a.lastRecallTokens > 0 ? " ★RECALL=\(a.lastRecallTokens)tok" : ""
                print("[SP] CONVO [\(r.intent)] gen=\(a.genTokenCount) kept=\(a.keptTokenCount) recallTok=\(a.lastRecallTokens)\(fired)  Q: \(q)")
                print("[SP] CONVO    A: \(r.answer.replacingOccurrences(of: "\n", with: " ").prefix(200))")
            }
            print("[SP] CONVO_TEST DONE")
        }
    }

    /// SP_RETR_PROBE: reproduce the user's two-turn calc conversation and show EXACTLY what the recall
    /// gate pulls each turn (src + injected chunks), plus the live memory contents and genState size —
    /// to confirm nothing stale / cross-session is retrieved (the "1985/2019" worry). Translation is
    /// orthogonal (retrieval runs on the English core), so we feed the English equivalents.
    func runRetrievalProbe() {
        guard let a = assistant else { return }
        let turns = [
            "An apple is 120 yen each. If I buy 5 and pay with a 1000 yen bill, what's the change?",
            "And if I also buy 4 pencils at 50 yen each, what's the change?",
        ]
        Task { @MainActor in
            a.resetConversation()
            var opts = SPModel.Options()
            opts.temp = 0.6; opts.rw = 512; opts.maxD = 4096; opts.genLen = 220; opts.seed = 0
            let c0 = a.mem.counts
            print("[SP] RETR START — fresh memory: session=\(c0.l1) pins=\(c0.pins) L2=\(c0.l2)  (L2 disk must be 0 = no cross-session)")
            for (i, q) in turns.enumerated() {
                let before = a.mem.counts
                let r = await a.turnRF(q, webEnabled: false, options: opts)
                print("[SP] RETR turn\(i+1)  Q: \(q)")
                print("[SP] RETR   memory BEFORE retrieval: session=\(before.l1) pins=\(before.pins) L2=\(before.l2)")
                print("[SP] RETR   >>> RETRIEVED src=\(r.source ?? "NONE")  injected chunks=\(r.injected.count)")
                for c in r.injected { print("[SP] RETR        chunk: \(c.prefix(120))") }
                print("[SP] RETR   genState tokens=\(a.genTokenCount) (in-session SP continuity)")
                print("[SP] RETR   A: \(r.answer.replacingOccurrences(of: "\n", with: " ").prefix(160))")
            }
            print("[SP] RETR DONE")
        }
    }

    /// SP_JP_TEST: Option B probe — feed Japanese DIRECTLY through turnRF (no translation shim) to
    /// gauge whether the model can be used natively in Japanese (would let us drop Apple Translation).
    func runJPDirectTest() {
        guard let a = assistant else { return }
        let qs = [
            "こんにちは、今日の調子はどう？",
            "フランスの首都はどこ？",
            "私のフライト番号は JL412 だよ。覚えておいて。",
            "さっきのフライト番号、何だっけ？",
            "簡単な俳句を作って。",
        ]
        Task { @MainActor in
            a.resetConversation()
            var opts = SPModel.Options()
            opts.temp = 0.6; opts.rw = 512; opts.maxD = 4096; opts.genLen = 220; opts.seed = 0
            print("[SP] JP_DIRECT (no translation — model handles Japanese natively)")
            for q in qs {
                let r = await a.turnRF(q, webEnabled: false, options: opts)
                print("[SP] JP Q: \(q)")
                print("[SP] JP A: \(r.answer.replacingOccurrences(of: "\n", with: " ").prefix(180))")
            }
            print("[SP] JP_TEST DONE")
        }
    }

    /// SP_RF_BATTERY: verify the router-free OPERATIONS_SPEC pipeline on Mac. Exercises the §5
    /// capabilities: no-refusal chitchat, offline known-fact, save→recall (with distraction), and Web
    /// (set SP_RF_NOWEB=1 to skip the network turn). Math/local/cross-session paths are not present.
    func runRFBattery() {
        guard let a = assistant else { return }
        // FULLY OFFLINE: no Web tier. Checks = no-refusal chitchat, offline known-facts, save→recall.
        let turns: [(q: String, web: Bool, expect: String?)] = [
            ("Hey, how's your day going?", false, nil),
            ("What is the capital of France?", false, "Paris"),
            ("Please remember that my flight number is JL412.", false, nil),
            ("I've been meaning to repaint my office lately.", false, nil),
            ("Remind me — what was my flight number?", false, "JL412"),
            ("Who wrote the play Romeo and Juliet?", false, "Shakespeare"),
        ]
        Task { @MainActor in
            a.resetConversation()
            var opts = SPModel.Options()
            opts.temp = 0.6; opts.rw = 512; opts.maxD = 4096; opts.genLen = 220; opts.seed = 0
            print("[SP] RF_BATTERY start (fully offline, no web)  persona=on")
            var pass = 0, checks = 0
            for t in turns {
                let r = await a.turnRF(t.q, webEnabled: t.web, options: opts)
                var verdict = ""
                if let exp = t.expect {
                    checks += 1
                    let ok = r.answer.lowercased().contains(exp.lowercased())
                    if ok { pass += 1 }
                    verdict = ok ? " ✓\(exp)" : " ✗(want \(exp))"
                }
                let refused = r.answer.lowercased().contains("can't assist") || r.answer.lowercased().contains("cannot assist")
                print("[SP] RF [\(r.intent)] src=\(r.source ?? "-")\(verdict)\(refused ? " ⚠REFUSAL" : "")  Q: \(t.q)")
                print("[SP] RF    A: \(r.answer.replacingOccurrences(of: "\n", with: " ").prefix(160))")
            }
            print("[SP] RF_BATTERY DONE  expected-checks \(pass)/\(checks)")
        }
    }

    /// SP_REPLY_RECALL: faithful port of `recall_gen.py` multi-fact eval on 4-bit MLX — the owner's
    /// reference decode (NO think scaffolding, pure greedy, no salvage, thresh −3.5, BGE bridge). Plants
    /// 3 distinct facts in 3 early turns, buries them past the 512 window with filler, then asks each
    /// back. Reports recall (value in answer) / retrieved (value in a pulled block) / casual-fire.
    func runReplyRecallTest() {
        guard let a = assistant, let rv = a.recallV2 else { print("[SP] REPLYREC no recallV2"); return }
        let sp = a.sp
        let thresh = ProcessInfo.processInfo.environment["SP_V2_THRESH"].flatMap { Float($0) } ?? -3.5
        let facts: [(kw: String, val: String)] = [
            ("project codename", "Bluefin"), ("vault code", "7382"), ("flight number", "JL412"),
            ("hotel confirmation number", "88231"), ("locker number", "514"), ("meeting room", "C12"),
        ]
        let casual: [(String, String)] = [
            ("How's your day going so far?", "Pretty good, thanks for asking! It's been a steady kind of day, nothing too dramatic. I got through my morning tasks earlier than expected, so I'm feeling fairly relaxed about the afternoon. How has yours been treating you?"),
            ("I had great ramen for lunch today.", "Oh that sounds wonderful, ramen is such a comforting meal. Tonkotsu broth on a cool day is honestly hard to beat, especially with a soft egg and some extra scallions. Did you go to a regular spot or try somewhere new this time?"),
            ("The weather's been grey and rainy all week.", "Ugh, I hear you, long stretches of grey skies can really weigh on your mood after a while. It makes it so tempting to just stay curled up indoors with something warm. Hopefully there's a bit of sunshine coming to break it up soon."),
            ("I'm trying to read more before bed lately.", "That's a really nice habit to build, much kinder to your brain than scrolling on a bright screen late at night. Even a few pages can help you wind down and sleep better. What kind of books are you reaching for these days?"),
            ("My neighbor just got a golden retriever puppy.", "Aw, that's adorable, golden retriever puppies are pure sunshine, though they do come with a serious amount of chaotic energy. Lots of zoomies and chewed shoes in their future, I'd guess."),
            ("I'm thinking of repainting my home office.", "Ooh, a fresh coat of paint can completely change how a room feels, that sounds like a fun little project. A calm, soft color can really help you focus and feel settled while you work."),
            ("I started learning to play the guitar.", "That's awesome, picking up an instrument is such a rewarding thing to do. The first few weeks can be tough on the fingertips, but the chords start coming together faster than you'd expect once muscle memory kicks in."),
            ("Traffic this morning was an absolute nightmare.", "Oh no, there's really no worse way to start the day than being stuck bumper to bumper when you just want to get where you're going. It drains your energy before anything has even happened."),
            ("I tried a new coffee place downtown.", "Nice, discovering a cozy new cafe is one of life's small but genuine pleasures. There's something really satisfying about finding a spot with good coffee and a comfortable corner to sit in."),
            ("I've been listening to a lot of jazz lately.", "Jazz is such a great choice, it has this wonderful way of filling a room without demanding all your attention. It makes for perfect background music whether you're working, cooking, or just relaxing in the evening."),
            ("I might take a short trip next month.", "That sounds refreshing, a little change of scenery can do wonders for resetting your head. Even a couple of days away can feel like a proper reset."),
        ]
        Task { @MainActor in
            print("[SP] REPLYREC thresh=\(thresh) (reply()-faithful: no-think, greedy, no-salvage)")
            var recallOK = 0, retrOK = 0, casualFire = 0, qN = 0, cN = 0
            // 3 facts, planted, buried, asked back (mirrors build_problem)
            let chosen = Array(facts.prefix(3))
            var history = [Int]()
            func add(_ u: String, _ aa: String) {
                history += sp.tokenizer.encode(text: (history.isEmpty ? "" : "<｜end▁of▁sentence｜>")
                    + "<｜User｜>\(u)<｜Assistant｜>\(aa)", addSpecialTokens: false)
            }
            for f in chosen { add("Important to note: the \(f.kw) is \(f.val).", "Got it, noted.") }
            for j in 0 ..< 9 { add(casual[j % casual.count].0, casual[j % casual.count].1) }
            // 2 casual probes (must NOT fire) — generated
            for j in [9, 10] {
                let probe = casual[j % casual.count].0
                let feed = history + sp.tokenizer.encode(
                    text: "<｜end▁of▁sentence｜><｜User｜>\(probe)<｜Assistant｜>", addSpecialTokens: false)
                let r = await sp.replyRecall(feed: feed, recall: rv, thresh: thresh)
                history = feed + r.genTokens
                cN += 1; if r.nFire > 0 { casualFire += 1 }
                print("[SP] REPLYREC ── CASUAL probe (fire=\(r.nFire) maxS=\(String(format: "%.2f", r.maxScore)))  Q: \(probe)")
                print("[SP] REPLYREC    FULL A: \(r.answer.replacingOccurrences(of: "\n", with: " "))")
            }
            // 3 questions (reverse order), each generated
            for f in chosen.reversed() {
                let q = "Quick — what was the \(f.kw) I mentioned? Just the value."
                let feed = history + sp.tokenizer.encode(
                    text: "<｜end▁of▁sentence｜><｜User｜>\(q)<｜Assistant｜>", addSpecialTokens: false)
                let r = await sp.replyRecall(feed: feed, recall: rv, thresh: thresh)
                history = feed + r.genTokens
                let norm = r.answer.replacingOccurrences(of: ",", with: "").replacingOccurrences(of: " ", with: "").lowercased()
                let ok = norm.contains(f.val.replacingOccurrences(of: ",", with: "").lowercased())
                let pulledOK = r.pulls.contains { sp.tokenizer.decode(tokens: $0).lowercased().contains(f.val.lowercased()) }
                qN += 1; if ok { recallOK += 1 }; if pulledOK { retrOK += 1 }
                print("[SP] REPLYREC ── RECALL Q[\(f.kw)=\(f.val)] fire=\(r.nFire) maxS=\(String(format: "%.2f", r.maxScore)) recall=\(ok ? "OK" : "x") retr=\(pulledOK ? "OK" : "x")")
                print("[SP] REPLYREC    Q: \(q)")
                print("[SP] REPLYREC    FULL A: \(r.answer.replacingOccurrences(of: "\n", with: " "))")
            }
            print("[SP] REPLYREC SUMMARY recall=\(recallOK)/\(qN) retrieved=\(retrOK)/\(qN) casual-fire=\(casualFire)/\(cN)")
            print("[SP] REPLYREC_TEST DONE")
        }
    }

    /// SP_V2_GENNEEDLE: the FAIR end-to-end recall number — the in-generation V2 loop on CLEAN context
    /// (no self-poisoning). Each needle gets a fresh genState seeded with the buried-fact history; we
    /// generate the answer through genOnceRecall (gate fires mid-stream → bridge retrieve+inject) at the
    /// DYNAMICALLY calibrated threshold, and check fired / recalled-block-hit / answer-contains-value.
    func runV2GenNeedleTest() {
        guard let a = assistant, let rv = a.recallV2 else { print("[SP] V2GEN no recallV2"); return }
        let sp = a.sp
        let needles: [(user: String, q: String, val: String)] = [
            ("By the way, the wifi password at the mountain cabin was trout-9182.", "What was the wifi password at the mountain cabin?", "9182"),
            ("I finally reached Dr. Imai today - her office extension is 4408.", "What is Dr. Imai's office extension?", "4408"),
            ("The rover prototype came in at 18.4 kg on the bench scale.", "How much did the rover prototype weigh?", "18.4"),
            ("Parking at the venue was a maze - we ended up in spot B-37.", "Which parking spot did we end up in at the venue?", "37"),
            ("That limited edition sake was 6800 yen.", "How much did the limited edition sake cost?", "6800"),
            ("The night train to Aomori leaves at 23:46.", "What time does the night train to Aomori leave?", "23:46"),
            ("The projector in conference room A is an XV-310.", "What model is the projector in conference room A?", "310"),
            ("Grandma's bread recipe needs exactly 320 grams of flour.", "How many grams of flour does grandma's bread recipe need?", "320"),
            ("We parked the camper at site 51 by the river.", "Which site did we park the camper at?", "51"),
            ("We turned back at trail marker 58 because of the fog.", "At which trail marker did we turn back?", "58"),
        ]
        let filler = "We talked about the weather, which had been all over the place, and about a quiet "
            + "ramen place near the station. Work stayed busy but manageable, and the garden tomatoes "
            + "were finally turning red. The library extended its evening hours, which was nice."
        func buildHistory(_ needleIdx: Int) -> [Int] {
            // one needle buried among the others' statements + filler, all pushed past the 512 window.
            var ids = [Int]()
            func emit(_ u: String, _ aa: String) {
                let t = (ids.isEmpty ? "" : "<｜end▁of▁sentence｜>") + "<｜User｜>\(u)<｜Assistant｜>\(aa)"
                ids.append(contentsOf: sp.tokenizer.encode(text: t, addSpecialTokens: false))
            }
            for (i, n) in needles.enumerated() {
                emit(n.user, "Noted - thanks for telling me.")
                if i == needleIdx, ids.count < 200 { emit(filler, "Got it.") }
            }
            while ids.count < 1400 { emit(filler, "Got it.") }
            return ids
        }
        let envThresh = ProcessInfo.processInfo.environment["SP_V2_THRESH"].flatMap { Float($0) }
        Task { @MainActor in
            if let t = envThresh {
                rv.threshOverride = t
            } else if a.recallV2?.threshOverride == nil {
                let t = await a.calibrateRecallThreshold()
                print("[SP] V2GEN calibrated thresh=\(t.map { String(format: "%.2f", $0) } ?? "nil")")
            }
            print("[SP] V2GEN thresh=\(String(format: "%.2f", rv.threshOverride ?? rv.thresh))")
            // PRODUCTION form: CoT open (forceThink:true) so the gate reaches the recall state, then
            // SALVAGE force-closes </think> and the returned answer is the EXTRACTED post-think reply.
            var opts = SPModel.Options(); opts.temp = 0.6; opts.rw = 512; opts.maxD = 4096
            opts.genLen = 260; opts.seed = 0
            var fireN = 0, hitN = 0, ansN = 0
            for (i, n) in needles.enumerated() {
                let hist = buildHistory(i)
                var st = SPModel.GenState(); st.gen = hist
                var emb = [[Float]]()
                let blocks = rv.makeBlocks(hist)
                let r = await sp.genOnceRecall(n.q, state: &st, options: opts, firstTurn: false,
                                               recall: rv, recallBlocks: blocks, blockEmb: &emb,
                                               forceThink: true, salvage: "", salvageBudget: 80)
                let ans = r.answer.lowercased()
                let ansHit = ans.contains(n.val.lowercased())     // value in the CLEAN extracted answer
                let blockHit = sp.tokenizer.decode(tokens: r.recIds).lowercased().contains(n.val.lowercased())
                if r.fired { fireN += 1 }
                if blockHit { hitN += 1 }      // injected the CORRECT block (contains the value)
                if ansHit { ansN += 1 }
                print("[SP] V2GEN \(r.fired ? "FIRE" : "skip") block=\(blockHit ? "HIT" : "miss") ans=\(ansHit ? "HIT" : "miss") val=\(n.val)")
                print("[SP] V2GEN    CLEAN A: \(r.answer.replacingOccurrences(of: "\n", with: " "))")
            }
            print("[SP] V2GEN SUMMARY fired=\(fireN)/\(needles.count) block-correct=\(hitN)/\(needles.count) answer-correct=\(ansN)/\(needles.count)")
            print("[SP] V2GEN_TEST DONE")
        }
    }

    /// SP_V2_NEEDLE: clean-context measurement of the RECALL V2 artifacts (gate.npz + bge_head.npz),
    /// the way the handoff measured them — explicit recall questions over CLEAN buried-fact history (no
    /// self-poisoning). Separates two things: (1) GATE — does the V2 gate FIRE on explicit recall and
    /// SKIP chitchat; (2) RETRIEVAL — does the bridge head's top-2 contain the needle (force-retrieve,
    /// gate-independent). 10 needles buried in ~4.2k clean filler tokens (past the 512 window).
    func runV2NeedleTest() {
        guard let a = assistant, let rv = a.recallV2 else { print("[SP] V2NEEDLE no recallV2"); return }
        let sp = a.sp
        let v2Thresh = ProcessInfo.processInfo.environment["SP_V2_THRESH"].flatMap { Float($0) }
        rv.threshOverride = v2Thresh
        // `lead` = a primed answer stub ending exactly where the recalled value would be emitted — a
        // by-construction TRUE-recall position, to test the gate at the right position (the artifacts
        // are a per-position in-generation gate, not a single pre-gen check).
        let needles: [(user: String, q: String, val: String, lead: String)] = [
            ("By the way, the wifi password at the mountain cabin was trout-9182. Took us forever to find it.", "What was the wifi password at the mountain cabin?", "9182", "The wifi password at the mountain cabin was "),
            ("I finally reached Dr. Imai today - her office extension is 4408.", "What is Dr. Imai's office extension?", "4408", "Dr. Imai's office extension is "),
            ("The rover prototype came in at 18.4 kg on the bench scale this morning.", "How much did the rover prototype weigh?", "18.4", "The rover prototype weighed "),
            ("My cousin finally named his fishing boat: the Yume-Maru.", "What is my cousin's fishing boat called?", "Yume", "Your cousin's fishing boat is called the "),
            ("Parking at the venue was a maze - we ended up in spot B-37.", "Which parking spot did we end up in at the venue?", "37", "You ended up in parking spot "),
            ("That limited edition sake was 6800 yen, totally worth it though.", "How much did the limited edition sake cost?", "6800", "The limited edition sake cost "),
            ("The night train to Aomori leaves at 23:46, I double checked.", "What time does the night train to Aomori leave?", "23:46", "The night train to Aomori leaves at "),
            ("The projector in conference room A is an XV-310, in case anyone asks.", "What model is the projector in conference room A?", "310", "The projector in conference room A is an "),
            ("Grandma's bread recipe needs exactly 320 grams of flour, she was very firm about that.", "How many grams of flour does grandma's bread recipe need?", "320", "Grandma's bread recipe needs "),
            ("We turned back at trail marker 58 because of the fog.", "At which trail marker did we turn back?", "58", "We turned back at trail marker "),
        ]
        let filler: [(String, String)] = [
            ("The weather this week has been all over the place, sunny then storms.", "It really has. Those sudden shifts make it hard to plan anything outdoors, though the cool evenings after the rain have been pleasant for walks."),
            ("I tried that new ramen place near the station.", "How was it? A good bowl of ramen after a long day is hard to beat - the broth makes or breaks it in my experience."),
            ("Work has been busy but manageable lately.", "That sounds like a decent balance. Busy enough to stay engaged without burning out is the sweet spot most people aim for."),
            ("I started reading a novel about Antarctic explorers.", "Polar exploration stories can be gripping - the isolation and the small daily decisions that decide everything give them real tension."),
            ("My neighbor's dog learned to open the gate somehow.", "Clever dog! Once they figure out latches there is no stopping them - you may need a second clip higher up where paws cannot reach."),
            ("We are thinking about repainting the kitchen.", "A fresh coat changes the whole feel of a room. Lighter colors tend to make small kitchens feel bigger, especially with warm lighting."),
            ("The train was delayed twenty minutes this morning.", "Annoying way to start the day. At least with a book or some music the platform wait goes a little faster."),
            ("I have been trying to drink more water instead of coffee.", "A solid habit. Keeping a full bottle on the desk where you can see it does most of the work, honestly."),
            ("The garden tomatoes are finally turning red.", "Nothing beats a sun-warmed tomato straight off the vine. The first ripe one of the season always feels like a small victory."),
            ("I watched a documentary about deep sea creatures last night.", "The deep sea is wilder than fiction - bioluminescent lures, pressure-proof bodies, whole ecosystems around vents. What stuck with you most?"),
        ]
        // chitchat probes (should NOT fire — false-positive measure)
        let chats = ["Hello, how are you today?", "Can you tell me a fun fact?",
                     "What should I cook for dinner tonight?", "That sounds nice, thanks!",
                     "What's your favorite season and why?"]
        func emit(_ ids: inout [Int], _ u: String, _ aa: String) {
            let t = (ids.isEmpty ? "" : "<｜end▁of▁sentence｜>") + "<｜User｜>\(u)<｜Assistant｜>\(aa)"
            ids.append(contentsOf: sp.tokenizer.encode(text: t, addSpecialTokens: false))
        }
        func buildHistory() -> [Int] {
            var ids = [Int](); var fi = 0
            for n in needles {
                emit(&ids, filler[fi % filler.count].0, filler[fi % filler.count].1); fi += 1
                emit(&ids, n.user, "Noted - that sounds like quite a day. Thanks for telling me.")
            }
            var lap = 0
            while ids.count < 4200 {
                var (u, aa) = filler[fi % filler.count]
                if lap > 0, let f = u.first { u = "One more thing about that - " + String(f).lowercased() + u.dropFirst() }
                emit(&ids, u, aa); fi += 1
                if fi % filler.count == 0 { lap += 1 }
            }
            return ids
        }
        Task { @MainActor in
            let hist = buildHistory()
            print("[SP] V2NEEDLE history=\(hist.count) tok  thresh=\(v2Thresh.map { String($0) } ?? "default -2.82")")
            var gateFire = 0, retrHit = 0, firedAndHit = 0, primedFire = 0, primedHit = 0
            for n in needles {
                let seq = hist + sp.tokenizer.encode(text: "<｜end▁of▁sentence｜><｜User｜>\(n.q)<｜Assistant｜>",
                                                     addSpecialTokens: false)
                let (fired, ids) = await rv.recall(seq: seq, ignoreGate: true)   // always retrieve to score retrieval
                let startScore = rv.lastScore
                let txt = sp.tokenizer.decode(tokens: ids).lowercased()
                let hit = txt.contains(n.val.lowercased())
                if fired { gateFire += 1 }
                if hit { retrHit += 1 }
                if fired && hit { firedAndHit += 1 }
                // PRIMED: gate at the by-construction true-recall position (answer stub ends at the fact).
                let pseq = hist + sp.tokenizer.encode(
                    text: "<｜end▁of▁sentence｜><｜User｜>\(n.q)<｜Assistant｜><think>\n\n</think>\n\n\(n.lead)",
                    addSpecialTokens: false)
                let (pfired, pids) = await rv.recall(seq: pseq, ignoreGate: true)
                let primeScore = rv.lastScore
                let phit = sp.tokenizer.decode(tokens: pids).lowercased().contains(n.val.lowercased())
                if pfired { primedFire += 1 }
                if phit { primedHit += 1 }
                print(String(format: "[SP] V2NEEDLE start=%.2f %@  PRIMED=%.2f %@  retr@start=%@ retr@primed=%@ val=%@  Q: %@",
                             startScore, fired ? "FIRE" : "skip", primeScore, pfired ? "FIRE" : "skip",
                             hit ? "HIT" : "miss", phit ? "HIT" : "miss", n.val, n.q))
            }
            for n in needles.prefix(4) {
                let pseq = hist + sp.tokenizer.encode(
                    text: "<｜end▁of▁sentence｜><｜User｜>\(n.q)<｜Assistant｜><think>\n\n</think>\n\n\(n.lead)",
                    addSpecialTokens: false)
                let d = await rv.debugRetrieve(seq: pseq, value: n.val, queryText: n.q)
                print("[SP] V2NEEDLE DBG nB=\(d.nBlocks) trueBlk=\(d.trueIdx) bridgeRank=\(d.bridgeRank) rawBgeRank=\(d.rawRank) gate=\(String(format: "%.2f", d.gateScore))  val=\(n.val)")
            }
            var chatFire = 0
            for q in chats {
                let seq = hist + sp.tokenizer.encode(text: "<｜end▁of▁sentence｜><｜User｜>\(q)<｜Assistant｜>",
                                                     addSpecialTokens: false)
                let (fired, _) = await rv.recall(seq: seq, ignoreGate: false)
                if fired { chatFire += 1 }
                print(String(format: "[SP] V2NEEDLE chat   score=%.2f %@  Q: %@", rv.lastScore, fired ? "FIRE" : "skip", q))
            }
            print("[SP] V2NEEDLE SUMMARY start-fire=\(gateFire)/\(needles.count) PRIMED-fire=\(primedFire)/\(needles.count) retr-hit@primed=\(primedHit)/\(needles.count) retr-hit@start=\(retrHit)/\(needles.count) fired&hit=\(firedAndHit)/\(needles.count) chat-falsefire=\(chatFire)/\(chats.count)")
            print("[SP] V2NEEDLE_TEST DONE")
        }
    }

    /// SP_RECALL_ONLY: routing-free recall measurement (`ルーティングは切って 単純にリコールだけの性能を見たい`).
    /// Same offsite conversation as SP_CONVO_TEST, but EVERY turn runs through `a.recallTurn` — no
    /// routeIntent, no cleanQuote/recall/lookup/fact branches. Recall is handled solely by the gate +
    /// archive. Logs gate fire/score + recallTok + answer per turn so pure recall can be judged
    /// against the router-driven SP_CONVO_TEST.
    func runRecallOnlyTest() {
        guard let a = assistant else { return }
        let convo = [
            "I'm organizing a 3-day team offsite to Hakone next month.",
            "There are 12 people on the team.",
            "The total budget is 4200 dollars.",
            "We want a place with an onsen and a meeting room.",
            "What team-building activities would you suggest?",
            "Any tips for the food, since some of the team are vegetarian?",
            "How far is Hakone from Tokyo by train, roughly?",
            "What should everyone pack for March weather there?",
            "What's a good way to split the budget across lodging, food, and activities?",
            "Can you summarize the plan so far?",
            "Remind me, how many people are coming again?",        // recall-needing
            "And what was the total budget I mentioned earlier?",  // recall-needing
            "So how much is that per person?",                     // CoT using recalled facts
            "Thanks — what was our destination again?",            // recall-needing
        ]
        let noGate = ProcessInfo.processInfo.environment["SP_RECALL_NOGATE"] != nil
        let useV2 = ProcessInfo.processInfo.environment["SP_RECALL_V2"] != nil
        // RECALL_V2 §4: 4-bit shifts the gate logit DOWN (separation unchanged). Threshold is set by
        // DYNAMIC startup calibration (10th-percentile of needle recall-peaks). SP_V2_THRESH overrides
        // (skips calibration); SP_V2_CALIB_PCT sets the percentile.
        let v2Thresh = ProcessInfo.processInfo.environment["SP_V2_THRESH"].flatMap { Float($0) }
        let calibPct = ProcessInfo.processInfo.environment["SP_V2_CALIB_PCT"].flatMap { Double($0) } ?? 10
        Task { @MainActor in
            a.recallGateEnabled = !noGate     // SP_RECALL_NOGATE → force a retrieve EVERY turn
            a.useRecallV2 = useV2 && a.recallV2 != nil
            if a.useRecallV2 {
                if let t = v2Thresh { a.recallV2!.threshOverride = t }
                else {
                    let t = await a.calibrateRecallThreshold(percentile: calibPct)
                    if let pk = a.lastCalibration?.peaks {
                        print("[SP] RECALL_ONLY calib peaks=[\(pk.map { String(format: "%.1f", $0) }.joined(separator: ","))] → thresh=\(t.map { String(format: "%.2f", $0) } ?? "nil")")
                    }
                }
            }
            a.resetConversation()
            let mode = a.useRecallV2
                ? "V2(in-gen gate+bridge, thresh=\(v2Thresh.map { String($0) } ?? String(format: "%.2f(calib)", a.recallV2?.threshOverride ?? 0)))"
                : (noGate ? "FORCED-OFF (retrieve every turn)" : (a.gate != nil ? "v3-gate" : "OFF"))
            var opts = SPModel.Options()
            opts.temp = 0.6; opts.rw = 512; opts.maxD = 4096; opts.genLen = 300; opts.seed = 0
            print("[SP] RECALL_ONLY  mode=\(mode) archive=\(a.archive != nil ? "on" : "OFF") recallV2=\(a.recallV2 != nil ? "loaded" : "MISSING")")
            for q in convo {
                let r = await a.recallTurn(q, options: opts)
                let g = String(format: "gate=%@(%.2f)", a.lastGateFired ? "FIRE" : "skip", a.lastGateScore)
                print("[SP] RECALL_ONLY gen=\(a.genTokenCount) kept=\(a.keptTokenCount) \(g) recallTok=\(a.lastRecallTokens)  Q: \(q)")
                print("[SP] RECALL_ONLY    A: \(r.answer.replacingOccurrences(of: "\n", with: " ").prefix(220))")
            }
            print("[SP] RECALL_ONLY_TEST DONE")
        }
    }

    /// SP_GATE_TEST: with evicted facts in the history, the recall gate should FIRE on recall
    /// questions and SKIP chitchat (addendum: recall 10/10 fire, negatives ~8/10 skip).
    func runGateTest() {
        guard let a = assistant, let gate = a.gate else { print("[SP] GATE no gate loaded"); return }
        let sp = a.sp
        var hist = [Int]()
        func emit(_ t: String) {
            hist.append(contentsOf: sp.tokenizer.encode(
                text: (hist.isEmpty ? "" : "<｜end▁of▁sentence｜>") + t, addSpecialTokens: false))
        }
        emit("<｜User｜>By the way, the wifi password at the mountain cabin was trout-9182.<｜Assistant｜>Noted, thanks for telling me.")
        let filler = "<｜User｜>The weather has been all over the place lately.<｜Assistant｜>It really has — the shifts make planning hard, though the cool evenings after rain are pleasant for a walk."
        while hist.count < 2000 { emit(filler) }
        emit("<｜User｜>Dr. Imai's office extension is 4408.<｜Assistant｜>Got it, noted.")
        while hist.count < 4000 { emit(filler) }
        let kept = Array(hist.prefix(max(0, hist.count - 512)))
        let window = Array(hist.suffix(512))
        let recallQs = ["What was the wifi password at the mountain cabin?",
                        "What is Dr. Imai's office extension?"]
        let chatQs = ["Hello, how are you today?", "Can you tell me a fun fact?", "What should I cook for dinner?"]
        Task { @MainActor in
            print(String(format: "[SP] GATE thresh=%.2f", gate.thresh))
            for q in recallQs {
                let s = gate.score(kept: kept, window: window, query: q)
                print(String(format: "[SP] GATE recall score=%.2f %@  %@", s, s > gate.thresh ? "FIRE" : "skip", q))
            }
            for q in chatQs {
                let s = gate.score(kept: kept, window: window, query: q)
                print(String(format: "[SP] GATE chat   score=%.2f %@  %@", s, s > gate.thresh ? "FIRE" : "skip", q))
            }
            print("[SP] GATE_TEST DONE")
        }
    }

    /// SP_ROUTE_TEST: route CLEAN English oil-price phrasings through intent_head (no translation,
    /// no generation) to isolate router-vs-translation. If these don't all go to `lookup`, the
    /// router itself is unstable regardless of translation quality.
    func runRouteTest() {
        guard let a = assistant else { return }
        let phrasings = [
            "What is today's oil price?",
            "I want to know today's oil price.",
            "I'm asking about today's oil price.",
            "Can you tell me the crude oil price today?",
            "Please search the web for the current oil price.",
            "I want to know the oil price on June 12, 2026.",
            "tell me the current price of crude oil",
            "what's the price of crude oil right now",
        ]
        Task {
            for p in phrasings {
                let doc = await a.bge.encode(p, isQuery: false)
                let intent = routeIntent(p, intentHead: a.intentHead, docEmbedding: doc)
                print("[SP] ROUTE intent=\(intent)  q=\"\(p)\"")
            }
            print("[SP] ROUTE_TEST DONE")
        }
    }

    /// SP_NEEDLE_TEST: Block Recall acceptance test (port of needle_recall_test.py). 10 needles
    /// buried in ~4.2k tokens of filler (so they're past the 512 window = SP-gist only). Compares
    /// `sp` (no recall) vs `bge` (Block Recall). Expect sp ≈ 0/10, bge ≈ 7/10.
    func runNeedleTest() {
        guard let a = assistant else { return }
        let sp = a.sp
        let bgeArch = BlockArchive(tokenizer: sp.tokenizer, bge: a.bge)
        let ensArch: EnsembleArchive? = {
            guard let ix = Bundle.main.url(forResource: "phaseb_indexer", withExtension: "safetensors")
            else { return nil }
            return try? EnsembleArchive(model: sp.model, tokenizer: sp.tokenizer, bge: a.bge,
                                        bos: sp.tokenizer.bosTokenId ?? 151646, indexerURL: ix)
        }()
        let needles: [(user: String, q: String, val: String)] = [
            ("By the way, the wifi password at the mountain cabin was trout-9182. Took us forever to find it.", "What was the wifi password at the mountain cabin?", "9182"),
            ("I finally reached Dr. Imai today - her office extension is 4408.", "What is Dr. Imai's office extension?", "4408"),
            ("The rover prototype came in at 18.4 kg on the bench scale this morning.", "How much did the rover prototype weigh?", "18.4"),
            ("My cousin finally named his fishing boat: the Yume-Maru.", "What is my cousin's fishing boat called?", "yume"),
            ("Parking at the venue was a maze - we ended up in spot B-37.", "Which parking spot did we end up in at the venue?", "37"),
            ("That limited edition sake was 6800 yen, totally worth it though.", "How much did the limited edition sake cost?", "6800"),
            ("The night train to Aomori leaves at 23:46, I double checked.", "What time does the night train to Aomori leave?", "23:46"),
            ("The projector in conference room A is an XV-310, in case anyone asks.", "What model is the projector in conference room A?", "310"),
            ("Grandma's bread recipe needs exactly 320 grams of flour, she was very firm about that.", "How many grams of flour does grandma's bread recipe need?", "320"),
            ("We turned back at trail marker 58 because of the fog.", "At which trail marker did we turn back?", "58"),
        ]
        let filler: [(String, String)] = [
            ("The weather this week has been all over the place, sunny then storms.", "It really has. Those sudden shifts make it hard to plan anything outdoors, though the cool evenings after the rain have been pleasant for walks."),
            ("I tried that new ramen place near the station.", "How was it? A good bowl of ramen after a long day is hard to beat - the broth makes or breaks it in my experience."),
            ("Work has been busy but manageable lately.", "That sounds like a decent balance. Busy enough to stay engaged without burning out is the sweet spot most people aim for."),
            ("I started reading a novel about Antarctic explorers.", "Polar exploration stories can be gripping - the isolation and the small daily decisions that decide everything give them real tension."),
            ("My neighbor's dog learned to open the gate somehow.", "Clever dog! Once they figure out latches there is no stopping them - you may need a second clip higher up where paws cannot reach."),
            ("We are thinking about repainting the kitchen.", "A fresh coat changes the whole feel of a room. Lighter colors tend to make small kitchens feel bigger, especially with warm lighting."),
            ("The train was delayed twenty minutes this morning.", "Annoying way to start the day. At least with a book or some music the platform wait goes a little faster."),
            ("I have been trying to drink more water instead of coffee.", "A solid habit. Keeping a full bottle on the desk where you can see it does most of the work, honestly."),
            ("The garden tomatoes are finally turning red.", "Nothing beats a sun-warmed tomato straight off the vine. The first ripe one of the season always feels like a small victory."),
            ("I watched a documentary about deep sea creatures last night.", "The deep sea is wilder than fiction - bioluminescent lures, pressure-proof bodies, whole ecosystems around vents. What stuck with you most?"),
            ("My phone battery barely lasts the afternoon now.", "Batteries do fade after a couple of years. Lowering screen brightness and trimming background apps can buy you a few more months."),
            ("We played board games with friends over the weekend.", "Board game nights are underrated. The slow conversational pace between turns is half the fun, win or lose."),
            ("There was a small earthquake here yesterday, nothing serious.", "Glad it was minor. Always a good prompt to check the emergency kit and strap down the tall furniture, just in case."),
            ("I am learning to make sourdough bread.", "A rewarding rabbit hole. Keeping the starter happy is the hard part - consistent feeding times matter more than fancy flour."),
            ("The library extended its evening hours this month.", "That is great news for anyone who works late. Quiet evening hours at a library are some of the best focused time there is."),
            ("My bicycle needs a new chain, it keeps slipping.", "A slipping chain usually means it has stretched past its service life. Replacing it early also saves the cassette from extra wear."),
        ]
        func buildHistory() -> [Int] {
            var pairs: [(String, String)] = []
            var fi = 0
            for n in needles {
                pairs.append(filler[fi % filler.count]); fi += 1
                pairs.append((n.user, "Noted - that sounds like quite a day. Thanks for telling me."))
            }
            var ids = [Int]()
            func emit(_ u: String, _ a: String) {
                let t = (ids.isEmpty ? "" : "<｜end▁of▁sentence｜>") + "<｜User｜>\(u)<｜Assistant｜>\(a)"
                ids.append(contentsOf: sp.tokenizer.encode(text: t, addSpecialTokens: false))
            }
            for (u, a) in pairs { emit(u, a) }
            var lap = 0
            while ids.count < 4200 {
                var (u, a) = filler[fi % filler.count]
                if lap > 0, let f = u.first { u = "One more thing about that - " + String(f).lowercased() + u.dropFirst() }
                emit(u, a); fi += 1
                if fi % filler.count == 0 { lap += 1 }
            }
            return ids
        }
        Task { @MainActor in
            let hist = buildHistory()
            print("[SP] NEEDLE history=\(hist.count) tokens")
            var opts = SPModel.Options()
            opts.temp = 0.05; opts.rw = 512; opts.maxD = 4096; opts.genLen = 160; opts.seed = 0
            print("[SP] NEEDLE ensemble-indexer loaded=\(ensArch != nil)")
            for cond in ["sp", "bge", "ens"] {
                let archive: RecallArchive? = cond == "bge" ? bgeArch : (cond == "ens" ? ensArch : nil)
                if cond == "ens", archive == nil { print("[SP] NEEDLE ens skipped (no indexer)"); continue }
                var hits = 0
                for n in needles {
                    var st = SPModel.GenState(); st.gen = hist
                    var rec: [Int] = []
                    if let archive {
                        archive.reset()
                        let target = max(0, hist.count - 512)
                        if target > 0 { await archive.sync(absorbed: Array(hist.prefix(target))) }
                        rec = await archive.retrieve(query: n.q)
                    }
                    // match reference needle harness: direct answer (force_think=false), no salvage prime
                    let (ans, _) = sp.genOnce(n.q, state: &st, options: opts, firstTurn: false,
                                              recIds: rec, forceThink: false, salvage: "", salvageBudget: 56)
                    let ok = ans.lowercased().replacingOccurrences(of: ",", with: "").contains(n.val.lowercased())
                    if ok { hits += 1 }
                    print("[SP] NEEDLE \(cond) [\(ok ? "OK  " : "MISS")] val=\(n.val) rec=\(rec.count) :: \(ans.replacingOccurrences(of: "\n", with: " ").prefix(90))")
                }
                print("[SP] NEEDLE === \(cond): \(hits)/\(needles.count) ===")
            }
            print("[SP] NEEDLE_TEST DONE")
        }
    }

    /// SP_STUCK_TEST: does a persistent SP-evict genState contaminate later chitchat turns?
    /// Same input ("Hello") run before vs after a poisoning fragment — if the later answers
    /// regurgitate the fragment's answer, the conversation stream is the cause (not the model alone).
    func runStuckTest() {
        guard let a = assistant else { return }
        // canonical acceptance procedure: a BARE-NUMERIC answer (recall "8042") then an immediate
        // greeting — the bare token must NOT become a copy attractor the greeting echoes.
        let seq = ["My locker code is 8042.", "What is my locker code?", "Hello", "Hello"]
        Task { @MainActor in
            print("[SP] STUCK triggerEnabled=\(a.triggerEnabled)")
            var opts = SPModel.Options()
            opts.temp = 0.6; opts.rw = 512; opts.maxD = 4096; opts.genLen = 300; opts.seed = 0
            for (i, q) in seq.enumerated() {
                let r = await a.turn(q, options: opts)
                print("[SP] STUCK [\(i)] q=\"\(q)\" → \(r.answer.replacingOccurrences(of: "\n", with: " ").prefix(120))")
            }
            print("[SP] STUCK_TEST DONE")
        }
    }

    /// SP_ANAPHORA_TEST: verify value-anaphora web-query expansion (finding B fix).
    func runAnaphoraTest() {
        let cases: [(q: String, session: [String])] = [
            ("Can you verify the number?", ["Brent crude is $82.40 today."]),
            ("best temples to visit there?", ["Tell me about Kyoto."]),
            ("is that price right?", ["The current price of crude oil is $99.29/barrel."]),
            ("what is 12 times 3?", ["Earlier computed result: 36."]),   // has own content → no expand
        ]
        for c in cases {
            let out = expandWebQuery(c.q, pins: [], session: c.session)
            print("[SP] ANAPH in=\"\(c.q)\"  ctx=\(c.session)  =>  \"\(out)\"")
        }
        print("[SP] ANAPHORA_TEST DONE")
    }

    /// Index a folder of local files (.txt/.md/.pdf) as a `lookup` knowledge tier. Cross-platform.
    func indexFolder(_ url: URL, then: (@MainActor @Sendable () -> Void)? = nil) {
        guard let a = assistant, !busy else { return }
        busy = true
        let bge = a.bge
        let needsScope = url.startAccessingSecurityScopedResource()
        messages.append(ChatMessage(role: .system, text: "📁 Indexing \(url.lastPathComponent)…"))
        Task.detached(priority: .userInitiated) {
            let idx = LocalFileIndex(bge: bge)
            let n = await idx.build(from: url)
            await MainActor.run {
                a.localIndex = idx
                if !self.messages.isEmpty {
                    self.messages[self.messages.count - 1] = ChatMessage(role: .system,
                        text: "📁 Indexed \(idx.indexedFiles) files (\(n) chunks) from \(url.lastPathComponent). Your files are now searched for questions.")
                }
                if needsScope { url.stopAccessingSecurityScopedResource() }
                self.busy = false
                then?()
            }
        }
    }

    /// Demo: save a fact → flood the context (SP compresses it) → recall it verbatim.
    func autoDemo() {
        sendTurn("Remember: my locker code is 4821.") { [weak self] in
            guard let self else { return }
            self.sendTurn(self.ragContext()) { [weak self] in
                self?.sendTurn("What is my locker code?")
            }
        }
    }

    private func ragContext() -> String {
        let passage = "The east wing corridor connects to the atrium and the supply room is on level 2. "
        // session-only distractor (no persist verb) — just pushes the saved fact out of the raw window
        return "Background reading: " + String(repeating: passage, count: 120)
            + " Anyway, here is a long passage to read."
    }

    static func deviceLabel() -> String {
        switch MLX.Device.defaultDevice().deviceType {
        case .gpu: return "GPU (Metal)"
        case .cpu: return "CPU"
        case .none: return "unknown"
        }
    }

    struct AppError: Error, CustomStringConvertible {
        let msg: String
        init(_ m: String) { msg = m }
        var description: String { msg }
    }
}
