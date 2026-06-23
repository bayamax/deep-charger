import SwiftUI

@main
struct HypernetSPAppApp: App {
    init() {
        // Unbuffered stdout so `print()` reaches `devicectl --console` live on device.
        setvbuf(stdout, nil, _IONBF, 0)
    }
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
