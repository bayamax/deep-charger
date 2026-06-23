// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "HypernetSP",
    platforms: [
        .iOS(.v16),
        .macOS(.v14),
    ],
    products: [
        .library(name: "HypernetSP", targets: ["HypernetSP"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.29.1")),
        .package(url: "https://github.com/ml-explore/mlx-swift-examples", exact: "2.29.1"),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.0.0")),
    ],
    targets: [
        .target(
            name: "HypernetSP",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
                .product(name: "MLXLLM", package: "mlx-swift-examples"),
                .product(name: "MLXEmbedders", package: "mlx-swift-examples"),
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        .testTarget(
            name: "HypernetSPTests",
            dependencies: ["HypernetSP"]
        ),
    ]
)
