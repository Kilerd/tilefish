// A consumer of the installed package: it loads a manifest with its pin through tilefish.hpp.
#include <cstdio>

#include "tilefish.hpp"

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    try {
        auto manifest = tilefish::Manifest::load(argv[1], argv[2]);
        std::printf("consumer: manifest %s (tilefish %s, ABI %u)\n", manifest.digest().c_str(), tilefish_version(), tilefish_abi_version() >> 16);
        return manifest.digest() == argv[2] ? 0 : 1;
    } catch (const tilefish::Error& e) {
        std::fprintf(stderr, "consumer: %s: %s\n", e.status_name(), e.what());
        return 1;
    }
}
