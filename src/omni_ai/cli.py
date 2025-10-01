import sys

SAFE_BANNER = (
    "Omni AI (safe mode)\n"
    "This tool focuses on ethical, lawful, and beneficial use cases.\n"
    "Harmful functionality (e.g., cyber attacks, malware, unauthorized access)\n"
    "is strictly out of scope and not supported.\n"
)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in {"-h", "--help"}:
        print(SAFE_BANNER)
        print("Usage: omni-ai [command]")
        print("Commands:")
        print("  hello     Print a friendly greeting")
        return 0

    cmd = argv[0]
    if cmd == "hello":
        print("Hello from Omni AI — staying safe and helpful!")
        return 0

    print(f"Unknown command: {cmd}")
    print("Use --help for usage.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
