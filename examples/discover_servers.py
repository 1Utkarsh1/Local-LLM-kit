"""Find running local LLM servers and report what they can actually do.

Runs offline: it only probes loopback endpoints and never downloads or infers.
If nothing is listening it explains what to start instead of failing.

    python examples/discover_servers.py
    python examples/discover_servers.py --url localhost:9000
    python examples/discover_servers.py --json
    python examples/discover_servers.py --require-local   # exit 1 if not on-device
"""

import argparse
import json
import sys

from local_llm_kit import LLM, discover, find_local_server


def describe(server):
    print("\n%s  [%s]  %sms" % (server.base_url, server.provider, server.latency_ms))
    if server.context_window:
        print("  server context window: %d tokens" % server.context_window)
    if server.vision is not None:
        print("  server reports vision: %s" % ("yes" if server.vision else "no"))
    for model in server.models:
        bits = []
        if model.context_length:
            bits.append("ctx %d" % model.context_length)
        if model.capabilities:
            bits.append(", ".join(model.capabilities))
        suffix = ""
        if model.remote:
            suffix = "   OFF-DEVICE -> %s" % (model.remote_host or "remote")
        print("  %-40s %s%s" % (model.id[:40], "  ".join(bits), suffix))
    for note in server.notes:
        print("  note: %s" % note)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        dest="urls",
        action="append",
        default=None,
        help="extra endpoint to probe (repeatable)",
    )
    parser.add_argument("--timeout", type=float, default=0.75)
    parser.add_argument("--json", action="store_true", help="print the raw report")
    parser.add_argument(
        "--require-local",
        action="store_true",
        help="exit non-zero unless an on-device model is available",
    )
    parser.add_argument(
        "--chat", action="store_true", help="also run one prompt against the first on-device model"
    )
    args = parser.parse_args(argv)

    report = discover(extra_urls=args.urls, timeout=args.timeout)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(
            "Probed %d endpoints in %sms; %d responded."
            % (report.probed, report.elapsed_ms, len(report.live))
        )
        for server in report.live:
            describe(server)
        for server in report.offline:
            if server.status is not None:
                print("\nReachable but not ready: %s (%s)" % (server.base_url, server.error))

    on_device = report.on_device()
    if not report.live:
        print("\nNo local LLM server answered. Start one, for example:")
        print("  ollama serve && ollama pull qwen2.5:0.5b")
        print("  python examples/discover_servers.py")
    elif not on_device:
        print("\nServers are up, but none offers a model that runs on this machine.")
    else:
        print("\nOn-device model available on %s" % on_device[0].provider)

    if args.chat:
        # find_local_server() is the convenience form of the loop above.
        server = find_local_server(extra_urls=args.urls, timeout=args.timeout)
        if server is None:
            print("\n--chat needs a server offering an on-device model.")
            return 1
        model = server.local_models[0]
        print("\nPrompting %r (this loads the model and can take a moment)..." % model.id)
        llm = LLM(model.id, backend=server.provider, require_local=True)
        response = llm.chat(
            [{"role": "user", "content": "Say hello in five words."}], max_tokens=32
        )
        print("  ->", response["choices"][0]["message"]["content"].strip())

    if args.require_local:
        return 0 if on_device else 1
    return 0 if report.live else 1


if __name__ == "__main__":
    sys.exit(main())
