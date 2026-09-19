import sys

from src.agent import run_agent


def main():
    message = " ".join(sys.argv[1:]).strip()
    if not message:
        message = input("Enter a support request: ").strip()

    if not message:
        print("No support request was provided.")
        return

    try:
        result = run_agent(message)
    except Exception as error:
        print(f"Support agent failed: {error}")
        return

    print(result["answer"])
    print(f"Trace ID: {result['trace_id']}")
    print(f"Latency: {result['latency_ms']} ms")


if __name__ == "__main__":
    main()
