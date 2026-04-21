import sys
from datetime import datetime
from model import SalesLLM


def main():
    print("🚀 Initializing Sales Assistant...")
    llm = SalesLLM()

    today = datetime.today().strftime("%d/%m/%Y (%A)")
    print(f"\n📅 Today: {today}")
    print("\n" + "=" * 60)
    print("🤖 SALES CRM AI IS ONLINE!")
    print("Ask anything about enquiries, appointments, or feedback.")
    print("Type 'exit' or 'quit' to close.")
    print("=" * 60 + "\n")

    while True:
        user_input = input("👤 You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("👋 Shutting down. Goodbye!")
            sys.exit()

        if not user_input:
            continue

        print("🤖 AI: ", end="")
        llm.chat_with_data(user_input)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()