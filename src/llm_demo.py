# from llm.llm import chat_by_api, chat_by_langchain
import llm.finetune as finetune


def main():
    # chat_by_api()
    # chat_by_langchain()
    finetune.run()


if __name__ == "__main__":
    main()
