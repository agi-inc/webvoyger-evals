import time
from typing import TYPE_CHECKING

from browser_use import AgentHistoryList
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI

if TYPE_CHECKING:
    from run_browser_use import EvalResult


SYSTEM_PROMPT = """As an evaluator, you will be presented with three primary components to assist you in your role:

1. Web Task Instruction: This is a clear and specific directive provided in natural language, detailing the online activity to be carried out. These requirements may include conducting searches, verifying information, comparing prices, checking availability, or any other action relevant to the specified web service (such as Amazon, Apple, ArXiv, BBC News, Booking etc).

2. Result Screenshots: This is a visual representation of the screen showing the result or intermediate state of performing a web task. It serves as visual proof of the actions taken in response to the instruction, and may not represent everything the agent sees.

3. Result Response: This is a textual response obtained after the execution of the web task. It serves as textual result in response to the instruction.

-- You DO NOT NEED to interact with web pages or perform actions such as booking flights or conducting searches on websites.
-- You SHOULD NOT make assumptions based on information not presented in the screenshot when comparing it to the instructions. If you cannot find any information in the screenshot that matches the instruction, you can believe the information in the response.
-- Your primary responsibility is to conduct a thorough assessment of the web task instruction against the outcome depicted in the screenshot and in the response, evaluating whether the actions taken align with the given instructions.
-- NOTE that the instruction may involve more than one task, for example, locating the garage and summarizing the review. Failing to complete either task, such as not providing a summary, should be considered unsuccessful.
-- NOTE that the screenshot is authentic, but the response provided by LLM is generated at the end of web browsing, and there may be discrepancies between the text and the screenshots.
-- Note the difference: 1) Result response may contradict the screenshot, then the content of the screenshot prevails, 2) The content in the Result response is not mentioned on the screenshot, choose to believe the content.
-- If you are not sure whether you should believe the content in the response, you should choose unknown.

You should elaborate on how you arrived at your final evaluation and then provide a definitive verdict on whether the task has been successfully accomplished, either as 'SUCCESS', 'NOT SUCCESS', or 'UNKNOWN'."""

USER_PROMPT = """TASK: <task>
Result Response: <answer>
<num> screenshot at the end: """


async def auto_eval_by_gpt4o(
    history: AgentHistoryList,
    task: str,
    openai_client: ChatOpenAI | AzureChatOpenAI | ChatAnthropic | ChatGoogleGenerativeAI,
) -> tuple["EvalResult", str]:
    # print(f"--------------------- {process_dir} ---------------------")

    if not history.is_done():
        return "failed", ""

    answer = history.final_result()
    if answer is None:
        return "failed", ""

    screenshots = history.screenshots()[-4:]
    screenshot_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot}"},
        }
        for screenshot in screenshots
    ]

    # Prepare GPT-4V messages
    user_prompt_tmp = USER_PROMPT.replace("<task>", task)
    user_prompt_tmp = user_prompt_tmp.replace("<answer>", answer)
    user_prompt_tmp = user_prompt_tmp.replace("<num>", str(len(screenshots)))

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {"type": "text", "text": user_prompt_tmp},
                *screenshot_content,
                {"type": "text", "text": "Your verdict:\n"},
            ]
        ),
    ]

    while True:
        try:
            # print("Calling gpt4v API to get the auto evaluation......")
            response = await openai_client.ainvoke(messages)
            # print("API call complete...")
            break
        except Exception as e:
            print(e)
            if type(e).__name__ == "RateLimitError":
                time.sleep(10)
            elif type(e).__name__ == "APIError":
                time.sleep(15)
            elif type(e).__name__ == "InvalidRequestError":
                exit(0)
            else:
                time.sleep(10)

    gpt_4v_res = str(response.content)

    if gpt_4v_res is None:
        return "unknown", ""
    elif "NOT SUCCESS" in gpt_4v_res:
        auto_eval_res = "failed"
    elif "SUCCESS" in gpt_4v_res:
        auto_eval_res = "success"
    elif "UNKNOWN" in gpt_4v_res:
        auto_eval_res = "unknown"
    else:
        auto_eval_res = "failed"

    return auto_eval_res, gpt_4v_res
