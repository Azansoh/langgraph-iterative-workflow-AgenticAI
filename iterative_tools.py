
import os
from typing import TypedDict, Annotated

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# TOOLS
# ============================================================

search_tool = TavilySearch(max_results=3)

tools = [search_tool]


# ============================================================
# LLM CONFIGURATION
# ============================================================

writer_llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=0.7
)

writer_llm_with_tools = writer_llm.bind_tools(tools)


reviewer_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2
)


# ============================================================
# STATE
# ============================================================

class State(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int


# ============================================================
# WRITER SYSTEM PROMPT
# ============================================================

WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. "
    "Your job is to write engaging and professional LinkedIn posts "
    "about the given topic. "

    "If the topic requires up-to-date information, statistics, "
    "or current trends, use the web search tool to gather fresh "
    "context before writing. "

    "If you have received reviewer feedback, carefully address "
    "every point in the feedback. "

    "Rules for good LinkedIn posts: "
    "strong hook in the first line, "
    "one clear takeaway, "
    "easy to skim with short paragraphs, "
    "around 150-200 words, "
    "and end with a question or call-to-action "
    "to invite engagement. "

    "Do not use hashtags."
)


# ============================================================
# WRITER NODE
# ============================================================

def writer_node(state: State) -> dict:

    attempt = state.get("attempt", 0) + 1

    topic = state["topic"]

    previous_feedback = state.get(
        "review_feedback",
        ""
    )

    # --------------------------------------------------------
    # FIRST ATTEMPT
    # --------------------------------------------------------

    if attempt == 1:

        user_message = (
            f"Write a LinkedIn post on this topic:\n\n"
            f"{topic}\n\n"
            f"If current information is required, "
            f"use the web search tool first."
        )

    # --------------------------------------------------------
    # REWRITE AFTER REVIEW
    # --------------------------------------------------------

    else:

        user_message = (
            f"Write a new and improved LinkedIn post about:\n\n"
            f"{topic}\n\n"

            f"The previous draft was rejected.\n\n"

            f"Reviewer feedback:\n"
            f"{previous_feedback}\n\n"

            f"Fix every issue mentioned in the feedback. "
            f"Do not repeat the same mistakes. "
            f"Return only the new LinkedIn post."
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # DO NOT SEND OLD MESSAGE HISTORY
    # --------------------------------------------------------

    messages = [
        ("system", WRITER_SYSTEM_PROMPT),
        ("human", user_message)
    ]

    response = writer_llm_with_tools.invoke(messages)

    return {
        "messages": [
            ("human", user_message),
            response
        ],
        "attempt": attempt
    }


# ============================================================
# TOOL NODE
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# EXTRACT DRAFT NODE
# ============================================================

def extract_draft_node(state: State) -> dict:

    last_message = state["messages"][-1]

    draft = last_message.content

    print("\n\nGenerated Post:\n")
    print(draft)
    print()

    return {
        "draft": draft
    }


# ============================================================
# REVIEWER SYSTEM PROMPT
# ============================================================

REVIEWER_SYSTEM_PROMPT = (
    "You are a strict LinkedIn content reviewer. "
    "Your job is to determine whether a LinkedIn post "
    "is ready to publish. "

    "Evaluate the post using these criteria:\n"

    "1. Strong hook in the first line\n"
    "2. One clear and valuable takeaway\n"
    "3. Easy to skim with short paragraphs\n"
    "4. Roughly 150-200 words\n"
    "5. Ends with an engaging question or CTA\n"
    "6. Professional but human tone\n"
    "7. No hashtags\n\n"

    "Respond EXACTLY in this format:\n"

    "VERDICT: APPROVED or REJECTED\n"
    "FEEDBACK: <maximum 3 short sentences>\n\n"

    "Do not rewrite the post. "
    "Do not repeat the post. "
    "Do not provide a long explanation. "

    "Be strict but fair. "
    "Approve only if the post genuinely meets all criteria. "
    "Reject if any criterion is clearly missing."
)


# ============================================================
# REVIEWER NODE
# ============================================================

def reviewer_node(state: State) -> dict:

    draft = state["draft"]

    # --------------------------------------------------------
    # SAFETY LIMIT
    # --------------------------------------------------------
    # The reviewer only needs the final LinkedIn post.
    # This also prevents accidentally sending huge content
    # to the Groq API.
    # --------------------------------------------------------

    if len(draft) > 12000:

        draft = draft[:12000]

    prompt = (
        "Review ONLY the following LinkedIn post.\n\n"
        "Do not consider previous messages, "
        "previous drafts, or search results.\n\n"

        "LINKEDIN POST:\n"
        f"{draft}\n\n"

        "Give your review according to the system instructions."
    )

    response = reviewer_llm.invoke(
        [
            ("system", REVIEWER_SYSTEM_PROMPT),
            ("human", prompt)
        ]
    )

    review_text = response.content.strip()

    # --------------------------------------------------------
    # EXTRACT VERDICT
    # --------------------------------------------------------

    verdict_section = review_text.upper().split(
        "FEEDBACK",
        1
    )[0]

    is_approved = "APPROVED" in verdict_section

    # --------------------------------------------------------
    # EXTRACT FEEDBACK
    # --------------------------------------------------------

    if "FEEDBACK:" in review_text:

        feedback = review_text.split(
            "FEEDBACK:",
            1
        )[1].strip()

    else:

        feedback = review_text

    verdict = (
        "APPROVED"
        if is_approved
        else "REJECTED"
    )

    print(f"[Verdict: {verdict}]")
    print(f"[Feedback: {feedback}]")

    return {
        "review_feedback": feedback,
        "is_approved": is_approved
    }


# ============================================================
# WRITER ROUTING
# ============================================================

def should_use_tool(state: State):

    last_message = state["messages"][-1]

    if getattr(
        last_message,
        "tool_calls",
        None
    ):

        return "tools"

    return "extract_draft"


# ============================================================
# REVIEWER ROUTING
# ============================================================

def should_stop_looping(state: State):

    # --------------------------------------------------------
    # APPROVED
    # --------------------------------------------------------

    if state["is_approved"]:

        print("\nPost has been approved!\n")

        return END

    # --------------------------------------------------------
    # MAXIMUM ATTEMPTS
    # --------------------------------------------------------

    if state["attempt"] >= 3:

        print(
            "\nReached maximum attempts (3).\n"
        )

        return END

    # --------------------------------------------------------
    # REWRITE
    # --------------------------------------------------------

    return "writer"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph = StateGraph(State)


graph.add_node(
    "writer",
    writer_node
)

graph.add_node(
    "tools",
    tool_node
)

graph.add_node(
    "extract_draft",
    extract_draft_node
)

graph.add_node(
    "reviewer",
    reviewer_node
)


# ============================================================
# GRAPH EDGES
# ============================================================

graph.add_edge(
    START,
    "writer"
)


graph.add_conditional_edges(
    "writer",
    should_use_tool
)


# Tool result goes back to writer
graph.add_edge(
    "tools",
    "writer"
)


# Draft goes to reviewer
graph.add_edge(
    "extract_draft",
    "reviewer"
)


# Reviewer decides:
# END or writer
graph.add_conditional_edges(
    "reviewer",
    should_stop_looping
)


# ============================================================
# COMPILE
# ============================================================

app = graph.compile()


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 55)

    print(
        "Welcome to the LinkedIn Post Generator"
    )

    print("=" * 55)

    print(
        "\nThis tool will draft a LinkedIn post, "
        "review it, and iterate until it is approved."
    )

    print("=" * 55)


    topic = input(
        "\nWhat topic do you want a LinkedIn post about?\n> "
    ).strip()


    if not topic:

        print(
            "\nNo topic given. Exiting."
        )

    else:

        print(
            "\nStarting generation...\n"
        )


        initial_state = {

            "topic": topic,

            "messages": [],

            "draft": "",

            "review_feedback": "",

            "is_approved": False,

            "attempt": 0
        }


        final_state = app.invoke(
            initial_state
        )


        print(
            "\n" + "=" * 55
        )

        print(
            "FINAL LINKEDIN POST"
        )

        print(
            "=" * 55
        )

        print(
            final_state["draft"]
        )

        print(
            "=" * 55
        )

        print(
            f"Total attempts: "
            f"{final_state['attempt']}"
        )

        print(
            f"Approved: "
            f"{final_state['is_approved']}"
        )

        print(
            "=" * 55
        )

