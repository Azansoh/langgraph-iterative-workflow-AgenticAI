
import streamlit as st
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
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LinkedIn AI Writer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main page */
    .stApp {
        background-color: #f6f8fc;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* Header */
    .top-badge {
        display: inline-block;
        background-color: #e8f1ff;
        color: #1769e0;
        padding: 7px 14px;
        border-radius: 30px;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .main-title {
        font-size: 44px;
        font-weight: 800;
        color: #172033;
        line-height: 1.1;
        margin-bottom: 8px;
    }

    .main-subtitle {
        font-size: 17px;
        color: #667085;
        line-height: 1.6;
        margin-bottom: 30px;
    }


    /* Cards */
    .custom-card {
        background-color: white;
        border: 1px solid #e4e7ec;
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 18px rgba(16, 24, 40, 0.05);
    }

    .card-title {
        color: #172033;
        font-size: 20px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .card-subtitle {
        color: #667085;
        font-size: 14px;
        margin-bottom: 15px;
    }


    /* Workflow */
    .workflow-step {
        background-color: #f8fafc;
        border: 1px solid #eaecf0;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 10px;
    }

    .workflow-number {
        font-size: 13px;
        color: #1769e0;
        font-weight: 700;
    }

    .workflow-name {
        color: #172033;
        font-weight: 700;
        font-size: 15px;
    }

    .workflow-description {
        color: #667085;
        font-size: 12px;
        margin-top: 3px;
    }


    /* Statistics */
    .stat-card {
        background-color: white;
        border: 1px solid #e4e7ec;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0px 3px 12px rgba(16, 24, 40, 0.04);
    }

    .stat-value {
        color: #1769e0;
        font-size: 28px;
        font-weight: 800;
    }

    .stat-label {
        color: #667085;
        font-size: 13px;
        margin-top: 3px;
    }


    /* Final post */
    .post-container {
        background-color: white;
        border: 1px solid #e4e7ec;
        border-radius: 18px;
        padding: 30px;
        box-shadow: 0px 5px 20px rgba(16, 24, 40, 0.06);
        color: #172033;
        font-size: 16px;
        line-height: 1.75;
        white-space: pre-wrap;
    }


    /* Approved */
    .approved-box {
        background-color: #ecfdf3;
        border: 1px solid #abefc6;
        color: #067647;
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 650;
        margin-bottom: 20px;
    }


    /* Rejected */
    .rejected-box {
        background-color: #fff7ed;
        border: 1px solid #fed7aa;
        color: #c2410c;
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 650;
        margin-bottom: 20px;
    }


    /* Feedback */
    .feedback-box {
        background-color: #f8fafc;
        border-left: 4px solid #1769e0;
        padding: 18px;
        border-radius: 10px;
        color: #344054;
        line-height: 1.6;
    }


    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #e4e7ec;
    }


    /* Text area */
    textarea {
        border-radius: 12px !important;
        border: 1px solid #d0d5dd !important;
    }


    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 45px;
    }


    /* Footer */
    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 13px;
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid #e4e7ec;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# TOOLS
# ============================================================

search_tool = TavilySearch(
    max_results=3
)

tools = [search_tool]


# ============================================================
# LLM CONFIGURATION
# ============================================================

writer_llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=0.7
)

writer_llm_with_tools = writer_llm.bind_tools(
    tools
)


reviewer_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2
)


# ============================================================
# STATE
# ============================================================

class State(TypedDict):

    topic: str

    messages: Annotated[
        list,
        add_messages
    ]

    draft: str

    review_feedback: str

    is_approved: bool

    attempt: int


# ============================================================
# WRITER PROMPT
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

    attempt = state.get(
        "attempt",
        0
    ) + 1

    topic = state["topic"]

    previous_feedback = state.get(
        "review_feedback",
        ""
    )


    if attempt == 1:

        user_message = (
            f"Write a LinkedIn post on this topic:\n\n"
            f"{topic}\n\n"
            f"If current information is required, "
            f"use the web search tool first."
        )

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


    # IMPORTANT:
    # Only send the current request.
    # Do not send accumulated message history.

    messages = [
        ("system", WRITER_SYSTEM_PROMPT),
        ("human", user_message)
    ]


    response = writer_llm_with_tools.invoke(
        messages
    )


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

tool_node = ToolNode(
    tools
)


# ============================================================
# EXTRACT DRAFT
# ============================================================

def extract_draft_node(
    state: State
) -> dict:

    last_message = state["messages"][-1]

    draft = last_message.content

    return {
        "draft": draft
    }


# ============================================================
# REVIEWER PROMPT
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

def reviewer_node(
    state: State
) -> dict:

    draft = state["draft"]


    # Safety limit
    if len(draft) > 12000:

        draft = draft[:12000]


    prompt = (
        "Review ONLY the following LinkedIn post.\n\n"

        "Do not consider previous messages, "
        "previous drafts, or search results.\n\n"

        "LINKEDIN POST:\n"

        f"{draft}\n\n"

        "Give your review according to "
        "the system instructions."
    )


    response = reviewer_llm.invoke(
        [
            (
                "system",
                REVIEWER_SYSTEM_PROMPT
            ),

            (
                "human",
                prompt
            )
        ]
    )


    review_text = response.content.strip()


    # ========================================================
    # VERDICT
    # ========================================================

    verdict_section = (
        review_text.upper()
        .split(
            "FEEDBACK",
            1
        )[0]
    )


    is_approved = (
        "APPROVED"
        in verdict_section
    )


    # ========================================================
    # FEEDBACK
    # ========================================================

    if "FEEDBACK:" in review_text:

        feedback = review_text.split(
            "FEEDBACK:",
            1
        )[1].strip()

    else:

        feedback = review_text


    return {

        "review_feedback": feedback,

        "is_approved": is_approved
    }


# ============================================================
# ROUTING
# ============================================================

def should_use_tool(
    state: State
):

    last_message = state["messages"][-1]


    if getattr(
        last_message,
        "tool_calls",
        None
    ):

        return "tools"


    return "extract_draft"


def should_stop_looping(
    state: State
):

    if state["is_approved"]:

        return END


    if state["attempt"] >= 3:

        return END


    return "writer"


# ============================================================
# BUILD GRAPH
# ============================================================

graph = StateGraph(
    State
)


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
# EDGES
# ============================================================

graph.add_edge(
    START,
    "writer"
)


graph.add_conditional_edges(
    "writer",
    should_use_tool
)


graph.add_edge(
    "tools",
    "writer"
)


graph.add_edge(
    "extract_draft",
    "reviewer"
)


graph.add_conditional_edges(
    "reviewer",
    should_stop_looping
)


# ============================================================
# COMPILE
# ============================================================

app = graph.compile()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ✍️ LinkedIn AI Writer"
    )

    st.markdown(
        """
        An iterative AI workflow that creates,
        reviews, and improves LinkedIn posts.
        """
    )


    st.divider()


    st.markdown(
        "### 🔄 Workflow"
    )


    st.markdown(
        """
        <div class="workflow-step">
            <div class="workflow-number">STEP 01</div>
            <div class="workflow-name">📝 AI Writer</div>
            <div class="workflow-description">
                Gemini creates the initial LinkedIn post.
            </div>
        </div>

        <div class="workflow-step">
            <div class="workflow-number">STEP 02</div>
            <div class="workflow-name">🌐 Web Search</div>
            <div class="workflow-description">
                Tavily provides fresh information when required.
            </div>
        </div>

        <div class="workflow-step">
            <div class="workflow-number">STEP 03</div>
            <div class="workflow-name">🔎 Reviewer</div>
            <div class="workflow-description">
                Groq evaluates the generated post.
            </div>
        </div>

        <div class="workflow-step">
            <div class="workflow-number">STEP 04</div>
            <div class="workflow-name">🔄 Iteration</div>
            <div class="workflow-description">
                Rejected posts are rewritten using feedback.
            </div>
        </div>

        <div class="workflow-step">
            <div class="workflow-number">STEP 05</div>
            <div class="workflow-name">✅ Approval</div>
            <div class="workflow-description">
                Workflow stops when the post is approved.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.divider()


    st.markdown(
        "### 🤖 Models"
    )

    st.write("**Writer:** Gemini")

    st.write("**Reviewer:** GPT-OSS 120B")

    st.write("**Search:** Tavily")


    st.divider()


    st.info(
        "Maximum 3 writing attempts per request."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="top-badge">'
    '🤖 AGENTIC AI • ITERATIVE WORKFLOW'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title">'
    'LinkedIn AI Post Generator'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-subtitle">'
    'Create professional LinkedIn content using an AI writer, '
    'real-time web search, and an automated reviewer.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT CARD
# ============================================================

st.markdown(
    '<div class="custom-card">',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="card-title">'
    '📝 Enter Your Topic'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="card-subtitle">'
    'Describe what you want your LinkedIn post to be about.'
    '</div>',
    unsafe_allow_html=True
)


topic = st.text_area(
    "Topic",
    placeholder=(
        "Example:\n"
        "I recently learned LangGraph and built my first "
        "iterative AI workflow for generating LinkedIn posts."
    ),
    height=140,
    label_visibility="collapsed"
)


generate_button = st.button(
    "✨ Generate LinkedIn Post",
    type="primary",
    use_container_width=True
)


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# GENERATE
# ============================================================

if generate_button:

    if not topic.strip():

        st.warning(
            "⚠️ Please enter a topic first."
        )

    else:

        initial_state = {

            "topic": topic.strip(),

            "messages": [],

            "draft": "",

            "review_feedback": "",

            "is_approved": False,

            "attempt": 0
        }


        status_container = st.container()


        try:

            with status_container:

                st.info(
                    "🚀 Starting the iterative AI workflow..."
                )

                progress = st.progress(
                    0
                )

                status_text = st.empty()


            final_state = None


            for state in app.stream(
                initial_state,
                stream_mode="values"
            ):

                final_state = state

                attempt = state.get(
                    "attempt",
                    0
                )


                if attempt == 1:

                    progress.progress(
                        25
                    )

                    status_text.write(
                        "📝 Gemini is creating your LinkedIn post..."
                    )


                elif attempt == 2:

                    progress.progress(
                        50
                    )

                    status_text.write(
                        "🔄 Reviewer rejected the first draft. "
                        "Gemini is improving it..."
                    )


                elif attempt == 3:

                    progress.progress(
                        75
                    )

                    status_text.write(
                        "🔄 Final improvement attempt..."
                    )


                if state.get(
                    "is_approved",
                    False
                ):

                    progress.progress(
                        100
                    )

                    status_text.write(
                        "✅ Post approved by the AI reviewer!"
                    )


            if final_state:

                st.session_state["final_state"] = final_state

                st.session_state["topic"] = topic.strip()

                st.rerun()


        except Exception as e:

            progress.empty()

            status_text.empty()

            st.error(
                f"❌ An error occurred:\n\n{str(e)}"
            )


# ============================================================
# RESULTS
# ============================================================

if "final_state" in st.session_state:

    final_state = st.session_state[
        "final_state"
    ]


    draft = final_state.get(
        "draft",
        ""
    )


    feedback = final_state.get(
        "review_feedback",
        ""
    )


    approved = final_state.get(
        "is_approved",
        False
    )


    attempts = final_state.get(
        "attempt",
        0
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    st.markdown(
        "## 📊 Generation Summary"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-value">
                    {attempts}
                </div>
                <div class="stat-label">
                    Writing Attempts
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        if approved:

            value = "✓"

            label = "Approved"

        else:

            value = "!"

            label = "Needs Review"


        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-value">
                    {value}
                </div>
                <div class="stat-label">
                    {label}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-value">
                    AI
                </div>
                <div class="stat-label">
                    Iterative Workflow
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("")


    # ========================================================
    # STATUS
    # ========================================================

    if approved:

        st.markdown(
            """
            <div class="approved-box">
                ✅ <strong>Post Approved!</strong><br>
                The AI reviewer found that the post meets
                the defined LinkedIn content criteria.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="rejected-box">
                ⚠️ <strong>Maximum attempts reached.</strong><br>
                The latest generated draft is shown below.
                Check the reviewer feedback before publishing.
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # FINAL POST
    # ========================================================

    st.markdown(
        "## 🚀 Final LinkedIn Post"
    )


    st.markdown(
        """
        <div class="card-subtitle">
            Your latest AI-generated LinkedIn content.
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div class="post-container">
{draft}
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("")


    # ========================================================
    # ACTION BUTTONS
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        st.download_button(
            label="📥 Download Post",
            data=draft,
            file_name="linkedin_post.txt",
            mime="text/plain",
            use_container_width=True
        )


    with col2:

        if st.button(
            "🆕 Create New Post",
            use_container_width=True
        ):

            st.session_state.pop(
                "final_state",
                None
            )

            st.session_state.pop(
                "topic",
                None
            )

            st.rerun()


    # ========================================================
    # REVIEWER FEEDBACK
    # ========================================================

    st.markdown("")


    with st.expander(
        "🔎 View AI Reviewer Feedback",
        expanded=False
    ):

        if feedback:

            st.markdown(
                f"""
                <div class="feedback-box">
                    {feedback}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.write(
                "No reviewer feedback available."
            )


    # ========================================================
    # WORKFLOW INFORMATION
    # ========================================================

    with st.expander(
        "🔄 How this post was generated",
        expanded=False
    ):

        st.write(
            f"**Topic:** {st.session_state.get('topic', '')}"
        )

        st.write(
            f"**Attempts:** {attempts}"
        )

        st.write(
            f"**Approved:** {'Yes' if approved else 'No'}"
        )

        st.write(
            "**Writer:** Gemini"
        )

        st.write(
            "**Reviewer:** GPT-OSS 120B"
        )

        st.write(
            "**Web Search:** Tavily"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Built with LangGraph • Gemini • Groq • Tavily • Streamlit
        <br><br>
        ✍️ Iterative AI Workflow for LinkedIn Content Generation
    </div>
    """,
    unsafe_allow_html=True
)

