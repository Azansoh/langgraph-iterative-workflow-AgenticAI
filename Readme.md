# 🔄 LangGraph Iterative Workflow – AI LinkedIn Post Generator

An **Agentic AI Iterative Workflow** built with **LangGraph** that generates, reviews, and continuously improves LinkedIn posts through an iterative feedback loop.

The workflow allows an AI writer to create a LinkedIn post, sends the generated draft to an AI reviewer, and if the reviewer rejects it, the workflow sends the feedback back to the writer to improve the post. This process continues until the post is approved or the maximum number of attempts is reached.

---

## 🚀 Project Overview

This project demonstrates an **Iterative Workflow using LangGraph**.

The system follows this process:

**Topic → AI Writer → Tool Check → Draft Extraction → AI Reviewer → Approval / Feedback → Writer Again**

If the reviewer approves the post, the workflow ends.

If the reviewer rejects the post, the writer receives the feedback and generates an improved version.

The workflow supports a maximum of **3 attempts** to prevent infinite looping.

---

## 🧠 How the Iterative Workflow Works

```text
                    ┌─────────────────┐
                    │      START      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    AI Writer    │
                    └────────┬────────┘
                             │
                     Tool Required?
                       /          \
                     Yes           No
                      │             │
                      ▼             │
               ┌──────────────┐     │
               │ Tavily Search│     │
               └──────┬───────┘     │
                      │             │
                      └──────┬──────┘
                             ▼
                    ┌─────────────────┐
                    │ Extract Draft   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   AI Reviewer    │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                 Approved          Rejected
                    │                 │
                    ▼                 ▼
                  END          Check Attempts
                                      │
                                      ▼
                                  Writer Again
                                      │
                                      └──────────► 🔄





Key Features
🔄 Iterative AI Workflow
📝 Automatic LinkedIn post generation
🤖 AI-powered post reviewing
💬 Reviewer feedback loop
🔁 Automatic post improvement
🌐 Tavily Search tool for current information
🧠 LangGraph StateGraph
🛑 Maximum 3 attempts to prevent infinite loops
✅ Automatic approval detection
🔀 Conditional workflow routing
🔐 Environment variable based API key management



🛠️ Technologies Used
Python
LangGraph
LangChain
Google Gemini
Groq
Tavily Search
Python-dotenv