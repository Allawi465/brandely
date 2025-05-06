import os
import re
import asyncio
from dotenv import load_dotenv
import chainlit as cl
from agents import Agent, Runner, GuardrailFunctionOutput, input_guardrail
from pydantic import BaseModel
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# In-memory store for user session data
user_profiles = {}
user_history = {}

# ---------------- INSTRUCTIONS ---------------- #
instructions = """
You are Brandely, an AI branding strategist. Think, speak, and act like a seasoned creative director who guides founders through brand discovery in a calm, conversational way.
You always have access to the entire current conversation. Use what the user has already told you. Never ask the user to repeat something you can read above. Never say you have “no memory,” never reset the chat unless the user explicitly asks you to start over.
Your purpose is to help each user create or refine a cohesive brand identity. Work in logical order, but keep the flow natural—do not label steps or sound robotic. Move forward only after the current topic is clear and confirmed.
Hidden reasoning sequence (do not reveal as steps):
1. Understand the business—what it does, whom it serves, what makes it unique. If unclear, ask open questions until clear.
2. Define brand personality, tone, and core values. If none provided, suggest a few that genuinely fit and ask for feedback.
3. Handle naming. If a name exists, reflect on its fit; offer alternatives only if invited. If no name, propose two or three ideas with one-sentence rationales, based on confirmed business, tone, and values.
4. Address visual identity. Ask whether colors, fonts, or logos already exist. If they do, assess their fit; if not, suggest:
- a palette of three HEX colors with brief reasoning
- two or three Google Fonts with mood notes
- a concise visual-style sentence that ties back to values and tone.
5. Present a brand snapshot: name, business description, values, tone, color palette, font suggestions, and visual-style direction. Invite final tweaks.

Conversation rules:
- Tackle one theme at a time; wait for the user’s reply before moving on.
- Do not repeat questions already answered.
- Do not fall back to generic lines like “Tell me about your business” when you already know it—reference what you know and ask a precise follow-up.
- Speak in short, natural paragraphs; use bold labels or short lists only when they improve clarity.
- Avoid buzzwords, filler, or forced enthusiasm; be concise, confident, and genuinely helpful.
- Do not generate or promise logos or images—this is a text-only branding consultation.
- Stay focused on branding; politely decline requests outside that scope.

Make the branding process feel easy, collaborative, and intelligent.
"""

# ---------------- Brand Name Extraction ---------------- #
def extract_brand_name(text: str):
    match = re.search(r"(?:brand name is|my brand is)\s+([A-Za-z0-9\- ]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None

# ---------------- Guardrail Function ---------------- #
class InputCheck(BaseModel):
    is_safe: bool
    reason: str
    rewritten_prompt: str

@input_guardrail
async def no_sensitive_topics(ctx, agent, input_data):
    banned_keywords = [
        "president", "election", "government", "politics", "campaign", "senator", "parliament",
        "kill", "murder", "weapon", "violence", "how to kill", "plan to kill", "kill a", "shoot", "stab",
        "drugs", "cocaine", "heroin", "weed", "meth", "hasj", "hash", "coke", "crack",
        "crime", "criminal", "jail", "prison", "steal", "cartel", "smuggle", "rape", "kidnap",
        "money laundering", "launder money"
    ]

    lowered = input_data.lower()
    if any(phrase in lowered for phrase in banned_keywords):
        return GuardrailFunctionOutput(
            output_info=InputCheck(
                is_safe=False,
                reason="Blocked due to banned phrase.",
                rewritten_prompt=input_data
            ),
            tripwire_triggered=False
        )

    return GuardrailFunctionOutput(
        output_info=InputCheck(
            is_safe=True,
            reason="OK",
            rewritten_prompt=input_data
        ),
        tripwire_triggered=False
    )

# ---------------- Brandely Agent ---------------- #
my_agent = Agent(
    name="Brandely",
    instructions=instructions,
    input_guardrails=[no_sensitive_topics]
)

# ---------------- STARTERS ---------------- #
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Clarify my brand positioning",
            message="I'm building a new brand and need help clarifying what makes it distinct. Can you guide me through defining my positioning and target audience?",
        ),
        cl.Starter(
            label="Define tone and personality",
            message="I want my brand to have a clear personality and tone of voice. Can you help me shape that so it feels consistent across all channels?",
        ),
        cl.Starter(
            label="Develop a strong brand name",
            message="I need a compelling brand name that reflects my values and market. Can you guide me through generating and evaluating options?",
        ),
        cl.Starter(
            label="Create a visual identity foundation",
            message="I'm ready to define my visual style. Can you help me choose a color palette and fonts that align with my brand's values and tone?",
        ),
    ]

# ---------------- MESSAGE HANDLER ---------------- #
@cl.on_message
async def handle_message(message: cl.Message):
    try:
        session_id = message.author

        # Use in-memory chat history instead of Redis
        if session_id not in user_history:
            user_history[session_id] = []

        # Add the new user input to the in-memory history
        user_history[session_id].append({"role": "user", "content": message.content})

        # Format history into OpenAI-compatible messages
        formatted_history = [{"role": "user", "content": msg["content"]} for msg in user_history[session_id]]

        # Optional: Trim history to avoid token limits
        formatted_history = formatted_history[-40:]

        # Typing indicator in Chainlit
        thinking = cl.Message(author="Brandely", content="Brandely is thinking…")
        await thinking.send()

        # Directly specify the GPT-4o model here
        openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

        # Make the chat completion request using GPT-4o
        response = await openai_client.chat.completions.create(
            model="gpt-4o",  # Specify the GPT-4o model
            messages=[ 
                {"role": "system", "content": instructions},
                *formatted_history
            ],
            temperature=0.7
        )

        # Extract reply content from OpenAI's response
        reply_content = response.choices[0].message.content
        print(f"[DEBUG] OpenAI reply: {reply_content}")

        # Stream the response to the UI
        response_msg = cl.Message(author="Brandely", content="", id=thinking.id)
        await asyncio.sleep(0.3)
        for tok in reply_content:
            await asyncio.sleep(0.01)  # Simulated typing speed
            await response_msg.stream_token(tok)

        await response_msg.update()

        # Save user and assistant messages back into in-memory history
        user_history[session_id].append({"role": "assistant", "content": reply_content})

    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()