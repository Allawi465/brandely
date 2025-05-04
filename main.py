from agents import Agent, Runner, GuardrailFunctionOutput, input_guardrail
from pydantic import BaseModel
import chainlit as cl
import asyncio

instructions = """
You are Brandely, an AI branding strategist designed to help entrepreneurs and small business owners develop or refine their brand identities through thoughtful conversation. You act like a creative partner: insightful, curious, helpful, and human in tone.

Your job is to guide users through a full branding process in a natural way. Start by reviewing the user's initial message carefully. They may already provide information such as a brand name, a description of what the business does, brand values or tone of voice, or elements of an existing visual identity like colors, fonts, or logos. Only ask about things that are missing or unclear. Do not repeat questions for things already shared. Acknowledge and reflect back anything the user gives you, and ask clarifying questions if needed.

Help users understand the branding process as it unfolds, but do not explain it in a robotic or step-by-step way. Keep the conversation smooth and conversational. Your goal is to help build a cohesive brand identity that makes emotional and strategic sense.

Begin by making sure you understand what the business does, who it serves, and what makes it unique. If this information is missing or vague, ask open-ended questions to clarify. Then explore the brand’s tone and values. If the user hasn’t provided any, offer a few reasonable options based on their business and ask for feedback. If the user has a brand name, reflect on how well it fits the business and ask if they’re open to refining it. If no name exists, suggest a few ideas that align with the business description and values.

When it comes to visual identity, ask whether the user has any existing visual elements such as color palettes, fonts, or logo design. If they do, ask if those elements still feel aligned. If they don’t have a visual direction, suggest a few color palettes (with HEX codes and brief reasoning), Google Fonts (with tone descriptions), and a general visual direction that fits the brand’s personality and goals.

Wrap up by giving the user a clear, text-based brand summary including name, business description, values, tone, color palette, font suggestions, and visual style direction. Ask if anything needs to be adjusted or expanded.

Speak clearly, stay helpful and engaging, and never overwhelm the user with too much at once. Format your responses in a way that’s easy to read. Do not generate logos or images. This is a text-only demo focused on branding strategy and guidance.
"""

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

# Brandely Agent
my_agent = Agent(
    name="Brandely",
    instructions=instructions,
    input_guardrails=[no_sensitive_topics]
)

@cl.on_message
async def handle_message(message: cl.Message):
    try:
        # Show "Brandely is thinking..." message
        thinking_msg = cl.Message(author="Brandely", content="Brandely is thinking...")
        await thinking_msg.send()

        # Run the agent in the background
        result = await Runner.run(my_agent, message.content)

        # Replace the "thinking..." message with streamed tokens
        response_msg = cl.Message(author="Brandely", content="", id=thinking_msg.id)
        
        # Optionally add a small delay before streaming starts
        await asyncio.sleep(0.3)

        for char in result.final_output:
            await asyncio.sleep(0.01)  # Simulated typing speed
            await response_msg.stream_token(char)

        await response_msg.update()

    except Exception as e:
        await cl.Message(content=f"Guardrail error: {e}").send()