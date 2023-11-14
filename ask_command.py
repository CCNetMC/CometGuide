import discord
import pandas as pd
from discord import app_commands
from discord.ext import commands
from openai import AsyncOpenAI
from scipy import spatial

from util import num_tokens, EMBEDDING_MODEL, GPT_MODEL

PROMPT = """
You are a wiki staff member on CCNet, a Minecraft server. You refer to it as 'CCNet'. 
Your role is to clearly and concisely advise players on mechanics and answer queries concerning them.
Anything else is out of your scope and irrelevant. You must redirect these to support tickets on the CCNet Discord server.
You must never judge whether an action is permitted or disallowed by the server rules; you must defer to CCNet staff.
You must always respond in a positive and encouraging tone. 
"""


class AskCommand(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    # 30 second per-server cooldown on using this command.
    @app_commands.checks.cooldown(1, 30.0, key=lambda i: i.guild_id)
    @app_commands.command(name="askwiki", description="Ask a question to the CCNet Wiki.")
    @app_commands.guild_only()
    @app_commands.describe(question="The question to ask.",)
    async def ask_command(self, interaction: discord.Interaction, question: str):
        await interaction.response.defer(thinking=True)
        flagged = await is_flagged(self.bot.openai_client, question)
        if flagged:
            await interaction.followup.send("Sorry, your question was flagged as being inappropriate by OpenAI.")
            return
        try:
            response = await ask(self.bot.openai_client, question, self.bot.wiki_embeddings)
        except Exception as e:
            return await interaction.followup.send(f"Sorry, there was an error: {e}")

        em = discord.Embed(title=f"Question: {question}", description=response, colour=0xebab34)
        em.set_footer(text="CometGuide may make mistakes. Always check the wiki if unsure.")
        await interaction.followup.send(embed=em)

    @ask_command.error
    async def on_ask_error(self, interaction: discord.Interaction, error: discord.app_commands.AppCommandError):
        if isinstance(error, discord.app_commands.CommandOnCooldown):
            await interaction.response.send_message(str(error), ephemeral=True)
        else:
            await interaction.response.send_message(str(error))


async def strings_ranked_by_relatedness(
        openai_client: AsyncOpenAI,
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = await openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


async def query_message(
        openai_client: AsyncOpenAI,
        query: str,
        df: pd.DataFrame,
        model: str,
        token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = await strings_ranked_by_relatedness(openai_client, query, df)
    introduction = 'Use the below articles about the CCNet Minecraft server to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer to that on the CCNet wiki."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    index = 0
    for string in strings:
        # Ignore less relevant text
        if relatednesses[index] < 0.8:
            continue
        index += 1
        next_article = f'\n\nCCNet Wiki section:\n"""\n{string}\n"""'
        if num_tokens(message + next_article + question, model=model) > token_budget:
            break
        else:
            message += next_article
    return message + question


async def is_flagged(openai_client: AsyncOpenAI, query: str) -> bool:
    response = await openai_client.moderations.create(input=query)
    return response.results[0].flagged


async def ask(
    openai_client: AsyncOpenAI,
    query: str,
    df: pd.DataFrame,
    model: str = GPT_MODEL,
    token_budget: int = 2048,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = await query_message(openai_client, query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": message},
    ]
    response = await openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.5
    )
    response_message = response.choices[0].message.content
    return response_message


async def setup(bot):
    await bot.add_cog(AskCommand(bot))
