import ast

import discord
import pandas as pd
from discord.ext import commands
from openai import AsyncOpenAI

intents = discord.Intents.default()
intents.message_content = True

extensions = ["ask_command", "update_embeddings_command", "reload_command"]

openai_api_key_file = open("openai_api_key.txt", "r")
openai_api_key = openai_api_key_file.read()
openai_api_key_file.close()


class CCNetGuideBot(commands.Bot):
    def __init__(self) -> None:
        super().__init__(command_prefix='$', description="Queries the CCNet wiki.", case_insensitive=True, intents=intents)
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        self.wiki_embeddings = pd.read_csv("wiki_embeddings.csv")

    async def on_ready(self):
        print(f'Logged in as {self.user}!')

        # Load embeddings
        print("Loading embeddings..")
        self.wiki_embeddings['embedding'] = self.wiki_embeddings['embedding'].apply(ast.literal_eval)
        print("Embeddings loaded!")

        # Sync commands
        await self.tree.sync()

        # Update presence
        game = discord.Game(f"/askwiki")
        await bot.change_presence(activity=game)

    async def setup_hook(self):
        for extension in extensions:
            await self.load_extension(extension)


token_file = open("token.txt", "r")
token = token_file.read()
token_file.close()
bot = CCNetGuideBot()
bot.run(token)
