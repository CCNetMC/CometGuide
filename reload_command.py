from discord.ext import commands


class ReloadCommand(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    @commands.guild_only()
    @commands.has_permissions(administrator=True)
    async def reload(self, ctx, arg: str):
        if arg in self.bot.extensions.keys():
            await self.bot.reload_extension(arg)
            await ctx.send(f":white_check_mark: **Reloaded the cog `{arg}`.**")
        else:
            await ctx.send(f":warning: **Invalid extension**. Valid extensions: `{list(self.bot.extensions.keys())}`")

    @reload.error
    async def on_reload_error(self, ctx, error):
        await ctx.send(f"Error: {error}")


async def setup(bot):
    await bot.add_cog(ReloadCommand(bot))
