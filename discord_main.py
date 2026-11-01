import os

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv


load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
TEST_GUILD_ID = int(os.getenv("TEST_GUILD_ID", "0"))

if not DISCORD_TOKEN:
    raise ValueError("Не найден DISCORD_TOKEN в файле .env")


class MaratDiscordBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()

        super().__init__(
            command_prefix="!",
            intents=intents
        )

    async def setup_hook(self):
        if TEST_GUILD_ID:
            guild = discord.Object(id=TEST_GUILD_ID)

            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)

            print(f"Зарегистрировано команд: {len(synced)}")
        else:
            synced = await self.tree.sync()

            print(f"Глобально зарегистрировано команд: {len(synced)}")


bot = MaratDiscordBot()


@bot.event
async def on_ready():
    print(f"Бот {bot.user} успешно запущен!")


@bot.tree.command(
    name="hello",
    description="Поздороваться с ботом"
)
async def hello(interaction: discord.Interaction):
    await interaction.response.send_message(
        f"Привет, {interaction.user.mention}! Я Марат-бот."
    )


@bot.tree.command(
    name="ping",
    description="Проверить соединение с Discord"
)
async def ping(interaction: discord.Interaction):
    latency = round(bot.latency * 1000)

    await interaction.response.send_message(
        f"Задержка бота: {latency} мс"
    )


@bot.tree.command(
    name="server",
    description="Получить информацию о сервере"
)
@app_commands.guild_only()
async def server(interaction: discord.Interaction):
    guild = interaction.guild

    embed = discord.Embed(
        title=f"Информация о сервере: {guild.name}",
        color=discord.Color.blue()
    )

    embed.add_field(
        name="Название",
        value=guild.name,
        inline=False
    )

    embed.add_field(
        name="ID сервера",
        value=str(guild.id),
        inline=False
    )

    embed.add_field(
        name="Количество участников",
        value=str(guild.member_count),
        inline=False
    )

    embed.add_field(
        name="Дата создания",
        value=discord.utils.format_dt(guild.created_at, style="F"),
        inline=False
    )

    if guild.icon:
        embed.set_thumbnail(url=guild.icon.url)

    await interaction.response.send_message(embed=embed)


@bot.tree.command(
    name="avatar",
    description="Показать аватар пользователя"
)
@app_commands.describe(
    user="Пользователь, чей аватар нужно показать"
)
async def avatar(
    interaction: discord.Interaction,
    user: discord.User = None
):
    target = user or interaction.user

    embed = discord.Embed(
        title=f"Аватар пользователя {target.display_name}",
        color=discord.Color.blue()
    )

    embed.set_image(url=target.display_avatar.url)

    await interaction.response.send_message(embed=embed)


bot.run(DISCORD_TOKEN)