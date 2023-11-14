import os

import pandas as pd
import tiktoken
from bs4 import BeautifulSoup, Comment
from discord.ext import commands

from util import num_tokens, GPT_MODEL, EMBEDDING_MODEL, SAVE_PATH


class UpdateEmbeddingsCommand(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    @commands.guild_only()
    @commands.has_permissions(administrator=True)
    async def update_embeddings(self, ctx):
        await ctx.send("Beginning embeddings update process.")
        sections = await sectionize_articles()
        strings = []
        for section in sections:
            strings.extend(split_strings_from_subsection(section))

        await ctx.send("Wiki articles have been split into strings.")
        embeddings = []
        BATCH_SIZE = 2048
        for batch_start in range(0, len(sections), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            # batch = sections[batch_start:batch_end]
            response = await self.bot.openai_client.embeddings.create(input=strings, model=EMBEDDING_MODEL)
            for i, be in enumerate(response.data):
                assert i == be.index  # double check embeddings are in same order as input
            batch_embeddings = [e.embedding for e in response.data]
            embeddings.extend(batch_embeddings)

        await ctx.send("Embeddings have been creating. Now saving to data frame.")
        df = pd.DataFrame({"text": strings, "embedding": embeddings})
        df.to_csv(SAVE_PATH, index=False)
        await ctx.send("Data frame save complete. Finished!")

    @update_embeddings.error
    async def on_embeddings_error(self, ctx, error):
        await ctx.send(f"Error: {error}")


# See https://help.openai.com/en/articles/6643167-how-to-use-openai-api-for-q-a-and-chatbot-apps
async def process_headings(soup, page_title: str) -> list[tuple[list[str], str]]:
    """Splits article into sections and subsections using its headings"""
    headings = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    elements = soup.find_all(headings)

    sections = []
    path = [f"{page_title} - "]  # Keeps track of the current path of headings

    for element in elements:
        # Determine the level of the current heading
        current_level = int(element.name[1])

        # Update the path to reflect the current heading's level
        path = path[:current_level - 1] + [element.get_text(strip=True)]

        # Gather all sibling elements until the next heading
        content = []
        for sibling in element.next_siblings:
            if sibling.name in headings and int(sibling.name[1]) <= current_level:
                break  # Stop at a heading of equal or higher level

            if sibling.name is not None and sibling.name != 'img':
                content.append(sibling.get_text(strip=True, separator=" "))

        # Add the current section to the list
        sections.append((path, ' '.join(content).replace("\n", " ")))

    return sections


async def sectionize(directory) -> list[tuple[list[str], str]]:
    """Sectionizes the page at the given directory"""
    with open(directory, 'r', encoding="utf8") as file:
        # Load the HTML file
        html_content = file.read()

        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find page title
        title = None
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            if 'title:' in comment:
                # Split the comment into lines
                lines = comment.split('\n')
                # Find the line with the title
                for line in lines:
                    if line.strip().startswith('title:'):
                        # Extract the text after "title:"
                        title = line.split('title:', 1)[1].strip()
                        break

        if title is None:
            return []

        processed_sections = await process_headings(soup, title)
        return processed_sections


async def sectionize_articles() -> list[tuple[list[str], str]]:
    sections = []
    for root, dirs, files in os.walk("wiki"):
        for file in files:
            if file.endswith('.html'):
                file_path = os.path.join(root, file)
                page_sections = await sectionize(file_path)
                sections.extend(page_sections)
    return sections


def halved_by_delimiter(string: str, delimiter: str = "\n") -> list[str, str]:
    """Split a string in two, on a delimiter, trying to balance tokens on each side."""
    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""]  # no delimiter found
    elif len(chunks) == 2:
        return chunks  # no need to search for halfway point
    else:
        total_tokens = num_tokens(string)
        halfway = total_tokens // 2
        best_diff = halfway
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[: i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)
            if diff >= best_diff:
                break
            else:
                best_diff = diff
        left = delimiter.join(chunks[:i])
        right = delimiter.join(chunks[i:])
        return [left, right]


def truncated_string(
    string: str,
    model: str,
    max_tokens: int,
    print_warning: bool = True,
) -> str:
    """Truncate a string to a maximum number of tokens."""
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")
    return truncated_string


# See https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_Wikipedia_articles_for_search.ipynb
def split_strings_from_subsection(
    subsection: tuple[list[str], str],
    max_tokens: int = 1000,
    model: str = GPT_MODEL,
    max_recursion: int = 5,
) -> list[str]:
    """
    Split a subsection into a list of subsections, each with no more than max_tokens.
    Each subsection is a tuple of parent titles [H1, H2, ...] and text (str).
    """
    titles, text = subsection
    string = "\n\n".join(titles + [text])
    num_tokens_in_string = num_tokens(string)
    # if length is fine, return string
    if num_tokens_in_string <= max_tokens:
        return [string]
    # if recursion hasn't found a split after X iterations, just truncate
    elif max_recursion == 0:
        return [truncated_string(string, model=model, max_tokens=max_tokens)]
    # otherwise, split in half and recurse
    else:
        titles, text = subsection
        for delimiter in ["\n\n", "\n", ". "]:
            left, right = halved_by_delimiter(text, delimiter=delimiter)
            if left == "" or right == "":
                # if either half is empty, retry with a more fine-grained delimiter
                continue
            else:
                # recurse on each half
                results = []
                for half in [left, right]:
                    half_subsection = (titles, half)
                    half_strings = split_strings_from_subsection(
                        half_subsection,
                        max_tokens=max_tokens,
                        model=model,
                        max_recursion=max_recursion - 1,
                    )
                    results.extend(half_strings)
                return results
    # otherwise no split was found, so just truncate (should be very rare)
    return [truncated_string(string, model=model, max_tokens=max_tokens)]


async def setup(bot):
    await bot.add_cog(UpdateEmbeddingsCommand(bot))
