Discord bot that uses the OpenAI embeddings and chat completion APIs to answer queries using the CCNet wiki.

User input is filtered using the OpenAI moderation API, which is free.

This bot requires three files:
1. `openai_api_key.txt`: contains the OpenAI API key
2. `token.txt`: contains the bot token
3. `wiki_embeddings.csv`: embeddings generated from the CCNet Wiki.

There are three commands:
1. `$reload <file name>` - reloads the cog in the given file
2. `$update_embeddings` - regenerates `wiki_embeddings.csv` using a wiki.js [local file system dump](https://wiki.ccnetmc.com/a/storage). This must be located in a "wiki" subdirectory in the project root.
3. `/askwiki <query>`

Updating the wiki involves updating the `wiki` subdirectory with a fresh wiki dump and running `$update_embeddings`.

`requirements.txt` is updated using `pipreqs`.