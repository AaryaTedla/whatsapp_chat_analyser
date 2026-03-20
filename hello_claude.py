import anthropic

# This creates a client that knows your API key
client = anthropic.Anthropic()

# This is one API call — you send a message, you get a message back
message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain what a REST API is in 3 sentences."}
    ]
)

# The response lives here
print(message.content[0].text)
