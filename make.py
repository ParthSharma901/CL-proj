import re
import os


def clean_whatsapp_chat(chat_text):
    # This pattern matches different WhatsApp timestamp formats
    timestamp_pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}(?::\d{2})?\s+(?:AM|PM)\s+-\s+'

    # Split by timestamp but keep the timestamp
    chunks = re.split(f'({timestamp_pattern})', chat_text)

    if chunks and not chunks[0].strip():
        chunks.pop(0)

    messages = []

    # Process chunks in pairs (timestamp + message)
    i = 0
    while i < len(chunks) - 1:
        timestamp = chunks[i]
        message_text = chunks[i + 1]
        i += 2

        if not re.match(timestamp_pattern, timestamp):
            continue

        # Extract the sender and message content
        sender_message_match = re.match(r'(.+?):\s+(.*)', message_text, re.DOTALL)

        if not sender_message_match:
            continue

        sender = sender_message_match.group(1).strip()
        content = sender_message_match.group(2).strip()

        if (any(skip_text in content for skip_text in [
            '<Media omitted>',
            'This message was deleted',
            '<This message was edited>',
            'Follow this link to join',
            'https://chat.whatsapp.com/',
            'Using this group\'s invite link'
        ]) or
                any(skip_text in message_text for skip_text in [
                    'joined using this group',
                    'created group',
                    'Messages and calls are end-to-end encrypted',
                    'added',
                    'removed',
                    'left',
                    'changed this group\'s settings',
                    'pinned a message',
                    'changed the group description'
                ])):
            continue

        if re.match(r'\+\d{1,3}\s+\d+', sender):
            continue

        # Skip messages with only links
        if re.match(r'^https?://\S+$', content.strip()):
            continue

        # Clean up the content - replace multiple spaces and newlines
        content = re.sub(r'\s+', ' ', content).strip()

        if content:
            messages.append(content)

    return '\n'.join(messages)


# Input and output files
input_file = "Whatsapp_official.txt"
output_file = "cleaned_chat_official.txt"

# Ensure the input file exists
if not os.path.exists(input_file):
    print(f"Error: Input file '{input_file}' not found.")
    exit(1)

try:
    # Read, clean, and write
    with open(input_file, "r", encoding="utf-8") as f:
        chat_data = f.read()

    cleaned_chat = clean_whatsapp_chat(chat_data)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_chat)

    print(f"Cleaned chat saved to {output_file}")
    print(f"Original characters: {len(chat_data)}")
    print(f"Cleaned characters: {len(cleaned_chat)}")
    print(f"Extracted messages: {len(cleaned_chat.strip().split('\\n'))}")

    # Print the first few lines of the cleaned chat for verification
    if cleaned_chat:
        print("\nFirst few cleaned messages:")
        for i, line in enumerate(cleaned_chat.split('\n')[:5]):
            print(f"{i + 1}: {line}")
    else:
        print("\nWarning: No messages were extracted!")

except Exception as e:
    print(f"Error processing file: {e}")