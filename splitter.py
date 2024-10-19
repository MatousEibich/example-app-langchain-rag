# Split documents into chunks
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.docstore.document import Document


def split_documents(docs):
    # Define the headers to split on
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")
    ]

    # Initialize the MarkdownHeaderTextSplitter
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    contents = docs
    if docs and isinstance(docs[0], Document):
        # Extract the content from Document objects
        contents = [doc.page_content for doc in docs]

    # Split the markdown content using the MarkdownHeaderTextSplitter
    texts = []
    for content in contents:
        split_texts = text_splitter.split_text(content)
        texts.extend(split_texts)  # Combine all splits into a single list

    n_chunks = len(texts)
    print(f"Split into {n_chunks} chunks")
    return texts
