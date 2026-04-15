from fastmcp import FastMCP
import wikipedia
import duckduckgo_search

mcp = FastMCP("Search Server")

@mcp.tool()
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for a given topic to get factual background information."""
    try:
        return wikipedia.summary(query, sentences=3)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Ambiguous term. Options: {', '.join(e.options[:5])}"
    except Exception as e:
        return f"Error searching Wikipedia: {e}"

@mcp.tool()
def search_web(query: str) -> str:
    """Search the web for real-time information using DuckDuckGo."""
    try:
        ddgs = duckduckgo_search.DDGS()
        results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        formatted_results = "\n\n".join([f"Title: {r['title']}\nSummary: {r['body']}\nURL: {r['href']}" for r in results])
        return formatted_results
    except Exception as e:
        return f"Error searching web: {e}"

@mcp.tool()
def create_markdown_report(filename: str, content: str) -> str:
    """Creates and saves a markdown report with the given content to the hard drive."""
    try:
        if not filename.endswith('.md'):
            filename += '.md'
        
        import os
        # Ensure we write to a safe directory, e.g. the current directory 'data' or root
        filepath = os.path.join(".", filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully saved markdown report to {filepath}"
    except Exception as e:
        return f"Error saving markdown report: {e}"

@mcp.tool()
def add_to_database(text: str, source: str = "External web source") -> str:
    """Adds a new text snippet to the vector database so the internal researcher can retrieve it later."""
    try:
        import sys
        import os
        from pathlib import Path
        import faiss
        import json
        import numpy as np
        
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
            
        from scripts.rag.embedding import get_embeddings
        from scripts.rag.indexing import load_index
        
        index_path = str(project_root / "data" / "my_index.faiss")
        chunks_path = str(project_root / "data" / "chunks.json")
        
        # Create a new chunk entry
        new_chunk = {
            "source": source,
            "page": 1,
            "text": text,
            "char_count": len(text)
        }
        
        # Initialize or load existing index
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            index, chunks = load_index(index_path, chunks_path)
        else:
            return "Error: Database index files not found."
            
        # Get embeddings from openai
        embedding = get_embeddings([text])[0]
        vector = np.array([embedding]).astype("float32")
        faiss.normalize_L2(vector)
        
        # Update
        index.add(vector)
        chunks.append(new_chunk)
        
        # Save
        faiss.write_index(index, index_path)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
            
        return f"Successfully added new information from '{source}' to the vector database."
    except Exception as e:
        return f"Error adding to database: {e}"

@mcp.tool()
def load_pdf_to_database(pdf_path: str, source_name: str) -> str:
    """Reads a PDF file from the local file system or an absolute path, extracts text, chunks it, embeds it, and adds it to the vector database."""
    try:
        import sys
        import os
        import fitz  # PyMuPDF
        import faiss
        import json
        import numpy as np
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
            
        from scripts.rag.embedding import get_embeddings
        from scripts.rag.indexing import load_index
        from scripts.rag.chunking import chunk_corpus
        
        if not os.path.exists(pdf_path):
            return f"Error: The PDF file {pdf_path} was not found."
            
        # 1. Extract text
        pdf_document = fitz.open(pdf_path)
        pages_data = []
        for page_num in range(len(pdf_document)):
            text = pdf_document[page_num].get_text()
            cleaned_text = ' '.join(text.split())
            if len(cleaned_text) > 50:
                pages_data.append({
                    "source": source_name,
                    "page": page_num + 1,
                    "char_count": len(cleaned_text),
                    "text": cleaned_text
                })
        pdf_document.close()
        
        if not pages_data:
            return "No text could be extracted from this PDF."
            
        # 2. Chunk text
        chunks = chunk_corpus(pages_data, chunk_size=1000, overlap_size=200)
        
        # 3. Load DB
        index_path = str(project_root / "data" / "my_index.faiss")
        chunks_path = str(project_root / "data" / "chunks.json")
        index, existing_chunks = load_index(index_path, chunks_path)
        
        # 4. Embed and Insert
        embeddings = get_embeddings([c["text"] for c in chunks])
        vectors = np.array(embeddings).astype("float32")
        faiss.normalize_L2(vectors)
        index.add(vectors)
        existing_chunks.extend(chunks)
        
        # 5. Save DB
        faiss.write_index(index, index_path)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(existing_chunks, f, indent=2)
            
        return f"Successfully processed PDF '{source_name}', added {len(chunks)} new chunks to the database."
    except Exception as e:
        return f"Error loading PDF into database: {e}"

@mcp.tool()
def generate_graph(data_json: str, title: str, chart_type: str = "bar", filename: str = "chart.png") -> str:
    """Generates a graph from a JSON string dictionary mapping labels to numerical values (e.g. '{"Apple": 150, "Google": 200}').
    chart_type can be 'bar', 'line', or 'pie'.
    filename must end with .png. 
    It saves the graph to disk and returns a confirmation message with the file path."""
    import json
    import matplotlib
    matplotlib.use('Agg') # Safe for non-interactive backend
    import matplotlib.pyplot as plt
    try:
        data = json.loads(data_json)
        labels = list(data.keys())
        values = list(data.values())
        
        plt.figure(figsize=(10, 6))
        
        if chart_type == 'bar':
            plt.bar(labels, values, color='skyblue')
        elif chart_type == 'line':
            plt.plot(labels, values, marker='o', color='green')
        elif chart_type == 'pie':
            plt.pie(values, labels=labels, autopct='%1.1f%%')
        else:
            return "Error: chart_type must be 'bar', 'line', or 'pie'."
            
        plt.title(title)
        if chart_type != 'pie':
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
        if not filename.endswith('.png'):
            filename += '.png'
            
        plt.savefig(filename)
        plt.close()
        return f"Successfully generated '{chart_type}' chart and saved to {filename}"
        
    except Exception as e:
        return f"Error generating chart: {e}"

if __name__ == "__main__":
    mcp.run()
