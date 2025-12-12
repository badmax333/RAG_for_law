import argparse
from src.pipeline import RAGPipeline

def main():
    """Main function to run the CLI."""
    parser = argparse.ArgumentParser(description="RAG for Law: Ask questions about Russian Traffic Laws.")
    parser.add_argument("query", type=str, nargs="?", help="Your question about the traffic laws.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of context documents to retrieve.")
    parser.add_argument("--trace", action="store_true", help="Show the full reasoning trace of the generator.")
    
    args = parser.parse_args()

    # Initialize the pipeline
    # This will take some time on the first run as it downloads models and creates indexes.
    try:
        rag_pipeline = RAGPipeline()
    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}")
        return

    if args.query:
        # Run in single-shot mode
        query = args.query
        print(f"\n🔍 Query: {query}\n")
        result = rag_pipeline.run(query, top_k_content=args.top_k, include_trace=args.trace)
        
        print("--- Answer ---")
        print(result['answer'])
        print("\n--- Sources ---")
        for i, source in enumerate(result['context']):
            print(f"{i+1}. [{source['source']}] (Score: {source['score']:.4f})")
            print(f"   \"{source['text'][:100]}...\"")
        
        if args.trace and "trace" in result:
            print("\n--- Reasoning Trace ---")
            for i, step in enumerate(result['trace']):
                print(f"\nStep {i+1}: {step['step']}")
                print(f"Result: {step['result']}")
    else:
        # Run in interactive mode
        print("🚀 RAG for Law is ready! Type 'exit' to quit.")
        while True:
            try:
                query = input("\n🔍 Enter your question: ")
                if query.lower() in ['exit', 'quit']:
                    break
                
                if not query:
                    continue

                result = rag_pipeline.run(query, top_k_content=args.top_k, include_trace=args.trace)
                
                print("\n--- Answer ---")
                print(result['answer'])
                print("\n--- Sources ---")
                for i, source in enumerate(result['context']):
                    print(f"{i+1}. [{source['source']}] (Score: {source['score']:.4f}")

                if args.trace and "trace" in result:
                    print("\n--- Reasoning Trace ---")
                    for i, step in enumerate(result['trace']):
                        print(f"\nStep {i+1}: {step['step']}")
                        print(f"Result: {step['result']}")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
