import argparse
from src.pipeline import RAGPipeline

def main():
    """Main function to run the CLI."""
    parser = argparse.ArgumentParser(description="RAG for Law: Ask questions about Russian Traffic Laws.")
    parser.add_argument("query", type=str, nargs="?", help="Your question about the traffic laws.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of context documents to retrieve.")
    parser.add_argument("--trace", action="store_true", help="Show the full reasoning trace of the generator.")
    parser.add_argument("--no-cache", action="store_true", help="Disable query caching.")
    parser.add_argument("--cache-stats", action="store_true", help="Show cache statistics and exit.")

    args = parser.parse_args()

    # Initialize the pipeline
    # This will take some time on the first run as it downloads models and creates indexes.
    try:
        rag_pipeline = RAGPipeline(use_cache=not args.no_cache)
    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}")
        return

    # Show cache stats if requested
    if args.cache_stats:
        stats = rag_pipeline.get_cache_stats()
        print("\n=== CACHE STATISTICS ===")
        for key, value in stats.items():
            if key == "hit_rate":
                print(f"{key}: {value:.1%}")
            else:
                print(f"{key}: {value}")
        return

    if args.query:
        # Run in single-shot mode
        query = args.query
        print(f"\n🔍 Query: {query}\n")
        result = rag_pipeline.run(query, top_k_content=args.top_k, include_trace=args.trace)

        from_cache = "[CACHED]" if result.get('from_cache', False) else ""
        print(f"Query Type: {result.get('query_type', 'N/A')} | Latency: {result.get('latency_ms', 0):.1f}ms {from_cache}\n")

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

                from_cache = "[CACHED]" if result.get('from_cache', False) else ""
                print(f"\nQuery Type: {result.get('query_type', 'N/A')} | Latency: {result.get('latency_ms', 0):.1f}ms {from_cache}")
                print("\n--- Answer ---")
                print(result['answer'])
                print("\n--- Sources ---")
                for i, source in enumerate(result['context']):
                    print(f"{i+1}. [{source['source']}] (Score: {source['score']:.4f})")

                if args.trace and "trace" in result:
                    print("\n--- Reasoning Trace ---")
                    for i, step in enumerate(result['trace']):
                        print(f"\nStep {i+1}: {step['step']}")
                        print(f"Result: {step['result']}")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                # Show cache stats on exit
                stats = rag_pipeline.get_cache_stats()
                if stats.get("enabled", True):
                    print("\n=== CACHE STATISTICS ===")
                    print(f"Hits: {stats['hits']}, Misses: {stats['misses']}, Hit Rate: {stats['hit_rate']:.1%}")
                    print(f"Cache size: {stats['size']}/{stats['max_size']}")
                break
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
