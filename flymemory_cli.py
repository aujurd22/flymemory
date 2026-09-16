"""FlyMemory CLI — persistent associative memory for the AI agent.

Usage:
    python flymemory_cli.py remember "text to remember" [--tags tag1,tag2]
    python flymemory_cli.py recall "query text" [--top 3]
    python flymemory_cli.py stats
"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.path.join(os.path.dirname(__file__), "flymemory.pkl")

def load_store():
    from flymemory import FlyMemoryStore
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return FlyMemoryStore(n_compartments=8, n_bits=4096)

def save_store(store):
    with open(DB_PATH, "wb") as f:
        pickle.dump(store, f)

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    store = load_store()
    if cmd == "remember" and len(sys.argv) >= 3:
        text = sys.argv[2]
        mid = store.remember(text, "")
        save_store(store)
        print(f"[remembered #{mid}] {text[:80]}")
    elif cmd == "recall" and len(sys.argv) >= 3:
        query = sys.argv[2]
        results = store.recall(query, top_k=3)
        if results:
            for mem, ov in results:
                print(f"  [{ov:.2f}] {mem.context[:100]}")
        else:
            print("  no matching memories")
    elif cmd == "stats":
        print(f"Total memories: {store.size}")
    else:
        print(__doc__)

if __name__ == "__main__":
    main()
