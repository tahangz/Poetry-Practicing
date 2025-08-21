"""
poetry_main.py — Entry point for Poetry Pronunciation Learning App
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import time
import argparse

from poetry.transcriber import PoetryTranscriber
from poetry.config import WORD_SIMILARITY_THRESHOLD, MAX_WORDS_PER_CHUNK


def parse_args():
    p = argparse.ArgumentParser(description="Poetry Pronunciation Learning App with Multi-word Support")
    p.add_argument("--model", default="small", help="Whisper model size")
    p.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    p.add_argument("--compute", default="int8_float16", help="Compute type")
    p.add_argument("--chunk", type=float, default=1.0, help="Audio chunk seconds")
    p.add_argument("--overlap", type=float, default=0.5, help="Overlap seconds")
    p.add_argument("--poems", default="data/poems_1.json", help="Path to poems JSON file")
    p.add_argument("--lang", default="fr", help="Language code (fr for French)")
    p.add_argument("--out", default=None, help="Output file for logs")
    p.add_argument("--word-threshold", type=float, default=WORD_SIMILARITY_THRESHOLD, help="Word similarity threshold (0.0-1.0)")
    p.add_argument("--max-words", type=int, default=MAX_WORDS_PER_CHUNK, help="Maximum words to process per chunk")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.poems):
        print(f"Error: Poems file '{args.poems}' not found!")
        sys.exit(1)

    # Update global settings from args
    from poetry import config
    config.WORD_SIMILARITY_THRESHOLD = args.word_threshold
    config.MAX_WORDS_PER_CHUNK = args.max_words

    transcriber = PoetryTranscriber(
        model_name=args.model,
        device=args.device,
        compute_type=args.compute,
        chunk_seconds=args.chunk,
        overlap_seconds=args.overlap,
        poems_file=args.poems,
        language=args.lang,
        out_file=args.out
    )

    try:
        transcriber.start()
        print("\n🎭 POETRY PRACTICE - Application is running!")
        print("⏹️  Press Ctrl+C to stop\n")

        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Keep practicing your poetry!")
    finally:
        transcriber.stop()


if __name__ == "__main__":
    main()
