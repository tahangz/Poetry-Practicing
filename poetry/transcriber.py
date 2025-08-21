import os
import sys
import time
import json
import threading
import tempfile
import numpy as np
from typing import Optional, List

from faster_whisper import WhisperModel

from .state import AppState
from .models import PoemData, MultiWordResult
from .audio import AudioProducer, NoiseEstimator, vad_decision
from .similarity import tokenize_speech, match_multiple_words, levenshtein_similarity
from .utils import float_to_int16, write_wav, rms_float32, normalize_text
from .config import *


class PoetryTranscriber:
    def __init__(self, model_name="small", device="cuda", compute_type="int8_float16",
                 chunk_seconds=1.0, overlap_seconds=0.5, poems_file="data/poems_1.json",
                 language="fr", out_file=None):

        self.chunk_seconds = float(chunk_seconds)
        self.overlap_seconds = float(overlap_seconds)
        self.language = language
        self.out_file = out_file

        # Load poems database
        self.poems = self._load_poems(poems_file)
        print(f"Loaded {len(self.poems)} poems from {poems_file}")

        # App state
        self.state = AppState.WAITING_FOR_TITLE
        self.current_poem: Optional[PoemData] = None
        self.current_word_index = 0
        self.correct_words = 0
        self.total_attempts = 0

        print(f"Loading Whisper model {model_name} on {device}...")
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        print("Model loaded.")

        self.producer = AudioProducer(samplerate=TARGET_SR)
        self.noise = NoiseEstimator(sr=TARGET_SR)
        self.prev_tail = np.array([], dtype=np.int16)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    # ---------------- Poem handling ----------------
    def _load_poems(self, poems_file: str) -> List[PoemData]:
        try:
            with open(poems_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            poems = [PoemData(poem_dict) for poem_dict in data]
            return poems
        except Exception as e:
            print(f"Error loading poems: {e}")
            return []

    def _find_poem_by_title(self, spoken_title: str) -> Optional[PoemData]:
        spoken_title = normalize_text(spoken_title)
        best_match, best_score = None, 0

        for poem in self.poems:
            title_score = levenshtein_similarity(spoken_title, poem.title)
            if title_score > TITLE_SIMILARITY_THRESHOLD and title_score > best_score:
                best_match, best_score = poem, title_score

            for title_word in poem.title_words:
                word_score = levenshtein_similarity(spoken_title, title_word)
                if word_score > 0.8 and title_score > 0.3:
                    combined_score = (word_score + title_score) / 2
                    if combined_score > best_score:
                        best_match, best_score = poem, combined_score

        return best_match if best_score > TITLE_SIMILARITY_THRESHOLD else None

    # ---------------- Word matching ----------------
    def _process_multi_word_input(self, spoken_text: str) -> MultiWordResult:
        if not self.current_poem or self.current_word_index >= len(self.current_poem.poem_words):
            return MultiWordResult()

        spoken_words = tokenize_speech(spoken_text)
        target_words = self.current_poem.poem_words

        result = match_multiple_words(
            spoken_words=spoken_words,
            target_words=target_words,
            start_position=self.current_word_index,
            similarity_threshold=WORD_SIMILARITY_THRESHOLD
        )
        return result

    def _print_match_results(self, result: MultiWordResult, spoken_text: str):
        print(f"\n🎤 You said: \"{spoken_text}\"")
        print(f"📊 Processed {result.total_words} word(s), {result.correct_count} correct")

        for i, match in enumerate(result.matches):
            status = "✅" if match.is_correct else "❌"
            print(f"   {status} Word {i+1}: \"{match.word}\" → \"{match.target}\" (similarity: {match.similarity:.2f})")

        if result.correct_count > 0:
            print(f"🎯 Advancing {result.advance_count} position(s)")

        if result.correct_count < result.total_words:
            failed_words = [m.target for m in result.matches if not m.is_correct]
            print(f"🔄 Try again for: {', '.join(failed_words)}")

    # ---------------- Status printing ----------------
    def _print_status(self):
        """Print current application status."""
        print("\n" + "="*60)
        if self.state == AppState.WAITING_FOR_TITLE:
            print(f"🎭 POETRY PRACTICE - Waiting for poem title")
            print(f"📝 Available poems ({len(self.poems)}):")
            for poem in self.poems[15:20]:  # Show 5
                print(f"   • \"{poem.title}\" by {poem.author}")
            if len(self.poems) > 5:
                print(f"   ... and {len(self.poems)-5} more")
            print(f"\n🗣️  Say the title of a poem you want to practice")
            
        elif self.state == AppState.TITLE_FOUND:
            print(f"✅ Found poem: \"{self.current_poem.title}\" by {self.current_poem.author}")
            print(f"📜 Full poem: {self.current_poem.poem}")
            print(f"\n🎯 Now recite the poem! You can say multiple words at once.")
            next_words = self.current_poem.poem_words[0:3]  # Show first 3 words
            print(f"🗣️  Start with: \"{' '.join(next_words)}\"")
            
        elif self.state == AppState.RECITING_POEM:
            if self.current_poem:
                progress = f"{self.current_word_index}/{len(self.current_poem.poem_words)}"
                accuracy = f"{self.correct_words}/{self.total_attempts}" if self.total_attempts > 0 else "0/0"
                
                print(f"📖 Poem: \"{self.current_poem.title}\"")
                print(f"📊 Progress: {progress} words | Accuracy: {accuracy}")
                
                if self.current_word_index < len(self.current_poem.poem_words):
                    # Show next few words
                    remaining = len(self.current_poem.poem_words) - self.current_word_index
                    next_count = min(3, remaining)
                    next_words = self.current_poem.poem_words[self.current_word_index:self.current_word_index + next_count]
                    print(f"🎯 Next word(s): \"{' '.join(next_words)}\"")
                    
                    # Show context (previous 3 and next 5 words)
                    context_start = max(0, self.current_word_index - 3)
                    context_end = min(len(self.current_poem.poem_words), self.current_word_index + 6)
                    context_words = self.current_poem.poem_words[context_start:context_end]
                    
                    context_display = []
                    for i, word in enumerate(context_words):
                        actual_idx = context_start + i
                        if actual_idx < self.current_word_index:
                            context_display.append(f"✓{word}")  # Completed words
                        elif actual_idx >= self.current_word_index and actual_idx < self.current_word_index + next_count:
                            context_display.append(f"[{word}]")  # Current target words
                        else:
                            context_display.append(word)  # Future words
                    print(f"📝 Context: {' '.join(context_display)}")
                
        elif self.state == AppState.POEM_COMPLETE:
            accuracy = (self.correct_words / self.total_attempts * 100) if self.total_attempts > 0 else 0
            print(f"🎉 POEM COMPLETE!")
            print(f"📊 Final score: {self.correct_words}/{self.total_attempts} ({accuracy:.1f}%)")
            print(f"🔄 Say another poem title to continue practicing")
        print("="*60)

    # ---------------- Control ----------------
    def start(self):
        self.producer.start()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self._print_status()

    def stop(self):
        self.stop_event.set()
        self.producer.stop()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)

    def _reset_for_new_title(self):
        self.state = AppState.WAITING_FOR_TITLE
        self.current_poem = None
        self.current_word_index = 0
        self.correct_words = 0
        self.total_attempts = 0

    # ---------------- Transcription ----------------
    def _process_transcription(self, text: str):
        if not text.strip():
            return

        if self.state == AppState.WAITING_FOR_TITLE:
            print(f"\n🎤 Heard: \"{text}\"")
            poem = self._find_poem_by_title(text)
            if poem:
                self.current_poem = poem
                self.state = AppState.TITLE_FOUND
                print(f"✅ Found poem: \"{poem.title}\"!")
                time.sleep(1)
                self.state = AppState.RECITING_POEM
                self._print_status()
            else:
                print(f"❌ No poem found matching \"{text}\"")

        elif self.state == AppState.RECITING_POEM:
            result = self._process_multi_word_input(text)
            if result.total_words == 0:
                return
            self.total_attempts += result.total_words
            self.correct_words += result.correct_count
            self._print_match_results(result, text)
            if result.advance_count > 0:
                self.current_word_index += result.advance_count
                if self.current_word_index >= len(self.current_poem.poem_words):
                    self.state = AppState.POEM_COMPLETE
                    time.sleep(1)
                    self._print_status()
                    time.sleep(3)
                    self._reset_for_new_title()
                    self._print_status()
                else:
                    self._print_status()

    # ---------------- Worker loop ----------------
    def _worker_loop(self):
        while not self.stop_event.is_set():
            float_chunk = self.producer.read_seconds(self.chunk_seconds)
            if float_chunk.size == 0:
                time.sleep(0.05)
                continue

            rms_val = rms_float32(float_chunk)
            pcm16 = float_to_int16(float_chunk)
            pcm_bytes = pcm16.tobytes()

            voiced_frames, total_frames = vad_decision(pcm_bytes, sample_rate=TARGET_SR,
                                                       aggressiveness=VAD_AGGRESSIVENESS,
                                                       frame_ms=VAD_FRAME_MS)
            voiced_ratio = (voiced_frames / total_frames) if total_frames > 0 else 0.0
            voiced_seconds = voiced_frames * (VAD_FRAME_MS / 1000.0)

            noise_floor = self.noise.estimated_noise_floor()
            energy_ok = (rms_val > noise_floor * NOISE_MULTIPLIER)
            vad_ok = (voiced_ratio >= SPEECH_RATIO_THRESHOLD) and (voiced_seconds >= MIN_VOICED_SECONDS)

            if not (vad_ok and energy_ok):
                if not vad_ok and NOISE_UPDATE_SKIP_VOICED:
                    self.noise.update_if_silent(rms_val)
                continue

            overlap_samples = int(self.overlap_seconds * TARGET_SR)
            concat = np.concatenate([self.prev_tail[-overlap_samples:], pcm16]) if self.prev_tail.size >= overlap_samples else pcm16

            tmpf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmpf.name
            tmpf.close()
            write_wav(tmp_path, concat, sr=TARGET_SR)

            try:
                segments, _ = self.model.transcribe(tmp_path, beam_size=5,
                                                    language=self.language,
                                                    temperature=0.0,
                                                    without_timestamps=True,
                                                    no_speech_threshold=0.6)

                transcribed_text = " ".join(seg.text.strip() for seg in segments
                                            if seg.end - seg.start >= MIN_SEGMENT_DURATION).strip()

                if transcribed_text:
                    self._process_transcription(transcribed_text)

            except Exception as e:
                print(f"Transcription error: {e}", file=sys.stderr)
            finally:
                os.unlink(tmp_path)

            self.prev_tail = pcm16
