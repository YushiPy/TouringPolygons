from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from faster_whisper import WhisperModel


def timestamp(seconds: float) -> str:
	milliseconds = int(round(seconds * 1000))
	hours, remainder = divmod(milliseconds, 3_600_000)
	minutes, remainder = divmod(remainder, 60_000)
	secs, millis = divmod(remainder, 1000)
	return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def extract_audio(video: Path, audio: Path) -> None:
	audio.parent.mkdir(parents=True, exist_ok=True)
	subprocess.run(
		[
			"ffmpeg",
			"-y",
			"-i",
			str(video),
			"-vn",
			"-map",
			"0:a:0",
			"-c:a",
			"copy",
			str(audio),
		],
		check=True,
	)


def transcribe(
	audio: Path,
	transcript_txt: Path,
	transcript_json: Path,
	model_name: str,
	language: str,
	prompt: str | None,
) -> None:
	model = WhisperModel(model_name, device="cpu", compute_type="int8")
	segments_iterator, info = model.transcribe(
		str(audio),
		language=language,
		beam_size=5,
		vad_filter=True,
		word_timestamps=False,
		initial_prompt=prompt,
		condition_on_previous_text=True,
	)

	segments: list[dict[str, float | str]] = []
	for segment in segments_iterator:
		text = segment.text.strip()
		if not text:
			continue
		item: dict[str, float | str] = {
			"start": segment.start,
			"end": segment.end,
			"text": text,
		}
		segments.append(item)
		print(f"[{segment.start:8.2f} -> {segment.end:8.2f}] {text}", flush=True)

	transcript_txt.parent.mkdir(parents=True, exist_ok=True)
	transcript = "\n\n".join(
		f"[{timestamp(float(item['start']))} - {timestamp(float(item['end']))}] {item['text']}"
		for item in segments
	)
	transcript_txt.write_text(transcript + "\n", encoding="utf-8")
	payload = {
		"audio": str(audio),
		"model": model_name,
		"language": info.language,
		"language_probability": info.language_probability,
		"duration": info.duration,
		"segments": segments,
	}
	transcript_json.write_text(
		json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
	)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Extrai o áudio de uma reunião em vídeo e o transcreve."
	)
	parser.add_argument("video", type=Path)
	parser.add_argument("--model", default="small")
	parser.add_argument("--language", default="pt")
	parser.add_argument("--output-dir", type=Path)
	parser.add_argument("--prompt")
	parser.add_argument(
		"--skip-audio",
		action="store_true",
		help="Reutiliza o arquivo gravação-áudio.m4a existente.",
	)
	args = parser.parse_args()

	video = args.video.resolve()
	output_dir = (args.output_dir or video.parent).resolve()
	audio = output_dir / "gravação-áudio.m4a"
	transcript_txt = output_dir / "transcrição-bruta.txt"
	transcript_json = output_dir / "transcrição-bruta.json"
	if not args.skip_audio:
		extract_audio(video, audio)
	transcribe(
		audio,
		transcript_txt,
		transcript_json,
		args.model,
		args.language,
		args.prompt,
	)


if __name__ == "__main__":
	main()
