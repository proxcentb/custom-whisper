from custom import get_out_dir, models_dir, print_function_call, get_audio
from subprocess import run, CalledProcessError 
from json import load
import os

@print_function_call
def transcribe(filename, model, threads , language, **kwargs):
    get_audio(filename)
    out_dir = get_out_dir(filename)

    model_full = f'ggml-{model}.bin'
    if not os.path.isfile(models_dir + model_full):
        from urllib.request import urlretrieve
        from tqdm import tqdm

        class DownloadProgressBar(tqdm):
            def update_to(self, b = 1, bsize = 1, tsize = None):
                if tsize is not None: self.total = tsize
                self.update(b * bsize - self.n)

        with DownloadProgressBar(unit = 'B', unit_scale = True, miniters = 1, desc = model_full) as t:
            urlretrieve(f'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{model_full}', filename = models_dir + model_full, reporthook = t.update_to)

    try: run([
       "./whisper.cpp/main", 
       "--model", models_dir + model_full,
        "-f", out_dir + f'{filename}.wav', 
        "-of", out_dir + f'{filename}-transcribed',
        "--threads", str(threads),
        # "--max-len", str(max_len),
        "--language", str(language),
        "--output-json-full",
        "--split-on-word",
        # "--print-progress",
        "--flash-attn",
    ])
    except CalledProcessError as e: raise RuntimeError(f"Failed transcribe audio: {e.stderr.decode()}") from e

    with open(out_dir + f'{filename}-transcribed.json', 'r') as file: result = load(file)

    return result

parser_arguments = {
  "--model": {
    "choices": ["tiny-q5_1", "tiny.en-q5_1", "base-q5_1", "base.en-q5_1", "small-q5_1", "small.en-q5_1", "medium-q5_0", "medium.en-q5_0", "large-v3-q5_0"],
    "default": "large-v3-q5_0",
    "required": False,
  },
  "--language": {
    "default": 'auto',
    "help": "Spoken language ('auto' for auto-detect).",
    "required": False,
  },
  "--threads": {
    "default": 4,
    "help": "Number of threads to use when running on CPU (4 by default)",
    "type": int,
    "required": False,
  },
#   "--max_len": {
#     "default": 50,
#     "help": "Maximum segment length in characters.",
#     "type": int,
#     "required": False,
#   },
}

def process_args(args):
  return {
    "filename": args['filename'],
    "model": args['model'],
    "language": args['language'],
    "threads": args['threads'],
    # "max_len": args['max_len'],
    "output-json-full": True,
    "split-on-word": True,
    "print-progress": True,
  }

if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser("transcribe.py")
  parser.add_argument("filename")
  for arg_name, arg_params in parser_arguments.items(): parser.add_argument(arg_name, **arg_params)

  args = vars(parser.parse_args())
  transcription = transcribe(**process_args(args))

  print(transcription)
