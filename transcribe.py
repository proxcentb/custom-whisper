from whisperx import load_model
from torch.cuda import is_available
from whisperx.asr import WhisperModel
from custom import save_json, clean, models_dir, get_audio, print_function_call

@print_function_call
def transcribe(filename, asr_options, vad_options, whisper_model_kwargs, whisper_transcribe_kwargs):
  whisper_model = load_model(
    whisper_arch = whisper_model_kwargs['model_size_or_path'],
    device = whisper_model_kwargs['device'],
    asr_options = asr_options,
    vad_options = vad_options,
    model = WhisperModel(
      download_root = models_dir,
      local_files_only = False,
      **whisper_model_kwargs
    ),
  )
  transcription = whisper_model.transcribe(audio = get_audio(filename), **whisper_transcribe_kwargs)
  clean(whisper_model)

  save_json(transcription, filename, 'transcribed')
  return transcription

parser_arguments = {
  "--model": {
    "choices": ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v1", "large-v2", "large-v3"],
    "default": "medium",
    "required": False,
  },
  "--compute_type": {
    "choices": ["int8", "int8_float32", "int8_float16", "int8_bfloat16", "int16", "float16", "bfloat16", "float32"],
    "help": "Change to int8 if low on GPU memory (may reduce accuracy).",
    "default": "int8",
    "required": False,
  },
  "--batch_size": {
    "help": "Reduce if low on GPU memory.",
    "default": 8,
    "type": int,
    "required": False,
  },
  "--cpu_threads": {
    "default": 4,
    "help": "Number of threads to use when running on CPU (4 by default)",
    "type": int,
    "required": False,
  },
}

def process_args(args):
  return {
    "filename": args['filename'],
    "whisper_model_kwargs": {
      "device": "cuda" if is_available() else "cpu",
      "model_size_or_path":  args['model'],
      "compute_type": args['compute_type'],
      "cpu_threads": args["cpu_threads"],

      # When transcribe() is called from multiple Python threads, having multiple workers 
      # enables true parallelism when running the model (concurrent calls to self.model.generate() will run in parallel).
      # This can improve the global throughput at the cost of increased memory usage.
      # "num_workers": 1,

      # Device ID to use. The model can also be loaded on multiple GPUs by passing a list of IDs (e.g. [0, 1, 2, 3]). 
      # In that case, multiple transcriptions can run in parallel when transcribe() is called from multiple Python threads (see also num_workers).
      # "device_index": 0,
    },
    "whisper_transcribe_kwargs": {
      "batch_size": args['batch_size'],
      "print_progress": True,
      # "num_workers": 0,
      # "chunk_size": 30,
    },
    "asr_options": {
      # "beam_size": 5,
      # "best_of": 5,
      # "patience": 1,
      # "length_penalty": 1,
      # "repetition_penalty": 1,
      # "no_repeat_ngram_size": 0,
      # "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
      # "compression_ratio_threshold": 2.4,
      # "log_prob_threshold": -1.0,
      # "no_speech_threshold": 0.6,
      # "condition_on_previous_text": False,
      # "prompt_reset_on_temperature": 0.5,
      # "initial_prompt": None,
      # "prefix": None,
      # "suppress_blank": True,
      # "suppress_tokens": [-1],
      # "without_timestamps": True,
      # "max_initial_timestamp": 0.0,
      # "word_timestamps": False,
      # "prepend_punctuations": "\"'“¿([{-",
      # "append_punctuations": "\"'.。,，!！?？:：”)]}、",
      # "suppress_numerals": False,
      # "max_new_tokens": None,
      # "clip_timestamps": None,
      # "hallucination_silence_threshold": None,
    },
    "vad_options": {
      # "device": device, !
      # "vad_onset": 0.5,
      # "vad_offset": 0.363,
      # "use_auth_token": None,
      # "model_fp": None,
    },
  }

if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser("transcribe.py")
  parser.add_argument("filename")
  for arg_name, arg_params in parser_arguments.items(): parser.add_argument(arg_name, **arg_params)

  args = vars(parser.parse_args())
  transcription = transcribe(**process_args(args))

  print(transcription)
