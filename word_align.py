from torchaudio import pipelines
from torch.cuda import is_available
from whisperx.alignment import align, DEFAULT_ALIGN_MODELS_TORCH, DEFAULT_ALIGN_MODELS_HF, Wav2Vec2Processor, Wav2Vec2ForCTC
from custom import models_dir, get_audio, clean, save_json, load_json, print_function_call

@print_function_call
def word_align(filename, align_kwargs):
  transcription = load_json(filename, 'transcribed')
  audio = get_audio(filename)

  align_model, metadata = load_align_model(language_code = transcription["language"], device = align_kwargs['device'], model_dir = models_dir)
  aligned_transcription = align(transcription["segments"], align_model, metadata, audio, **align_kwargs)
  clean(align_model)

  save_json(aligned_transcription, filename, 'word_aligned')
  return aligned_transcription

def load_align_model(language_code, device, model_name = None, model_dir = None):
    if model_name is None:
        # use default model
        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        elif language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            print(f"There is no default alignment model set for this language ({language_code}).\
                Please find a wav2vec2.0 model finetuned on this language in https://huggingface.co/models, then pass the model name in --align_model [MODEL_NAME]")
            raise ValueError(f"No default align-model for language: {language_code}")

    if model_name in pipelines.__all__:
        pipeline_type = "torchaudio"
        bundle = pipelines.__dict__[model_name]
        align_model = bundle.get_model(dl_kwargs={"model_dir": model_dir}).to(device)
        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
    else:
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir = model_dir)
            align_model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir = model_dir)
        except Exception as e:
            print(e)
            print(f"Error loading model from huggingface, check https://huggingface.co/models for finetuned wav2vec2.0 models")
            raise ValueError(f'The chosen align_model "{model_name}" could not be found in huggingface (https://huggingface.co/models) or torchaudio (https://pytorch.org/audio/stable/pipelines.html#id14)')
        pipeline_type = "huggingface"
        align_model = align_model.to(device)
        labels = processor.tokenizer.get_vocab()
        align_dictionary = {char.lower(): code for char,code in processor.tokenizer.get_vocab().items()}

    align_metadata = {"language": language_code, "dictionary": align_dictionary, "type": pipeline_type}

    return align_model, align_metadata

def process_args(args):
  return {
    "filename": args['filename'],
    "align_kwargs": {
      'device': "cuda" if is_available() else "cpu",
      "print_progress": True,
      # "return_char_alignments": False,
      # "interpolate_method": "nearest",
    },
  }

if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser("word_align.py")
  parser.add_argument("filename")

  args = vars(parser.parse_args())
  transcription = word_align(**process_args(args))

  print(transcription)
