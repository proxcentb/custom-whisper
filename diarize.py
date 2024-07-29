from os.path import exists
from torch.cuda import is_available
from numpy import ndarray
from pandas import DataFrame
from torch import device as torch_device, from_numpy
from pyannote.audio import Pipeline
from custom import get_out_dir, diarization_config_path, get_audio, initialize, clean, print_function_call

@print_function_call
def diarize(filename, device, diarize_model_kwargs):
  out_dir = get_out_dir(filename)
  audio = get_audio(filename)

  diarize_model = DiarizationPipeline(device = device, config_path = diarization_config_path)
  diarized_transcription = diarize_model(audio, **diarize_model_kwargs)
  clean(diarize_model)

  diarized_transcription.to_json(out_dir + f'{filename}-diarization.json', indent=4)
  return diarized_transcription

class DiarizationPipeline:
  def __init__(self, config_path = None, device = "cpu"):
    if isinstance(device, str): device = torch_device(device)
    if not exists(config_path): initialize()
    self.model = Pipeline.from_pretrained(config_path).to(device)

  def __call__(self, audio: ndarray, **kwargs):
    audio_data = { 'waveform': from_numpy(audio[None, :]), 'sample_rate': 16000 }
    segments = self.model(audio_data, **kwargs)
    diarize_df = DataFrame(segments.itertracks(yield_label = True), columns = ['segment', 'label', 'speaker'])
    diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
    diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
    return diarize_df

parser_arguments = {
  "--num_speakers": { "type": int, "required": False },
  "--min_speakers": { "type": int, "required": False },
  "--max_speakers": { "type": int, "required": False },
}

def process_args(args):
  return {
    "filename": args['filename'],
    "device": "cuda" if is_available() else "cpu",
    "diarize_model_kwargs" : {
      "num_speakers": args['num_speakers'],
      "min_speakers": args['min_speakers'],
      "max_speakers": args['max_speakers'],
    }
  }

if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser("diarize.py")
  parser.add_argument("filename")
  for arg_name, arg_params in parser_arguments.items(): parser.add_argument(arg_name, **arg_params)

  args = vars(parser.parse_args())
  transcription = diarize(**process_args(args))

  print(transcription)
