from whisperx.audio import load_audio
from torch.cuda import empty_cache
from subprocess import run, CalledProcessError 
from gc import collect
from os import makedirs
from os.path import exists
from json import load, dump
from functools import wraps

models_dir = './cache/'
diarization_config_path = models_dir + 'pyannote/speaker-diarization-3.1.yaml'
hf_token = ''
# get it from here: https://huggingface.co/docs/hub/en/security-tokens
# you must agree with terms of
# https://huggingface.co/pyannote/speaker-diarization-3.1
# https://huggingface.co/pyannote/segmentation-3.0

def get_out_dir(filename): 
  out_dir = f'./files/{filename}_out/'
  if not exists(out_dir): makedirs(out_dir)
  return out_dir

def get_audio(filename):
  out_dir = get_out_dir(filename)

  if not exists(out_dir + f'{filename}.wav'):
    try: run(["ffmpeg", "-i", f"./files/{filename}", '-ac', '1', "-acodec", "pcm_s16le", "-ar", '16000', '-y', out_dir + f'{filename}.wav'])
    except CalledProcessError as e: raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

  audio_file_name = out_dir + f'{filename}.wav'
  return load_audio(audio_file_name)

def clean(object):
  collect(); empty_cache(); del object

def load_json(filename, step):
  out_dir = get_out_dir(filename)
  with open(out_dir + f'{filename}-{step}.json', 'r', encoding = 'utf-8') as file: data = load(file)
  return data

def save_json(data, filename, step):
  out_dir = get_out_dir(filename)
  with open(out_dir + f'{filename}-{step}.json', 'w', encoding = 'utf-8') as file:
    dump(data, file, ensure_ascii = False, indent = 4)

def initialize():
  from huggingface_hub import snapshot_download
  wespeaker_voxceleb_model_path = snapshot_download(token=hf_token, repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",  local_dir = models_dir + "pyannote/wespeaker-voxceleb-resnet34-LM")
  segmentation_model_path = snapshot_download(token=hf_token, repo_id="pyannote/segmentation-3.0",  local_dir = models_dir + "pyannote/segmentation-3.0")

  # from https://huggingface.co/pyannote/speaker-diarization-3.1/blob/main/config.yaml
  with open(diarization_config_path, "w") as file:
      file.write(f'''
version: 3.1.0

pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    embedding: {wespeaker_voxceleb_model_path}/pytorch_model.bin
    embedding_batch_size: 32
    embedding_exclude_overlap: true
    segmentation: {segmentation_model_path}/pytorch_model.bin
    segmentation_batch_size: 32

params:
  clustering:
    method: centroid
    min_cluster_size: 12
    threshold: 0.7045654963945799
  segmentation:
    min_duration_off: 0.0
      ''')
    
def print_function_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      print(f"\nExecuting {func.__name__} script.")
      for i, arg in enumerate(args): print(f"Argument {i}: {arg}")
      for key, value in kwargs.items(): print(f"{key}: {value}")
      print('')
      result = func(*args, **kwargs)
      # print(f"\nFinished {func.__name__} script with this result:\n{result}")
      return result
    return wrapper
