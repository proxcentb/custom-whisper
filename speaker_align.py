from pandas import read_json
from whisperx import assign_word_speakers
from custom import get_out_dir, save_json, load_json, print_function_call

@print_function_call
def speaker_align(filename):
  diarized_transcription = read_json(get_out_dir(filename) + f'{filename}-diarization.json')
  aligned_transcription = load_json(filename, 'word_aligned')

  combined_transcription = assign_word_speakers(diarized_transcription, aligned_transcription)

  save_json(combined_transcription, filename, 'diarized')
  return combined_transcription

if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser("speaker_align.py")
  parser.add_argument("filename")

  args = vars(parser.parse_args())
  transcription = speaker_align(filename = args['filename'])

  print(transcription)
