from custom import get_out_dir, print_function_call, save_json
from json import load

@print_function_call
def word_align(filename):
    out_dir = get_out_dir(filename)
    with open(out_dir + f'{filename}-transcribed.json', 'r') as file: result = load(file)

    new_result = { 'segments': [] }
    for segment in result['transcription']:
        new_result['segments'].append({
           'start': segment['offsets']['from'] / 1000,
           'end': segment['offsets']['to'] / 1000,
           'text': segment['text'],
           'words': [
                {
                    'start': token['offsets']['from'] / 1000,
                    'end': token['offsets']['to'] / 1000,
                    'score': token['p'],
                    'word': token['text'],
                }
                for token
                in segment['tokens']
           ],
        })

    save_json(new_result, filename, 'word_aligned')
    return new_result

if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser("word_align.py")
  parser.add_argument("filename")

  args = vars(parser.parse_args())
  transcription = word_align(filename = args['filename'])

  print(transcription)
