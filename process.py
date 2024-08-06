from argparse import ArgumentParser
import transcribe
import word_align
import diarize 
import speaker_align
import generate_html

if __name__ == "__main__":
  parser = ArgumentParser("process.py")
  parser.add_argument("filename")
  parser.add_argument('--transcribe', action='store_true', help='Transcribe the audio', default=False)
  parser.add_argument('--word_align', action='store_true', help='Align words in the transcript', default=False)
  parser.add_argument('--diarize', action='store_true', help='Diarize the audio', default=False)
  parser.add_argument('--speaker_align', action='store_true', help='Align speakers in the audio', default=False)
  parser.add_argument('--generate_html', action='store_true', help='Generate HTML file', default=False)
  parser.add_argument('--all', action='store_true', help='Perform all actions', default=False)

  for arg_name, arg_params in transcribe.parser_arguments.items(): parser.add_argument(arg_name, **arg_params)
  for arg_name, arg_params in diarize.parser_arguments.items(): parser.add_argument(arg_name, **arg_params)

  args = vars(parser.parse_args())
  if args['all']:
    args['transcribe'] = True
    args['word_align'] = True
    args['diarize'] = True
    args['speaker_align'] = True
    args['generate_html'] = True
  
  if args['transcribe']: 
    transcribe.transcribe(
      **transcribe.process_args({
        'filename': args['filename'],
        **{ arg_name[2:]: args[arg_name[2:]] for arg_name in list(transcribe.parser_arguments) },
      })
    )

  if args['word_align']: 
    word_align.word_align(filename = args['filename'])

  if args['diarize']: 
    diarize.diarize(
      **diarize.process_args({
        'filename': args['filename'],
        **{ arg_name[2:]: args[arg_name[2:]] for arg_name in list(diarize.parser_arguments) },
      })
    )

  if args['speaker_align']: 
    speaker_align.speaker_align(filename = args['filename'])

  if args['generate_html']: 
    generate_html.generate_html(filename = args['filename'])
