from custom import get_out_dir, load_json, print_function_call

@print_function_call
def generate_html(filename):
  out_dir = get_out_dir(filename)
  with open(f'./template.html', 'r') as file:
    html_template = file.read()

  with open(out_dir + f'{filename}.html', 'w') as file:
    file.write(
      html_template.replace(
        '// python replace',
        f'''
          setDialog({load_json(filename, 'diarized')});
          setMedia("{f"../{filename}" if filename.split('.')[-1].lower() in ['mpg', 'mp2', 'mpeg', 'mpe', 'mpv', 'mp4'] else f"{filename}.wav"}")
        ''' ,
      )
    )

if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser("generate_html.py")
  parser.add_argument("filename")

  args = vars(parser.parse_args())
  generate_html(filename = args['filename'])
