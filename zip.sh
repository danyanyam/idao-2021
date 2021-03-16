# скрипт, который архивирует сабмишен, исключая перечисленные папки
zip -r ../track_2_submission.zip * -x "venv/*" -x "__pycache__/*" -x "tests/*" -x "tests2/*" -x ".vscode"