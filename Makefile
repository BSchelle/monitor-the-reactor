default: pylint pytest

pylint:
	find . -iname "*.py" -not -path "./tests/*" | xargs -n1 -I {} pylint --output-format=colorized {}; true

pytest:
	PYTHONDONTWRITEBYTECODE=1 pytest -v --color=yes

run_main:
	python -c 'from scripts.main import main; main()'

run_load_eval:
	python -c 'from scripts.main import load_eval; load_eval()'
