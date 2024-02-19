all_files=$(git ls-tree --full-tree -r --name-only HEAD .)
py_files=$(echo "$all_files" | grep ".*\.py$")
nb_files=$(echo "$all_files" | grep ".*\.ipynb$" | grep -v "^notebooks")
python -m black $py_files $nb_files
python -m isort $py_files $nb_files
python -m flake8 $py_files
