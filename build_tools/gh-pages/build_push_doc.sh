#!/bin/bash

set -e

# this is a hack, but we have to make sure we're only ever running this from
# the top level of the package and not in the subdirectory...
if [[ ! -d skoot/__check_build ]]; then
    echo "This must be run from the skoot project directory"
    exit 3
fi

# get the running branch
branch=$(git symbolic-ref --short HEAD)

# we only really want to do this from master
if [[ ${branch} != "master" ]]; then
    echo "This must be run from the master branch"
    exit 5
fi

# make sure no untracked changes in git
if [[ -n $(git status -s) ]]; then
    echo "You have untracked changes in git"
    exit 7
fi

# setup the project
python setup.py install

# cd into docs, make them
cd doc
make clean html EXAMPLES_PATTERN=ex_*
cd ..

# move the docs to the top-level directory, stash for checkout
mv doc/_build/html ./
# html/ will stay there actually...
git stash

# checkout gh-pages, remove everything but .git, pop the stash
git checkout gh-pages
# remove all files that are not in the .git dir
find . -not -name ".git/*" -type f -maxdepth 1 -delete
# remove the remaining directories
rm -r .cache/ build/ build_tools/ doc/ examples/ skoot/
# we need this empty file for git not to try to build a jekyll project
touch .nojekyll
mv html/* ./
rm -r html/

# add everything, get ready for commit
git add --all
git commit -m "[ci skip] publishing updated documentation..."
git push origin gh-pages

# switch back to master
git checkout master
