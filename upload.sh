#!/bin/bash

rm trie/trie-c trie/trie.pyc
git config --global user.email xiaohu@iastate.edu
git config --global user.name Rinoahu


git remote rm origin

git add -A .
git commit -m 'try to add remove function'
git remote add origin https://github.com/Rinoahu/trie
git pull origin master
git push origin master

git checkout master
