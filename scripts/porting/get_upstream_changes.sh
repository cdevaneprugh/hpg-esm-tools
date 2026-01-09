#!/bin/bash

git -c core.quotePath=false log refs/tags/${OLD_TAG}..refs/tags/${NEW_TAG}   --no-merges   --date=iso   --pretty='format:@@@ %H | %ad | %an | %s'   --name-only > upstream_changes.txt
