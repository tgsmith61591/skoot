#!/bin/bash

set -e -x

pip install twine wheel
twine upload --skip-existing dist/skoot-*
