#!/bin/bash

FILE=$1
nbLineToDelete=$2

tail -n +"$2" "$FILE" > "$FILE.tmp" && mv "$FILE.tmp" "$FILE"
