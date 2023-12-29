#!/bin/sh
LINE=$1
PET=$(sed -n ${LINE}p /tmp/pets.input)
echo My favorite pet is $PET.