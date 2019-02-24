#!/bin/bash
cd $2
while read line;\
     do echo $line;\
     if [[ $line == *"dataset"* ]];\
          then numbers=$(echo $line | grep -Eo '[0-9]+([.][0-9]+)?' | tr '\n' ' ' | sed 's/^[ \t]*//;s/[ \t]*$//' | cut -d ' ' -f 2,3 | tr ' ' ';');\
          echo $numbers >> $1;\
     fi;\
done
