#!/bin/bash

i=0
hosts_file_lines=`grep -cv '$=' hosts_file.config`
while [ $i -lt $hosts_file_lines ]
do
	j=`expr $i + 1`
	if [ $j -eq $1 ]
	then
		cat hosts_file.config | head -$j | tail -1 | sed -e 's/:1:/:0:/g' >> hosts_reset.tmp
	else
		cat hosts_file.config | head -$j | tail -1 >> hosts_reset.tmp
	fi
	i=`expr $i + 1`
done
mv hosts_reset.tmp hosts_file.config
